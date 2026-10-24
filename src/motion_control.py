import torch
import torch.nn as nn
import torch.nn.functional as F

from src.adapter import MotionAdapter
from src.condition_encoder import ConditionEncoder, EntityConditionEncoder
from src.conditioning import rasterize_trajectories, render_entity_condition


def pack_latents(latents):
    """[B, C, F, H, W] -> [B, F*H*W, C] (LTX uses patch_size 1 for both space and time)."""
    B, C, F_, H, W = latents.shape
    return latents.permute(0, 2, 3, 4, 1).reshape(B, F_ * H * W, C)


def unpack_latents(tokens, num_frames, height, width):
    """[B, F*H*W, C] -> [B, C, F, H, W]."""
    B, _, C = tokens.shape
    return tokens.reshape(B, num_frames, height, width, C).permute(0, 4, 1, 2, 3)


class MotionControl(nn.Module):
    """Trajectory -> condition tokens -> injected into a frozen LTX transformer.

    mode="input":      the condition is added once to the (noisy) video tokens.
    mode="multilayer": zero-initialised projections add the condition to the hidden
                       states after every `inject_every`-th transformer block
                       (ControlNet / T2I-Adapter style, without copying the blocks).

    condition="points":  heatmap + dx + dy of the tracked point, rasterized in pixel space.
    condition="entity":  object-sized Gaussian footprint + displacement + the object's own
                         feature (masked mean of the image latent), built on the latent grid.

    These are the only trainable weights; LTX stays frozen.
    """

    def __init__(self, dim=128, mode="input", num_blocks=0, hidden_dim=0, inject_every=4, condition="points",
                 relative=False):
        super().__init__()
        if mode not in ("input", "multilayer"):
            raise ValueError(f"unknown adapter mode {mode}")
        if condition not in ("points", "entity"):
            raise ValueError(f"unknown condition type {condition}")

        self.mode = mode
        self.condition = condition
        self.relative = relative  # multilayer: inject in units of the hidden state's own RMS
        self.condition_encoder = (
            ConditionEncoder() if condition == "points" else EntityConditionEncoder(in_channels=3 + dim)
        )

        if mode == "input":
            self.adapter = MotionAdapter(dim=dim)
        else:
            self.block_ids = list(range(0, num_blocks, inject_every))
            self.projs = nn.ModuleList()
            for _ in self.block_ids:
                proj = nn.Linear(dim, hidden_dim)
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
                self.projs.append(proj)

        self._ctx = None  # per-block residuals for the forward pass in flight

    # ------------------------------------------------------------------
    # Condition
    # ------------------------------------------------------------------

    def condition_tokens(self, trajectory, visible, latent_shape, entity=None):
        """
        trajectory: (B, F, 2) normalized, visible: (B, F)
        latent_shape: (F', H', W') of the video latent grid
        entity: {"embedding": (B, 128), "radius": (B,)}, required for condition="entity"

        Returns (B, F'*H'*W', 128) float32 tokens.
        """
        Fl, Hl, Wl = latent_shape

        if self.condition == "entity":
            if entity is None:
                raise ValueError("condition='entity' needs entity={'embedding', 'radius'}")
            condition = render_entity_condition(
                trajectory, visible, entity["embedding"], entity["radius"], Hl, Wl
            )
        else:
            # rasterize at exactly 32x the latent grid so the encoder lands on it
            condition = rasterize_trajectories(
                trajectory.float(), visible, height=Hl * 32, width=Wl * 32
            )
            condition = condition.permute(0, 2, 1, 3, 4).float()  # B,3,F,H,W

        latent = self.condition_encoder(condition)

        if latent.shape[2:] != (Fl, Hl, Wl):
            latent = F.interpolate(
                latent, size=(Fl, Hl, Wl), mode="trilinear", align_corners=False
            )

        return pack_latents(latent)

    @staticmethod
    def _gate(out, keep_mask, cond_mask):
        """Zero the residual on the clean image tokens and on dropped-out samples."""
        if keep_mask is not None:
            out = out * (~keep_mask).unsqueeze(-1).to(out.dtype)
        if cond_mask is not None:
            out = out * cond_mask.view(-1, 1, 1).to(out.dtype)
        return out

    def residuals(self, trajectory, visible, latent_shape, keep_mask=None, cond_mask=None, entity=None):
        """What gets injected. A tensor for mode=input, a list (one per block) for multilayer.

        keep_mask: (B, S) bool, True for tokens that must stay untouched (the image frame)
        cond_mask: (B,) 1 = use the trajectory, 0 = drop it (condition dropout)
        """
        cond = self.condition_tokens(trajectory, visible, latent_shape, entity)

        if self.mode == "input":
            out = self.adapter.scale * self.adapter.net(cond)
            return self._gate(out, keep_mask, cond_mask)

        return [self._gate(proj(cond), keep_mask, cond_mask) for proj in self.projs]

    # ------------------------------------------------------------------
    # Injection into LTX
    # ------------------------------------------------------------------

    def attach(self, transformer):
        """Install forward hooks that add the stored residuals after the chosen blocks."""
        if self.mode != "multilayer":
            return

        def make_hook(i):
            def hook(module, args, output):
                if self._ctx is None:
                    return output
                r = self._ctx[i].to(output.dtype)
                if self.relative:
                    # the residual stream's scale is unknown and grows with depth; expressing the
                    # injection relative to it makes a weight of ~0.1 mean "10% of the activations"
                    r = r * output.detach().float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt().to(output.dtype)
                return output + r
            return hook

        self._handles = [
            transformer.transformer_blocks[b].register_forward_hook(make_hook(i))
            for i, b in enumerate(self.block_ids)
        ]

    def set_context(self, residuals):
        """Must stay set until backward() finishes: gradient checkpointing re-runs the
        blocks (and these hooks) then. Call set_context(None) to switch injection off."""
        self._ctx = residuals if self.mode == "multilayer" else None

    def apply_input(self, video_tokens, residual):
        """Token-space injection for mode=input; a no-op for multilayer."""
        if self.mode == "input":
            return video_tokens.float() + residual
        return video_tokens.float()

    # ------------------------------------------------------------------
    # Logging / checkpoints
    # ------------------------------------------------------------------

    def gate_params(self):
        return [self.adapter.scale] if self.mode == "input" else []

    def gate_value(self):
        """One number to watch in the training log."""
        if self.mode == "input":
            return self.adapter.scale.item()
        return float(torch.stack([p.weight.abs().mean() for p in self.projs]).mean())

    def trainable_state(self):
        state = {"condition_encoder": self.condition_encoder.state_dict()}
        if self.mode == "input":
            state["adapter"] = self.adapter.state_dict()
        else:
            state["projs"] = self.projs.state_dict()
        return state

    def load_trainable_state(self, state):
        self.condition_encoder.load_state_dict(state["condition_encoder"])
        if self.mode == "input":
            self.adapter.load_state_dict(state["adapter"])
        else:
            self.projs.load_state_dict(state["projs"])
