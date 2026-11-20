import torch
import torch.nn as nn
import torch.nn.functional as F

from src.condition_encoder import EntityConditionEncoder
from src.conditioning import entity_footprint, render_entity_condition


def pack_latents(latents):
    """[B, C, F, H, W] -> [B, F*H*W, C] (LTX uses patch_size 1 for both space and time)."""
    B, C, F_, H, W = latents.shape
    return latents.permute(0, 2, 3, 4, 1).reshape(B, F_ * H * W, C)


def unpack_latents(tokens, num_frames, height, width):
    """[B, F*H*W, C] -> [B, C, F, H, W]."""
    B, _, C = tokens.shape
    return tokens.reshape(B, num_frames, height, width, C).permute(0, 4, 1, 2, 3)


def sincos_3d(f, h, w, dim, device):
    """(f*h*w, dim) fixed sin/cos position code over (time, y, x); dim is split in three parts."""
    d = dim // 3 // 2 * 2
    def axis(n, k):
        pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
        freq = torch.exp(-torch.arange(0, k, 2, device=device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / k))
        a = pos * freq[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)  # (n, k)
    t, y, x = axis(f, d), axis(h, d), axis(w, d)
    grid = torch.cat([
        t[:, None, None].expand(f, h, w, d), y[None, :, None].expand(f, h, w, d), x[None, None].expand(f, h, w, d)
    ], dim=-1).reshape(f * h * w, 3 * d)
    if grid.shape[1] < dim:
        grid = F.pad(grid, (0, dim - grid.shape[1]))
    return grid


class MotionControl(nn.Module):
    """Trajectory + object -> condition tokens -> injected into a frozen LTX transformer.

    The condition (object-sized footprint, displacement, the object's own feature; built on the latent grid)
    goes through a small conv encoder and `cond_blocks` small transformer blocks. Zero-initialised projections
    add it to the hidden states after every `inject_every`-th LTX block (ControlNet style, via forward hooks).
    These are the only trainable weights; LTX stays frozen.
    """

    def __init__(self, dim=128, num_blocks=0, hidden_dim=0, inject_every=4, relative=False, max_inject=0.0,
                 cond_blocks=0, cond_heads=4, local_inject=0.0, reference=False):
        super().__init__()
        # >0: inject only around the object: the residual is multiplied by a Gaussian footprint whose radius is
        # local_inject x the object's radius (the background stays exactly as LTX would produce it). 0 = everywhere
        self.local_inject = local_inject
        # reference: the VAE latents of a rendered "drag video" (src/reference.py) are added to the encoded condition,
        # a pixel-aligned hint of what the output should look like (like a ControlNet's canny / depth image)
        self.reference = reference
        if reference:
            self.ref_proj = nn.Conv3d(dim, 128, 1)
        self.relative = relative  # inject in units of the hidden state's own RMS (absolute injection is easily drowned out)
        self.max_inject = max_inject  # never inject more than this fraction of the hidden RMS (0 = no cap)
        self.condition_encoder = EntityConditionEncoder(in_channels=3 + dim)
        # small randomly initialised transformer blocks: the condition tokens see each other before injection
        self.cond_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=cond_heads, dim_feedforward=4 * dim,
                                       dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(cond_blocks)
        ])
        self.block_ids = list(range(0, num_blocks, inject_every))
        self.projs = nn.ModuleList()
        for _ in self.block_ids:
            proj = nn.Linear(dim, hidden_dim)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.projs.append(proj)

        self._ctx = None  # per-block residuals for the forward pass in flight
        self._ratio = {}  # block index -> |injection| / |hidden state| from the latest forward (diagnostic)

    def condition_tokens(self, trajectory, visible, latent_shape, entity):
        """trajectory (B, F, 2) normalised, visible (B, F), latent_shape (F', H', W'),
        entity {"embedding": (B, 128), "radius": (B,), optional "shape": (B, 1, 4H', 4W')}.
        Returns (B, F'*H'*W', 128) float32 tokens."""
        Fl, Hl, Wl = latent_shape
        condition = render_entity_condition(
            trajectory, visible, entity["embedding"], entity["radius"], Hl, Wl, shape=entity.get("shape")
        )
        latent = self.condition_encoder(condition)
        if latent.shape[2:] != (Fl, Hl, Wl):
            latent = F.interpolate(latent, size=(Fl, Hl, Wl), mode="trilinear", align_corners=False)
        if self.reference:
            latent = latent + self.ref_proj(entity["ref_latent"].to(latent.dtype))  # (B, 128, F', H', W')

        tokens = pack_latents(latent)
        if len(self.cond_blocks):
            tokens = tokens + sincos_3d(Fl, Hl, Wl, tokens.shape[-1], tokens.device)[None].to(tokens.dtype)
            for blk in self.cond_blocks:
                tokens = blk(tokens)
        return tokens

    def near_object(self, trajectory, visible, latent_shape, entity):
        """(B, S, 1) in [0, 1]: how close each latent token is to the object (widened by local_inject)."""
        Fl, Hl, Wl = latent_shape
        fp = entity_footprint(trajectory, visible, entity["radius"] * self.local_inject, Hl, Wl)  # (B, F, Hl, Wl)
        idx = torch.linspace(0, fp.shape[1] - 1, Fl).round().long().to(fp.device)  # latent frame k shows video frame ~8k
        return fp[:, idx].reshape(fp.shape[0], -1, 1)

    def residuals(self, trajectory, visible, latent_shape, keep_mask=None, cond_mask=None, entity=None):
        """One residual per injection point. keep_mask (B, S) bool: tokens that stay untouched (the image frame);
        cond_mask (B,): 1 = use the trajectory, 0 = drop it (condition dropout)."""
        cond = self.condition_tokens(trajectory, visible, latent_shape, entity)
        near = self.near_object(trajectory, visible, latent_shape, entity) if self.local_inject > 0 else None
        out = []
        for proj in self.projs:
            r = proj(cond)
            if near is not None:
                r = r * near
            if keep_mask is not None:
                r = r * (~keep_mask).unsqueeze(-1).to(r.dtype)
            if cond_mask is not None:
                r = r * cond_mask.view(-1, 1, 1).to(r.dtype)
            out.append(r)
        return out

    def attach(self, transformer):
        """Install forward hooks that add the stored residuals after the chosen blocks."""

        def make_hook(i):
            def hook(module, args, output):
                if self._ctx is None:
                    return output
                r = self._ctx[i].to(output.dtype)
                if self.relative:
                    rms_h = output.detach().float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
                    r = r * rms_h.to(output.dtype)
                    if self.max_inject > 0:
                        # safety brake: a high-lr Adam step can multiply the injection in a few updates and
                        # wreck LTX's activations; this bounds the damage per forward pass
                        rms_r = r.detach().float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
                        r = r * (self.max_inject * rms_h / rms_r.clamp(min=1e-8)).clamp(max=1.0).to(r.dtype)
                with torch.no_grad():
                    num = r.float().pow(2).mean().sqrt()
                    den = output.float().pow(2).mean().sqrt().clamp(min=1e-6)
                    if float(num) > 0:  # skip condition-dropout forwards, which inject exactly 0
                        self._ratio[i] = num / den
                return output + r
            return hook

        self._handles = [
            transformer.transformer_blocks[b].register_forward_hook(make_hook(i))
            for i, b in enumerate(self.block_ids)
        ]

    def set_context(self, residuals):
        """Must stay set until backward() finishes (gradient checkpointing re-runs the hooks then).
        set_context(None) switches injection off."""
        self._ctx = residuals

    def gate_value(self):
        """Mean |projection weight|: one number to watch in the training log."""
        return float(torch.stack([p.weight.detach().abs().mean() for p in self.projs]).mean())

    def injection_ratio(self):
        """RMS of what the adapter adds / RMS of the hidden state, averaged over the injection points (latest
        forward pass). ~0 = ignored, ~0.05-0.3 = a real influence."""
        vals = [float(v) for v in self._ratio.values()]
        return sum(vals) / len(vals) if vals else float("nan")

    def trainable_state(self):
        state = {"condition_encoder": self.condition_encoder.state_dict(), "projs": self.projs.state_dict()}
        if len(self.cond_blocks):
            state["cond_blocks"] = self.cond_blocks.state_dict()
        if self.reference:
            state["ref_proj"] = self.ref_proj.state_dict()
        return state

    def load_trainable_state(self, state):
        self.condition_encoder.load_state_dict(state["condition_encoder"])
        self.projs.load_state_dict(state["projs"])
        if len(self.cond_blocks):
            if "cond_blocks" not in state:
                raise ValueError("checkpoint has no cond_blocks but the model was built with them")
            self.cond_blocks.load_state_dict(state["cond_blocks"])
        if self.reference:
            if "ref_proj" not in state:
                raise ValueError("checkpoint was trained without the reference condition")
            self.ref_proj.load_state_dict(state["ref_proj"])
