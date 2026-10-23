import torch
import torch.nn as nn
import torch.nn.functional as F

from src.adapter import MotionAdapter
from src.condition_encoder import ConditionEncoder
from src.conditioning import rasterize_trajectories


def pack_latents(latents):
    """[B, C, F, H, W] -> [B, F*H*W, C] (LTX uses patch_size 1 for both space and time)."""
    B, C, F_, H, W = latents.shape
    return latents.permute(0, 2, 3, 4, 1).reshape(B, F_ * H * W, C)


def unpack_latents(tokens, num_frames, height, width):
    """[B, F*H*W, C] -> [B, C, F, H, W]."""
    B, _, C = tokens.shape
    return tokens.reshape(B, num_frames, height, width, C).permute(0, 4, 1, 2, 3)


class MotionControl(nn.Module):
    """Trajectory -> condition tokens -> added onto the (noisy) LTX video tokens.

    These are the only trainable weights; LTX stays frozen.
    """

    def __init__(self, dim=128):
        super().__init__()
        self.condition_encoder = ConditionEncoder()
        self.adapter = MotionAdapter(dim=dim)

    def condition_tokens(self, trajectory, visible, latent_shape):
        """
        trajectory: (B, F, 2) normalized, visible: (B, F)
        latent_shape: (F', H', W') of the video latent grid

        Returns (B, F'*H'*W', 128) float32 tokens.
        """
        Fl, Hl, Wl = latent_shape

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

    def residual(self, trajectory, visible, latent_shape, keep_mask=None):
        """What the adapter adds to the video tokens; zero on tokens in keep_mask
        (the clean conditioning image must stay exactly as encoded)."""
        cond = self.condition_tokens(trajectory, visible, latent_shape)
        out = self.adapter.scale * self.adapter.net(cond)

        if keep_mask is not None:
            out = out * (~keep_mask).unsqueeze(-1).to(out.dtype)

        return out

    def forward(self, video_tokens, trajectory, visible, latent_shape, keep_mask=None):
        residual = self.residual(trajectory, visible, latent_shape, keep_mask)
        return video_tokens.float() + residual

    def trainable_state(self):
        return {
            "condition_encoder": self.condition_encoder.state_dict(),
            "adapter": self.adapter.state_dict(),
        }

    def load_trainable_state(self, state):
        self.condition_encoder.load_state_dict(state["condition_encoder"])
        self.adapter.load_state_dict(state["adapter"])
