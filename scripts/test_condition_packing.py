import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.dataset import TrajectoryDataset
from src.conditioning import rasterize_trajectories
from src.condition_encoder import ConditionEncoder


def pack_latents(latents, patch_size=1, patch_size_t=1):
    B, C, F, H, W = latents.shape

    latents = latents.reshape(
        B,
        C,
        F // patch_size_t,
        patch_size_t,
        H // patch_size,
        patch_size,
        W // patch_size,
        patch_size,
    )

    latents = latents.permute(
        0, 2, 4, 6, 1, 3, 5, 7
    )

    return latents.flatten(4, 7).flatten(1, 3)


device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 1. Load trajectory
# -------------------------

dataset = TrajectoryDataset(
    split="train",
    num_frames=33,
    random_crop=False,
)

item = dataset[0]

trajectory = item["trajectory"].unsqueeze(0).to(device)
visible = item["visible"].unsqueeze(0).to(device)

_, _, H, W = item["video"].shape

# -------------------------
# 2. Rasterize trajectory
# -------------------------

condition = rasterize_trajectories(
    trajectory,
    visible,
    H,
    W,
)

# B,F,C,H,W -> B,C,F,H,W
condition = condition.permute(0, 2, 1, 3, 4)

print("Condition:", condition.shape)

# -------------------------
# 3. Encode condition
# -------------------------

encoder = ConditionEncoder().to(device)

with torch.no_grad():
    condition_latent = encoder(condition)

print("Condition latent:", condition_latent.shape)

# -------------------------
# 4. Pack into LTX tokens
# -------------------------

condition_tokens = pack_latents(condition_latent)

print("Condition tokens:", condition_tokens.shape)