import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import TrajectoryDataset
from src.conditioning import rasterize_trajectories


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TrajectoryDataset(
    split="train",
    num_frames=33,
    random_crop=False,
)

item = dataset[0]

trajectory = item["trajectory"].unsqueeze(0).to(device)
visible = item["visible"].unsqueeze(0).to(device)

video = item["video"].unsqueeze(0).to(device)

print("trajectory:", trajectory.shape)
print("video:", video.shape)

B, F, _, H, W = video.shape

condition = rasterize_trajectories(
    trajectory,
    visible,
    H,
    W,
)

print("condition:", condition.shape)