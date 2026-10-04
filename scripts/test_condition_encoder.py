import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.dataset import TrajectoryDataset
from src.conditioning import rasterize_trajectories
from src.condition_encoder import ConditionEncoder


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TrajectoryDataset(
    split="train",
    num_frames=33,
    random_crop=True,
)

item = dataset[0]

trajectory = item["trajectory"].unsqueeze(0).to(device)
visible = item["visible"].unsqueeze(0).to(device)

video = item["video"]
_, _, height, width = video.shape

condition = rasterize_trajectories(
    trajectory,
    visible,
    height,
    width,
).permute(0,2,1,3,4)

print("Condition:", condition.shape)


encoder = ConditionEncoder().to(device)

with torch.no_grad():
    encoded = encoder(condition)

print("Encoded condition:", encoded.shape)