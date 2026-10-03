import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dataset import TrajectoryDataset
from src.trajectory_encoder import TrajectoryEncoder

dataset = TrajectoryDataset(
    split="train",
    num_frames=33,
    random_crop=False,
    load_video=False,
)

sample = dataset[0]

trajectory = sample["trajectory"].unsqueeze(0)  # (1, 33, 2)

encoder = TrajectoryEncoder(input_dim=2, hidden_dim=128)

encoded = encoder(trajectory)

print("Input :", trajectory.shape)
print("Output:", encoded.shape)