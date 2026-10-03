import os
import imageio.v2 as imageio
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import TrajectoryDataset
from src.conditioning import rasterize_trajectories


def main():
    dataset = TrajectoryDataset(
        split="train",
        num_frames=33,
        random_crop=False,
    )

    sample = dataset[0]

    trajectory = sample["trajectory"].unsqueeze(0)
    visible = sample["visible"].unsqueeze(0)
    video = sample["video"].unsqueeze(0)

    print("trajectory:", trajectory.shape)
    print("visible:", visible.shape)
    print("video:", video.shape)

    _, _, _, H, W = video.shape

    condition = rasterize_trajectories(
        trajectory,
        visible,
        H,
        W,
    )

    print("condition:", condition.shape)

    # Visualize heatmap as red overlay
    frames = (
        video[0].permute(0, 2, 3, 1).numpy() + 1
    ) * 127.5

    heat = condition[0, :, 0].numpy()

    os.makedirs("results/debug", exist_ok=True)

    output = []

    for frame, h in zip(frames, heat):
        overlay = frame.copy()

        # Red where trajectory is located
        overlay[..., 0] = np.maximum(
            overlay[..., 0],
            h * 255
        )

        output.append(
            overlay.clip(0, 255).astype(np.uint8)
        )

    imageio.mimwrite(
        "results/debug/trajectory_condition.mp4",
        output,
        fps=8,
    )

    print("Saved: results/debug/trajectory_condition.mp4")


if __name__ == "__main__":
    main()