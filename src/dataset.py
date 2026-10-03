import os

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        root="data/dataset",
        split="train",
        num_frames=33,
        random_crop=True,
        load_video=True,
    ):
        self.num_frames = num_frames
        self.random_crop = random_crop
        self.load_video = load_video

        split_dir = os.path.join(root, split)
        self.samples = []

        for name in sorted(os.listdir(split_dir)):
            folder = os.path.join(split_dir, name)
            path = os.path.join(folder, "trajectory.npy")

            if not os.path.exists(path):
                continue

            length = np.load(path, mmap_mode="r").shape[0]

            if length >= num_frames:
                self.samples.append((name, folder, length))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, folder, length = self.samples[idx]
        F = self.num_frames

        start = (
            np.random.randint(0, length - F + 1)
            if self.random_crop
            else 0
        )

        trajectory = np.load(
            os.path.join(folder, "trajectory.npy")
        )[start:start + F]

        visibility_path = os.path.join(folder, "visibility.npy")
        if os.path.exists(visibility_path):
            visible = np.load(visibility_path)[start:start + F]
        else:
            visible = np.ones(F, dtype=bool)

        item = {
            "name": name,
            "start": start,
            "trajectory": torch.from_numpy(trajectory).float(),
            "visible": torch.from_numpy(visible).bool(),
        }

        if self.load_video:
            video = read_video(
                os.path.join(folder, "video.mp4"),
                start,
                F,
            )

            video = (
                torch.from_numpy(video)
                .permute(0, 3, 1, 2)
                .float()
                / 127.5
                - 1
            )

            item["video"] = video
            item["image"] = video[0]

        return item


def read_video(path, start, num_frames):
    reader = imageio.get_reader(path)
    frames = []

    try:
        for i, frame in enumerate(reader):
            if i >= start + num_frames:
                break
            if i >= start:
                frames.append(frame)
    finally:
        reader.close()

    return np.stack(frames)