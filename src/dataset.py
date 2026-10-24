import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F_nn
from PIL import Image
from torch.utils.data import Dataset

from src.segmentation import disk_mask


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        root="data/dataset",
        split="train",
        num_frames=33,
        random_crop=True,
        load_video=True,
        height=None,
        width=None,
        default_radius=0.06,
        repeat=1,
    ):
        self.repeat = repeat  # visit every clip this many times per epoch (each visit = a new random crop)
        self.default_radius = default_radius
        self.height = height
        self.width = width
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
        return len(self.samples) * self.repeat

    def __getitem__(self, idx):
        name, folder, length = self.samples[idx % len(self.samples)]
        F = self.num_frames

        start = (
            np.random.randint(0, length - F + 1)
            if self.random_crop
            else 0
        )

        full_trajectory = np.load(os.path.join(folder, "trajectory.npy"))
        trajectory = full_trajectory[start:start + F]

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

            # Trajectories are normalized to [0, 1], so resizing keeps them valid.
            if self.height and video.shape[-2:] != (self.height, self.width):
                video = F_nn.interpolate(
                    video,
                    size=(self.height, self.width),
                    mode="bilinear",
                    antialias=True,
                    align_corners=False,
                )

            item["video"] = video
            item["image"] = video[0]

            mask, area = self.load_mask(folder, full_trajectory, start, *video.shape[-2:])
            item["mask"] = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)
            item["mask_area"] = torch.tensor(area, dtype=torch.float32)

        return item


    def load_mask(self, folder, full_trajectory, start, H, W):
        """Mask of the controlled object at the conditioning frame, plus its area (fraction).

        mask.png (from scripts/make_masks.py) belongs to frame 0. The conditioning image is
        frame `start`, so the mask is moved by the object's displacement (an approximation: the
        shape is kept). Without a mask file a disk around the tracked point is used.
        """
        path = os.path.join(folder, "mask.png")

        if os.path.exists(path):
            mask = np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST)) > 127
        else:
            mask = np.zeros((H, W), dtype=bool)

        if mask.sum() == 0:
            mask = disk_mask(H, W, full_trajectory[0], self.default_radius)

        area = float(mask.mean())

        if start > 0:
            shift = (full_trajectory[start] - full_trajectory[0]) * np.array([W, H])
            mask = shift_mask(mask, int(round(shift[0])), int(round(shift[1])))

        return mask, area


def shift_mask(mask, dx, dy):
    """Translate a boolean mask by (dx, dy) pixels, filling with False."""
    H, W = mask.shape
    out = np.zeros_like(mask)
    if abs(dx) >= W or abs(dy) >= H:
        return out
    out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = mask[
        max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)
    ]
    return out


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