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
        augment=False,
    ):
        # augment (train only): a random tracked point per visit (tracks.npy) and a random mirror flip
        self.augment = augment
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
            full_visible = np.load(visibility_path)
            visible = full_visible[start:start + F]
        else:
            full_visible = np.ones(len(full_trajectory), dtype=bool)
            visible = np.ones(F, dtype=bool)

        swapped = False
        if self.augment:
            pick = self.pick_track(folder, start, F)
            if pick is not None:
                full_trajectory, full_visible = pick
                trajectory = full_trajectory[start:start + F]
                visible = full_visible[start:start + F]
                swapped = True

        flip = self.augment and np.random.rand() < 0.5
        if flip:
            trajectory = trajectory.copy()
            trajectory[:, 0] = 1.0 - trajectory[:, 0]
            full_trajectory = full_trajectory.copy()
            full_trajectory[:, 0] = 1.0 - full_trajectory[:, 0]

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

            mask, area = self.load_mask(
                folder, full_trajectory, start, *video.shape[-2:], use_file=not swapped, flipped=flip
            )

            if flip:
                video = video.flip(-1)
                item["video"] = video
                item["image"] = video[0]

            item["mask"] = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)
            item["mask_area"] = torch.tensor(area, dtype=torch.float32)

        return item


    def pick_track(self, folder, start, F, min_visible=0.8):
        """A random tracked point of this clip for the crop [start, start+F), favouring moving ones.

        Returns (trajectory, visibility) over the whole clip, or None when the clip has no
        tracks.npy (then the stored trajectory is used).
        """
        tp = os.path.join(folder, "tracks.npy")
        vp = os.path.join(folder, "tracks_visibility.npy")
        if not (os.path.exists(tp) and os.path.exists(vp)):
            return None

        tracks, vis = np.load(tp), np.load(vp)  # (N, T, 2), (N, T)
        window = slice(start, start + F)
        # visible in most of the crop AND in its first frame (the conditioning image shows the object)
        ok = np.where((vis[:, window].mean(1) >= min_visible) & vis[:, start])[0]
        if len(ok) == 0:
            return None

        steps = np.linalg.norm(np.diff(tracks[ok][:, window], axis=1), axis=-1)
        motion = (steps * (vis[ok][:, window][:, 1:] & vis[ok][:, window][:, :-1])).sum(1)
        weights = motion + 0.02  # static points still get picked now and then
        i = np.random.choice(ok, p=weights / weights.sum())
        return tracks[i], vis[i]

    def load_mask(self, folder, full_trajectory, start, H, W, use_file=True, flipped=False):
        """Mask of the controlled object at the conditioning frame, plus its area (fraction).

        mask.png (from scripts/make_masks.py) belongs to frame 0. The conditioning image is
        frame `start`, so the mask is moved by the object's displacement (an approximation: the
        shape is kept). Without a mask file a disk around the tracked point is used.
        """
        path = os.path.join(folder, "mask.png")

        if use_file and os.path.exists(path):
            mask = np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST)) > 127
            if flipped:
                mask = np.ascontiguousarray(mask[:, ::-1])  # full_trajectory is already mirrored; mask.png is not
        else:
            mask = np.zeros((H, W), dtype=bool)

        if mask.sum() == 0:
            # no mask for this point: a disk centred where the object is in the conditioning frame
            mask = disk_mask(H, W, full_trajectory[start], self.default_radius)
            return mask, float(mask.mean())

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