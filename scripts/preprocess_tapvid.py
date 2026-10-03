import argparse
import os
import pickle

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        return list(data.items())          # [(name, sample), ...]
    return [(f"sample_{i:04d}", s) for i, s in enumerate(data)]


def normalize_video(video):
    video = np.asarray(video)
    if video.dtype != np.uint8:
        video = ((video + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return video


def normalize_points(points, w, h):
    points = points.astype(np.float32)

    if points.max() > 1.5:
        points[..., 0] /= w
        points[..., 1] /= h

    return np.clip(points, 0, 1)


def choose_point(points, visible):
    """Choose the most-moving point that is visible most of the time."""
    visibility = visible.mean(axis=1)
    candidates = np.where(visibility >= 0.8)[0]

    if len(candidates) == 0:
        return np.argmax(visibility)

    def motion(i):
        p = points[i][visible[i]]
        return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()

    return max(candidates, key=motion)


def process(name, sample, out_dir):
    video = normalize_video(sample["video"])
    points = np.asarray(sample["points"])
    visible = ~np.asarray(sample["occluded"]).astype(bool)

    _, h, w, _ = video.shape
    points = normalize_points(points, w, h)

    os.makedirs(out_dir, exist_ok=True)

    Image.fromarray(video[0]).save(f"{out_dir}/image.png")
    imageio.mimwrite(f"{out_dir}/video.mp4", list(video), fps=24)

    idx = choose_point(points, visible)

    np.save(f"{out_dir}/trajectory.npy", points[idx])
    np.save(f"{out_dir}/visibility.npy", visible[idx])

    # Keep every TAP-Vid point for future multi-point conditioning
    np.save(f"{out_dir}/tracks.npy", points)
    np.save(f"{out_dir}/tracks_visibility.npy", visible)

    print(f"{name}: {len(video)} frames, {len(points)} points")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle_path", required=True)
    p.add_argument("--out_dir", default="data/dataset")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_samples", type=int)
    args = p.parse_args()

    data = load_data(args.pickle_path)

    if args.max_samples:
        data = data[:args.max_samples]

    n_val = max(1, int(len(data) * args.val_ratio))

    for i, (name, sample) in enumerate(data):
        split = "val" if i < n_val else "train"
        process(name, sample, os.path.join(args.out_dir, split, name))

    print(f"\nDone: {len(data)} samples")


if __name__ == "__main__":
    main()