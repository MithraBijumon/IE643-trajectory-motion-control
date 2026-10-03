"""
build_index.py
Scans a processed dataset folder (output of preprocess_tapvid.py) and writes
a single metadata.json listing every sample and its file paths.
Usage:
python build_index.py --dataset_dir dataset
python build_index.py --dataset_dir dataset_kubric
"""

import argparse
import os
import json
import numpy as np


def index_split(dataset_dir, split):
    split_dir = os.path.join(dataset_dir, split)
    entries = []

    if not os.path.isdir(split_dir):
        return entries

    for sample_id in sorted(os.listdir(split_dir)):
        sample_dir = os.path.join(split_dir, sample_id)
        if not os.path.isdir(sample_dir):
            continue

        image_path = os.path.join(split, sample_id, "image.png")
        video_path = os.path.join(split, sample_id, "video.mp4")
        traj_path = os.path.join(split, sample_id, "trajectory.npy")

        full_image = os.path.join(dataset_dir, image_path)
        full_video = os.path.join(dataset_dir, video_path)
        full_traj = os.path.join(dataset_dir, traj_path)

        if not (os.path.exists(full_image) and os.path.exists(full_video) and os.path.exists(full_traj)):
            print(f"[SKIPPED] {sample_id}: missing one or more files")
            continue

        # Record trajectory length so downstream code can filter/batch by it
        # without re-loading every .npy file.
        try:
            traj_len = int(np.load(full_traj).shape[0])
        except Exception as e:
            print(f"[SKIPPED] {sample_id}: could not read trajectory ({e})")
            continue

        entries.append({
            "sample_id": sample_id,
            "image": image_path,
            "video": video_path,
            "trajectory": traj_path,
            "trajectory_length": traj_len,
            "split": split,
        })

    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Root folder produced by preprocess_tapvid.py")
    parser.add_argument("--out_file", default=None, help="Output path (default: <dataset_dir>/metadata.json)")
    args = parser.parse_args()

    out_file = args.out_file or os.path.join(args.dataset_dir, "metadata.json")

    all_entries = []
    for split in ["train", "val"]:
        entries = index_split(args.dataset_dir, split)
        print(f"{split}: {len(entries)} samples indexed")
        all_entries.extend(entries)

    with open(out_file, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"\nWrote {len(all_entries)} total entries to {out_file}")


if __name__ == "__main__":
    main()
