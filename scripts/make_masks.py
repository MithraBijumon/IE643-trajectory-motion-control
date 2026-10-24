"""
make_masks.py

Writes mask.png (the controlled object, from SAM prompted with the tracked point) into every
sample folder, plus a contact sheet so the masks can be checked by eye.

    python scripts/make_masks.py --data_root data/dataset
    python scripts/make_masks.py --data_root data/dataset --limit 5 --splits val
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw

from src.segmentation import SamSegmenter
from src.utils import get_device

MIN_AREA, MAX_AREA = 0.001, 0.6  # fraction of the frame; outside this the mask is suspicious


def overlay(image, mask, point, label):
    base = np.array(image).astype(np.float32)
    base[mask] = 0.45 * base[mask] + 0.55 * np.array([255, 40, 40], dtype=np.float32)
    img = Image.fromarray(base.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    x, y = point
    draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(0, 255, 0), outline="black")
    draw.text((6, 6), label, fill=(255, 255, 0))
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/dataset")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--limit", type=int, default=None, help="only the first N samples per split")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--pick", choices=["largest", "best"], default="largest")
    p.add_argument("--preview", default=None, help="contact sheet path (default: <data_root>/masks_preview.jpg)")
    args = p.parse_args()

    segmenter = SamSegmenter(get_device())
    tiles, flagged, done = [], [], 0

    for split in args.splits.split(","):
        split_dir = os.path.join(args.data_root, split)
        names = [n for n in sorted(os.listdir(split_dir)) if os.path.exists(os.path.join(split_dir, n, "trajectory.npy"))]

        for name in names[: args.limit]:
            folder = os.path.join(split_dir, name)
            out_path = os.path.join(folder, "mask.png")

            image = Image.open(os.path.join(folder, "image.png")).convert("RGB")
            W, H = image.size
            point = np.load(os.path.join(folder, "trajectory.npy"))[0] * [W, H]

            if os.path.exists(out_path) and not args.overwrite:
                mask = np.array(Image.open(out_path)) > 127
                score = float("nan")
            else:
                mask, score = segmenter.segment(image, point, args.pick)
                Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)

            area = mask.mean()
            if not (MIN_AREA <= area <= MAX_AREA):
                flagged.append(f"{split}/{name} (area {area:.4f})")
            if len(tiles) < 24:
                tiles.append(overlay(image, mask, point, f"{split}/{name} area {area:.3f} iou {score:.2f}"))
            done += 1

    print(f"Wrote {done} masks")
    if flagged:
        print(f"{len(flagged)} suspicious masks (area outside {MIN_AREA}-{MAX_AREA}): {flagged[:10]}")

    if tiles:
        w, h = 320, int(320 * tiles[0].height / tiles[0].width)
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w, rows * h))
        for i, t in enumerate(tiles):
            sheet.paste(t.resize((w, h)), ((i % cols) * w, (i // cols) * h))
        preview = args.preview or os.path.join(args.data_root, "masks_preview.jpg")
        sheet.save(preview)
        print(f"Contact sheet: {preview}")


if __name__ == "__main__":
    main()
