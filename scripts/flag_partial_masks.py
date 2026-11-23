"""
flag_partial_masks.py
Finds clips whose stored mask.png probably covers only PART of the object (e.g. a rider's legs and bike but not the upper body).
For every clip SAM is prompted at the mask's centroid and its LARGEST candidate is taken as "the whole object"; a clip is flagged
when that candidate contains most of the mask and is clearly larger (but not frame-sized). Flags are suspects, not verdicts
(a large candidate can also be floor or sky): look at the contact sheet before acting.

    python scripts/flag_partial_masks.py --data_root data/dataset_got10k_big
Writes <out>/flagged.txt (clip, mask area, candidate area, ratio) and <out>/flagged.jpg (red = stored mask, green = candidate).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw

from src.segmentation import SamSegmenter
from src.utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/dataset_got10k")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--out", default="results/partial_masks")
    p.add_argument("--min_ratio", type=float, default=1.5, help="candidate area / mask area to flag")
    p.add_argument("--contain", type=float, default=0.8, help="share of the mask the candidate must contain")
    p.add_argument("--max_frac", type=float, default=0.35, help="candidates larger than this share of the frame are ignored")
    args = p.parse_args()

    seg = SamSegmenter(get_device())
    os.makedirs(args.out, exist_ok=True)
    flagged, tiles, total = [], [], 0
    for split in args.splits.split(","):
        sd = os.path.join(args.data_root, split)
        if not os.path.isdir(sd):
            continue
        for name in sorted(os.listdir(sd)):
            f = os.path.join(sd, name)
            if not (os.path.exists(os.path.join(f, "mask.png")) and os.path.exists(os.path.join(f, "image.png"))):
                continue
            img = Image.open(os.path.join(f, "image.png")).convert("RGB")
            m = np.array(Image.open(os.path.join(f, "mask.png")).convert("L")) > 127
            if m.sum() == 0:
                continue
            total += 1
            ys, xs = np.nonzero(m)
            cand, _ = seg.segment(img, (xs.mean(), ys.mean()), pick="largest")
            if (cand & m).sum() / m.sum() < args.contain or cand.mean() > args.max_frac:
                continue
            ratio = cand.sum() / m.sum()
            if ratio >= args.min_ratio:
                flagged.append((f"{split}/{name}", int(m.sum()), int(cand.sum()), float(ratio)))
                a = np.array(img).astype(np.float32)
                a[cand] = 0.6 * a[cand] + 0.4 * np.array([0, 255, 0])
                a[m] = 0.5 * a[m] + 0.5 * np.array([255, 0, 0])
                t = Image.fromarray(a.astype(np.uint8))
                ImageDraw.Draw(t).text((4, 4), f"{split}/{name[-6:]} x{ratio:.1f}", fill=(255, 255, 0))
                tiles.append(t)
    flagged.sort(key=lambda r: -r[3])
    with open(os.path.join(args.out, "flagged.txt"), "w") as fh:
        for r in flagged:
            fh.write(f"{r[0]} mask={r[1]} candidate={r[2]} ratio={r[3]:.2f}\n")
    print(f"{len(flagged)} of {total} clips flagged")
    if tiles:
        tiles = tiles[:48]
        w = 256
        h = int(w * tiles[0].height / tiles[0].width)
        cols = 6
        sheet = Image.new("RGB", (cols * w, ((len(tiles) + cols - 1) // cols) * h))
        for i, t in enumerate(tiles):
            sheet.paste(t.resize((w, h)), ((i % cols) * w, (i // cols) * h))
        sheet.save(os.path.join(args.out, "flagged.jpg"), quality=88)
        print("wrote", os.path.join(args.out, "flagged.jpg"), "(first 48 in scan order)")


if __name__ == "__main__":
    main()
