"""
camera_motion.py
Scores how much the CAMERA moves in every clip, so clips where the background slides around
(panning / tracking shots) can be left out. In those clips the drawn trajectory is ambiguous:
a target that is static on screen can still be "moving" through the scene.

For each clip: Farneback optical flow between frames `--stride` apart, the target region (box from
mask.png / box.npy, dilated) masked out, median flow of the rest = the camera's motion. The score is the
path length of that motion over the first --frames frames, as a fraction of the frame width.

    python scripts/camera_motion.py --data_root data/dataset_got10k
    python scripts/camera_motion.py --data_root data/dataset_got10k --max_score 0.15 --move_rejected
"""

import argparse
import json
import os
import shutil

import cv2
import imageio.v2 as imageio
import numpy as np


def clip_score(folder, frames, stride, width=192):
    reader = imageio.get_reader(os.path.join(folder, "video.mp4"))
    imgs = []
    try:
        for i, f in enumerate(reader):
            if i >= frames:
                break
            h, w = f.shape[:2]
            imgs.append(cv2.cvtColor(cv2.resize(f, (width, int(h * width / w))), cv2.COLOR_RGB2GRAY))
    finally:
        reader.close()
    if len(imgs) < stride + 1:
        return None

    H, W = imgs[0].shape
    bg = np.ones((H, W), dtype=bool)
    mask_png = os.path.join(folder, "mask.png")
    if os.path.exists(mask_png):
        from PIL import Image
        m = np.array(Image.open(mask_png).convert("L").resize((W, H), Image.NEAREST)) > 127
        if m.any():
            ys, xs = np.nonzero(m)
            pad = int(0.25 * max(np.ptp(xs), np.ptp(ys)) + 4)
            bg[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad] = False
    if bg.mean() < 0.2:
        return None  # the target covers (almost) the whole frame: no background to judge

    total = 0.0
    for a, b in zip(imgs[:-stride:stride], imgs[stride::stride]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        total += float(np.linalg.norm(np.median(flow[bg], axis=0))) / W
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/dataset_got10k")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--frames", type=int, default=33, help="score the first N frames (= the training crop length)")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max_score", type=float, default=None, help="clips above this are 'moving camera'")
    p.add_argument("--move_rejected", action="store_true", help="move those clips to <data_root>/rejected_camera/ (reversible)")
    args = p.parse_args()

    scores = {}
    for split in args.splits.split(","):
        d = os.path.join(args.data_root, split)
        for name in sorted(os.listdir(d)):
            s = clip_score(os.path.join(d, name), args.frames, args.stride)
            if s is not None:
                scores[f"{split}/{name}"] = round(s, 4)

    out = os.path.join(args.data_root, "camera_motion.json")
    json.dump(scores, open(out, "w"), indent=1)
    v = np.array(list(scores.values()))
    print(f"{len(v)} clips scored -> {out}")
    print("camera path (fraction of frame width over the crop) percentiles:",
          {q: round(float(np.percentile(v, q)), 3) for q in (10, 25, 50, 75, 90)})
    for t in (0.05, 0.1, 0.15, 0.25, 0.4):
        print(f"  <= {t:<4}: {(v <= t).sum():3d} clips kept")

    if args.max_score is not None:
        bad = [k for k, s in scores.items() if s > args.max_score]
        print(f"{len(bad)} clips above {args.max_score}")
        if args.move_rejected:
            for k in bad:
                split, name = k.split("/")
                dst = os.path.join(args.data_root, "rejected_camera", split)
                os.makedirs(dst, exist_ok=True)
                shutil.move(os.path.join(args.data_root, split, name), os.path.join(dst, name))
            print(f"moved {len(bad)} clips to {args.data_root}/rejected_camera/")


if __name__ == "__main__":
    main()
