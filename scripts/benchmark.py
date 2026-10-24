"""
benchmark.py

Controlled test of simple motion instructions (left / right / up / down / diagonal).
For each val image a straight path of fixed length is built from the object's starting
point, one video is generated per path with the adapter, and one plain-LTX baseline video
per image. Every video is tracked with CoTracker3 and scored for

    trajectory adherence : ATE, final-point error, direction cosine, progress
    image preservation   : CLIP similarity to the input image (+ LPIPS if installed)
    temporal consistency : CLIP similarity between neighbouring frames

    python scripts/benchmark.py --checkpoint checkpoints/best.pt --num_images 8
    python scripts/benchmark.py --tiny --checkpoint <tiny ckpt> --num_images 1 --motions left,up
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
from PIL import Image, ImageDraw

from src.ltx_wrapper import LTXBackbone
from src.metrics import VideoQuality, trajectory_metrics
from src.utils import (
    DIRECTIONS,
    apply_checkpoint_cfg,
    get_device,
    load_config,
    linear_path,
    load_image,
    overlay_trajectory,
    save_video,
)

def val_items(data_root, num_images, cfg):
    d = cfg["data"]
    split = os.path.join(data_root, "val")
    items = []

    for name in sorted(os.listdir(split)):
        folder = os.path.join(split, name)
        if not os.path.exists(os.path.join(folder, "trajectory.npy")):
            continue
        items.append(
            {
                "name": name,
                "image": load_image(os.path.join(folder, "image.png"), d["height"], d["width"]),
                "start": np.load(os.path.join(folder, "trajectory.npy"))[0],
            }
        )
        if len(items) == num_images:
            break
    return items


def draw_row(frames, target, picks):
    """Selected frames with the target path drawn on every one."""
    H, W = frames.shape[1:3]
    tiles = []
    for t in picks:
        img = Image.fromarray(frames[t]).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.line([(x * W, y * H) for x, y in target], fill=(255, 215, 0), width=3)
        x, y = target[t, 0] * W, target[t, 1] * H
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 0, 0), outline="black")
        tiles.append(img)
    return tiles


def montage(rows, labels, path):
    w, h = rows[0][0].size
    pad = 150
    canvas = Image.new("RGB", (pad + w * len(rows[0]), h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (tiles, label) in enumerate(zip(rows, labels)):
        draw.text((6, r * h + h // 2), label, fill="black")
        for c, tile in enumerate(tiles):
            canvas.paste(tile, (pad + c * w, r * h))
    canvas.save(path)


def aggregate(rows):
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, float)})
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--out", default="results/benchmark")
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--motions", default=",".join(DIRECTIONS))
    p.add_argument("--length", type=float, default=0.3, help="path length as a fraction of the frame")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--motion_guidance", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    motions = [m for m in args.motions.split(",") if m]
    for m in motions:
        if m not in DIRECTIONS:
            p.error(f"unknown motion {m}; choose from {list(DIRECTIONS)}")

    cfg = apply_checkpoint_cfg(load_config(args.config, tiny=args.tiny), args.checkpoint)
    data_root = args.data_root or cfg["data"]["root"]
    inf = cfg["inference"]
    nF = cfg["data"]["num_frames"]
    steps = args.steps or inf["steps"]
    mg = inf["motion_guidance"] if args.motion_guidance is None else args.motion_guidance
    device = get_device()

    from evaluate import load_cotracker, track_point

    backbone = LTXBackbone(cfg, device)
    if args.checkpoint:
        backbone.load_checkpoint(args.checkpoint)
    cotracker = load_cotracker(device)
    quality = VideoQuality(device)

    items = val_items(data_root, args.num_images, cfg)
    print(f"{len(items)} images x {len(motions)} motions | adapter: {cfg['adapter']['mode']} | motion_guidance {mg}")

    H, W = cfg["data"]["height"], cfg["data"]["width"]
    scale = np.array([W, H], dtype=np.float32)
    picks = sorted({0, nF // 2, nF - 1})
    records = {"adapter": [], "baseline": []}
    montage_rows, montage_labels = [], []

    def generate(image, target, use_motion):
        return backbone.generate(
            image,
            target,
            steps=steps,
            guidance_scale=inf["guidance_scale"],
            decode_timestep=inf["decode_timestep"],
            seed=args.seed,
            motion_guidance=mg,
            use_motion=use_motion,
        )

    for item in items:
        name, image = item["name"], item["image"]
        img_np = np.array(image)

        # one baseline video per image; it is scored against every target path
        first_target = linear_path(item["start"], DIRECTIONS[motions[0]], args.length, nF)
        base_frames = generate(image, first_target, use_motion=False)
        save_video(base_frames, os.path.join(args.out, "baseline", f"{name}.mp4"), fps=inf["fps"])
        base_track = track_point(cotracker, base_frames, first_target[0] * scale, device)
        base_q = quality.score(img_np, base_frames)

        for motion in motions:
            target = linear_path(item["start"], DIRECTIONS[motion], args.length, nF)
            tag = f"{name}_{motion}"

            frames = generate(image, target, use_motion=True)
            save_video(
                overlay_trajectory(frames, target),
                os.path.join(args.out, "adapter", f"{tag}.mp4"),
                fps=inf["fps"],
            )

            track = track_point(cotracker, frames, target[0] * scale, device)
            row = {"image": name, "motion": motion, **trajectory_metrics(target, track), **quality.score(img_np, frames)}
            base_row = {"image": name, "motion": motion, **trajectory_metrics(target, base_track), **base_q}
            records["adapter"].append(row)
            records["baseline"].append(base_row)

            print(
                f"{tag}: ATE adapter {row['ate']:.3f} | baseline {base_row['ate']:.3f} "
                f"| direction_cos {row.get('direction_cos', float('nan')):.2f}"
            )

            if len(montage_rows) < 12:
                montage_rows += [draw_row(base_frames, target, picks), draw_row(frames, target, picks)]
                montage_labels += [f"{tag}\nbaseline", f"{tag}\nadapter"]

    summary = {"overall": {}, "per_motion": {}}
    for label, rows in records.items():
        summary["overall"][label] = aggregate(rows)
        summary["per_motion"][label] = {m: aggregate([r for r in rows if r["motion"] == m]) for m in motions}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(
            {"args": vars(args), "adapter_mode": cfg["adapter"]["mode"], "summary": summary, "records": records},
            f,
            indent=2,
        )
    if montage_rows:
        montage(montage_rows, montage_labels, os.path.join(args.out, "montage.png"))

    cols = ["ate", "epe_final", "direction_cos", "progress", "clip_image_sim", "clip_temporal"]
    if "lpips_to_input" in summary["overall"]["adapter"]:
        cols.append("lpips_to_input")

    lines = ["| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for label in ("baseline", "adapter"):
        o = summary["overall"][label]
        lines.append(f"| {label} | " + " | ".join(f"{o.get(c, float('nan')):.3f}" for c in cols) + " |")
    table = "\n".join(lines)

    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
