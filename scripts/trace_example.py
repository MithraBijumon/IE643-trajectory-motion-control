"""
One example traced end to end: input image + object mask + drawn path -> drag video -> generated frames
with the SAM 2 mask and centroid path -> the metrics, each with what it measures. Plain LTX is shown for comparison.

    python scripts/trace_example.py --checkpoint best.pt --direction right --ref_init 0.8 [--clip <val clip>]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from benchmark import val_items
from src.ltx_wrapper import LTXBackbone
from src.metrics import background_motion, load_cotracker, object_trajectory_metrics, track_point, trajectory_metrics
from src.reference import render_reference
from src.segmentation import Sam2Tracker
from src.utils import DIRECTIONS, apply_checkpoint_cfg, get_device, linear_path, load_config, overlay_masks, save_video

MEANING = [
    ("ate", "mean distance between the tracked point and the drawn path (fraction of the frame; lower is better)"),
    ("direction_cos", "did it move the right way: 1 = exactly, 0 = no motion, negative = opposite"),
    ("progress", "share of the drawn distance actually travelled (1 = all of it)"),
    ("obj_ate", "the same error for the whole object (SAM 2 mask centroid) instead of one tracked point"),
    ("bg_drift", "how much the scene outside the object changed since frame 0 (0 = static background)"),
    ("bg_flow", "optical-flow motion outside the object per frame (fraction of the width; 0 = still)"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--clip", default=None, help="val clip name (default: the first one)")
    p.add_argument("--direction", default="right", choices=list(DIRECTIONS))
    p.add_argument("--length", type=float, default=0.3)
    p.add_argument("--ref_init", type=float, default=None)
    p.add_argument("--bg_comp", type=float, default=None)
    p.add_argument("--adapter_scale", type=float, default=None)
    p.add_argument("--adapter_min_sigma", type=float, default=None)
    p.add_argument("--motion_guidance", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/trace")
    args = p.parse_args()

    cfg = apply_checkpoint_cfg(load_config(args.config, tiny=args.tiny), args.checkpoint)
    root = args.data_root or cfg["data"]["root"]
    inf, nF = cfg["inference"], cfg["data"]["num_frames"]
    items = val_items(root, 10 ** 6, cfg)
    item = next((i for i in items if i["name"] == args.clip), items[0]) if args.clip else items[0]
    device = get_device()

    backbone = LTXBackbone(cfg, device, prompts=[item["prompt"]] if item["prompt"] else None)
    if args.checkpoint:
        backbone.load_checkpoint(args.checkpoint)
    target = linear_path(item["start"], DIRECTIONS[args.direction], args.length, nF)
    opts = dict(steps=args.steps or inf["steps"], guidance_scale=inf["guidance_scale"], decode_timestep=inf["decode_timestep"],
                seed=args.seed, mask=item["mask"], prompt=item["prompt"])
    base = backbone.generate(item["image"], target, use_motion=False, **opts)
    gen = backbone.generate(item["image"], target, use_motion=True, ref_init=args.ref_init, bg_comp=args.bg_comp, adapter_scale=args.adapter_scale, adapter_min_sigma=args.adapter_min_sigma,
                            motion_guidance=args.motion_guidance if args.motion_guidance is not None else inf["motion_guidance"], **opts)
    drag = render_reference(np.array(item["image"]), item["mask"], target)

    tracker = Sam2Tracker(device)
    cot = load_cotracker(device)
    H, W = gen.shape[1:3]
    scores = {}
    for label, frames in (("baseline", base), ("adapter", gen)):
        masks = tracker.track(frames, item["mask"]) if tracker.ok else np.repeat(item["mask"][None], nF, 0)
        track = track_point(cot, frames, target[0] * np.array([W, H], np.float32), device)
        scores[label] = {**trajectory_metrics(target, track), **object_trajectory_metrics(target, masks),
                         **background_motion(frames, masks)}
        scores[label]["_shown"] = overlay_masks(frames, masks, target)

    os.makedirs(args.out, exist_ok=True)
    save_video(gen, os.path.join(args.out, "adapter.mp4"), fps=inf["fps"])
    picks = sorted({0, nF // 2, nF - 1})
    mask_view = np.array(item["image"]).astype(np.float32)
    mask_view[item["mask"]] = 0.5 * mask_view[item["mask"]] + 0.5 * np.array([255, 40, 40], np.float32)
    mask_view = Image.fromarray(mask_view.astype(np.uint8))
    ImageDraw.Draw(mask_view).line([(x * W, y * H) for x, y in target], fill=(255, 215, 0), width=3)
    mask_view = np.array(mask_view)
    rows = [
        ("1  input: image, object mask (red),\n    drawn path (yellow)", [mask_view]),
        ("2  drag video: the object's cut-out\n    moved along the path (the condition)", [drag[i] for i in picks]),
        ("3  plain LTX, same prompt and seed\n    mask + tracked path (cyan)", [scores["baseline"]["_shown"][i] for i in picks]),
        ("4  adapter\n    mask + tracked path (cyan)", [scores["adapter"]["_shown"][i] for i in picks]),
    ]
    fig = plt.figure(figsize=(15, 4 * 3.3 + 3.2))
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 0.9], width_ratios=[1.15, 1, 1, 1])
    for r, (title, imgs) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0]); ax.axis("off"); ax.text(0, 0.5, title, va="center", fontsize=10)
        for c, im in enumerate(imgs):
            ax = fig.add_subplot(gs[r, c + 1]); ax.imshow(im); ax.axis("off")
    ax = fig.add_subplot(gs[4, :]); ax.axis("off")
    lines = [f"{'metric':<14}{'plain LTX':>10}{'adapter':>10}   what it measures"]
    for k, text in MEANING:
        lines.append(f"{k:<14}{scores['baseline'].get(k, float('nan')):>10.3f}{scores['adapter'].get(k, float('nan')):>10.3f}   {text}")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.suptitle(f"{item['name']}  |  path: {args.direction}  |  seed {args.seed}  |  prompt: {(item['prompt'] or 'config prompt')[:110]}", fontsize=10)
    fig.savefig(os.path.join(args.out, "trace.png"), dpi=110, bbox_inches="tight")
    print("wrote", os.path.join(args.out, "trace.png"))


if __name__ == "__main__":
    main()
