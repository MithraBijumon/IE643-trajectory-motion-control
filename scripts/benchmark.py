"""
benchmark.py

Controlled test of simple motion instructions (left / right / up / down / diagonal).
For each val image a straight path of fixed length is built from the object's starting
point, one video is generated per path with the adapter, and one plain-LTX baseline video
per image. Every video is tracked with CoTracker3 and scored for

    trajectory adherence : ATE, final-point error, direction cosine, progress
    image preservation   : CLIP similarity, first-frame PSNR/SSIM, colour drift (+ LPIPS if installed)
    temporal consistency : CLIP neighbour similarity, optical-flow warping error (+ temporal LPIPS)
    object level (SAM 2) : obj_* = tracked object-mask centroid path vs. the target, CLIP similarity of the
                           object crop to the input object, size drift
    video quality        : sharpness ratio, LAION aesthetic score, FVD-style distance (VideoMAE features) to real clips
    PCK@5% / @10%        : fraction of frames where the tracked point is near the target
    background (bg_*)    : unintended background movement outside the object: bg_drift (change vs. frame 0) and bg_flow

--seeds 0,1,2 repeats everything per seed (reports the spread); --ood_dir adds clean images from outside the training data
(<name>.jpg + optional <name>.txt prompt / <name>_mask.png mask); --ref_init / --bg_lock switch on the drag-video start and
the background lock for the adapter runs.

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

from src.dataset import collect_prompts, read_caption, shift_mask
from src.ltx_wrapper import LTXBackbone
from src.metrics import (AestheticScorer, VideoFeatures, VideoQuality, background_motion, frechet_distance,
                         object_appearance, object_trajectory_metrics, trajectory_metrics)
from src.segmentation import Sam2Tracker
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
    if num_images <= 0:
        return items

    for name in sorted(os.listdir(split)):
        folder = os.path.join(split, name)
        if not os.path.exists(os.path.join(folder, "trajectory.npy")):
            continue
        mask_path = os.path.join(folder, "mask.png")
        mask = None
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L").resize((d["width"], d["height"]), Image.NEAREST)) > 127

        items.append(
            {
                "mask": mask,
                "group": "val",
                "name": name,
                "prompt": read_caption(folder) or None,
                "video": os.path.join(folder, "video.mp4"),
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


def fit_image(path, H, W):
    """Cover-resize, then centre-crop to the target size (no stretching)."""
    img = Image.open(path).convert("RGB")
    s = max(W / img.width, H / img.height)
    img = img.resize((max(W, round(img.width * s)), max(H, round(img.height * s))), Image.BICUBIC)
    x, y = (img.width - W) // 2, (img.height - H) // 2
    return img.crop((x, y, x + W, y + H))


def ood_items(folder, cfg, device):
    """Clean images from outside the training data: <name>.jpg|png, optional <name>.txt (prompt) and <name>_mask.png."""
    from src.segmentation import SamSegmenter, disk_mask

    H, W = cfg["data"]["height"], cfg["data"]["width"]
    sam, items = None, []
    for fn in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp") or stem.endswith("_mask"):
            continue
        image = fit_image(os.path.join(folder, fn), H, W)
        mpath = os.path.join(folder, stem + "_mask.png")
        mask = None
        if os.path.exists(mpath):
            mask = np.array(fit_image(mpath, H, W).convert("L")) > 127
        else:
            sam = sam or SamSegmenter(device)
            for pick in ("largest", "best"):   # SAM from a click at the centre; keep a plausible object-sized mask
                m, _ = sam.segment(image, (W / 2, H / 2), pick=pick)
                if 0.02 <= m.mean() <= 0.6:
                    mask = m
                    break
        if mask is None:
            mask = disk_mask(H, W, (0.5, 0.5), 0.1)
        ys, xs = np.nonzero(mask)
        tpath = os.path.join(folder, stem + ".txt")
        items.append({
            "name": "ood_" + stem, "group": "ood", "mask": mask, "image": image, "video": None,
            "start": np.array([xs.mean() / W, ys.mean() / H], dtype=np.float32),
            "prompt": open(tpath, encoding="utf-8").read().strip() if os.path.exists(tpath) else None,
        })
    return items


def moved_masks(mask, target):
    """The first-frame mask carried along the target path (fallback when SAM 2 is not available)."""
    H, W = mask.shape
    d = (np.asarray(target) - np.asarray(target)[0]) * np.array([W, H])
    return np.stack([shift_mask(mask, int(round(x)), int(round(y))) for x, y in d])


def spread(records, key, seeds):
    """Std over seeds of the per-seed mean of `key`."""
    per = [np.mean([r[key] for r in records if r["seed"] == s and key in r]) for s in seeds]
    return float(np.std(per))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--out", default="results/benchmark")
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--ood_dir", default=None, help="folder of clean out-of-domain images (see the header)")
    p.add_argument("--motions", default=",".join(DIRECTIONS))
    p.add_argument("--length", type=float, default=0.3, help="path length as a fraction of the frame")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--motion_guidance", type=float, default=None)
    p.add_argument("--ref_init", type=float, default=None, help="start the adapter runs from the drag video noised to this sigma (e.g. 0.8)")
    p.add_argument("--bg_lock", type=float, default=None, help="lock the background outside this many object radii (e.g. 2.5)")
    p.add_argument("--adapter_scale", type=float, default=None, help="strength of the adapter residual at inference (1 = as trained, 0 = off; the drag video / composite still run)")
    p.add_argument("--adapter_min_sigma", type=float, default=None, help="the adapter acts only while sigma >= this (e.g. 0.4); plain LTX finishes the detail")
    p.add_argument("--bg_comp", type=float, default=None, help="composite the object-free scene outside this many object radii (e.g. 2.5; needs the drag video)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 0,1,2 (overrides --seed)")
    args = p.parse_args()

    motions = [m for m in args.motions.split(",") if m]
    for m in motions:
        if m not in DIRECTIONS:
            p.error(f"unknown motion {m}; choose from {list(DIRECTIONS)}")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    cfg = apply_checkpoint_cfg(load_config(args.config, tiny=args.tiny), args.checkpoint)
    data_root = args.data_root or cfg["data"]["root"]
    if args.ref_init is not None:
        cfg["inference"]["ref_init"] = args.ref_init
    if args.bg_lock is not None:
        cfg["inference"]["bg_lock"] = args.bg_lock
    if args.bg_comp is not None:
        cfg["inference"]["bg_comp"] = args.bg_comp
    if args.adapter_scale is not None:
        cfg["inference"]["adapter_scale"] = args.adapter_scale
    if args.adapter_min_sigma is not None:
        cfg["inference"]["adapter_min_sigma"] = args.adapter_min_sigma
    inf = cfg["inference"]
    nF = cfg["data"]["num_frames"]
    steps = args.steps or inf["steps"]
    mg = inf["motion_guidance"] if args.motion_guidance is None else args.motion_guidance
    device = get_device()

    items = val_items(data_root, args.num_images, cfg)
    if args.ood_dir:
        items += ood_items(args.ood_dir, cfg, device)
    print(f"{len(items)} images ({sum(i['group'] == 'ood' for i in items)} out-of-domain) x {len(motions)} motions x "
          f"{len(seeds)} seeds | motion_guidance {mg} | ref_init {inf.get('ref_init', 0)} | bg_lock {inf.get('bg_lock', 0)} | bg_comp {inf.get('bg_comp', 0)} | adapter_scale {inf.get('adapter_scale', 1.0)} | adapter_min_sigma {inf.get('adapter_min_sigma', 0.0)}")

    from evaluate import load_cotracker, track_point

    backbone = LTXBackbone(cfg, device, prompts=[i["prompt"] for i in items if i["prompt"]])
    if args.checkpoint:
        backbone.load_checkpoint(args.checkpoint)
    cotracker = load_cotracker(device)
    quality = VideoQuality(device)
    aesthetic = AestheticScorer(device)
    vfeat = VideoFeatures(device)
    feats = {"adapter": [], "baseline": [], "real": []}
    tracker = Sam2Tracker(device)   # object-level metrics: SAM 2 follows the frame-0 mask through each generated video

    def target_free(frames, mask, img_np, masks, fallback):
        """Scores that do not depend on the target path."""
        out = {}
        if masks is not None:
            out.update(object_appearance(quality, img_np, mask, frames, masks))
        if mask is not None:
            out.update(background_motion(frames, masks if masks is not None else fallback))
        if aesthetic.ok:
            out["aesthetic"] = aesthetic.score(frames)
        return out

    H, W = cfg["data"]["height"], cfg["data"]["width"]
    scale = np.array([W, H], dtype=np.float32)
    picks = sorted({0, nF // 2, nF - 1})
    records = {"adapter": [], "baseline": []}
    montage_rows, montage_labels = [], []

    def generate(image, target, use_motion, seed, mask=None, prompt=None):
        return backbone.generate(
            image, target, steps=steps, guidance_scale=inf["guidance_scale"], decode_timestep=inf["decode_timestep"],
            seed=seed, motion_guidance=mg, use_motion=use_motion, mask=mask, prompt=prompt,
        )

    for item in items:
        name, image, mask, prompt = item["name"], item["image"], item["mask"], item["prompt"]
        img_np = np.array(image)
        if vfeat.ok and item["video"]:  # the real clip of this val sample = the reference distribution for FVD
            import imageio.v2 as imageio
            real = np.stack([np.asarray(f) for f in imageio.mimread(item["video"], memtest=False)][:nF])
            feats["real"].append(vfeat(real))

        for seed in seeds:
            sfx = f"_s{seed}" if len(seeds) > 1 else ""
            # one baseline video per (image, seed); it is scored against every target path
            first_target = linear_path(item["start"], DIRECTIONS[motions[0]], args.length, nF)
            base_frames = generate(image, first_target, False, seed, mask, prompt)
            save_video(base_frames, os.path.join(args.out, "baseline", f"{name}{sfx}.mp4"), fps=inf["fps"])
            base_track = track_point(cotracker, base_frames, first_target[0] * scale, device)
            base_masks = tracker.track(base_frames, mask) if (tracker.ok and mask is not None) else None
            base_common = {**quality.score(img_np, base_frames),
                           **target_free(base_frames, mask, img_np, base_masks, np.repeat(mask[None], nF, 0) if mask is not None else None)}
            if vfeat.ok:
                feats["baseline"].append(vfeat(base_frames))

            for motion in motions:
                target = linear_path(item["start"], DIRECTIONS[motion], args.length, nF)
                tag = f"{name}_{motion}{sfx}"
                frames = generate(image, target, True, seed, mask, prompt)
                save_video(overlay_trajectory(frames, target), os.path.join(args.out, "adapter", f"{tag}.mp4"), fps=inf["fps"])

                track = track_point(cotracker, frames, target[0] * scale, device)
                masks = tracker.track(frames, mask) if (tracker.ok and mask is not None) else None
                key = {"image": name, "motion": motion, "seed": seed, "group": item["group"]}
                row = {**key, **trajectory_metrics(target, track), **quality.score(img_np, frames),
                       **target_free(frames, mask, img_np, masks, moved_masks(mask, target) if mask is not None else None)}
                base_row = {**key, **trajectory_metrics(target, base_track), **base_common}
                if masks is not None:
                    row.update(object_trajectory_metrics(target, masks))
                if base_masks is not None:
                    base_row.update(object_trajectory_metrics(target, base_masks))
                if vfeat.ok and motion == motions[0]:
                    feats["adapter"].append(vfeat(frames))
                records["adapter"].append(row)
                records["baseline"].append(base_row)

                print(f"{tag}: ATE adapter {row['ate']:.3f} | baseline {base_row['ate']:.3f} "
                      f"| direction_cos {row.get('direction_cos', float('nan')):.2f} "
                      f"| bg_flow adapter {row.get('bg_flow', float('nan')):.4f} baseline {base_row.get('bg_flow', float('nan')):.4f}")

                if len(montage_rows) < 12 and seed == seeds[0]:
                    montage_rows += [draw_row(base_frames, target, picks), draw_row(frames, target, picks)]
                    montage_labels += [f"{tag}\nbaseline", f"{tag}\nadapter"]

    groups = sorted({r["group"] for r in records["adapter"]})
    summary = {"overall": {}, "per_motion": {}, "by_group": {}}
    for label, rows in records.items():
        summary["overall"][label] = aggregate(rows)
        summary["per_motion"][label] = {m: aggregate([r for r in rows if r["motion"] == m]) for m in motions}
        summary["by_group"][label] = {g: aggregate([r for r in rows if r["group"] == g]) for g in groups}
    if len(seeds) > 1:   # how much the result depends on the random seed
        keys = ("ate", "direction_cos", "progress", "obj_ate", "bg_flow", "bg_drift")
        summary["seed_std"] = {label: {k: spread(rows, k, seeds) for k in keys if any(k in r for r in rows)}
                               for label, rows in records.items()}

    # FVD-style distance (VideoMAE features) to the real val clips; only meaningful with enough videos
    if vfeat.ok and len(feats["real"]) >= 2:
        fvd = {}
        for label in ("baseline", "adapter"):
            if len(feats[label]) >= 2:
                fvd[label] = frechet_distance(np.stack(feats[label]), np.stack(feats["real"]))
        summary["fvd_videomae"] = fvd
        summary["fvd_n"] = {k: len(v) for k, v in feats.items()}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"args": vars(args), "adapter_mode": "multilayer/entity", "summary": summary, "records": records}, f, indent=2)
    if montage_rows:
        montage(montage_rows, montage_labels, os.path.join(args.out, "montage.png"))

    cols = ["ate", "epe_final", "pck_5", "pck_10", "direction_cos", "progress",
            "clip_image_sim", "first_frame_psnr", "first_frame_ssim", "color_drift",
            "clip_temporal", "warp_error", "sharpness_ratio", "aesthetic",
            "obj_ate", "obj_direction_cos", "obj_progress", "obj_clip_sim", "obj_area_ratio", "bg_drift", "bg_flow"]
    for k in ("lpips_to_input", "lpips_temporal"):
        if k in summary["overall"]["adapter"]:
            cols.append(k)

    def table(rowsets, columns, first="method"):
        lines = [f"| {first} | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
        for label, o in rowsets:
            lines.append(f"| {label} | " + " | ".join(f"{o.get(c, float('nan')):.3f}" for c in columns) + " |")
        return "\n".join(lines)

    out = ["## overall", table([(l, summary["overall"][l]) for l in ("baseline", "adapter")], cols)]
    if len(groups) > 1:
        out += ["", "## by image group (val = held-out clips, ood = clean out-of-domain images)",
                table([(f"{l} / {g}", summary["by_group"][l][g]) for g in groups for l in ("baseline", "adapter")], cols)]
    short = ["ate", "direction_cos", "progress", "obj_ate", "obj_direction_cos", "bg_drift", "bg_flow"]
    out += ["", "## per motion (opposite directions show forward / backward asymmetry)",
            table([(f"{l} / {m}", summary["per_motion"][l][m]) for m in motions for l in ("baseline", "adapter")], short)]
    if "seed_std" in summary:
        out += ["", f"## std over seeds {seeds}", table([(l, summary["seed_std"][l]) for l in ("baseline", "adapter")], short)]
    text = "\n".join(out)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
