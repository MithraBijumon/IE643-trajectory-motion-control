"""
inference.py

Generate a video from a first-frame image and a target trajectory.

    # from a preprocessed sample folder (image.png + trajectory.npy [+ visibility.npy])
    python scripts/inference.py --checkpoint checkpoints/best.pt --sample_dir data/dataset/val/goat

    # from your own image + a (T, 2) normalized trajectory
    python scripts/inference.py --checkpoint checkpoints/best.pt --image a.png --trajectory t.npy

    # pipeline check with a miniature random LTX
    python scripts/inference.py --tiny --checkpoint <tiny ckpt> --sample_dir data/dataset/val/goat
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.ltx_wrapper import LTXBackbone
from src.utils import (
    apply_checkpoint_cfg,
    get_device,
    load_config,
    load_image,
    overlay_trajectory,
    resample_trajectory,
    save_video,
)


def load_sample(args, cfg):
    d = cfg["data"]

    if args.sample_dir:
        image_path = os.path.join(args.sample_dir, "image.png")
        traj_path = os.path.join(args.sample_dir, "trajectory.npy")
        vis_path = os.path.join(args.sample_dir, "visibility.npy")
    else:
        image_path, traj_path, vis_path = args.image, args.trajectory, None

    image = load_image(image_path, d["height"], d["width"])
    trajectory = np.load(traj_path)
    visible = np.load(vis_path) if vis_path and os.path.exists(vis_path) else None

    # Use the first num_frames of a long trajectory; stretch a short one.
    nF = d["num_frames"]
    if len(trajectory) >= nF and args.sample_dir:
        trajectory = trajectory[:nF]
        visible = visible[:nF] if visible is not None else None
    else:
        trajectory, visible = resample_trajectory(trajectory, nF, visible)

    return image, trajectory, visible


def run(backbone, cfg, image, trajectory, visible, out_path, seed=None, steps=None, guidance=None,
        motion_guidance=None, use_motion=True):
    inf = cfg["inference"]
    frames = backbone.generate(
        image,
        trajectory,
        visible,
        steps=steps or inf["steps"],
        guidance_scale=inf["guidance_scale"] if guidance is None else guidance,
        decode_timestep=inf["decode_timestep"],
        seed=inf["seed"] if seed is None else seed,
        motion_guidance=inf["motion_guidance"] if motion_guidance is None else motion_guidance,
        use_motion=use_motion,
    )

    save_video(frames, out_path, fps=inf["fps"])
    stem, _ = os.path.splitext(out_path)
    save_video(overlay_trajectory(frames, trajectory), f"{stem}_overlay.mp4", fps=inf["fps"])
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--checkpoint", default=None, help="Adapter checkpoint; omit to run the untrained adapter")
    p.add_argument("--sample_dir", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--trajectory", default=None)
    p.add_argument("--out", default="results/adapter/generated.mp4")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--motion_guidance", type=float, default=None, help=">1 amplifies the trajectory")
    p.add_argument("--no_motion", action="store_true", help="run plain LTX image-to-video (baseline)")
    args = p.parse_args()

    if not args.sample_dir and not (args.image and args.trajectory):
        p.error("give --sample_dir, or both --image and --trajectory")

    cfg = apply_checkpoint_cfg(load_config(args.config, tiny=args.tiny), args.checkpoint)
    backbone = LTXBackbone(cfg, get_device())

    if args.checkpoint:
        backbone.load_checkpoint(args.checkpoint)
        print(f"Loaded {args.checkpoint}")

    image, trajectory, visible = load_sample(args, cfg)
    run(backbone, cfg, image, trajectory, visible, args.out, args.seed, args.steps, args.guidance,
        args.motion_guidance, use_motion=not args.no_motion)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
