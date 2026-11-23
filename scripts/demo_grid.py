"""
demo_grid.py
Quick visual check: the same val clips and the same straight path, generated with several variants, in one contact sheet
(rows = clip x variant, columns = first / middle / last frame, target path drawn). Loads the model once.

    python scripts/demo_grid.py --config configs/ltx2b.yaml --num 3 --direction right \
        --variants base,adapter,ref0.8,ref0.8+comp2.5 [--checkpoint <best.pt>]

A variant is "base" (plain LTX, no adapter) or "adapter" plus options joined with "+":
    ref0.8    start denoising from the drag video noised to sigma 0.8 (training-free; works without a checkpoint)
    lock2.5   background lock: outside 2.5 object radii the velocity is plain LTX's (keeps whatever plain LTX shows, ghosts included)
    comp2.5   scene composite: outside 2.5 object radii the tokens are the drag video's object-free scene (removes the old copy)
    as0.3     adapter strength 0.3 of what was trained (as0 = adapter off but drag video/composite on: the training-free path
              with the same checkpoint loaded; every variant except 'base' otherwise uses the adapter when --checkpoint is given)
    until0.4  the adapter acts only while sigma >= 0.4; plain LTX finishes the detail
    rel0      with comp: keep compositing until the last step (default releases below sigma 0.25)
    mg3       motion guidance 3
Without --checkpoint the adapter is untrained (a no-op), so "ref0.8" shows the training-free effect alone.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
from PIL import Image

from benchmark import draw_row, montage, val_items
from src.ltx_wrapper import LTXBackbone
from src.utils import DIRECTIONS, apply_checkpoint_cfg, get_device, linear_path, load_config, save_video


def parse(variant):
    opts = {"use_motion": variant != "base", "ref_init": 0.0, "bg_lock": 0.0, "bg_comp": 0.0, "motion_guidance": 1.0, "release": None, "adapter_scale": None, "adapter_min_sigma": None}
    for part in variant.split("+")[0:]:
        if part.startswith("ref"):
            opts["ref_init"] = float(part[3:])
        elif part.startswith("lock"):
            opts["bg_lock"] = float(part[4:])
        elif part.startswith("comp"):
            opts["bg_comp"] = float(part[4:])
        elif part.startswith("as"):
            opts["adapter_scale"] = float(part[2:])
        elif part.startswith("until"):
            opts["adapter_min_sigma"] = float(part[5:])
        elif part.startswith("rel"):
            opts["release"] = float(part[3:])
        elif part.startswith("mg"):
            opts["motion_guidance"] = float(part[2:])
        elif part not in ("base", "adapter"):
            raise SystemExit(f"unknown variant part '{part}' in '{variant}'")
    return opts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--num", type=int, default=3)
    p.add_argument("--direction", default="right", choices=list(DIRECTIONS))
    p.add_argument("--length", type=float, default=0.3)
    p.add_argument("--variants", default="base,adapter,ref0.8,ref0.8+comp2.5")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/demo_grid")
    args = p.parse_args()

    cfg = apply_checkpoint_cfg(load_config(args.config, tiny=args.tiny), args.checkpoint)
    root = args.data_root or cfg["data"]["root"]
    inf, nF = cfg["inference"], cfg["data"]["num_frames"]
    items = val_items(root, args.num, cfg)
    backbone = LTXBackbone(cfg, get_device(), prompts=[i["prompt"] for i in items if i["prompt"]])
    if args.checkpoint:
        backbone.load_checkpoint(args.checkpoint)

    picks = sorted({0, nF // 2, nF - 1})
    rows, labels = [], []
    os.makedirs(args.out, exist_ok=True)
    for item in items:
        target = linear_path(item["start"], DIRECTIONS[args.direction], args.length, nF)
        for v in args.variants.split(","):
            o = parse(v)
            rel = o.pop("release")
            cfg["inference"]["bg_comp_release"] = 0.25 if rel is None else rel   # rel0 = keep compositing to the very end
            frames = backbone.generate(
                item["image"], target, steps=args.steps or inf["steps"], guidance_scale=inf["guidance_scale"],
                decode_timestep=inf["decode_timestep"], seed=args.seed, mask=item["mask"], prompt=item["prompt"], **o,
            )
            save_video(frames, os.path.join(args.out, f"{item['name']}_{args.direction}_{v}.mp4"), fps=inf["fps"])
            rows.append(draw_row(frames, target, picks))
            labels.append(f"{item['name'][-6:]} {args.direction}\n{v}")
            print("done", item["name"], v, flush=True)
    montage(rows, labels, os.path.join(args.out, "grid.png"))
    print("wrote", os.path.join(args.out, "grid.png"))


if __name__ == "__main__":
    main()
