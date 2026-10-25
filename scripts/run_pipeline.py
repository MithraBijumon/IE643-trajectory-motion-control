"""
run_pipeline.py

End-to-end: preprocess -> index -> train -> generate -> evaluate.

    python scripts/run_pipeline.py                         # full run on data/dataset
    python scripts/run_pipeline.py --pickle data/tapvid_davis.pkl   # build the dataset first
    python scripts/run_pipeline.py --tiny --skip_eval      # CPU smoke test of every stage
    python scripts/run_pipeline.py --stages train,generate # run only some stages

Stages: preprocess, index, masks, train, generate, evaluate  (masks only matter for --condition entity)
Generation runs every val sample twice: with the trained adapter (results/adapter)
and with the adapter switched off (results/baseline_ltx), which is plain
image-to-video LTX. Evaluation reports ATE for both so the adapter's effect is explicit.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import torch

from src.dataset import TrajectoryDataset  # noqa: F401  (fail early if the package is broken)
from src.utils import apply_checkpoint_cfg, get_device, load_config

STAGES = ["preprocess", "index", "masks", "train", "generate", "evaluate"]


def sh(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def val_samples(data_root, limit=None):
    split = os.path.join(data_root, "val")
    names = [
        d for d in sorted(os.listdir(split))
        if os.path.exists(os.path.join(split, d, "trajectory.npy"))
    ]
    return names[:limit] if limit else names


def stage_generate(args, cfg, ckpt, out_dir, base_dir):
    limit = args.max_val
    from inference import load_sample, run
    from src.ltx_wrapper import LTXBackbone

    apply_checkpoint_cfg(cfg, ckpt)
    backbone = LTXBackbone(cfg, get_device())
    backbone.load_checkpoint(ckpt)

    for name in val_samples(cfg["data"]["root"], limit):
        ns = argparse.Namespace(
            sample_dir=os.path.join(cfg["data"]["root"], "val", name), image=None, trajectory=None
        )
        image, trajectory, visible, mask = load_sample(ns, cfg)

        print(f"[generate] {name}: with adapter")
        run(backbone, cfg, image, trajectory, visible, os.path.join(out_dir, f"{name}.mp4"), mask=mask)

        print(f"[generate] {name}: baseline (adapter off)")
        run(backbone, cfg, image, trajectory, visible, os.path.join(base_dir, f"{name}.mp4"),
            use_motion=False, mask=mask)

    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_evaluate(cfg, out_dir, base_dir, metrics_path, limit=None):
    from evaluate import evaluate_one, load_clip, load_cotracker

    device = get_device()
    cotracker = load_cotracker(device)
    clip_bundle = load_clip(device)
    results = {}

    for name in val_samples(cfg["data"]["root"], limit):
        traj = os.path.join(cfg["data"]["root"], "val", name, "trajectory.npy")
        row = {}
        for label, folder in (("adapter", out_dir), ("baseline", base_dir)):
            row[label] = evaluate_one(
                os.path.join(folder, f"{name}.mp4"), traj, None, cotracker, clip_bundle, device
            )["ate"]
        results[name] = row
        print(f"[evaluate] {name}: ATE adapter {row['adapter']:.4f} | baseline {row['baseline']:.4f}")

    summary = {
        k: float(np.mean([r[k] for r in results.values()])) for k in ("adapter", "baseline")
    }
    print(f"\nMean ATE (lower is better): adapter {summary['adapter']:.4f} | baseline {summary['baseline']:.4f}")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"per_sample": results, "mean": summary}, f, indent=2)
    print(f"Wrote {metrics_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--pickle", default=None, help="TAP-Vid pickle; builds the dataset in --data_root")
    p.add_argument("--data_root", default=None)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--results_dir", default=None)
    p.add_argument("--stages", default=None, help="comma list; default: all (masks only with --condition entity)")
    p.add_argument("--skip_eval", action="store_true", help="skip CoTracker evaluation (needs a download)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--variant", choices=["input", "multilayer"], default=None)
    p.add_argument("--condition", choices=["points", "entity"], default=None)
    p.add_argument("--repeat", type=int, default=None)
    p.add_argument("--max_val", type=int, default=None, help="only generate/evaluate the first N val clips")
    args = p.parse_args()

    if args.stages is None:
        default = [s for s in STAGES if s != "masks" or args.condition == "entity"]
        args.stages = ",".join(default)
    stages = [s for s in args.stages.split(",") if s]
    if args.skip_eval and "evaluate" in stages:
        stages.remove("evaluate")
    for s in stages:
        if s not in STAGES:
            p.error(f"unknown stage {s}; choose from {STAGES}")

    cfg = load_config(args.config, tiny=args.tiny)
    data_root = args.data_root or cfg["data"]["root"]
    cfg["data"]["root"] = data_root
    ckpt_dir = args.ckpt_dir or ("checkpoints_tiny" if args.tiny else cfg["train"]["save_dir"])
    results_dir = args.results_dir or ("results/tiny" if args.tiny else "results")
    out_dir = os.path.join(results_dir, "adapter")
    base_dir = os.path.join(results_dir, "baseline_ltx")

    py = sys.executable
    common = ["--tiny"] if args.tiny else []
    if args.config:
        common += ["--config", args.config]

    if "preprocess" in stages:
        if not args.pickle:
            print("[preprocess] no --pickle given, using the existing dataset")
        else:
            sh([py, "scripts/preprocess_tapvid.py", "--pickle_path", args.pickle, "--out_dir", data_root])

    if "index" in stages:
        sh([py, "scripts/build_index.py", "--dataset_dir", data_root])

    if "masks" in stages:
        sh([py, "scripts/make_masks.py", "--data_root", data_root])

    if "train" in stages:
        cmd = [py, "scripts/train.py", *common, "--data_root", data_root, "--save_dir", ckpt_dir]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        if args.variant:
            cmd += ["--variant", args.variant]
        if args.condition:
            cmd += ["--condition", args.condition]
        if args.repeat:
            cmd += ["--repeat", str(args.repeat)]
        sh(cmd)

    ckpt = os.path.join(ckpt_dir, "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(ckpt_dir, "last.pt")

    if "generate" in stages:
        stage_generate(args, cfg, ckpt, out_dir, base_dir)

    if "evaluate" in stages:
        stage_evaluate(cfg, out_dir, base_dir, os.path.join(results_dir, "metrics.json"), args.max_val)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
