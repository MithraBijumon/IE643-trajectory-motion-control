"""
Trains the trajectory adapter (condition encoder + zero-init projections) on a frozen LTX-Video.

    python scripts/train.py                          # configs/config.yaml (needs a large GPU)
    python scripts/train.py --tiny                   # CPU smoke test with a miniature random LTX
    python scripts/train.py --resume checkpoints/last.pt
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import TrajectoryDataset, collect_prompts
from src.ltx_wrapper import LTXBackbone
from src.utils import apply_checkpoint_cfg, get_device, load_config, seed_everything

VAL_SIGMAS = (0.3, 0.6, 0.9)


def make_loader(cfg, split, train):
    d = cfg["data"]
    dataset = TrajectoryDataset(
        root=d["root"], split=split, num_frames=d["num_frames"], random_crop=train, height=d["height"], width=d["width"],
        repeat=d.get("repeat", 1) if train else 1, augment=bool(d.get("augment", False)) and train,
        mask_loss=bool(d.get("mask_loss", True)), captions=bool(d.get("captions", True)),
    )
    if not train and len(dataset) > cfg["train"]["max_val_samples"]:
        dataset = Subset(dataset, range(cfg["train"]["max_val_samples"]))
    workers = d["num_workers"]
    loader = DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"] if train else 1, shuffle=train, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
    )
    return dataset, loader


def lr_factor(step, total, warmup, floor=0.1):
    """Linear warmup, then cosine decay to `floor` x the base lr."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(backbone, loader, device, seed=1234):
    """Val loss with the trajectory and without it, on identical noise, at each of VAL_SIGMAS (motion is decided at
    high noise). gain = loss_without - loss_with; clearly > 0 means the model uses the trajectory. Also reported on the
    object's tokens only and on the background tokens (whose gain should stay ~0)."""
    backbone.motion.eval()
    loss = {True: {}, False: {}}
    parts = {True: {"obj": {}, "bg": {}}, False: {"obj": {}, "bg": {}}}
    for use in (True, False):
        gen = torch.Generator(device=device).manual_seed(seed)
        for batch in loader:
            batch["video"] = batch["video"].to(device)
            name = batch["name"][0]
            for sg in VAL_SIGMAS:
                l, pt = backbone.training_loss(batch, generator=gen, use_condition=use, sigma=sg, focus=0.0, return_parts=True)
                loss[use][(name, sg)] = l.item()
                for k in ("obj", "bg"):
                    parts[use][k][(name, sg)] = pt[k]
    backbone.motion.train()

    keys = list(loss[True])
    mean = lambda d, ks: sum(d[k] for k in ks) / max(len(ks), 1)
    gain = lambda ks: mean(loss[False], ks) - mean(loss[True], ks)
    return {
        "with": mean(loss[True], keys), "without": mean(loss[False], keys),
        "obj_with": mean(parts[True]["obj"], keys), "obj_without": mean(parts[False]["obj"], keys),
        "bg_with": mean(parts[True]["bg"], keys), "bg_without": mean(parts[False]["bg"], keys),
        "gain_by_sigma": {sg: gain([k for k in keys if k[1] == sg]) for sg in VAL_SIGMAS},
        "gain_by_clip": {n: gain([k for k in keys if k[0] == n]) for n in dict.fromkeys(k[0] for k in keys)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--data_root", default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--focus_loss", type=float, default=None, help="extra loss weight on the object tokens (0 = off)")
    p.add_argument("--max_inject", type=float, default=None, help="cap on the injection relative to the hidden RMS")
    p.add_argument("--inject_cap", choices=["global", "token"], default=None, help="cap the RMS over all tokens, or every token")
    args = p.parse_args()

    cfg = load_config(args.config, tiny=args.tiny)
    for value, section, key in [(args.data_root, "data", "root"), (args.save_dir, "train", "save_dir"), (args.epochs, "train", "epochs"),
                                (args.lr, "train", "lr"), (args.focus_loss, "train", "focus_loss"),
                                (args.max_inject, "adapter", "max_inject"), (args.inject_cap, "adapter", "inject_cap")]:
        if value is not None:
            cfg[section][key] = value
    if args.resume:
        apply_checkpoint_cfg(cfg, args.resume)

    t = cfg["train"]
    device = get_device()
    seed_everything(t["seed"])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(t["save_dir"], exist_ok=True)

    print(f"Device: {device} | tiny: {cfg['is_tiny']} | lr {t['lr']} | focus_loss {t.get('focus_loss', 0)} | accum {t['grad_accum']}")
    prompts = collect_prompts(cfg["data"]["root"]) if cfg["data"].get("captions", True) else []
    print(f"Loading LTX... ({len(prompts)} clip captions)")
    backbone = LTXBackbone(cfg, device, prompts=prompts)

    train_set, train_loader = make_loader(cfg, "train", train=True)
    val_set, val_loader = make_loader(cfg, "val", train=False)
    print(f"Train samples: {len(train_set)} | val samples: {len(val_set)}")
    if len(train_set) == 0:
        raise SystemExit(f"No training samples under {cfg['data']['root']}/train (clips shorter than data.num_frames are skipped).")

    motion = backbone.motion
    optimizer = torch.optim.AdamW(motion.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    print("Trainable parameters:", sum(q.numel() for q in motion.parameters() if q.requires_grad))

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ckpt = backbone.load_checkpoint(args.resume)
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer.param_groups[0]["lr"] = t["lr"]   # load_state_dict restores the old lr; re-apply the config's
        start_epoch, best_val = ckpt["epoch"], ckpt.get("best_val", best_val)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    log_path = os.path.join(t["save_dir"], "log.jsonl")
    accum = t["grad_accum"]
    motion.train()

    # an epoch is a random subset of at most max_visits_per_epoch visits, so validation and checkpoints come at a fixed interval
    visits = min(len(train_loader), t.get("max_visits_per_epoch") or len(train_loader))
    updates_per_epoch = math.ceil(visits / accum)
    total_updates = updates_per_epoch * t["epochs"]
    warmup = max(1, int(t.get("warmup_frac", 0.05) * total_updates))
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    global_update = start_epoch * updates_per_epoch
    print(f"{visits} visits/epoch -> {updates_per_epoch} updates/epoch, {total_updates} in total (warmup {warmup})")

    for epoch in range(start_epoch, t["epochs"]):
        t0 = time.time()
        running, n = torch.zeros((), device=device), 0
        norms, skipped = [], 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch["video"] = batch["video"].to(device, non_blocking=True)
            loss = backbone.training_loss(batch)
            (loss / accum).backward()
            running += loss.detach()
            n += 1

            if (step + 1) % accum == 0 or step + 1 == visits:
                f = lr_factor(global_update, total_updates, warmup, t.get("lr_floor", 0.1))
                for g, base in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = base * f
                gn = float(torch.nn.utils.clip_grad_norm_(motion.parameters(), t["grad_clip"]))
                if math.isfinite(gn):
                    norms.append(gn)
                    optimizer.step()
                else:
                    skipped += 1   # a NaN/inf gradient would poison the weights for good
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            if (step + 1) % t["log_every"] == 0:
                eta = (time.time() - t0) / (step + 1) * (visits - step - 1) / 60
                print(f"Epoch {epoch + 1}/{t['epochs']} step {step + 1}/{visits} ({eta:.1f} min left) loss {loss.item():.5f} "
                      f"gate {motion.gate_value():.5f} inject {motion.injection_ratio():.4f}")
            if step + 1 >= visits:
                break

        inject_ratio = motion.injection_ratio()   # read now: validation switches the trajectory off
        nan = float("nan")
        v = validate(backbone, val_loader, device) if len(val_set) else {
            "with": nan, "without": nan, "obj_with": nan, "obj_without": nan, "bg_with": nan, "bg_without": nan,
            "gain_by_sigma": {}, "gain_by_clip": {}}

        record = {
            "epoch": epoch + 1,
            "train_loss": float(running) / max(n, 1),
            "val_loss": v["with"], "val_loss_nocond": v["without"], "cond_gain": v["without"] - v["with"],
            "val_obj_gain": v["obj_without"] - v["obj_with"],   # cond_gain on the object's tokens only
            "val_bg_gain": v["bg_without"] - v["bg_with"],      # ... and on the background (should stay ~0)
            "val_obj_loss": v["obj_with"], "val_bg_loss": v["bg_with"],
            "gain_by_sigma": {str(k): round(x, 5) for k, x in v["gain_by_sigma"].items()},
            "gain_by_clip": {k: round(x, 5) for k, x in v["gain_by_clip"].items()},
            "adapter_scale": motion.gate_value(), "inject_ratio": inject_ratio,
            "grad_norm_mean": round(sum(norms) / max(len(norms), 1), 4), "grad_norm_max": round(max(norms, default=0.0), 4),
            "skipped_updates": skipped, "seconds": round(time.time() - t0, 1),
        }
        print(json.dumps(record))
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # best.pt = lowest loss on the object's tokens (the overall val_loss is dominated by background tokens)
        is_best = len(val_set) > 0 and v["obj_with"] < best_val
        if is_best:
            best_val = v["obj_with"]
        extra = {"epoch": epoch + 1, "optimizer": optimizer.state_dict(), "best_val": best_val}
        backbone.save_checkpoint(os.path.join(t["save_dir"], "last.pt"), **extra)
        if is_best:
            backbone.save_checkpoint(os.path.join(t["save_dir"], "best.pt"), **extra)
        if (epoch + 1) % t["save_every"] == 0:
            backbone.save_checkpoint(os.path.join(t["save_dir"], f"checkpoint_{epoch + 1}.pt"), **extra)

    print(f"Done. Checkpoints in {t['save_dir']}")


if __name__ == "__main__":
    main()
