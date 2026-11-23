"""
train.py

Trains the trajectory condition encoder + motion adapter on top of a frozen LTX-Video.

    python scripts/train.py                      # full model (needs a large GPU)
    python scripts/train.py --tiny               # CPU smoke test with a miniature random LTX
    python scripts/train.py --resume checkpoints/last.pt
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import TrajectoryDataset, collect_prompts
from src.ltx_wrapper import LTXBackbone
from src.utils import apply_checkpoint_cfg, get_device, load_config, seed_everything


def most_moving(dataset, k):
    """The k clips whose first num_frames of trajectory travel the furthest."""
    def travel(sample):
        t = np.load(os.path.join(sample[1], "trajectory.npy"))[: dataset.num_frames]
        return np.linalg.norm(np.diff(t, axis=0), axis=1).sum()

    return sorted(dataset.samples, key=travel, reverse=True)[:k]


def make_loader(cfg, split, train, overfit=0, limit=None):
    """overfit=k: use only the k most-moving train clips, always the same crop, no augmentation, and
    validate on those same clips. A sanity check: if the adapter cannot use the trajectory even here,
    the setup is wrong, not the amount of data."""
    d = cfg["data"]
    dataset = TrajectoryDataset(
        root=d["root"],
        split="train" if overfit else split,
        num_frames=d["num_frames"],
        random_crop=train and not overfit,
        load_video=True,
        height=d["height"],
        width=d["width"],
        repeat=d.get("repeat", 1) if train else 1,
        augment=bool(d.get("augment", False)) and train and not overfit,
        mask_loss=bool(d.get("mask_loss", True)),
        reference=bool(cfg["adapter"].get("reference", False)),
        captions=bool(d.get("captions", True)),
        speed_aug=bool(d.get("speed_aug", True)),
    )
    if overfit:
        dataset.samples = most_moving(dataset, overfit)
    cap = limit or cfg["train"]["max_val_samples"]
    if not train and len(dataset) > cap:
        dataset = Subset(dataset, range(cap))

    workers = d["num_workers"]
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"] if train else 1,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,          # video decoding runs ahead of the GPU instead of stalling it
        prefetch_factor=2 if workers > 0 else None,
    )
    return dataset, loader


VAL_SIGMAS = (0.3, 0.6, 0.9)


def lr_factor(step, total, warmup, floor=0.1):
    """Linear warmup, then cosine decay to `floor` x the base lr. Short runs (a few hundred
    updates) with zero-initialised projections are unstable without the warmup."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(backbone, loader, device, seed=1234, sde=None):
    """Val loss with the trajectory and without it, on identical noise, over a grid of noise levels.

    Motion is decided at high noise, and one random noise level per clip is too coarse to see a
    small effect, so every clip is scored at each of VAL_SIGMAS. Returns
        with / without : mean loss over clips and noise levels
        gain_by_sigma  : {sigma: loss_without - loss_with}
        gain_by_clip   : {clip name: loss_without - loss_with}   (one bad clip shows up here)
    cond_gain = without - with; clearly > 0 means the model uses the trajectory.
    sde=True scores SDEdit-style samples instead (input = the noised drag video, as in --ref_init generation).
    """
    backbone.motion.eval()

    # losses[mode][(clip, sigma)] = loss
    losses = {True: {}, False: {}}
    parts = {True: {"obj": {}, "bg": {}}, False: {"obj": {}, "bg": {}}}   # mean error inside / outside the object mask
    for use in (True, False):
        gen = torch.Generator(device=device).manual_seed(seed)
        for batch in loader:
            batch["video"] = batch["video"].to(device)
            name = batch["name"][0]
            for sg in VAL_SIGMAS:
                loss, pt = backbone.training_loss(
                    batch, generator=gen, use_condition=use, sigma=sg, focus=0.0, return_parts=True, sde=sde
                )
                losses[use][(name, sg)] = loss.item()
                for k in ("obj", "bg"):
                    parts[use][k][(name, sg)] = pt[k]

    backbone.motion.train()

    keys = list(losses[True])
    mean = lambda d, ks: sum(d[k] for k in ks) / max(len(ks), 1)
    gain = lambda ks: mean(losses[False], ks) - mean(losses[True], ks)

    return {
        "with": mean(losses[True], keys),
        "without": mean(losses[False], keys),
        # the same comparison restricted to the object's tokens / the background tokens
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
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--relative", action="store_true", help="multilayer: inject relative to the hidden-state RMS")
    p.add_argument("--no_augment", action="store_true", help="disable random-track + mirror augmentation")
    p.add_argument("--no_entity_embedding", action="store_true", help="entity condition without the per-object feature (footprint + flow only)")
    p.add_argument("--repeat", type=int, default=None, help="visits per clip per epoch (data.repeat)")
    p.add_argument("--focus_loss", type=float, default=None, help="extra training-loss weight around the controlled object: w = 1 + focus * footprint (0 = off)")
    p.add_argument("--max_inject", type=float, default=None, help="cap on the injected residual relative to the hidden state's RMS (config: adapter.max_inject)")
    p.add_argument("--inject_cap", choices=["global", "token"], default=None, help="cap per token (no hot spot) or on the global RMS (old)")
    p.add_argument("--sde_prob", type=float, default=None, help="share of training samples built from the NOISED DRAG VIDEO (SDEdit-style, matches --ref_init at inference), e.g. 0.5")
    p.add_argument("--no_reference", action="store_true", help="ablation: no drag-video condition (adapter.reference false)")
    p.add_argument("--no_captions", action="store_true", help="ablation: the one config prompt for every clip (data.captions false)")
    p.add_argument("--local_inject", type=float, default=None, help="ablation: 0 = inject everywhere, >0 = only near the object (x radius)")
    p.add_argument("--no_grad_ckpt", action="store_true", help="turn gradient checkpointing off: ~30%% faster, needs an 80 GB card")
    p.add_argument("--eval_only", action="store_true", help="with --resume: print cond_gain on val clips and on train clips, then exit")
    p.add_argument("--overfit", type=int, default=0, metavar="K", help="sanity check: train and validate on the K most-moving clips only, fixed crop, no augmentation/dropout")
    args = p.parse_args()

    cfg = load_config(args.config, tiny=args.tiny)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.save_dir:
        cfg["train"]["save_dir"] = args.save_dir
    if args.repeat:
        cfg["data"]["repeat"] = args.repeat
    if args.no_augment:
        cfg["data"]["augment"] = False
    if args.no_entity_embedding:
        cfg["adapter"]["entity_embedding"] = False
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.grad_accum:
        cfg["train"]["grad_accum"] = args.grad_accum
    if args.relative:
        cfg["adapter"]["relative"] = True
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.max_inject is not None:
        cfg["adapter"]["max_inject"] = args.max_inject
    if args.inject_cap is not None:
        cfg["adapter"]["inject_cap"] = args.inject_cap
    if args.sde_prob is not None:
        cfg["train"]["sde_prob"] = args.sde_prob
    if args.no_reference:
        cfg["adapter"]["reference"] = False
    if args.no_captions:
        cfg["data"]["captions"] = False
    if args.local_inject is not None:
        cfg["adapter"]["local_inject"] = args.local_inject
    if args.no_grad_ckpt:
        cfg["model"]["gradient_checkpointing"] = False
    if args.overfit:
        cfg["train"]["cond_dropout"] = 0.0
    if args.focus_loss is not None:
        cfg["train"]["focus_loss"] = args.focus_loss

    if args.resume:
        before = dict(cfg["adapter"])
        apply_checkpoint_cfg(cfg, args.resume)
        changed = {
            k: (before.get(k), cfg["adapter"].get(k))
            for k in set(before) | set(cfg["adapter"])
            if before.get(k) != cfg["adapter"].get(k)
        }
        if changed:
            print(f"NOTE: --resume replaces the adapter settings with the checkpoint's (config, checkpoint): {changed}")

    t = cfg["train"]
    device = get_device()
    seed_everything(t["seed"])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True   # the fp32 parts (condition encoder, losses) use tensor cores
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True          # the 3D convs of the condition encoder see fixed shapes
    os.makedirs(t["save_dir"], exist_ok=True)

    print(f"Device: {device} | tiny: {cfg['is_tiny']} | adapter: multilayer / entity"
          f" | relative: {cfg['adapter'].get('relative', False)} | augment: {cfg['data'].get('augment', False)} | embedding: {cfg['adapter'].get('entity_embedding', True)} | lr {cfg['train']['lr']} accum {cfg['train']['grad_accum']}")
    print("Loading LTX...")
    prompts = collect_prompts(cfg["data"]["root"]) if cfg["data"].get("captions", True) else []
    print(f"{len(prompts)} clip captions")
    backbone = LTXBackbone(cfg, device, prompts=prompts)

    train_set, train_loader = make_loader(cfg, "train", train=True, overfit=args.overfit)
    val_set, val_loader = make_loader(cfg, "val", train=False, overfit=args.overfit)
    # the same cond_gain measurement on training clips (fixed crop): train gain >> val gain = memorising,
    # both ~0 = the adapter has simply not learned yet
    train_eval = None
    if not args.overfit:
        _, train_eval = make_loader(cfg, "train", train=False, limit=6)
    if args.eval_only and not args.resume:
        raise SystemExit("--eval_only needs --resume <checkpoint>")
    if args.overfit:
        print(f"OVERFIT TEST on {[s[0] for s in train_set.samples]} (train and val are the same clips)")
    print(f"Train samples: {len(train_set)} | val samples: {len(val_set)}")
    if len(train_set) == 0:
        raise SystemExit(
            f"No training samples under {cfg['data']['root']}/train. Clips shorter than "
            f"data.num_frames ({cfg['data']['num_frames']}) are skipped; lower num_frames (8k+1)."
        )

    motion = backbone.motion
    optimizer = torch.optim.AdamW(motion.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    print(
        "Trainable parameters:",
        sum(q.numel() for q in motion.parameters() if q.requires_grad),
    )

    sde_val = t.get("sde_prob", 0.0) > 0 and motion.reference     # SDE-style validation + selection (the regime of --ref_init)
    best_on = "val_sde_obj_loss" if sde_val else "val_obj_loss"
    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ckpt = backbone.load_checkpoint(args.resume)
        optimizer.load_state_dict(ckpt["optimizer"])
        # load_state_dict restores the old lrs; re-apply the ones from the config
        optimizer.param_groups[0]["lr"] = t["lr"]
        start_epoch = ckpt["epoch"]
        if ckpt.get("best_on") == best_on:   # checkpoints that selected on another loss are not comparable
            best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if args.eval_only:
        for label, loader in (("val", val_loader), ("train", train_eval)):
            if loader is None:
                continue
            v = validate(backbone, loader, device)
            print(json.dumps({
                "split": label,
                "loss_with": v["with"],
                "loss_without": v["without"],
                "cond_gain": v["without"] - v["with"],
                "gain_by_sigma": {str(k): round(x, 5) for k, x in v["gain_by_sigma"].items()},
                "gain_by_clip": {k: round(x, 5) for k, x in v["gain_by_clip"].items()},
            }))
        return

    log_path = os.path.join(t["save_dir"], "log.jsonl")
    accum = t["grad_accum"]
    motion.train()

    # an epoch is a random subset of at most max_visits_per_epoch visits (the loader reshuffles every epoch),
    # so validation / checkpoints / divergence checks come at a predictable interval
    visits = min(len(train_loader), t.get("max_visits_per_epoch") or len(train_loader))
    updates_per_epoch = math.ceil(visits / accum)
    total_updates = updates_per_epoch * t["epochs"]
    warmup = max(1, int(t.get("warmup_frac", 0.05) * total_updates))
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    global_update = start_epoch * updates_per_epoch
    print(f"{visits} visits/epoch (of {len(train_loader)}) -> {updates_per_epoch} updates/epoch, {total_updates} in total (warmup {warmup})")

    for epoch in range(start_epoch, t["epochs"]):
        t0 = time.time()
        running, n = torch.zeros((), device=device), 0
        norms, skipped = [], 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch["video"] = batch["video"].to(device, non_blocking=True)

            loss = backbone.training_loss(batch)
            (loss / accum).backward()

            running += loss.detach()   # a tensor: no GPU->CPU sync every step
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
                    skipped += 1  # a NaN/inf gradient would poison the weights for good
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            if (step + 1) % t["log_every"] == 0:
                print(
                    f"Epoch {epoch + 1}/{t['epochs']} step {step + 1}/{visits} "
                    f"({(time.time() - t0) / (step + 1) * (visits - step - 1) / 60:.1f} min left) loss {loss.item():.5f} gate {motion.gate_value():.5f} inject {motion.injection_ratio():.4f}"
                )

            if step + 1 >= visits:
                break

        train_loss = float(running) / max(n, 1)
        inject_ratio = motion.injection_ratio()  # read now: validation below switches the trajectory off
        if len(val_set):
            v = validate(backbone, val_loader, device)
            val_loss, val_loss_nocond = v["with"], v["without"]
            vs = None
            if sde_val:   # the same measurement in the regime --ref_init generation runs in
                vs = validate(backbone, val_loader, device, sde=True)
        else:
            v = {"gain_by_sigma": {}, "gain_by_clip": {}, "obj_with": float("nan"), "obj_without": float("nan"),
                 "bg_with": float("nan"), "bg_without": float("nan")}
            vs = None
            val_loss = val_loss_nocond = float("nan")

        train_gain = None
        if train_eval is not None:
            tv = validate(backbone, train_eval, device)
            train_gain = tv["without"] - tv["with"]

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_loss_nocond": val_loss_nocond,
            "cond_gain": val_loss_nocond - val_loss,
            "train_cond_gain": train_gain,
            "val_obj_gain": v["obj_without"] - v["obj_with"],   # cond_gain on the object's tokens only
            "val_bg_gain": v["bg_without"] - v["bg_with"],      # ... and on the background (should stay ~0)
            "val_obj_loss": v["obj_with"], "val_bg_loss": v["bg_with"],
            "gain_by_sigma": {str(k): round(x, 5) for k, x in v["gain_by_sigma"].items()},
            "gain_by_clip": {k: round(x, 5) for k, x in v["gain_by_clip"].items()},
            **({"val_sde_obj_gain": vs["obj_without"] - vs["obj_with"], "val_sde_obj_loss": vs["obj_with"],
                "val_sde_gain": vs["without"] - vs["with"], "val_sde_bg_gain": vs["bg_without"] - vs["bg_with"],
                "sde_gain_by_sigma": {str(k): round(x, 5) for k, x in vs["gain_by_sigma"].items()}} if vs else {}),
            "adapter_scale": motion.gate_value(),
            "inject_ratio": inject_ratio,
            "grad_norm_mean": round(sum(norms) / max(len(norms), 1), 4),
            "grad_norm_max": round(max(norms, default=0.0), 4),
            "skipped_updates": skipped,
            "seconds": round(time.time() - t0, 1),
        }
        print(json.dumps(record))
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # best.pt = lowest loss on the object's tokens (val_obj_loss); the overall val_loss is dominated by background tokens
        # (with --sde_prob the model is used from the noised drag video, so that is the loss to select on)
        score = vs["obj_with"] if vs else v["obj_with"]
        is_best = len(val_set) > 0 and score < best_val
        if is_best:
            best_val = score

        extra = {
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
            "best_on": best_on,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        backbone.save_checkpoint(os.path.join(t["save_dir"], "last.pt"), **extra)
        if is_best:
            backbone.save_checkpoint(os.path.join(t["save_dir"], "best.pt"), **extra)
        if (epoch + 1) % t["save_every"] == 0:
            backbone.save_checkpoint(
                os.path.join(t["save_dir"], f"checkpoint_{epoch + 1}.pt"), **extra
            )

    print(f"Done. Checkpoints in {t['save_dir']}")


if __name__ == "__main__":
    main()
