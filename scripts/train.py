"""
train.py

Trains the trajectory condition encoder + motion adapter on top of a frozen LTX-Video.

    python scripts/train.py                      # full model (needs a large GPU)
    python scripts/train.py --tiny               # CPU smoke test with a miniature random LTX
    python scripts/train.py --resume checkpoints/last.pt
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import TrajectoryDataset
from src.ltx_wrapper import LTXBackbone
from src.utils import apply_checkpoint_cfg, get_device, load_config, seed_everything


def make_loader(cfg, split, train):
    d = cfg["data"]
    dataset = TrajectoryDataset(
        root=d["root"],
        split=split,
        num_frames=d["num_frames"],
        random_crop=train,
        load_video=True,
        height=d["height"],
        width=d["width"],
        repeat=d.get("repeat", 1) if train else 1,
        augment=bool(d.get("augment", False)) and train,
    )
    if not train and len(dataset) > cfg["train"]["max_val_samples"]:
        dataset = Subset(dataset, range(cfg["train"]["max_val_samples"]))

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"] if train else 1,
        shuffle=train,
        num_workers=d["num_workers"],
    )
    return dataset, loader


@torch.no_grad()
def validate(backbone, loader, device, seed=1234):
    """Val loss with the trajectory and without it, on identical noise and timesteps.

    cond_gain = loss_without - loss_with. Clearly > 0 means the model uses the trajectory.
    """
    backbone.motion.eval()

    totals = {}
    for name, use in (("with", True), ("without", False)):
        gen = torch.Generator(device=device).manual_seed(seed)
        losses = []
        for batch in loader:
            batch["video"] = batch["video"].to(device)
            losses.append(backbone.training_loss(batch, generator=gen, use_condition=use).item())
        totals[name] = sum(losses) / max(len(losses), 1)

    backbone.motion.train()
    return totals["with"], totals["without"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--data_root", default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--variant", choices=["input", "multilayer"], default=None, help="adapter mode (overrides config)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--relative", action="store_true", help="multilayer: inject relative to the hidden-state RMS")
    p.add_argument("--no_augment", action="store_true", help="disable random-track + mirror augmentation")
    p.add_argument("--no_entity_embedding", action="store_true", help="entity condition without the per-object feature (footprint + flow only)")
    p.add_argument("--repeat", type=int, default=None, help="visits per clip per epoch (data.repeat)")
    p.add_argument("--condition", choices=["points", "entity"], default=None, help="condition type (overrides config)")
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

    if args.variant:
        cfg["adapter"]["mode"] = args.variant
    if args.condition:
        cfg["adapter"]["condition"] = args.condition
    if args.resume:
        apply_checkpoint_cfg(cfg, args.resume)

    t = cfg["train"]
    device = get_device()
    seed_everything(t["seed"])
    os.makedirs(t["save_dir"], exist_ok=True)

    print(f"Device: {device} | tiny: {cfg['is_tiny']} | adapter: {cfg['adapter']['mode']} / {cfg['adapter'].get('condition', 'points')}"
          f" | relative: {cfg['adapter'].get('relative', False)} | augment: {cfg['data'].get('augment', False)} | embedding: {cfg['adapter'].get('entity_embedding', True)} | lr {cfg['train']['lr']} accum {cfg['train']['grad_accum']}")
    print("Loading LTX...")
    backbone = LTXBackbone(cfg, device)

    train_set, train_loader = make_loader(cfg, "train", train=True)
    val_set, val_loader = make_loader(cfg, "val", train=False)
    print(f"Train samples: {len(train_set)} | val samples: {len(val_set)}")
    if len(train_set) == 0:
        raise SystemExit(
            f"No training samples under {cfg['data']['root']}/train. Clips shorter than "
            f"data.num_frames ({cfg['data']['num_frames']}) are skipped; lower num_frames (8k+1)."
        )

    motion = backbone.motion
    scale_params = motion.gate_params()
    scale_ids = {id(q) for q in scale_params}
    other_params = [q for q in motion.parameters() if id(q) not in scale_ids]

    groups = [{"params": other_params, "lr": t["lr"]}]
    if scale_params:
        groups.append({"params": scale_params, "lr": t["scale_lr"], "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])

    print(
        "Trainable parameters:",
        sum(q.numel() for q in motion.parameters() if q.requires_grad),
    )

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ckpt = backbone.load_checkpoint(args.resume)
        optimizer.load_state_dict(ckpt["optimizer"])
        # load_state_dict restores the old lrs; re-apply the ones from the config
        optimizer.param_groups[0]["lr"] = t["lr"]
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]["lr"] = t["scale_lr"]
        start_epoch = ckpt["epoch"]
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    log_path = os.path.join(t["save_dir"], "log.jsonl")
    accum = t["grad_accum"]
    motion.train()

    for epoch in range(start_epoch, t["epochs"]):
        t0 = time.time()
        running, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch["video"] = batch["video"].to(device)

            loss = backbone.training_loss(batch)
            (loss / accum).backward()

            running += loss.item()
            n += 1

            if (step + 1) % accum == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(motion.parameters(), t["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % t["log_every"] == 0:
                print(
                    f"Epoch {epoch + 1}/{t['epochs']} step {step + 1}/{len(train_loader)} "
                    f"loss {loss.item():.5f} gate {motion.gate_value():.5f}"
                )

        train_loss = running / max(n, 1)
        if len(val_set):
            val_loss, val_loss_nocond = validate(backbone, val_loader, device)
        else:
            val_loss = val_loss_nocond = float("nan")

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_loss_nocond": val_loss_nocond,
            "cond_gain": val_loss_nocond - val_loss,
            "adapter_scale": motion.gate_value(),
            "seconds": round(time.time() - t0, 1),
        }
        print(json.dumps(record))
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        is_best = len(val_set) > 0 and val_loss < best_val
        if is_best:
            best_val = val_loss

        extra = {
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
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
