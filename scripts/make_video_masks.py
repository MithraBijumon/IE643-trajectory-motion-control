"""
make_video_masks.py

Per-frame object masks with SAM 2 (video tracking): the first-frame box (box.npy, written by
make_masks.py --use_box) prompts the object on frame 0 and SAM 2 propagates it through the clip.

Writes, per sample folder:
    masks_lowres.npy   (T, 4*GH, 4*GW) uint8, 0-255 = fraction of each low-res cell covered by the object
    (default 60x108 = 4 x the latent grid of the default 480x864 setting; any size is resampled at load time)

and a contact sheet <data_root>/video_masks_preview.jpg (first / middle / last frame of the first clips).

    python scripts/make_video_masks.py --data_root data/dataset_got10k
    python scripts/make_video_masks.py --data_root data/dataset_got10k --limit 3 --splits val   # try it first
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from src.utils import get_device

SAM2_ID = "facebook/sam2.1-hiera-small"
LOW_H, LOW_W = 60, 108


def read_frames(path):
    reader = imageio.get_reader(path)
    try:
        return [np.asarray(f) for f in reader]
    finally:
        reader.close()


@torch.no_grad()
def track(model, processor, frames, box, device, dtype):
    """frames: list of (H, W, 3) uint8; box: (x0, y0, x1, y1) on frame 0. Returns (T, H, W) bool."""
    H, W = frames[0].shape[:2]
    session = processor.init_video_session(video=frames, inference_device=device, dtype=dtype)
    processor.add_inputs_to_inference_session(
        inference_session=session, frame_idx=0, obj_ids=1,
        input_boxes=[[[float(v) for v in box]]],
    )
    masks = np.zeros((len(frames), H, W), dtype=bool)
    out = model(inference_session=session, frame_idx=0)
    m = processor.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
    masks[0] = m[0, 0].cpu().numpy().astype(bool)
    for out in model.propagate_in_video_iterator(session, start_frame_idx=0):
        m = processor.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
        masks[out.frame_idx] = m[0, 0].cpu().numpy().astype(bool)
    return masks


def to_lowres(masks):
    """(T, H, W) bool -> (T, LOW_H, LOW_W) uint8 coverage."""
    return (np.stack([
        cv2.resize(m.astype(np.float32), (LOW_W, LOW_H), interpolation=cv2.INTER_AREA) for m in masks
    ]) * 255).round().astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/dataset_got10k")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--limit", type=int, default=None, help="only the first N clips per split")
    p.add_argument("--model", default=SAM2_ID)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from transformers import Sam2VideoModel, Sam2VideoProcessor

    device = get_device()
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    model = Sam2VideoModel.from_pretrained(args.model).to(device, dtype).eval()
    processor = Sam2VideoProcessor.from_pretrained(args.model)

    tiles, report = [], []
    for split in args.splits.split(","):
        split_dir = os.path.join(args.data_root, split)
        names = [n for n in sorted(os.listdir(split_dir)) if os.path.exists(os.path.join(split_dir, n, "box.npy"))]
        for name in names[: args.limit]:
            folder = os.path.join(split_dir, name)
            out_path = os.path.join(folder, "masks_lowres.npy")
            if os.path.exists(out_path) and not args.overwrite:
                continue

            frames = read_frames(os.path.join(folder, "video.mp4"))
            box = np.load(os.path.join(folder, "box.npy"))
            masks = track(model, processor, frames, box, device, dtype)
            np.save(out_path, to_lowres(masks))

            area = masks.mean(axis=(1, 2))
            empty = int((area == 0).sum())
            jump = float(np.max(np.abs(np.diff(area)) / np.maximum(area[:-1], 1e-6))) if len(area) > 1 else 0.0
            report.append((f"{split}/{name}", empty, jump, float(area.mean())))
            if empty or jump > 3:
                print(f"[check] {split}/{name}: {empty} empty frames, max area jump x{jump:.1f}", flush=True)
            else:
                print(f"[ok]    {split}/{name}  mean area {area.mean():.3f}", flush=True)

            if len(tiles) < 8:
                row = []
                for i in (0, len(frames) // 2, len(frames) - 1):
                    base = frames[i].astype(np.float32)
                    base[masks[i]] = 0.45 * base[masks[i]] + 0.55 * np.array([255, 40, 40], dtype=np.float32)
                    row.append(cv2.resize(base.astype(np.uint8), (256, 160)))
                tiles.append(np.concatenate(row, axis=1))

    print(f"\n{len(report)} clips tracked; {sum(1 for r in report if r[1] or r[2] > 3)} flagged (empty frames or a sudden area jump)")
    if tiles:
        path = os.path.join(args.data_root, "video_masks_preview.jpg")
        Image.fromarray(np.concatenate(tiles, axis=0)).save(path, quality=90)
        print(f"Contact sheet: {path}  (rows = clips, columns = first / middle / last frame)")


if __name__ == "__main__":
    main()
