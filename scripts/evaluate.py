#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
evaluate.py

Evaluates how well a video follows a target trajectory and how good it looks.

Metrics (per video, then averaged over a set of videos):
  1. ATE  (Average Trajectory Error, lower = better)
       CoTracker3 tracks where the point really went in the video; we compare
       that to the target trajectory. Normalized [0,1] coordinates.
  2. Static-baseline ATE (reference only)
       The ATE you would get if the point never moved. Gives the ATE numbers
       some context: a good result should be clearly below this.
  3. Image similarity (higher = better)
       CLIP cosine similarity between the input image and every video frame,
       averaged. Measures "does the video still look like the input image".
  4. Temporal consistency (higher = better)
       CLIP cosine similarity between consecutive frames, averaged.
       Measures frame-to-frame smoothness / lack of flicker.

Two ways to run it
A) One video (the original usage works):
    python evaluate.py --video_path dataset_kubric/val/sample_0001/video.mp4 \
                       --trajectory_path dataset_kubric/val/sample_0001/trajectory.npy \
                       --image_path dataset_kubric/val/sample_0001/image.png

B) A whole split:
    # Reference run: scores the dataset's own ground-truth videos
    python evaluate.py --dataset_dir dataset_kubric --split val --max_samples 20

    # Scoring generated videos: one file per sample, named <sample_id>.mp4
    python evaluate.py --dataset_dir dataset_kubric --split val \
                       --generated_dir results/adapter --out_csv results/adapter.csv

In mode B the target trajectory and input image always come from the dataset
folder (<dataset_dir>/<split>/<sample_id>/trajectory.npy and image.png).
The video comes from --generated_dir if given, otherwise from the dataset
folder's own video.mp4.
"""

import argparse
import csv
import os

import numpy as np
import imageio.v3 as iio
from PIL import Image

# Models
def load_cotracker(device):
    """Loads CoTracker3 (offline mode) from torch.hub."""
    import torch
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    return model.to(device)


def load_clip(device, name="openai/clip-vit-base-patch32"):
    """Loads CLIP (via Hugging Face transformers) for the image-similarity and
    temporal-consistency metrics. Returns (model, processor)."""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(name)
    return model, processor

# Trajectory metrics
def track_point(cotracker, video_frames, start_xy_pixels, device):
    """Runs CoTracker3 on a video, tracking ONE point starting at a given
    pixel location in frame 0.

    video_frames: (T, H, W, 3) uint8 array
    start_xy_pixels: (x, y) in pixel coordinates, the point's location in frame 0

    Returns: (T, 2) array, (x, y) normalized to [0, 1]
    """
    import torch
    T, H, W, _ = video_frames.shape

    video_tensor = torch.tensor(video_frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

    # queries: (B, N, 3) -> each row is (frame_idx, x_pixel, y_pixel)
    queries = torch.tensor([[0.0, start_xy_pixels[0], start_xy_pixels[1]]], device=device)[None]  # (1, 1, 3)

    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker(video_tensor, queries=queries)
    # pred_tracks: (B, T, N, 2) in pixel coordinates

    tracks = pred_tracks[0, :, 0, :].detach().cpu().numpy()  # (T, 2), pixel coords
    tracks_normalized = tracks.copy()
    tracks_normalized[:, 0] /= W
    tracks_normalized[:, 1] /= H
    return tracks_normalized


def resample_trajectory(traj, num_frames):
    """Linearly resamples a (T, 2) trajectory to `num_frames` points, spaced
    evenly in time. Needed because a generated video can have a different
    frame count than the dataset trajectory, simply
    cutting off the end would misalign the two in time.
    """
    traj = np.asarray(traj, dtype=np.float32)
    if len(traj) == num_frames:
        return traj
    if len(traj) == 1:
        return np.repeat(traj, num_frames, axis=0)
    old_t = np.linspace(0.0, 1.0, len(traj))
    new_t = np.linspace(0.0, 1.0, num_frames)
    x = np.interp(new_t, old_t, traj[:, 0])
    y = np.interp(new_t, old_t, traj[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def compute_ate(target_traj, predicted_traj):
    """Average Trajectory Error: mean Euclidean distance between target and
    predicted (x, y) at each frame, in normalized [0,1] coordinates.
    Both trajectories should be the same length, if not, trims to the shorter.
    """
    T = min(len(target_traj), len(predicted_traj))
    dists = np.sqrt(((target_traj[:T] - predicted_traj[:T]) ** 2).sum(axis=1))
    return float(dists.mean())


def static_baseline_ate(target_traj):
    """ATE of a 'prediction' that never moves from the start position."""
    static = np.repeat(target_traj[:1], len(target_traj), axis=0)
    return compute_ate(target_traj, static)

# Appearance metrics (CLIP)
def embed_images(clip_bundle, images, device, batch_size=32):
    """CLIP-embeds a list of images (HxWx3 uint8 arrays or PIL images).
    Returns an (N, D) numpy array of L2-normalized embeddings."""
    import torch
    model, processor = clip_bundle
    out = []
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        pil = [im if isinstance(im, Image.Image) else Image.fromarray(im) for im in chunk]
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        if not torch.is_tensor(feats):  # some transformers versions return an output object
            feats = feats.image_embeds if hasattr(feats, "image_embeds") else feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def image_similarity(image_emb, frame_embs):
    """Mean cosine similarity between the input image and every frame.
    image_emb: (D,), frame_embs: (T, D); both L2-normalized."""
    return float((frame_embs @ image_emb).mean())


def temporal_consistency(frame_embs):
    """Mean cosine similarity between consecutive frames. frame_embs: (T, D)."""
    if len(frame_embs) < 2:
        return float("nan")
    return float((frame_embs[:-1] * frame_embs[1:]).sum(axis=1).mean())

# Evaluating videos
def evaluate_one(video_path, trajectory_path, image_path, cotracker, clip_bundle, device):
    """Scores one video. image_path may be None (image similarity then = NaN).
    Returns a dict of metrics."""
    video_frames = iio.imread(video_path, plugin="FFMPEG")  # (T, H, W, 3)
    target_traj = np.load(trajectory_path)                   # (T0, 2), normalized [0,1]

    T, H, W = video_frames.shape[0], video_frames.shape[1], video_frames.shape[2]

    # Put the target on the same time axis as the video before comparing
    target_traj = resample_trajectory(target_traj, T)

    start_xy_pixels = (target_traj[0, 0] * W, target_traj[0, 1] * H)
    predicted_traj = track_point(cotracker, video_frames, start_xy_pixels, device)

    result = {
        "num_frames": T,
        "ate": compute_ate(target_traj, predicted_traj),
        "static_baseline_ate": static_baseline_ate(target_traj),
        "image_similarity": float("nan"),
        "temporal_consistency": float("nan"),
    }

    frame_embs = embed_images(clip_bundle, list(video_frames), device)
    result["temporal_consistency"] = temporal_consistency(frame_embs)
    if image_path is not None:
        image_emb = embed_images(clip_bundle, [Image.open(image_path).convert("RGB")], device)[0]
        result["image_similarity"] = image_similarity(image_emb, frame_embs)

    return result


def list_sample_ids(dataset_dir, split, max_samples=None):
    split_dir = os.path.join(dataset_dir, split)
    ids = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
    return ids[:max_samples] if max_samples else ids


def summarize(rows):
    """Prints mean +/- std of each metric over all rows (NaNs ignored)."""
    def stat(key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return "n/a"
        return f"{vals.mean():.4f} +/- {vals.std():.4f}"

    print(f"\nVideos evaluated: {len(rows)}")
    print(f"  ATE (lower is better):            {stat('ate')}")
    print(f"  Static-baseline ATE (reference):  {stat('static_baseline_ate')}")
    print(f"  Image similarity (higher):        {stat('image_similarity')}")
    print(f"  Temporal consistency (higher):    {stat('temporal_consistency')}")


def write_csv(rows, path):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["sample_id", "num_frames", "ate", "static_baseline_ate",
              "image_similarity", "temporal_consistency"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"Saved per-video results to {path}")

# CLI
def main():
    parser = argparse.ArgumentParser()
    # Mode A: one video
    parser.add_argument("--video_path")
    parser.add_argument("--trajectory_path")
    parser.add_argument("--image_path", default=None)
    # Mode B: a split of a dataset
    parser.add_argument("--dataset_dir")
    parser.add_argument("--split", default="val")
    parser.add_argument("--generated_dir", default=None,
                        help="Folder of generated videos named <sample_id>.mp4. "
                             "If omitted, scores the dataset's own video.mp4 files.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading CoTracker3 and CLIP (downloads weights on first run)...")
    cotracker = load_cotracker(device)
    clip_bundle = load_clip(device)

    rows = []

    if args.dataset_dir:
        sample_ids = list_sample_ids(args.dataset_dir, args.split, args.max_samples)
        print(f"Evaluating {len(sample_ids)} samples from {args.dataset_dir}/{args.split}")
        for sid in sample_ids:
            sample_dir = os.path.join(args.dataset_dir, args.split, sid)
            if args.generated_dir:
                video_path = os.path.join(args.generated_dir, f"{sid}.mp4")
            else:
                video_path = os.path.join(sample_dir, "video.mp4")
            if not os.path.exists(video_path):
                print(f"[SKIPPED] {sid}: no video at {video_path}")
                continue
            try:
                r = evaluate_one(video_path,
                                 os.path.join(sample_dir, "trajectory.npy"),
                                 os.path.join(sample_dir, "image.png"),
                                 cotracker, clip_bundle, device)
            except Exception as e:
                print(f"[SKIPPED] {sid}: {e}")
                continue
            r["sample_id"] = sid
            rows.append(r)
            print(f"{sid}: ATE={r['ate']:.4f}  static={r['static_baseline_ate']:.4f}  "
                  f"img_sim={r['image_similarity']:.3f}  temporal={r['temporal_consistency']:.3f}")
    elif args.video_path and args.trajectory_path:
        r = evaluate_one(args.video_path, args.trajectory_path, args.image_path,
                         cotracker, clip_bundle, device)
        r["sample_id"] = os.path.basename(os.path.dirname(args.video_path)) or "video"
        rows.append(r)
    else:
        parser.error("Give either --dataset_dir, or both --video_path and --trajectory_path.")

    if not rows:
        print("Nothing was evaluated.")
        return

    summarize(rows)
    if args.out_csv:
        write_csv(rows, args.out_csv)


if __name__ == "__main__":
    main()

