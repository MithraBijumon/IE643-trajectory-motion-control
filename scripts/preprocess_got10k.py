"""
preprocess_got10k.py
Converts one GOT-10k clip (one annotated moving target) into a training sample folder
    <out_dir>/{image.png, video.mp4, trajectory.npy, visibility.npy, mask.png}
trajectory.npy = smoothed centre of the target box, normalised to [0, 1] as (x, y); mask.png = the first-frame box.
Used by stream_got10k.py (process(), add_filter_args()).
"""

import glob
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def read_boxes(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip().replace(",", " ").replace("\t", " ")
            if line:
                rows.append([float(v) for v in line.split()[:4]])
    return np.asarray(rows, dtype=np.float32)  # (N, 4) x, y, w, h


def read_labels(path, n):
    if not os.path.exists(path):
        return np.zeros(n, dtype=bool)
    with open(path) as f:
        vals = [int(float(v)) for v in f.read().replace(",", " ").split()]
    out = np.zeros(n, dtype=bool)
    m = min(n, len(vals))
    out[:m] = np.asarray(vals[:m]) > 0
    return out


def smooth(traj, k):
    if k <= 1 or len(traj) < k:
        return traj
    pad = k // 2
    padded = np.pad(traj, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)], axis=1)


def process(clip_dir, out_dir, args):
    frames = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
    boxes = read_boxes(os.path.join(clip_dir, "groundtruth.txt"))
    n = min(len(frames), len(boxes))
    if n < args.min_frames:
        return f"only {n} frames"

    frames, boxes = frames[:n], boxes[:n]
    absent = read_labels(os.path.join(clip_dir, "absence.label"), n)
    if absent.mean() > args.max_absent:
        return f"target absent in {absent.mean():.0%} of frames"

    # window: first max_frames frames
    n = min(n, args.max_frames)
    frames, boxes, absent = frames[:n], boxes[:n], absent[:n]

    first = Image.open(frames[0]).convert("RGB")
    W0, H0 = first.size
    scale = min(1.0, args.max_side / max(W0, H0))
    W, H = int(round(W0 * scale / 2)) * 2, int(round(H0 * scale / 2)) * 2  # even sizes for h264

    centres = np.stack([boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], axis=1)
    centres = smooth(centres, args.smooth)
    traj = np.clip(centres / np.array([W0, H0], dtype=np.float32), 0, 1).astype(np.float32)

    visible = (~absent) & (boxes[:, 2] > 1) & (boxes[:, 3] > 1)
    visible &= (centres[:, 0] >= 0) & (centres[:, 0] <= W0) & (centres[:, 1] >= 0) & (centres[:, 1] <= H0)
    if visible.mean() < 0.9:
        return f"target visible in only {visible.mean():.0%} of frames"

    # the target must really move (screen space, as a fraction of the frame diagonal)
    diag = np.hypot(W0, H0)
    net = np.linalg.norm(centres[-1] - centres[0]) / diag
    path = np.linalg.norm(np.diff(centres, axis=0), axis=1).sum() / diag
    if net < args.min_net_motion:
        return f"net motion {net:.3f} < {args.min_net_motion}"
    if path > args.max_path_ratio * max(net, 1e-6) and net < 2 * args.min_net_motion:
        return "jittery, no net motion"

    # target must not fill the frame (hard to condition on, usually camera-following)
    area = float(np.median(boxes[:, 2] * boxes[:, 3]) / (W0 * H0))
    if area > args.max_box_area:
        return f"box covers {area:.0%} of frame"

    os.makedirs(out_dir, exist_ok=True)
    video = [np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BICUBIC)) for p in frames]
    Image.fromarray(video[0]).save(os.path.join(out_dir, "image.png"))
    imageio.mimwrite(os.path.join(out_dir, "video.mp4"), video, fps=args.fps)
    np.save(os.path.join(out_dir, "trajectory.npy"), traj)
    np.save(os.path.join(out_dir, "visibility.npy"), visible)

    mask = Image.new("L", (W, H), 0)
    x, y, w, h = boxes[0] * np.array([W / W0, H / H0, W / W0, H / H0], dtype=np.float32)
    ImageDraw.Draw(mask).rectangle([x, y, x + w, y + h], fill=255)
    mask.save(os.path.join(out_dir, "mask.png"))
    return None


def add_filter_args(p):
    p.add_argument("--max_clips", type=int, default=None, help="per split, after filtering")
    p.add_argument("--min_frames", type=int, default=33)
    p.add_argument("--max_frames", type=int, default=65)
    p.add_argument("--max_side", type=int, default=512)
    p.add_argument("--fps", type=int, default=10, help="GOT-10k is ~10 fps")
    p.add_argument("--smooth", type=int, default=5, help="moving-average window on the box centre")
    p.add_argument("--min_net_motion", type=float, default=0.10,
                   help="min start-to-end displacement of the target (fraction of frame diagonal)")
    p.add_argument("--max_path_ratio", type=float, default=4.0)
    p.add_argument("--max_absent", type=float, default=0.1)
    p.add_argument("--max_box_area", type=float, default=0.4)
