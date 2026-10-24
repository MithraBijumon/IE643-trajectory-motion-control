import os
import random

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "config.yaml")


def load_config(path=None, tiny=False):
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    t = cfg.pop("tiny")
    cfg["is_tiny"] = bool(tiny)

    if tiny:
        cfg["model"]["dtype"] = t["dtype"]
        cfg["model"]["gradient_checkpointing"] = False
        for k in ("num_frames", "height", "width"):
            cfg["data"][k] = t[k]
        for k in ("epochs", "grad_accum"):
            cfg["train"][k] = t[k]
        cfg["inference"]["steps"] = t["steps"]

    validate_config(cfg)
    return cfg


def apply_checkpoint_cfg(cfg, ckpt_path):
    """Make cfg's adapter settings match the checkpoint that is about to be loaded."""
    if ckpt_path:
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("cfg", {})
        cfg["adapter"] = saved.get("adapter", {"mode": "input"})
    return cfg


def validate_config(cfg):
    d = cfg["data"]
    if (d["num_frames"] - 1) % 8 != 0:
        raise ValueError(f"num_frames must be 8k+1, got {d['num_frames']}")
    if d["height"] % 32 or d["width"] % 32:
        raise ValueError("height and width must be multiples of 32")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def torch_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


# ---------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------

def resample_trajectory(traj, num_frames, visible=None):
    """Linearly resample a (T, 2) trajectory over time to (num_frames, 2).

    Returns (trajectory, visible) with visible of shape (num_frames,).
    """
    traj = np.asarray(traj, dtype=np.float32)
    T = len(traj)

    if visible is None:
        visible = np.ones(T, dtype=bool)
    visible = np.asarray(visible).astype(bool)

    if T == num_frames:
        return traj, visible

    src = np.linspace(0, 1, T)
    dst = np.linspace(0, 1, num_frames)
    out = np.stack([np.interp(dst, src, traj[:, i]) for i in range(2)], axis=1)

    # a resampled frame is visible only if its nearest source frame was
    nearest = np.clip(np.round(dst * (T - 1)).astype(int), 0, T - 1)
    return out.astype(np.float32), visible[nearest]


DIRECTIONS = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "diag": (0.7071, 0.7071),  # down-right
}


def linear_path(start, direction, length, num_frames):
    """Constant-speed straight path; the start is shifted if the path would leave the frame."""
    d = np.asarray(direction, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = start + d * length

    lo, hi = 0.05, 0.95
    shift = np.where(end < lo, lo - end, 0) + np.where(end > hi, hi - end, 0)
    start, end = start + shift, end + shift
    start = np.clip(start, lo, hi)

    t = np.linspace(0, 1, num_frames)[:, None]
    return (start + (end - start) * t).astype(np.float32)


def smooth_trajectory(points, num_frames):
    """Turn a handful of clicked points (normalized) into a smooth per-frame path.

    Uses a Catmull-Rom spline through the points; two points give a straight line.
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, num_frames, axis=0)
    if len(pts) == 2:
        return resample_trajectory(pts, num_frames)[0]

    padded = np.concatenate([pts[:1], pts, pts[-1:]], axis=0)
    t = np.linspace(0, len(pts) - 1, num_frames)
    seg = np.clip(np.floor(t).astype(int), 0, len(pts) - 2)
    u = (t - seg)[:, None]

    p0, p1, p2, p3 = (padded[seg], padded[seg + 1], padded[seg + 2], padded[seg + 3])
    out = 0.5 * (
        2 * p1
        + (-p0 + p2) * u
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3
    )
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------

def video_tensor_to_uint8(video):
    """(B, 3, F, H, W) in [-1, 1] -> (B, F, H, W, 3) uint8."""
    video = ((video.float() + 1) * 127.5).clamp(0, 255).round().byte()
    return video.permute(0, 2, 3, 4, 1).cpu().numpy()


def save_video(frames, path, fps=24):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimwrite(path, list(frames), fps=fps)


def load_image(path, height, width):
    return Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)


def overlay_trajectory(frames, trajectory, trail=15):
    """Draw the target trajectory (normalized (T, 2)) on uint8 frames (T, H, W, 3)."""
    out = []
    H, W = frames.shape[1:3]

    for t in range(min(len(frames), len(trajectory))):
        img = Image.fromarray(frames[t])
        draw = ImageDraw.Draw(img)

        for i in range(max(0, t - trail), t):
            x, y = trajectory[i, 0] * W, trajectory[i, 1] * H
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 0))

        x, y = trajectory[t, 0] * W, trajectory[t, 1] * H
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 0, 0), outline="black")
        out.append(np.array(img))

    return np.stack(out)
