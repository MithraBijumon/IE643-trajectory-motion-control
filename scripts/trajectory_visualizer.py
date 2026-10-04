"""
trajectory_visualizer.py

Visual sanity check for preprocessed samples: draws the trajectory on top of
the video/image so we can SEE whether it actually follows the right object

Two modes: mode static and mode video 
"""

import argparse
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw


def draw_static_overlay(image_path, trajectory_path, out_path):
    """Draws the full trajectory path as a line + dots on the first frame."""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    traj = np.load(trajectory_path)  # (T, 2), normalized [0,1]

    draw = ImageDraw.Draw(img)

    # Convert normalized coords to pixel coords
    pixels = [(x * W, y * H) for x, y in traj]

    # Draw the path as a line, color fading from green (start) to red (end)
    for i in range(len(pixels) - 1):
        t = i / max(1, len(pixels) - 2)
        color = (int(255 * t), int(255 * (1 - t)), 0)  # green -> red
        draw.line([pixels[i], pixels[i + 1]], fill=color, width=3)

    # Mark start (green) and end (red) clearly
    r = 6
    draw.ellipse([pixels[0][0] - r, pixels[0][1] - r, pixels[0][0] + r, pixels[0][1] + r], fill=(0, 255, 0), outline="black")
    draw.ellipse([pixels[-1][0] - r, pixels[-1][1] - r, pixels[-1][0] + r, pixels[-1][1] + r], fill=(255, 0, 0), outline="black")

    img.save(out_path)
    print(f"Saved static overlay to {out_path}")


def draw_video_overlay(video_path, trajectory_path, out_path, fps=8):
    """Draws a moving dot (current position) + fading trail on every frame."""
    video = imageio.mimread(video_path)
    traj = np.load(trajectory_path)  # (T, 2), normalized [0,1]

    T = min(len(video), len(traj))
    H, W = video[0].shape[0], video[0].shape[1]

    frames_out = []
    for t in range(T):
        img = Image.fromarray(video[t]).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Trail of past points, fading
        trail_len = 15
        start_i = max(0, t - trail_len)
        for i in range(start_i, t):
            x, y = traj[i, 0] * W, traj[i, 1] * H
            alpha = (i - start_i) / max(1, t - start_i)
            r = 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, int(255 * alpha), 0))

        # Current position, bigger and bright
        x, y = traj[t, 0] * W, traj[t, 1] * H
        r = 6
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline="black")

        frames_out.append(np.array(img))

    imageio.mimwrite(out_path, frames_out, fps=fps)
    print(f"Saved video overlay to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", required=True, help="Folder containing image.png, video.mp4, trajectory.npy")
    parser.add_argument("--mode", choices=["static", "video"], default="static")
    parser.add_argument("--out_path", default=None)
    args = parser.parse_args()

    image_path = os.path.join(args.sample_dir, "image.png")
    video_path = os.path.join(args.sample_dir, "video.mp4")
    traj_path = os.path.join(args.sample_dir, "trajectory.npy")

    if args.mode == "static":
        out_path = args.out_path or os.path.join(args.sample_dir, "overlay_static.png")
        draw_static_overlay(image_path, traj_path, out_path)
    else:
        out_path = args.out_path or os.path.join(args.sample_dir, "overlay_video.mp4")
        draw_video_overlay(video_path, traj_path, out_path)


if __name__ == "__main__":
    main()
