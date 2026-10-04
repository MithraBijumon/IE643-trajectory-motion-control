"""
evaluate.py

Measures how closely a video's actual motion matches a target trajectory,
using CoTracker3 to track where the point really went, then computing
Average Trajectory Error (ATE) against the target.

"""

import argparse
import numpy as np
import torch
import imageio.v3 as iio


def load_cotracker(device):
    """Loads CoTracker3 (offline mode) from torch.hub."""
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    return model.to(device)


def track_point(cotracker, video_frames, start_xy_pixels, device):
    """Runs CoTracker3 on a video, tracking ONE point starting at a given
    pixel location in frame 0.

    video_frames: (T, H, W, 3) uint8 array
    start_xy_pixels: (x, y) in pixel coordinates, the point's location in frame 0

    Returns: pred_tracks_normalized, shape (T, 2), (x, y) normalized to [0, 1]
    """
    T, H, W, _ = video_frames.shape

    video_tensor = torch.tensor(video_frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

    # queries: (B, N, 3) -> each row is (frame_idx, x_pixel, y_pixel)
    queries = torch.tensor([[0.0, start_xy_pixels[0], start_xy_pixels[1]]], device=device)[None]  # (1, 1, 3)

    pred_tracks, pred_visibility = cotracker(video_tensor, queries=queries)
    # pred_tracks: (B, T, N, 2) in pixel coordinates

    tracks = pred_tracks[0, :, 0, :].detach().cpu().numpy()  # (T, 2), pixel coords
    tracks_normalized = tracks.copy()
    tracks_normalized[:, 0] /= W
    tracks_normalized[:, 1] /= H

    return tracks_normalized


def compute_ate(target_traj, predicted_traj):
    """Average Trajectory Error: mean Euclidean distance between target and
    predicted (x, y) at each frame, in normalized [0,1] coordinates.
    Both trajectories must be the same length. if not, trims to the shorter one.
    """
    T = min(len(target_traj), len(predicted_traj))
    target = target_traj[:T]
    pred = predicted_traj[:T]

    dists = np.sqrt(((target - pred) ** 2).sum(axis=1))  # (T,)
    return float(dists.mean())


def evaluate_one(video_path, trajectory_path, cotracker, device):
    video_frames = iio.imread(video_path, plugin="FFMPEG")  # (T, H, W, 3)
    target_traj = np.load(trajectory_path)  # (T, 2), normalized [0,1]

    H, W = video_frames.shape[1], video_frames.shape[2]
    start_xy_pixels = (target_traj[0, 0] * W, target_traj[0, 1] * H)

    predicted_traj = track_point(cotracker, video_frames, start_xy_pixels, device)
    ate = compute_ate(target_traj, predicted_traj)

    return ate, target_traj, predicted_traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--trajectory_path", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CoTracker3 (downloads weights on first run)...")
    cotracker = load_cotracker(device)

    ate, target_traj, predicted_traj = evaluate_one(args.video_path, args.trajectory_path, cotracker, device)

    print(f"\nTarget trajectory shape:    {target_traj.shape}")
    print(f"Predicted trajectory shape: {predicted_traj.shape}")
    print(f"\nATE (Average Trajectory Error): {ate:.4f}")
    print("(Lower is better. For a sanity check on a real dataset video, expect this close to 0.)")


if __name__ == "__main__":
    main()
