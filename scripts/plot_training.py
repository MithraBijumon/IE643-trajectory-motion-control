"""
plot_training.py

Draws the report's training figures from the epoch logs written by train.py (log.jsonl).

    python scripts/plot_training.py                       # results/logs/*.jsonl -> results/figures/*.png
    python scripts/plot_training.py --full path/to/log.jsonl --overfit path/to/other.jsonl

Colours follow the dataviz reference palette (blue, grey baseline, one-hue ordinal ramp for noise levels).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
RAMP = {"0.3": "#86b6ef", "0.6": "#2a78d6", "0.9": "#104281"}  # blue 250 / 450 / 650: noise level, ordinal
NOISE_FLOOR = 0.001  # spread of the loss without the trajectory between evaluations (see report)


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def style(ax, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_xlabel("epoch", color=INK2, fontsize=8)
    ax.set_ylabel(ylabel, color=INK2, fontsize=8)
    ax.set_title(title, color=INK, fontsize=9, loc="left", fontweight="semibold")


def finish(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def losses_panel(ax, rows, title):
    ep = [r["epoch"] for r in rows]
    ax.plot(ep, [r["val_loss_nocond"] for r in rows], color=MUTED, linewidth=2, marker="o", markersize=4, label="without trajectory")
    ax.plot(ep, [r["val_loss"] for r in rows], color=BLUE, linewidth=2, marker="o", markersize=4, label="with trajectory")
    style(ax, "validation loss", title)
    ax.set_xticks(ep)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="best")


def overfit_figure(rows, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.7))
    losses_panel(a, rows, "Loss on the training clip")

    ep = [r["epoch"] for r in rows]
    nudge = {"0.3": 4, "0.6": -4, "0.9": 0}  # the 0.3 and 0.6 curves end close together; keep their labels apart
    for sg, color in RAMP.items():
        ys = [r["gain_by_sigma"][sg] for r in rows]
        b.plot(ep, ys, color=color, linewidth=2, marker="o", markersize=4, label=f"$\\sigma$ = {sg}")
        b.annotate(f"$\\sigma$={sg}", (ep[-1], ys[-1]), xytext=(5, nudge[sg]), textcoords="offset points", fontsize=7, color=INK2, va="center")
    style(b, "cond_gain (loss without - with)", "Gain by noise level")
    b.set_xticks(ep)
    b.set_xlim(ep[0] - 0.2, ep[-1] + 1.4)
    b.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    finish(fig, path)


def full_figure(rows, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.7))
    losses_panel(a, rows, "Loss on the 6 held-out clips")
    a.set_ylim(0.332, 0.342)

    ep = [r["epoch"] for r in rows]
    gain = [r["cond_gain"] for r in rows]
    b.axhspan(-NOISE_FLOOR, NOISE_FLOOR, color=GRID, zorder=0)
    b.text(ep[-1] + 0.45, NOISE_FLOOR * 1.15, "noise floor", fontsize=7, color=MUTED, ha="right", va="bottom")
    b.bar(ep, gain, color=BLUE, width=0.6, zorder=2)
    b.axhline(0, color=AXIS, linewidth=1, zorder=3)
    style(b, "cond_gain (loss without - with)", "Gain from the trajectory, held-out")
    b.set_xticks(ep)
    b.set_ylim(min(gain) - 0.0004, max(max(gain), NOISE_FLOOR) + 0.0009)  # room above the tallest bar for the label
    finish(fig, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", default="results/logs/full_run_log.jsonl")
    p.add_argument("--overfit", default="results/logs/overfit_log.jsonl")
    p.add_argument("--out", default="results/figures")
    args = p.parse_args()

    overfit_figure(read(args.overfit), os.path.join(args.out, "fig_overfit.png"))
    full_figure(read(args.full), os.path.join(args.out, "fig_fullrun.png"))


if __name__ == "__main__":
    main()
