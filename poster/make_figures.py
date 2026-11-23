"""Poster figures. Frames are cropped from the demo_grid output (grid.png: rows base/adapter/as3/as6/ref0.8+comp2.5+as3 per clip, 3 time columns).
Figures are drawn at the size they have on the poster (1 inch here = 1 inch there), so font sizes are the real printed sizes (poster body text is about 25 pt).
usage: python poster/make_figures.py path/to/grid.png"""
import sys
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

Image.MAX_IMAGE_PIXELS = None
GRID = sys.argv[1] if len(sys.argv) > 1 else "grid.png"
OUT = "poster/figures/"
NAVY, GREEN, GREY, LIGHT = "#000080", "#1b7f2a", "#555555", "#e8eefc"
g = Image.open(GRID).convert("RGB")
W = g.size[0]; FW, FH = 864, 480; X0 = W - 3 * FW
CLIP = {"cyclist": 0, "cat": 5, "boat": 10}
ROW = {"base": 0, "adapter": 1, "as3": 2, "as6": 3, "drag": 4}


def frame(clip, row, col):
    r = CLIP[clip] + ROW[row]
    return g.crop((X0 + col * FW, r * FH, X0 + (col + 1) * FW, (r + 1) * FH))


def panel(ax, im, title, color=NAVY, fs=17):
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor(color); s.set_linewidth(2.5)
    if title: ax.set_title(title, fontsize=fs, color=color, pad=5, fontweight="bold")


# ---- Intro (box 17.6 x 11 cm = 6.9 x 4.3 in) ---------------------------------------------------------------------------
if len(sys.argv) > 2:   # benchmark montage: per motion two rows (baseline, adapter) x 3 time columns, same clip / path / seed
    bm = Image.open(sys.argv[2]).convert("RGB")
    MOT = {"left": 0, "right": 1, "up": 2, "down": 3, "diag": 4}
    def bframe(motion, adapter, col):
        r = 2 * MOT[motion] + int(adapter)
        return bm.crop((bm.size[0] - 3 * FW + col * FW, r * FH, bm.size[0] - 3 * FW + (col + 1) * FW, (r + 1) * FH))
    fig, axs = plt.subplots(3, 3, figsize=(6.9, 4.3), dpi=300)
    for i, m in enumerate(["right", "up", "diag"]):
        panel(axs[i, 0], bframe(m, False, 0), "Input + path" if i == 0 else "")
        panel(axs[i, 1], bframe(m, False, 2), "Plain LTX" if i == 0 else "", GREY)
        panel(axs[i, 2], bframe(m, True, 2), "Adapter x3" if i == 0 else "", GREEN)
        axs[i, 0].set_ylabel(m, fontsize=15, fontweight="bold", color=NAVY)
    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.3)
    fig.savefig(OUT + "Intro.png"); plt.close(fig)
else:
    fig, axs = plt.subplots(2, 2, figsize=(6.9, 4.3), dpi=300)
    panel(axs[0, 0], frame("boat", "base", 0), "Input + path", fs=15)
    panel(axs[0, 1], frame("boat", "base", 2), "Plain LTX-Video", GREY, fs=15)
    panel(axs[1, 0], frame("boat", "as6", 2), "Adapter, x6", GREEN, fs=15)
    panel(axs[1, 1], frame("boat", "drag", 2), "Adapter + drag start", GREEN, fs=15)
    plt.tight_layout(pad=0.6, h_pad=1.3, w_pad=0.5)
    fig.savefig(OUT + "Intro.png"); plt.close(fig)

# ---- Qualitative (optional; 3 clips x 4 columns, last frame) --------------------------------------------------------
fig, axs = plt.subplots(3, 4, figsize=(11.0, 5.5), dpi=220)
cols = [("Input + path", "base", 0, NAVY), ("Plain LTX-Video", "base", 2, GREY),
        ("Adapter, x6", "as6", 2, GREEN), ("Adapter + drag start", "drag", 2, GREEN)]
for i, c in enumerate(["cyclist", "cat", "boat"]):
    for j, (t, row, col, colr) in enumerate(cols):
        panel(axs[i, j], frame(c, row, col), t if i == 0 else "", colr, fs=17)
        if j == 0: axs[i, j].set_ylabel(c, fontsize=17, fontweight="bold", color=NAVY)
plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.3)
fig.savefig("poster/optional/qualitative.png"); plt.close(fig)


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=20, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02", fc=fc, ec=ec, lw=2.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color=NAVY, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=26, lw=2.8, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---- Workflow (box 0.95 linewidth x 12 cm = about 13.0 x 4.7 in): two rows of three ------------------------------------
fig = plt.figure(figsize=(13.0, 4.7), dpi=250); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
steps = [
    ("1. Data", "GOT-10k, 714 clips,\nstatic camera only"),
    ("2. Labels", "SAM 2 masks, point tracks,\nQwen2-VL captions"),
    ("3. Condition", "object footprint + path\n+ object feature"),
    ("4. Train adapter", "frozen LTX-Video 2B,\nfocus loss on the object"),
    ("5. Generate", "image + path -> video,\nstrength dial"),
    ("6. Evaluate", "object error, PCK,\nbackground drift"),
]
bw, bh = 0.285, 0.40
xs = [0.02, 0.357, 0.695]
ys = [0.55, 0.06]
for k, (t, body) in enumerate(steps):
    x, y = xs[k % 3], ys[k // 3]
    ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.004,rounding_size=0.02", fc=LIGHT, ec=NAVY, lw=2.5))
    ax.text(x + bw / 2, y + bh * 0.74, t, ha="center", va="center", fontsize=22, fontweight="bold", color=NAVY)
    ax.text(x + bw / 2, y + bh * 0.30, body, ha="center", va="center", fontsize=19, linespacing=1.25)
for row in range(2):
    y = ys[row] + bh / 2
    arrow(ax, (xs[0] + bw + 0.004, y), (xs[1] - 0.004, y)); arrow(ax, (xs[1] + bw + 0.004, y), (xs[2] - 0.004, y))
arrow(ax, (xs[2] + bw / 2, ys[0] - 0.005), (xs[0] + bw / 2, ys[1] + bh + 0.005), rad=0.0)
fig.savefig(OUT + "Workflow.png"); plt.close(fig)

# ---- Architecture (box 0.95 linewidth x 14 cm = about 13.0 x 5.5 in) ---------------------------------------------------
fig = plt.figure(figsize=(13.0, 5.5), dpi=250); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.5, 0.955, "frozen LTX-Video 2B", ha="center", va="center", fontsize=22, fontweight="bold", color=GREY)
box(ax, 0.01, 0.58, 0.19, 0.30, "image +\nnoisy latents\n+ prompt", fs=19)
lx = [0.25, 0.43, 0.61]
for x in lx:
    box(ax, x, 0.58, 0.15, 0.30, "frozen\nLTX\nblocks", fc="#dddddd", ec=GREY, fs=19)
box(ax, 0.79, 0.58, 0.09, 0.30, "VAE", fs=19)
box(ax, 0.90, 0.58, 0.09, 0.30, "video", fs=19, ec=GREEN, bold=True)
arrow(ax, (0.205, 0.73), (0.245, 0.73), color=NAVY)
for a in lx: arrow(ax, (a + 0.155, 0.73), (a + 0.175, 0.73), color=GREY)
arrow(ax, (0.765, 0.73), (0.785, 0.73), color=GREY); arrow(ax, (0.885, 0.73), (0.895, 0.73))
ax.text(0.02, 0.43, "trainable adapter (5.9M parameters)", ha="left", va="center", fontsize=22, fontweight="bold", color=NAVY)
bx = [0.01, 0.275, 0.54, 0.775]; bws = [0.23, 0.23, 0.20, 0.215]
btxt = ["path + mask\n+ object", "conv encoder\n+ 2 blocks", "zero-init\nprojections", "x footprint,\ncap 5%, add"]
for x, w, tx in zip(bx, bws, btxt):
    box(ax, x, 0.03, w, 0.30, tx, fs=19, ec=GREEN if tx.startswith("x ") else NAVY)
for i in range(3): arrow(ax, (bx[i] + bws[i] + 0.004, 0.18), (bx[i + 1] - 0.004, 0.18))
for x in lx:
    arrow(ax, (0.87, 0.345), (x + 0.075, 0.575), color=GREEN, ls="--", rad=0.0)
fig.savefig(OUT + "Deep-Neural-Network-Model.jpg"); plt.close(fig)
print("ok")
