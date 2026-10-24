"""
Gradio demo: upload an image, click points to draw a motion path, generate a video.

    python app/app.py --checkpoint checkpoints/best.pt
    python app/app.py --tiny --checkpoint checkpoints_tiny/best.pt     # pipeline check only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from src.ltx_wrapper import LTXBackbone
from src.utils import (
    DIRECTIONS,
    apply_checkpoint_cfg,
    get_device,
    linear_path,
    load_config,
    overlay_trajectory,
    save_video,
    smooth_trajectory,
)

_state = {}


def get_backbone():
    if "backbone" not in _state:
        cfg = apply_checkpoint_cfg(load_config(_state["config"], tiny=_state["tiny"]), _state["checkpoint"])
        backbone = LTXBackbone(cfg, get_device())
        if _state["checkpoint"]:
            backbone.load_checkpoint(_state["checkpoint"])
        _state["backbone"], _state["cfg"] = backbone, cfg
    return _state["backbone"], _state["cfg"]


def get_mask(image, start_norm):
    """Object mask for the first click (SAM), or None to fall back to a disk."""
    try:
        if "sam" not in _state:
            from src.segmentation import SamSegmenter

            _state["sam"] = SamSegmenter(get_device())
        W, H = image.size
        mask, _ = _state["sam"].segment(image, (start_norm[0] * W, start_norm[1] * H))
        return mask
    except Exception as e:  # SAM weights unavailable, etc.
        print(f"Segmentation failed ({e}); using a disk around the point instead.")
        return None


def draw_points(image, points):
    """Preview: clicked points joined by the smoothed path."""
    img = Image.open(image).convert("RGB") if isinstance(image, str) else image.copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)

    if len(points) > 1:
        path = smooth_trajectory(points, 60)
        draw.line([(x * W, y * H) for x, y in path], fill=(255, 200, 0), width=3)

    for i, (x, y) in enumerate(points):
        r = 7
        color = (0, 255, 0) if i == 0 else (255, 0, 0)
        draw.ellipse([x * W - r, y * H - r, x * W + r, y * H + r], fill=color, outline="black")
    return img


def on_image(image_path):
    return [], image_path


def on_click(image_path, points, evt: gr.SelectData):
    if image_path is None:
        return points, None

    img = Image.open(image_path).convert("RGB")
    x, y = evt.index
    points = points + [[x / img.width, y / img.height]]
    return points, draw_points(img, points)


def on_clear(image_path):
    return [], image_path


PRESETS = ["Draw my own path"] + list(DIRECTIONS)


def generate(image_path, points, preset, length, steps, seed, motion_guidance, progress=gr.Progress()):
    if image_path is None:
        raise gr.Error("Upload an image first.")
    if preset == PRESETS[0] and len(points) < 2:
        raise gr.Error("Click at least two points on the image, or pick a preset motion.")

    backbone, cfg = get_backbone()
    d = cfg["data"]

    image = Image.open(image_path).convert("RGB").resize((d["width"], d["height"]), Image.BICUBIC)
    if preset == PRESETS[0]:
        trajectory = smooth_trajectory(points, d["num_frames"])
    else:
        # presets start at the first clicked point (put it on the object), default: centre
        start = points[0] if points else [0.5, 0.5]
        trajectory = linear_path(start, DIRECTIONS[preset], length, d["num_frames"])

    mask, mask_view = None, None
    if backbone.motion.condition == "entity":
        mask = get_mask(image, trajectory[0])
        if mask is not None:
            shown = np.array(image).astype(np.float32)
            shown[mask] = 0.5 * shown[mask] + 0.5 * np.array([255, 40, 40], dtype=np.float32)
            mask_view = Image.fromarray(shown.astype(np.uint8))

    frames = backbone.generate(
        image,
        trajectory,
        mask=mask,
        steps=int(steps),
        guidance_scale=cfg["inference"]["guidance_scale"],
        decode_timestep=cfg["inference"]["decode_timestep"],
        seed=int(seed),
        motion_guidance=float(motion_guidance),
        progress=lambda f: progress(f, desc="Denoising"),
    )

    out = os.path.join("results", "app", "generated.mp4")
    over = os.path.join("results", "app", "generated_overlay.mp4")
    save_video(frames, out, fps=cfg["inference"]["fps"])
    save_video(overlay_trajectory(frames, trajectory), over, fps=cfg["inference"]["fps"])
    return out, over, mask_view


def build_demo():
    with gr.Blocks(title="Motion Controlled Video") as demo:
        gr.Markdown(
            "# Motion Controlled Video\n"
            "1. Upload a first-frame image. 2. Click along the path you want the subject to follow "
            "(first click = start), or pick a preset direction and click once on the subject. 3. Generate."
        )
        points = gr.State([])
        source = gr.State(None)

        with gr.Row():
            with gr.Column():
                upload = gr.Image(type="filepath", label="Upload image", sources=["upload"])
                preview = gr.Image(label="Click to add trajectory points", interactive=False)
                clear = gr.Button("Clear points")
                preset = gr.Radio(PRESETS, value=PRESETS[0], label="Motion")
                length = gr.Slider(0.1, 0.6, value=0.3, step=0.05, label="Preset path length (fraction of the frame)")
                guidance = gr.Slider(1.0, 4.0, value=1.0, step=0.5, label="Motion strength (guidance)")
                steps = gr.Slider(2, 50, value=30, step=1, label="Denoising steps")
                seed = gr.Number(value=0, label="Seed", precision=0)
                run = gr.Button("Generate video", variant="primary")

            with gr.Column():
                video = gr.Video(label="Generated video")
                overlay = gr.Video(label="With target trajectory overlay")
                mask_img = gr.Image(label="Detected object (entity condition only)", interactive=False)

        upload.change(on_image, upload, [points, preview])
        preview.select(on_click, [upload, points], [points, preview])
        clear.click(on_clear, upload, [points, preview])
        run.click(generate, [upload, points, preset, length, steps, seed, guidance], [video, overlay, mask_img])
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    _state.update(config=args.config, checkpoint=args.checkpoint, tiny=args.tiny)
    build_demo().launch(share=args.share)
