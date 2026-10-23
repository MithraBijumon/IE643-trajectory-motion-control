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
from src.utils import get_device, load_config, overlay_trajectory, save_video, smooth_trajectory

_state = {}


def get_backbone():
    if "backbone" not in _state:
        cfg = load_config(_state["config"], tiny=_state["tiny"])
        backbone = LTXBackbone(cfg, get_device())
        if _state["checkpoint"]:
            backbone.load_checkpoint(_state["checkpoint"])
        _state["backbone"], _state["cfg"] = backbone, cfg
    return _state["backbone"], _state["cfg"]


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


def generate(image_path, points, steps, seed, progress=gr.Progress()):
    if image_path is None:
        raise gr.Error("Upload an image first.")
    if len(points) < 2:
        raise gr.Error("Click at least two points on the image to define a path.")

    backbone, cfg = get_backbone()
    d = cfg["data"]

    image = Image.open(image_path).convert("RGB").resize((d["width"], d["height"]), Image.BICUBIC)
    trajectory = smooth_trajectory(points, d["num_frames"])

    frames = backbone.generate(
        image,
        trajectory,
        steps=int(steps),
        guidance_scale=cfg["inference"]["guidance_scale"],
        decode_timestep=cfg["inference"]["decode_timestep"],
        seed=int(seed),
        progress=lambda f: progress(f, desc="Denoising"),
    )

    out = os.path.join("results", "app", "generated.mp4")
    over = os.path.join("results", "app", "generated_overlay.mp4")
    save_video(frames, out, fps=cfg["inference"]["fps"])
    save_video(overlay_trajectory(frames, trajectory), over, fps=cfg["inference"]["fps"])
    return out, over


def build_demo():
    with gr.Blocks(title="Motion Controlled Video") as demo:
        gr.Markdown(
            "# Motion Controlled Video\n"
            "1. Upload a first-frame image. 2. Click along the path you want the subject to follow "
            "(first click = start). 3. Generate."
        )
        points = gr.State([])
        source = gr.State(None)

        with gr.Row():
            with gr.Column():
                upload = gr.Image(type="filepath", label="Upload image", sources=["upload"])
                preview = gr.Image(label="Click to add trajectory points", interactive=False)
                clear = gr.Button("Clear points")
                steps = gr.Slider(2, 50, value=30, step=1, label="Denoising steps")
                seed = gr.Number(value=0, label="Seed", precision=0)
                run = gr.Button("Generate video", variant="primary")

            with gr.Column():
                video = gr.Video(label="Generated video")
                overlay = gr.Video(label="With target trajectory overlay")

        upload.change(on_image, upload, [points, preview])
        preview.select(on_click, [upload, points], [points, preview])
        clear.click(on_clear, upload, [points, preview])
        run.click(generate, [upload, points, steps, seed], [video, overlay])
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
