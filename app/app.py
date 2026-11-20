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
from src.metrics import object_trajectory_metrics
from src.utils import (
    DIRECTIONS,
    apply_checkpoint_cfg,
    get_device,
    linear_path,
    load_config,
    overlay_masks,
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


MODES = ["Trajectory point", "Object: box corner", "Object: + click (add)", "Object: - click (remove)"]


def get_segmenter():
    if "sam" not in _state:
        from src.segmentation import SamSegmenter

        _state["sam"] = SamSegmenter(get_device())
    return _state["sam"]


def segment_from_prompts(img, box, pos, neg):
    """SAM mask from the box (two corners) and/or the +/- clicks, or None while there is no prompt."""
    box_xyxy = None
    if len(box) == 2:
        (x0, y0), (x1, y1) = box
        box_xyxy = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    points = [tuple(p) for p in pos] + [tuple(p) for p in neg]
    if box_xyxy is None and not points:
        return None
    labels = [1] * len(pos) + [0] * len(neg)
    try:
        mask, _ = get_segmenter().segment_prompts(img, box=box_xyxy, points=points or None, labels=labels or None)
        return mask
    except Exception as e:  # SAM weights unavailable, etc.
        print(f"Segmentation failed ({e})")
        return None


def render(img, points, box, pos, neg, mask):
    """Preview: object mask, box, +/- clicks, and the smoothed trajectory."""
    img = img.copy()
    W, H = img.size
    if mask is not None:
        arr = np.array(img).astype(np.float32)
        arr[mask] = 0.5 * arr[mask] + 0.5 * np.array([255, 40, 40], dtype=np.float32)
        img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    if len(box) == 2:
        (x0, y0), (x1, y1) = box
        draw.rectangle([x0, y0, x1, y1], outline=(30, 144, 255), width=3)
    for x, y in (box if len(box) < 2 else []):
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(30, 144, 255), outline="white")
    for x, y in pos:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(0, 220, 0), outline="white")
    for x, y in neg:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(220, 0, 0), outline="white")

    if len(points) > 1:
        path = smooth_trajectory(points, 60)
        draw.line([(x * W, y * H) for x, y in path], fill=(255, 200, 0), width=3)
    for i, (x, y) in enumerate(points):
        r = 7
        color = (0, 255, 0) if i == 0 else (255, 0, 0)
        draw.ellipse([x * W - r, y * H - r, x * W + r, y * H + r], fill=color, outline="black")
    return img


def on_image(image_path):
    return [], [], [], [], None, image_path


def on_click(image_path, mode, points, box, pos, neg, mask, evt: gr.SelectData):
    if image_path is None:
        return points, box, pos, neg, mask, None

    img = Image.open(image_path).convert("RGB")
    x, y = evt.index
    points, box, pos, neg = list(points), list(box), list(pos), list(neg)

    if mode == MODES[0]:
        points.append([x / img.width, y / img.height])
    elif mode == MODES[1]:
        box = [[x, y]] if len(box) >= 2 else box + [[x, y]]   # a third click starts a new box
        mask = segment_from_prompts(img, box, pos, neg) if len(box) == 2 else mask
    elif mode == MODES[2]:
        pos.append([x, y])
        mask = segment_from_prompts(img, box, pos, neg)
    else:
        neg.append([x, y])
        mask = segment_from_prompts(img, box, pos, neg)
    return points, box, pos, neg, mask, render(img, points, box, pos, neg, mask)


def on_clear_path(image_path, box, pos, neg, mask):
    if image_path is None:
        return [], None
    return [], render(Image.open(image_path).convert("RGB"), [], box, pos, neg, mask)


def on_clear_object(image_path, points):
    if image_path is None:
        return [], [], [], None, None
    return [], [], [], None, render(Image.open(image_path).convert("RGB"), points, [], [], [], None)


PRESETS = ["Draw my own path"] + list(DIRECTIONS)


def centroid_of(mask):
    ys, xs = np.nonzero(mask)
    return [float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0])]


def get_tracker():
    if "sam2" not in _state:
        from src.segmentation import Sam2Tracker

        _state["sam2"] = Sam2Tracker(get_device())
    return _state["sam2"]


def generate(image_path, points, mask_native, preset, length, steps, seed, motion_guidance, progress=gr.Progress()):
    if image_path is None:
        raise gr.Error("Upload an image first.")
    if preset == PRESETS[0] and len(points) < 2:
        raise gr.Error("Click at least two trajectory points on the image, or pick a preset motion.")

    backbone, cfg = get_backbone()
    d = cfg["data"]
    size = (d["width"], d["height"])

    image = Image.open(image_path).convert("RGB").resize(size, Image.BICUBIC)

    # the object mask chosen in the interface (box / +- clicks), resized to the generation resolution
    mask = None
    if mask_native is not None and np.asarray(mask_native).any():
        mask = np.array(Image.fromarray(np.asarray(mask_native).astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127

    if preset == PRESETS[0]:
        trajectory = smooth_trajectory(points, d["num_frames"])
    else:
        # presets start at the selected object's centre, else the first clicked point, else the image centre
        start = centroid_of(mask) if mask is not None else (points[0] if points else [0.5, 0.5])
        trajectory = linear_path(start, DIRECTIONS[preset], length, d["num_frames"])

    if mask is None:
        mask = get_mask(image, trajectory[0])   # no object selected: SAM from a click at the path start

    mask_view = None
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
    track_path = os.path.join("results", "app", "generated_tracked.mp4")
    save_video(frames, out, fps=cfg["inference"]["fps"])
    save_video(overlay_trajectory(frames, trajectory), over, fps=cfg["inference"]["fps"])

    # follow the object through the generated video with SAM 2 and score the path
    tracked, report = None, "Object tracking unavailable (needs a selected / detected object and SAM 2)."
    if mask is not None:
        try:
            tracker = get_tracker()
            if tracker.ok:
                progress(0.99, desc="Tracking the object")
                masks = tracker.track(frames, mask)
                save_video(overlay_masks(frames, masks, trajectory), track_path, fps=cfg["inference"]["fps"])
                tracked = track_path
                m = object_trajectory_metrics(trajectory, masks)
                area = masks.reshape(len(masks), -1).sum(1) / max(int(mask.sum()), 1)
                report = (
                    "**Object follows the path?** (SAM 2 mask centroid vs. your trajectory)\n\n"
                    f"- mean path error (ATE, fraction of frame): **{m['obj_ate']:.3f}**\n"
                    f"- final-point error: {m['obj_epe_final']:.3f}\n"
                    f"- within 5% of the frame on {m['obj_pck_5']:.0%} of frames\n"
                    f"- direction match (cosine): {m.get('obj_direction_cos', float('nan')):.2f}, "
                    f"distance covered: {m.get('obj_progress', float('nan')):.0%} of the drawn length\n"
                    f"- object size vs. the input: {float(area[1:].mean()):.2f}x, lost in {m['obj_empty_frac']:.0%} of frames"
                )
        except Exception as e:
            report = f"Object tracking failed: {e}"
    return out, over, tracked, report, mask_view


def build_demo():
    with gr.Blocks(title="Motion Controlled Video") as demo:
        gr.Markdown(
            "# Motion Controlled Video\n"
            "1. Upload a first-frame image. 2. Select the **object**: set *Click mode* to *Object: box corner* and click two "
            "opposite corners, then fix the mask with *+ click* (add) / *- click* (remove); the red overlay is the mask the model "
            "will use. 3. Switch to *Trajectory point* and click the path (first click = start), or pick a preset direction. "
            "4. Generate. The result is tracked with SAM 2 to show whether the object really followed the path."
        )
        points, box, pos, neg, mask = gr.State([]), gr.State([]), gr.State([]), gr.State([]), gr.State(None)

        with gr.Row():
            with gr.Column():
                upload = gr.Image(type="filepath", label="Upload image", sources=["upload"])
                mode = gr.Radio(MODES, value=MODES[0], label="Click mode")
                preview = gr.Image(label="Click here (object mask in red, path in yellow)", interactive=False)
                with gr.Row():
                    clear_path = gr.Button("Clear path")
                    clear_obj = gr.Button("Clear object")
                preset = gr.Radio(PRESETS, value=PRESETS[0], label="Motion")
                length = gr.Slider(0.1, 0.6, value=0.3, step=0.05, label="Preset path length (fraction of the frame)")
                guidance = gr.Slider(1.0, 4.0, value=1.0, step=0.5, label="Motion strength (guidance)")
                steps = gr.Slider(2, 50, value=30, step=1, label="Denoising steps")
                seed = gr.Number(value=0, label="Seed", precision=0)
                run = gr.Button("Generate video", variant="primary")

            with gr.Column():
                video = gr.Video(label="Generated video")
                overlay = gr.Video(label="With target trajectory overlay")
                tracked = gr.Video(label="Object tracked by SAM 2 (red) vs. target path (yellow) and its path (cyan)")
                report = gr.Markdown()
                mask_img = gr.Image(label="Object mask used", interactive=False)

        states = [points, box, pos, neg, mask]
        upload.change(on_image, upload, states + [preview])
        preview.select(on_click, [upload, mode] + states, states + [preview])
        clear_path.click(on_clear_path, [upload, box, pos, neg, mask], [points, preview])
        clear_obj.click(on_clear_object, [upload, points], [box, pos, neg, mask, preview])
        run.click(generate, [upload, points, mask, preset, length, steps, seed, guidance],
                  [video, overlay, tracked, report, mask_img])
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
