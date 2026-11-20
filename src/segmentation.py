import numpy as np
import torch

SAM_ID = "facebook/sam-vit-base"


class SamSegmenter:
    """Point-prompted object mask with SAM. The prompt is the tracked point, which sits on
    the object we want to control."""

    def __init__(self, device, model_id=SAM_ID):
        from transformers import SamModel, SamProcessor

        self.device = device
        self.model = SamModel.from_pretrained(model_id).to(device).eval()
        self.processor = SamProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def segment(self, image, point_xy, pick="largest"):
        """
        image:    PIL RGB image
        point_xy: (x, y) in pixels
        pick:     "best" = candidate with the highest predicted IoU (often just a part under
                  the point); "largest" = the biggest candidate, usually the whole object
        Returns (mask (H, W) bool, score float).
        """
        inputs = self.processor(
            image, input_points=[[[float(point_xy[0]), float(point_xy[1])]]], return_tensors="pt"
        ).to(self.device)

        out = self.model(**inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0][0]  # (3, H, W) bool
        scores = out.iou_scores.cpu()[0, 0]  # (3,)

        if pick == "largest":
            best = int(masks.flatten(1).sum(1).argmax())
        else:
            best = int(scores.argmax())
        return masks[best].numpy(), float(scores[best])


    @torch.no_grad()
    def segment_box(self, image, box_xyxy):
        """Box-prompted mask (a much stronger prompt than a point). Returns (mask (H, W) bool, score)."""
        inputs = self.processor(
            image, input_boxes=[[[float(v) for v in box_xyxy]]], return_tensors="pt"
        ).to(self.device)
        out = self.model(**inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0][0]  # (3, H, W) bool
        scores = out.iou_scores.cpu()[0, 0]
        # of SAM's candidates, keep the one that agrees best with the box (it must not spill far outside it)
        x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
        box = torch.zeros_like(masks[0])
        box[max(y0, 0):y1, max(x0, 0):x1] = True
        iou = [float((m & box).sum() / max((m | box).sum(), 1)) for m in masks]
        best = int(np.argmax(iou))
        return masks[best].numpy(), float(scores[best])


    @torch.no_grad()
    def segment_prompts(self, image, box=None, points=None, labels=None):
        """Interactive refinement: an optional box plus positive (label 1) / negative (label 0) clicks, all in
        pixels. Returns (mask (H, W) bool, score). Falls back to the box alone if the combined prompt fails."""
        kw = {}
        if box is not None:
            kw["input_boxes"] = [[[float(v) for v in box]]]
        if points:
            kw["input_points"] = [[[float(x), float(y)] for x, y in points]]
            kw["input_labels"] = [[int(v) for v in labels]]
        try:
            inputs = self.processor(image, return_tensors="pt", **kw).to(self.device)
        except Exception:
            if box is None or not points:
                raise
            return self.segment_prompts(image, box=box)
        out = self.model(**inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0][0]  # (3, H, W) bool
        scores = out.iou_scores.cpu()[0, 0]
        if box is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            bx = torch.zeros_like(masks[0])
            bx[max(y0, 0):y1, max(x0, 0):x1] = True
            agree = [float((m & bx).sum() / max((m | bx).sum(), 1)) for m in masks]
            best = int(np.argmax(np.array(agree) + 0.25 * scores.numpy()))  # box agreement first, SAM's own score as tie-break
        elif points and len(points) == 1:
            best = int(masks.flatten(1).sum(1).argmax())
        else:
            best = int(scores.argmax())
        return masks[best].numpy(), float(scores[best])


SAM2_ID = "facebook/sam2.1-hiera-small"


class Sam2Tracker:
    """SAM 2 video tracking from the first-frame mask (its bounding box is the prompt). Used on GENERATED videos
    to measure whether the object follows the path and keeps its look, and to draw the mask in the interface.
    `.ok` is False when SAM 2 is unavailable (needs transformers with Sam2VideoModel); callers then skip."""

    def __init__(self, device, model_id=SAM2_ID):
        self.ok = False
        try:
            from transformers import Sam2VideoModel, Sam2VideoProcessor

            self.device = device
            self.dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
            self.model = Sam2VideoModel.from_pretrained(model_id).to(device, self.dtype).eval()
            self.processor = Sam2VideoProcessor.from_pretrained(model_id)
            self.ok = True
        except Exception as e:
            print(f"[sam2] unavailable ({type(e).__name__}: {e})")

    @torch.no_grad()
    def track(self, frames, first_mask):
        """frames: (T, H, W, 3) uint8; first_mask: (H, W) bool on frame 0. Returns (T, H, W) bool."""
        frames = [np.asarray(f) for f in frames]
        H, W = frames[0].shape[:2]
        ys, xs = np.where(first_mask)
        if len(xs) == 0:
            return np.zeros((len(frames), H, W), dtype=bool)
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        session = self.processor.init_video_session(video=frames, inference_device=self.device, dtype=self.dtype)
        self.processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=1, input_boxes=[[box]]
        )
        masks = np.zeros((len(frames), H, W), dtype=bool)
        for out in self.model.propagate_in_video_iterator(session, start_frame_idx=0):
            m = self.processor.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
            masks[out.frame_idx] = m[0, 0].cpu().numpy().astype(bool)
        masks[0] = first_mask  # frame 0 is the user's / SAM's own mask, not re-predicted
        return masks


def disk_mask(height, width, center_xy_norm, radius_norm):
    """Fallback object mask: a disk of radius `radius_norm` * sqrt(H*W) around the point."""
    ys, xs = np.mgrid[0:height, 0:width]
    cx, cy = center_xy_norm[0] * width, center_xy_norm[1] * height
    r = radius_norm * np.sqrt(height * width)
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r**2
