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


def disk_mask(height, width, center_xy_norm, radius_norm):
    """Fallback object mask: a disk of radius `radius_norm` * sqrt(H*W) around the point."""
    ys, xs = np.mgrid[0:height, 0:width]
    cx, cy = center_xy_norm[0] * width, center_xy_norm[1] * height
    r = radius_norm * np.sqrt(height * width)
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r**2
