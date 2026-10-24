"""Metrics for trajectory adherence, image preservation and temporal consistency."""

import numpy as np
import torch

CLIP_ID = "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------------
# Trajectory adherence (tracked point vs. target path, both normalized (T, 2))
# ---------------------------------------------------------------------

def trajectory_metrics(target, tracked, min_motion=1e-3, min_tracked_motion=0.01):
    T = min(len(target), len(tracked))
    target, tracked = np.asarray(target[:T]), np.asarray(tracked[:T])

    dist = np.linalg.norm(target - tracked, axis=1)
    out = {"ate": float(dist.mean()), "epe_final": float(dist[-1])}

    want = target[-1] - target[0]
    got = tracked[-1] - tracked[0]
    want_len = np.linalg.norm(want)

    if want_len > min_motion:
        got_len = np.linalg.norm(got)
        # cosine between the intended and the observed displacement (1 = right direction);
        # 0 when the tracked point moved < 1% of the frame, where a direction is just noise
        out["direction_cos"] = float(got @ want / (got_len * want_len)) if got_len > min_tracked_motion else 0.0
        # how far it moved along the intended direction, as a fraction of the intended distance
        out["progress"] = float(got @ want / want_len**2)
    return out


# ---------------------------------------------------------------------
# Image preservation and temporal consistency
# ---------------------------------------------------------------------

class VideoQuality:
    """CLIP-based scores, plus LPIPS when the `lpips` package is installed."""

    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_ID)

        try:
            import lpips

            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception:
            self.lpips = None

    @torch.no_grad()
    def embed(self, frames):
        """uint8 (T, H, W, 3) -> L2-normalised CLIP embeddings (T, D)."""
        inputs = self.processor(images=list(frames), return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        if not torch.is_tensor(feats):  # newer transformers return an output object
            feats = feats.pooler_output
        return torch.nn.functional.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def score(self, input_image, frames):
        """
        input_image: uint8 (H, W, 3), the conditioning image
        frames:      uint8 (T, H, W, 3), the generated video
        """
        emb = self.embed(np.concatenate([input_image[None], frames]))
        ref, vid = emb[0], emb[1:]

        out = {
            # similarity of every later frame to the input image (appearance kept?)
            "clip_image_sim": float((vid[1:] @ ref).mean()),
            # similarity between neighbouring frames (flicker / jumps -> lower)
            "clip_temporal": float((vid[1:] * vid[:-1]).sum(-1).mean()),
        }

        if self.lpips is not None:
            def prep(x):
                return torch.from_numpy(x).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.device)

            ref_t = prep(input_image[None]).expand(len(frames) - 1, -1, -1, -1)
            out["lpips_to_input"] = float(self.lpips(prep(frames[1:]), ref_t).mean())

        return out
