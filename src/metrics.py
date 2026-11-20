"""Metrics for trajectory adherence, image preservation and temporal consistency."""

import cv2
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
    # PCK: fraction of frames where the tracked point is within 5% / 10% of the frame of the target
    out["pck_5"] = float((dist < 0.05).mean())
    out["pck_10"] = float((dist < 0.10).mean())

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
# Object-level metrics from SAM 2 masks tracked through a generated video
# ---------------------------------------------------------------------

def mask_centroids(masks):
    """(T, H, W) bool -> ((T, 2) centroids normalised to [0, 1] as (x, y), fraction of frames with an empty mask).
    Empty frames repeat the last known centroid."""
    T, H, W = masks.shape
    out, last, empty = np.zeros((T, 2), dtype=np.float32), None, 0
    for t in range(T):
        ys, xs = np.nonzero(masks[t])
        if len(xs):
            last = np.array([xs.mean() / W, ys.mean() / H], dtype=np.float32)
        else:
            empty += 1
        out[t] = last if last is not None else (0.5, 0.5)
    return out, empty / T


def object_trajectory_metrics(target, masks):
    """Whole-object trajectory adherence: the mask centroid's path vs. the target path. The centroid is shifted
    to start where the target starts (the user's click is rarely exactly the mask centre), so this scores the
    displacement. Keys are prefixed obj_."""
    cen, empty = mask_centroids(masks)
    T = min(len(target), len(cen))
    cen = cen[:T] - cen[0] + np.asarray(target[0])
    m = {f"obj_{k}": v for k, v in trajectory_metrics(np.asarray(target)[:T], cen).items()}
    m["obj_empty_frac"] = float(empty)
    return m


def _bbox(mask, pad=2):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return max(ys.min() - pad, 0), ys.max() + 1 + pad, max(xs.min() - pad, 0), xs.max() + 1 + pad


def object_appearance(quality, input_image, first_mask, frames, masks, max_frames=8, min_px=8):
    """Is the OBJECT still the same object? CLIP similarity between the object's crop in the input image and its
    crop (tracked mask bounding box) in later frames, plus how much its size drifts (area vs. frame 0)."""
    box0 = _bbox(first_mask)
    if box0 is None:
        return {}
    y0, y1, x0, x1 = box0
    ref = input_image[y0:y1, x0:x1]
    idx = np.linspace(1, len(frames) - 1, min(max_frames, len(frames) - 1)).round().astype(int)
    crops, ratios = [], []
    for t in idx:
        b = _bbox(masks[t])
        if b is None or (b[1] - b[0]) < min_px or (b[3] - b[2]) < min_px:
            continue
        crops.append(frames[t][b[0]:b[1], b[2]:b[3]])
        ratios.append(masks[t].sum() / max(first_mask.sum(), 1))
    if not crops or ref.shape[0] < min_px or ref.shape[1] < min_px:
        return {}
    emb = quality.embed([ref] + crops)
    return {
        "obj_clip_sim": float((emb[1:] @ emb[0]).mean()),
        "obj_area_ratio": float(np.mean(ratios)),  # 1 = same size as in the input image
    }


# ---------------------------------------------------------------------
# Pixel-level helpers: image fidelity, flow warping error, sharpness, colour drift
# ---------------------------------------------------------------------

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-10 else float(10 * np.log10(255.0**2 / mse))


def ssim_gray(a, b):
    """Global-window-free SSIM on grey images (gaussian 11x11), no extra dependency."""
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    va, vb, cov = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2, blur(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
    return float(m.mean())


def warp_error(frames):
    """Temporal flicker: warp frame t+1 back onto t with Farneback flow and measure what is left
    (mean abs difference, 0-1). Smooth real motion -> small; flicker / popping -> large."""
    errs = []
    for t in range(len(frames) - 1):
        g0 = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(frames[t + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = g0.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        warped = cv2.remap(frames[t + 1], gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        errs.append(np.abs(warped.astype(np.float32) - frames[t].astype(np.float32)).mean() / 255.0)
    return float(np.mean(errs))


def sharpness(frames):
    """Mean variance of the Laplacian (no-reference; blur / diffusion mush -> low)."""
    return float(np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames]))


# ---------------------------------------------------------------------
# Image preservation, temporal consistency and video quality
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

        # pixel-level image preservation: how close is the generated first frame / the whole clip to the input
        out["first_frame_psnr"] = psnr(frames[0], input_image)
        out["first_frame_ssim"] = ssim_gray(frames[0], input_image)
        # colour drift: mean-colour change of the last frame vs. the input (0-1)
        out["color_drift"] = float(np.abs(frames[-1].reshape(-1, 3).mean(0) - input_image.reshape(-1, 3).mean(0)).mean() / 255.0)
        # temporal consistency, pixel level
        out["warp_error"] = warp_error(frames)
        # video quality (no reference): sharpness relative to the input image's own sharpness
        out["sharpness_ratio"] = sharpness(frames) / max(sharpness(input_image[None]), 1e-6)

        if self.lpips is not None:
            def prep(x):
                return torch.from_numpy(x).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.device)

            ref_t = prep(input_image[None]).expand(len(frames) - 1, -1, -1, -1)
            out["lpips_to_input"] = float(self.lpips(prep(frames[1:]), ref_t).mean())
            out["lpips_temporal"] = float(self.lpips(prep(frames[1:]), prep(frames[:-1])).mean())

        return out


# ---------------------------------------------------------------------
# Aesthetics (LAION aesthetic predictor v2, optional) and FVD-style distribution distance
# ---------------------------------------------------------------------

AESTHETIC_CLIP = "openai/clip-vit-large-patch14"
AESTHETIC_HEAD = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
VIDEO_FEATURE_MODEL = "MCG-NJU/videomae-base"


class AestheticScorer:
    """LAION aesthetic predictor v2 (CLIP ViT-L/14 + linear head, ~1-10, higher = prettier), averaged over
    up to `max_frames` frames. Needs `pip install simple-aesthetics-predictor`; `.ok` is False if the
    package or weights are unavailable, and callers should then skip the metric."""

    def __init__(self, device, max_frames=8):
        self.device, self.max_frames, self.ok = device, max_frames, False
        try:
            from aesthetics_predictor import AestheticsPredictorV2Linear
            from transformers import CLIPProcessor

            self.model = AestheticsPredictorV2Linear.from_pretrained(AESTHETIC_HEAD).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(AESTHETIC_CLIP)
            self.ok = True
        except Exception as e:  # missing package / no network: the benchmark just leaves the column out
            print(f"[aesthetics] unavailable ({type(e).__name__}: {e})")

    @torch.no_grad()
    def score(self, frames):
        idx = np.linspace(0, len(frames) - 1, min(self.max_frames, len(frames))).round().astype(int)
        inputs = self.processor(images=[frames[i] for i in idx], return_tensors="pt").to(self.device)
        return float(self.model(**inputs).logits.float().mean())


def frechet_distance(a, b):
    """Frechet distance between two feature sets (N1, D), (N2, D). Needs N >= 2 each; with N << D the
    covariances are rank-deficient, so only compare values computed with the same N."""
    from scipy import linalg

    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps).dot(s2 + eps))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


class VideoFeatures:
    """Clip-level features for an FVD-style score. The classic FVD uses I3D (needs a separate weight
    download); this uses VideoMAE-base (16 frames -> mean token, 768-d), so numbers are NOT comparable with
    published FVD, only between runs/methods of this project."""

    def __init__(self, device):
        self.ok = False
        try:
            from transformers import VideoMAEImageProcessor, VideoMAEModel

            self.device = device
            self.model = VideoMAEModel.from_pretrained(VIDEO_FEATURE_MODEL).to(device).eval()
            self.processor = VideoMAEImageProcessor.from_pretrained(VIDEO_FEATURE_MODEL)
            self.ok = True
        except Exception as e:
            print(f"[fvd] unavailable ({type(e).__name__}: {e})")

    @torch.no_grad()
    def __call__(self, frames):
        """uint8 (T, H, W, 3) -> (768,) numpy"""
        idx = np.linspace(0, len(frames) - 1, 16).round().astype(int)
        inputs = self.processor([frames[i] for i in idx], return_tensors="pt").to(self.device)
        return self.model(**inputs).last_hidden_state.float().mean(1)[0].cpu().numpy()
