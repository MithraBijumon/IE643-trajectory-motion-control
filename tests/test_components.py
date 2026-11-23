"""Shape and invariant checks for the adapter pieces. Runs on CPU in seconds, no data or LTX weights.

    python -m pytest tests
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from src.condition_encoder import EntityConditionEncoder
from src.conditioning import (SHAPE_FACTOR, entity_footprint, render_entity_condition,
                              shape_footprint)
from src.dataset import mask_from_lowres, shift_mask
from src.metrics import trajectory_metrics
from src.motion_control import MotionControl, pack_latents, unpack_latents
from src.utils import linear_path, load_config, resample_trajectory

# the tiny config: 17 frames at 64x96 -> a (3, 2, 3) latent grid
cfg = load_config(tiny=True)
NF, H, W = (cfg["data"][k] for k in ("num_frames", "height", "width"))
LATENT = ((NF - 1) // 8 + 1, H // 32, W // 32)
B = 2


def trajectory(batch=B):
    t = torch.linspace(0.2, 0.7, NF)
    return torch.stack([t, 1 - t], dim=-1).expand(batch, -1, -1).contiguous(), torch.ones(batch, NF, dtype=torch.bool)


def test_config_frames_and_size_are_valid():
    assert (NF - 1) % 8 == 0 and H % 32 == 0 and W % 32 == 0


def test_entity_condition_and_encoder_shapes():
    traj, vis = trajectory(1)
    emb = torch.randn(1, 128)
    cond = render_entity_condition(traj, vis, emb, torch.tensor([0.1]), LATENT[1], LATENT[2])
    assert cond.shape == (1, 3 + 128, NF, LATENT[1], LATENT[2])
    assert EntityConditionEncoder()(cond).shape == (1, 128, *LATENT)


def test_entity_footprint_peaks_at_the_object_and_vanishes_when_hidden():
    traj, vis = trajectory(1)
    vis[:, 5] = False
    fp = entity_footprint(traj, vis, torch.tensor([0.1]), LATENT[1], LATENT[2])
    assert fp.shape == (1, NF, LATENT[1], LATENT[2])
    assert fp[0, 5].abs().max() == 0

    y, x = np.unravel_index(int(fp[0, 0].argmax()), fp[0, 0].shape)
    assert abs((x + 0.5) / LATENT[2] - float(traj[0, 0, 0])) < 1 / LATENT[2]
    assert abs((y + 0.5) / LATENT[1] - float(traj[0, 0, 1])) < 1 / LATENT[1]


def test_pack_unpack_roundtrip():
    z = torch.randn(B, 128, *LATENT)
    assert torch.equal(unpack_latents(pack_latents(z), *LATENT).contiguous(), z)


ENTITY = lambda: {"embedding": torch.randn(1, 128), "radius": torch.tensor([0.1])}


def test_multilayer_injection_starts_as_noop():
    mc = MotionControl(num_blocks=8, hidden_dim=32, inject_every=4)
    res = mc.residuals(*trajectory(1), LATENT, entity=ENTITY())
    assert len(res) == 2  # blocks 0 and 4
    for r in res:
        assert r.shape == (1, int(np.prod(LATENT)), 32)
        assert r.abs().max() == 0


def test_residuals_are_gated_off_on_image_tokens_and_dropped_samples():
    mc = MotionControl(num_blocks=4, hidden_dim=32, inject_every=4)
    torch.nn.init.normal_(mc.projs[0].weight, std=0.1)  # make the residual non-zero
    S = int(np.prod(LATENT))
    keep = torch.zeros(B, S, dtype=torch.bool)
    keep[:, : LATENT[1] * LATENT[2]] = True  # first latent frame
    entity = {"embedding": torch.randn(B, 128), "radius": torch.full((B,), 0.1)}

    out = mc.residuals(*trajectory(), LATENT, keep_mask=keep, cond_mask=torch.tensor([1.0, 0.0]), entity=entity)[0]

    assert out[:, keep[0]].abs().max() == 0  # clean image tokens untouched
    assert out[1].abs().max() == 0  # dropped sample
    assert out[0, ~keep[0]].abs().max() > 0  # kept sample still gets the condition


def test_local_injection_leaves_the_background_alone():
    shape = (3, 8, 12)
    traj = torch.full((1, NF, 2), 0.1)  # object stays in the top-left corner
    vis = torch.ones(1, NF, dtype=torch.bool)
    entity = {"embedding": torch.randn(1, 128), "radius": torch.tensor([0.05])}
    outs = []
    for local in (0.0, 2.0):
        mc = MotionControl(num_blocks=4, hidden_dim=16, inject_every=4, local_inject=local)
        torch.nn.init.normal_(mc.projs[0].weight, std=0.1)
        outs.append(mc.residuals(traj, vis, shape, entity=entity)[0].view(3, 8, 12, 16))
    everywhere, local = outs
    assert everywhere[:, 6:, 9:].abs().max() > 0          # without it the far corner gets a residual
    assert local[:, 6:, 9:].abs().max() < 1e-6            # with it the far corner is untouched
    assert local[:, :2, :3].abs().max() > 0               # the object's own cells still get it


def test_resample_and_linear_path():
    traj = np.stack([np.linspace(0, 1, 5), np.linspace(1, 0, 5)], axis=1)
    out, vis = resample_trajectory(traj, NF)
    assert out.shape == (NF, 2) and vis.shape == (NF,)
    assert np.allclose(out[0], traj[0]) and np.allclose(out[-1], traj[-1])

    path = linear_path([0.97, 0.5], (1.0, 0.0), 0.3, NF)  # would leave the frame on the right
    assert path.min() >= 0.05 - 1e-6 and path.max() <= 0.95 + 1e-6


def test_shift_mask_moves_and_clips():
    m = np.zeros((6, 8), dtype=bool)
    m[2, 3] = True
    s = shift_mask(m, 2, 1)
    assert s[3, 5] and s.sum() == 1
    assert shift_mask(m, 20, 0).sum() == 0


def test_trajectory_metrics():
    t = np.stack([np.linspace(0.2, 0.6, NF), np.full(NF, 0.5)], axis=1)
    perfect = trajectory_metrics(t, t)
    assert perfect["ate"] == 0 and perfect["epe_final"] == 0
    assert perfect["direction_cos"] == pytest.approx(1.0) and perfect["progress"] == pytest.approx(1.0)

    static = trajectory_metrics(t, np.repeat(t[:1], NF, axis=0))
    assert static["ate"] > 0 and static["direction_cos"] == 0.0 and static["progress"] == 0.0


def _blob_mask(gh, gw, cell_y, cell_x):
    """(1, 1, 4*gh, 4*gw) coverage with the object filling one latent cell."""
    m = torch.zeros(1, 1, SHAPE_FACTOR * gh, SHAPE_FACTOR * gw)
    m[..., cell_y * SHAPE_FACTOR:(cell_y + 1) * SHAPE_FACTOR, cell_x * SHAPE_FACTOR:(cell_x + 1) * SHAPE_FACTOR] = 1
    return m


def test_shape_footprint_follows_the_trajectory_and_hides_with_the_object():
    gh, gw = 4, 6
    mask = _blob_mask(gh, gw, 1, 1)
    traj = torch.zeros(1, 5, 2) + 0.5
    traj[0, :, 0] += torch.arange(5) / gw          # one latent cell to the right per frame
    vis = torch.ones(1, 5, dtype=torch.bool)
    vis[0, 3] = False

    fp = shape_footprint(mask, traj, vis, gh, gw)
    assert fp.shape == (1, 5, gh, gw) and 0 <= fp.min() and fp.max() <= 1
    for t in (0, 1, 2, 4):
        y, x = np.unravel_index(int(fp[0, t].argmax()), (gh, gw))
        assert (y, x) == (1, 1 + t)               # same row, moves right with the path
    assert fp[0, 3].abs().max() == 0               # hidden -> nothing


def test_shape_footprint_keeps_a_tiny_object_visible():
    gh, gw = 4, 6
    mask = torch.zeros(1, 1, SHAPE_FACTOR * gh, SHAPE_FACTOR * gw)
    mask[..., 5, 5] = 0.2                          # a speck much smaller than a latent cell
    traj = torch.full((1, 3, 2), 0.5)
    fp = shape_footprint(mask, traj, torch.ones(1, 3, dtype=torch.bool), gh, gw)
    assert float(fp[0, 0].max()) == pytest.approx(1.0)


def test_render_entity_condition_with_shape_has_the_same_layout():
    traj, vis = trajectory(1)
    gh, gw = LATENT[1], LATENT[2]
    emb = torch.randn(1, 128)
    plain = render_entity_condition(traj, vis, emb, torch.tensor([0.1]), gh, gw)
    shaped = render_entity_condition(traj, vis, emb, torch.tensor([0.1]), gh, gw, shape=_blob_mask(gh, gw, 0, 0))
    assert shaped.shape == plain.shape == (1, 3 + 128, NF, gh, gw)
    assert not torch.equal(shaped, plain)


def test_mask_from_lowres_and_empty_mask():
    low = np.zeros((6, 10), dtype=np.uint8)
    low[2:4, 3:6] = 255
    mask, area = mask_from_lowres(low, 24, 40)
    assert mask.shape == (24, 40) and 0 < area < 0.5
    assert mask_from_lowres(np.zeros((6, 10), dtype=np.uint8), 24, 40) == (None, 0.0)


def _fake_transformer(n):
    t = torch.nn.Module()
    t.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(n)])
    return t


def test_max_inject_caps_the_injection_even_with_huge_weights():
    S = int(np.prod(LATENT))
    mc = MotionControl(num_blocks=8, hidden_dim=32, inject_every=4, relative=True, max_inject=0.05)
    for proj in mc.projs:
        torch.nn.init.normal_(proj.weight, std=5.0)           # a diverged adapter
    tr = _fake_transformer(8)
    mc.attach(tr)
    mc.set_context(mc.residuals(*trajectory(1), LATENT, entity=ENTITY()))

    h = torch.randn(1, S, 32)
    out = tr.transformer_blocks[0](h)                          # the hook adds the (capped) residual
    ratio = ((out - h).pow(2).mean().sqrt() / h.pow(2).mean().sqrt()).item()
    assert ratio <= 0.05 + 1e-4
    assert mc.injection_ratio() <= 0.05 + 1e-4


def test_injection_ratio_ignores_dropped_condition_forwards():
    S = int(np.prod(LATENT))
    mc = MotionControl(num_blocks=4, hidden_dim=16, inject_every=4, relative=True)
    torch.nn.init.normal_(mc.projs[0].weight, std=0.1)
    tr = _fake_transformer(4)
    mc.attach(tr)
    h = torch.randn(1, S, 16)

    mc.set_context(mc.residuals(*trajectory(1), LATENT, entity=ENTITY()))
    tr.transformer_blocks[0](h)
    seen = mc.injection_ratio()
    assert seen > 0

    mc.set_context(mc.residuals(*trajectory(1), LATENT, cond_mask=torch.zeros(1), entity=ENTITY()))   # condition dropped
    tr.transformer_blocks[0](h)
    assert mc.injection_ratio() == seen                        # the 0 is not recorded


def test_cond_blocks_add_trainable_params_and_roundtrip():
    import torch
    from src.motion_control import MotionControl

    kw = dict(num_blocks=8, hidden_dim=32, inject_every=2)
    plain = MotionControl(**kw)
    mc = MotionControl(**kw, cond_blocks=2, cond_heads=4)
    assert len(mc.projs) == 4  # inject_every=2 over 8 blocks
    assert sum(p.numel() for p in mc.parameters()) > sum(p.numel() for p in plain.parameters())

    traj = torch.rand(1, 9, 2)
    vis = torch.ones(1, 9, dtype=torch.bool)
    tok = mc.condition_tokens(traj, vis, (2, 2, 3), ENTITY())
    assert tok.shape == (1, 12, 128)

    other = MotionControl(**kw, cond_blocks=2, cond_heads=4)
    other.load_trainable_state(mc.trainable_state())
    other.eval(); mc.eval()
    e = ENTITY()
    assert torch.allclose(other.condition_tokens(traj, vis, (2, 2, 3), e), mc.condition_tokens(traj, vis, (2, 2, 3), e), atol=1e-6)


def test_extra_quality_metrics():
    import numpy as np
    from src.metrics import psnr, ssim_gray, warp_error, sharpness, trajectory_metrics

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
    assert psnr(img, img) == 99.0 and abs(ssim_gray(img, img) - 1) < 1e-6
    static = np.stack([img] * 4)
    assert warp_error(static) < 1e-3
    flicker = rng.integers(0, 255, (4, 64, 96, 3), dtype=np.uint8)
    assert warp_error(flicker) > warp_error(static)
    assert sharpness(static) > 0
    t = np.linspace([0.2, 0.5], [0.6, 0.5], 9)
    assert trajectory_metrics(t, t)["pck_5"] == 1.0


def _tiny_batch(nf=17, h=64, w=96):
    import torch

    g = torch.Generator().manual_seed(0)
    lm = torch.zeros(1, nf, 15, 24)  # per-frame SAM-style masks at low res
    for t in range(nf):
        lm[0, t, 5:9, 3 + t // 2: 8 + t // 2] = 1.0
    mask = torch.zeros(1, 1, h, w)
    mask[..., 20:36, 12:32] = 1.0
    return {
        "name": ["x"],
        "video": torch.rand(1, nf, 3, h, w, generator=g) * 2 - 1,
        "trajectory": torch.linspace(0.2, 0.6, nf).view(1, nf, 1).repeat(1, 1, 2),
        "visible": torch.ones(1, nf, dtype=torch.bool),
        "mask": mask,
        "mask_area": torch.tensor([0.1]),
        "loss_masks": lm,
    }


def test_split_loss_time_ramp_and_parts_run_on_the_tiny_model():
    import copy
    import torch
    from src.ltx_wrapper import LTXBackbone

    c = copy.deepcopy(cfg)
    c["adapter"].update(cond_blocks=1, inject_every=2)
    c["train"].update(focus_loss=5.0, split_loss=True, time_ramp=1.0)
    bb = LTXBackbone(c, "cpu")
    batch = _tiny_batch()
    def g():
        torch.manual_seed(1)  # the VAE samples its latent from the global RNG
        return torch.Generator().manual_seed(1)

    split, parts = bb.training_loss(batch, generator=g(), use_condition=True, sigma=0.6, return_parts=True)
    assert torch.isfinite(split) and parts["obj"] > 0 and parts["bg"] >= 0  # the tiny 2x3 grid is all "near the object"

    c["train"]["split_loss"] = False
    flat = bb.training_loss(batch, generator=g(), use_condition=True, sigma=0.6)
    plain = bb.training_loss(batch, generator=g(), use_condition=True, sigma=0.6, focus=0.0)
    assert torch.isfinite(flat) and torch.isfinite(plain)
    # with focus 0 the loss is the plain mean: the same number whichever mode is selected
    c["train"]["split_loss"] = True
    assert torch.allclose(plain, bb.training_loss(batch, generator=g(), use_condition=True, sigma=0.6, focus=0.0))


def test_object_level_metrics_and_overlay():
    import numpy as np
    from src.metrics import mask_centroids, object_trajectory_metrics
    from src.utils import overlay_masks

    T, H, W = 9, 40, 60
    masks = np.zeros((T, H, W), dtype=bool)
    for t in range(T):
        masks[t, 15:25, 5 + 4 * t: 15 + 4 * t] = True          # a box sliding right
    cen, empty = mask_centroids(masks)
    assert empty == 0 and cen[-1, 0] > cen[0, 0]

    target = np.stack([np.linspace(0.1, 0.1 + 32 / W, T), np.full(T, 0.5)], axis=1)
    m = object_trajectory_metrics(target, masks)
    assert m["obj_ate"] < 0.02 and m["obj_direction_cos"] > 0.99 and abs(m["obj_progress"] - 1) < 0.05

    masks[5:] = False                                            # object lost: centroid freezes, reported
    m2 = object_trajectory_metrics(target, masks)
    assert m2["obj_empty_frac"] > 0.4 and m2["obj_ate"] > m["obj_ate"]

    frames = np.zeros((T, H, W, 3), dtype=np.uint8)
    assert overlay_masks(frames, masks, target).shape == frames.shape


def test_render_reference_moves_the_cutout_and_keeps_frame_zero():
    from src.reference import render_reference

    H, W = 48, 64
    img = np.full((H, W, 3), 60, np.uint8)
    img[10:22, 8:20] = (250, 30, 30)                    # a red object
    mask = np.zeros((H, W), bool)
    mask[10:22, 8:20] = True
    traj = np.stack([np.linspace(0.2, 0.6, 5), np.full(5, 0.3)], 1)  # moves right by 0.4 W = 25 px
    ref = render_reference(img, mask, traj, np.array([1, 1, 1, 0, 1], bool))
    assert ref.shape == (5, H, W, 3) and ref.dtype == np.uint8
    assert (ref[0] == img).all()                        # frame 0 is the input image
    red = lambda f: np.argwhere((ref[f, ..., 0] > 200) & (ref[f, ..., 1] < 80))[:, 1].mean()
    assert red(4) - red(0) > 20                         # the cut-out moved right
    assert not ((ref[3, ..., 0] > 200) & (ref[3, ..., 1] < 80)).any()  # hidden frame: the object is gone
    assert not ((ref[4, :, :6, 0] > 200)).any()         # and its old place is filled with scene, not left red


def test_drag_start_and_composite_generate_on_the_tiny_model():
    import copy
    import torch
    from PIL import Image
    from src.ltx_wrapper import LTXBackbone

    c = copy.deepcopy(cfg)
    c["adapter"].update(cond_blocks=1, inject_every=1)
    bb = LTXBackbone(c, "cpu")
    batch = _tiny_batch()
    torch.manual_seed(0)
    assert torch.isfinite(bb.training_loss(batch, generator=torch.Generator().manual_seed(0), use_condition=True, sigma=0.5))

    traj = np.stack([np.linspace(0.2, 0.6, NF), np.full(NF, 0.4)], 1).astype(np.float32)
    mask = np.zeros((H, W), bool)
    mask[20:36, 12:32] = True
    img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (H, W, 3), dtype=np.uint8))
    for kw in ({}, {"ref_init": 0.7}, {"ref_init": 0.7, "bg_comp": 2.5}):
        frames = bb.generate(img, traj, steps=4, mask=mask, **kw)
        assert frames.shape == (NF, H, W, 3)


def test_per_clip_prompts_are_used_when_cached_and_fall_back_otherwise():
    import copy
    import torch
    from src.ltx_wrapper import LTXBackbone
    from src.dataset import collect_prompts, read_caption
    import tempfile

    bb = LTXBackbone(copy.deepcopy(cfg), "cpu")
    L, D = bb.prompt[0].shape[1:]
    bb.prompt_cache["a red car"] = (torch.randn(L, D), torch.ones(L, dtype=bb.prompt[1].dtype))
    emb, mask = bb._text(3, captions=["a red car", "unseen text", ""])
    assert emb.shape == (3, L, D) and mask.shape == (3, L)
    assert torch.allclose(emb[0].float(), bb.prompt_cache["a red car"][0].float(), atol=1e-2)   # cached caption
    assert torch.allclose(emb[1].float(), bb.prompt[0][0].float()) and torch.allclose(emb[2].float(), bb.prompt[0][0].float())

    batch = _tiny_batch()
    batch["caption"] = ["a red car"]
    torch.manual_seed(0)
    assert torch.isfinite(bb.training_loss(batch, generator=torch.Generator().manual_seed(0), use_condition=True, sigma=0.5))
    bb.add_prompts(["never encoded"])                 # the tiny model has no text encoder: ignored, the config prompt is used
    assert "never encoded" not in bb.prompt_cache
    bb._t5_pipe = None
    bb.cfg = dict(bb.cfg, is_tiny=False)              # a real model without a kept text encoder must refuse, not guess
    try:
        bb.add_prompts(["never encoded"])
        raise AssertionError("expected a RuntimeError without a text encoder")
    except RuntimeError:
        pass

    with tempfile.TemporaryDirectory() as root:
        for split, name, text in (("train", "a", "one cat."), ("val", "b", "one cat."), ("train", "c", "")):
            d = os.path.join(root, split, name)
            os.makedirs(d)
            if text:
                open(os.path.join(d, "caption.txt"), "w").write(text + chr(10))
        assert collect_prompts(root) == ["one cat."] and read_caption(os.path.join(root, "train", "c")) == ""


def test_background_motion_separates_object_motion_from_background_motion():
    import cv2
    from src.metrics import background_motion

    T, H, W = 6, 128, 192
    rng = np.random.RandomState(0)
    base = cv2.GaussianBlur(rng.randint(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 3)   # a smooth textured scene
    base = cv2.normalize(base, None, 0, 200, cv2.NORM_MINMAX)
    frames, masks = [], []
    for t in range(T):                                  # a static scene with a bright square moving right
        f = base.copy()
        f[50:80, 10 + 12 * t: 40 + 12 * t] = 255
        m = np.zeros((H, W), bool)
        m[50:80, 10 + 12 * t: 40 + 12 * t] = True
        frames.append(f)
        masks.append(m)
    still = background_motion(np.stack(frames), np.stack(masks))
    assert still["bg_drift"] < 0.02 and still["bg_flow"] < 0.006   # object motion is not background motion

    shifted = np.stack([np.roll(base, 4 * t, axis=1) for t in range(T)])      # the whole scene slides
    moving = background_motion(shifted, np.stack(masks))
    assert moving["bg_drift"] > 5 * still["bg_drift"] + 0.01 and moving["bg_flow"] > 3 * still["bg_flow"]


def test_near_object_scale_widens_the_footprint():
    mc = MotionControl(num_blocks=4, hidden_dim=16, inject_every=4, local_inject=2.0)
    traj = torch.full((1, NF, 2), 0.1)
    vis = torch.ones(1, NF, dtype=torch.bool)
    entity = {"embedding": torch.randn(1, 128), "radius": torch.tensor([0.05])}
    shape = (3, 8, 12)
    small = mc.near_object(traj, vis, shape, entity, scale=1.0)
    large = mc.near_object(traj, vis, shape, entity, scale=4.0)
    assert small.shape == (1, 3 * 8 * 12, 1) and large.sum() > small.sum()


def test_benchmark_helpers_for_clean_ood_images():
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from benchmark import fit_image, moved_masks
    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.png")
        Image.fromarray(np.zeros((100, 60, 3), np.uint8)).save(p)               # a portrait image
        out = fit_image(p, 48, 96)
        assert out.size == (96, 48)                                              # centre-cropped to 16:9-ish, not stretched
    m = np.zeros((40, 80), bool)
    m[10:20, 10:20] = True
    path = np.stack([np.linspace(0.2, 0.6, 5), np.full(5, 0.4)], 1)
    mm = moved_masks(m, path)
    assert mm.shape == (5, 40, 80) and (mm[0] == m).all()
    assert np.argwhere(mm[4])[:, 1].mean() - np.argwhere(mm[0])[:, 1].mean() > 25   # moved 0.4 * 80 = 32 px right


def test_benchmark_end_to_end_with_stubbed_heavy_models(tmp_path, monkeypatch):
    """Runs benchmark.main() on the tiny LTX with fake CLIP / CoTracker / SAM 2 (not available offline):
    two seeds, a val clip and a clean out-of-domain image, ref_init on."""
    import json
    import imageio.v2 as imageio
    from PIL import Image

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import benchmark

    rng = np.random.RandomState(0)
    clip = tmp_path / "data" / "val" / "clipA"
    clip.mkdir(parents=True)
    Image.fromarray(rng.randint(0, 255, (H, W, 3), dtype=np.uint8)).save(clip / "image.png")
    m = np.zeros((H, W), np.uint8)
    m[20:36, 12:32] = 255
    Image.fromarray(m).save(clip / "mask.png")
    np.save(clip / "trajectory.npy", np.stack([np.linspace(0.3, 0.5, NF), np.full(NF, 0.45)], 1).astype(np.float32))
    imageio.mimsave(clip / "video.mp4", [np.zeros((H, W, 3), np.uint8)] * 4, fps=4)
    (clip / "caption.txt").write_text("a grey square on noise.")
    ood = tmp_path / "ood"
    ood.mkdir()
    Image.fromarray(rng.randint(0, 255, (80, 120, 3), dtype=np.uint8)).save(ood / "thing.png")
    Image.fromarray(m[::1]).resize((120, 80)).save(ood / "thing_mask.png")        # user-given mask: no SAM needed

    class FakeQuality:
        def __init__(self, device): pass
        def score(self, img, frames):
            return {"clip_image_sim": 0.9, "first_frame_psnr": 30.0, "first_frame_ssim": 0.9, "color_drift": 0.01,
                    "clip_temporal": 0.95, "warp_error": 0.01, "sharpness_ratio": 1.0}

    class FakeTracker:
        ok = True
        def __init__(self, device): pass
        def track(self, frames, mask): return np.repeat(mask[None], len(frames), 0)

    monkeypatch.setattr(benchmark, "load_cotracker", lambda device: None)
    monkeypatch.setattr(benchmark, "track_point", lambda ct, frames, start, device: np.stack(
        [np.linspace(start[0] / frames.shape[2], start[0] / frames.shape[2] + 0.1, len(frames)),
         np.full(len(frames), start[1] / frames.shape[1])], 1))
    monkeypatch.setattr(benchmark, "VideoQuality", FakeQuality)
    monkeypatch.setattr(benchmark, "Sam2Tracker", FakeTracker)
    monkeypatch.setattr(benchmark, "object_appearance", lambda *a, **k: {})
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--tiny", "--data_root", str(tmp_path / "data"), "--num_images", "1",
                                      "--ood_dir", str(ood), "--motions", "left,right", "--seeds", "0,1", "--steps", "3",
                                      "--ref_init", "0.7", "--out", str(out)])
    benchmark.main()

    res = json.loads((out / "results.json").read_text())
    assert len(res["records"]["adapter"]) == 2 * 2 * 2          # images x motions x seeds
    assert {r["group"] for r in res["records"]["adapter"]} == {"val", "ood"}
    row = res["records"]["adapter"][0]
    assert "bg_flow" in row and "bg_drift" in row and "obj_ate" in row
    assert set(res["summary"]["by_group"]["adapter"]) == {"val", "ood"} and "seed_std" in res["summary"]
    text = (out / "summary.md").read_text()
    assert "## per motion" in text and "## by image group" in text and "std over seeds" in text


def test_scene_composite_replaces_background_tokens_with_the_object_free_scene():
    import copy
    import torch
    from PIL import Image
    from src.ltx_wrapper import LTXBackbone

    c = copy.deepcopy(cfg)
    c["adapter"].update(cond_blocks=1, inject_every=1)
    c["inference"]["bg_comp_release"] = 0.0           # never release: the composite stays on to the last step
    bb = LTXBackbone(c, "cpu")
    Fl, Hl, Wl = LATENT
    gen = torch.Generator().manual_seed(0)
    fixed_img = torch.randn(1, 128, 1, Hl, Wl, generator=gen)
    fixed_ref = torch.randn(1, 128, Fl, Hl, Wl, generator=gen)
    bb.encode_video = lambda v: fixed_img if v.shape[2] == 1 else fixed_ref   # deterministic stand-in for the VAE
    seen = {}

    def grab(latents, *a, **k):
        seen["lat"] = latents
        return np.zeros((1, NF, H, W, 3), np.uint8)

    bb.decode_latents = grab
    bb.motion.near_object = lambda *a, **k: torch.zeros(1, Fl * Hl * Wl, 1)      # every token counts as background
    img = Image.fromarray(np.zeros((H, W, 3), np.uint8))
    path = np.stack([np.linspace(0.2, 0.6, NF), np.full(NF, 0.4)], 1).astype(np.float32)
    mask = np.zeros((H, W), bool)
    mask[20:36, 12:32] = True
    bb.generate(img, path, steps=4, mask=mask, bg_comp=2.5)
    lat = seen["lat"]
    assert torch.allclose(lat[:, :, 0], fixed_img[:, :, 0], atol=1e-5)         # the input image frame stays clean
    assert torch.allclose(lat[:, :, 1:], fixed_ref[:, :, 1:], atol=1e-5)       # everything else = the object-free scene


def test_token_cap_has_no_hot_spot_and_adapter_scale_is_a_plain_dial():
    S = int(np.prod(LATENT))

    def run(cap, scale=1.0):
        torch.manual_seed(0)
        mc = MotionControl(num_blocks=4, hidden_dim=32, inject_every=4, relative=True, max_inject=0.1, local_inject=1.0, cap=cap)
        for proj in mc.projs:
            torch.nn.init.normal_(proj.weight, std=5.0)
        mc.scale = scale
        near = torch.zeros(1, S, 1)
        near[:, :3] = 1.0                                    # a small footprint: 3 of the 18 tokens
        mc.near_object = lambda *a, **k: near
        tr = _fake_transformer(4)
        mc.attach(tr)
        mc.set_context(mc.residuals(*trajectory(1), LATENT, entity=ENTITY()))
        h = torch.randn(1, S, 32)
        d = tr.transformer_blocks[0](h) - h
        per_token = d.pow(2).mean(-1).sqrt() / h.pow(2).mean().sqrt()      # each token's residual relative to the hidden RMS
        return d, per_token, mc.injection_ratio()

    d_tok, pt_tok, ratio_tok = run("token")
    d_glob, pt_glob, _ = run("global")
    assert (d_tok.abs().sum(-1) > 0).any() and (d_tok.abs().sum(-1) == 0).any()      # local: some tokens get nothing
    assert pt_tok.max() <= 0.1 + 1e-4                                                  # per-token cap holds everywhere
    assert ratio_tok <= 0.1 + 1e-4                                                     # ... and the logged strength agrees
    assert pt_glob.max() > 0.1 + 1e-3                                                  # the old cap lets the footprint exceed it (hot spot)
    d_half, _, _ = run("token", scale=0.5)
    assert torch.allclose(d_half, 0.5 * d_tok, atol=1e-6)
    d_off, _, _ = run("token", scale=0.0)
    assert float(d_off.abs().max()) == 0.0


def test_adapter_scale_and_min_sigma_gate_the_adapter_during_generation():
    import copy
    from PIL import Image
    from src.ltx_wrapper import LTXBackbone

    c = copy.deepcopy(cfg)
    c["adapter"].update(cond_blocks=1, inject_every=1)
    bb = LTXBackbone(c, "cpu")
    for proj in bb.motion.projs:                                  # make the adapter do something visible
        torch.nn.init.normal_(proj.weight, std=0.5)
    img = Image.fromarray(np.full((H, W, 3), 90, np.uint8))
    path = np.stack([np.linspace(0.2, 0.6, NF), np.full(NF, 0.4)], 1).astype(np.float32)
    mask = np.zeros((H, W), bool)
    mask[20:36, 12:32] = True
    seen = []
    bb.decode_latents = lambda latents, *a, **k: (seen.append(latents.clone()) or np.zeros((1, NF, H, W, 3), np.uint8))
    gen = lambda **k: bb.generate(img, path, steps=4, mask=mask, seed=3, **k)
    gen()                                  # full strength
    gen(adapter_scale=0.0)                 # off
    gen(adapter_min_sigma=2.0)             # never on (sigma <= 1)
    gen(adapter_scale=0.5)
    bb.generate(img, path, steps=4, mask=mask, seed=3, use_motion=False)   # plain LTX
    full, off, never, half, plain = seen
    assert torch.allclose(off, never, atol=1e-6) and torch.allclose(off, plain, atol=1e-6)   # off == no adapter at all
    assert not torch.allclose(full, off, atol=1e-4) and not torch.allclose(half, full, atol=1e-5)
    assert bb.motion.scale == 1.0          # the dial is restored after a call
