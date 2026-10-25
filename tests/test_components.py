"""Shape and invariant checks for the adapter pieces. Runs on CPU in seconds, no data or LTX weights.

    python -m pytest tests
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from src.adapter import MotionAdapter
from src.condition_encoder import ConditionEncoder, EntityConditionEncoder
from src.conditioning import rasterize_trajectories, render_entity_condition
from src.dataset import shift_mask
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


def test_rasterize_shape_and_peak_at_point():
    traj, vis = trajectory()
    cond = rasterize_trajectories(traj, vis, H, W)
    assert cond.shape == (B, NF, 3, H, W)

    heat = cond[0, 0, 0]
    y, x = np.unravel_index(int(heat.argmax()), heat.shape)
    assert abs(x / W - float(traj[0, 0, 0])) < 2 / W + 0.02
    assert abs(y / H - float(traj[0, 0, 1])) < 2 / H + 0.02


def test_rasterize_hidden_point_has_no_heat():
    traj, vis = trajectory()
    vis[:, 3] = False
    assert rasterize_trajectories(traj, vis, H, W)[:, 3, 0].abs().max() == 0


def test_condition_encoder_lands_on_latent_grid():
    cond = rasterize_trajectories(*trajectory(1), H, W).permute(0, 2, 1, 3, 4)
    out = ConditionEncoder()(cond)
    assert out.shape == (1, 128, *LATENT)


def test_entity_condition_and_encoder_shapes():
    traj, vis = trajectory(1)
    emb = torch.randn(1, 128)
    cond = render_entity_condition(traj, vis, emb, torch.tensor([0.1]), LATENT[1], LATENT[2])
    assert cond.shape == (1, 3 + 128, NF, LATENT[1], LATENT[2])
    assert EntityConditionEncoder()(cond).shape == (1, 128, *LATENT)


def test_pack_unpack_roundtrip():
    z = torch.randn(B, 128, *LATENT)
    assert torch.equal(unpack_latents(pack_latents(z), *LATENT).contiguous(), z)


def test_adapter_starts_as_identity():
    adapter = MotionAdapter(16)
    x, c = torch.randn(1, 5, 16), torch.randn(1, 5, 16)
    assert adapter.scale.item() == 0.0
    assert torch.equal(adapter(x, c), x)


@pytest.mark.parametrize("condition", ["points", "entity"])
def test_multilayer_injection_starts_as_noop(condition):
    mc = MotionControl(mode="multilayer", num_blocks=8, hidden_dim=32, inject_every=4, condition=condition)
    entity = {"embedding": torch.randn(1, 128), "radius": torch.tensor([0.1])} if condition == "entity" else None
    res = mc.residuals(*trajectory(1), LATENT, entity=entity)

    assert len(res) == 2  # blocks 0 and 4
    for r in res:
        assert r.shape == (1, int(np.prod(LATENT)), 32)
        assert r.abs().max() == 0


def test_residuals_are_gated_off_on_image_tokens_and_dropped_samples():
    mc = MotionControl(mode="input", condition="points")
    mc.adapter.scale.data.fill_(1.0)  # make the residual non-zero
    S = int(np.prod(LATENT))
    keep = torch.zeros(B, S, dtype=torch.bool)
    keep[:, : LATENT[1] * LATENT[2]] = True  # first latent frame

    traj, vis = trajectory()
    out = mc.residuals(traj, vis, LATENT, keep_mask=keep, cond_mask=torch.tensor([1.0, 0.0]))

    assert out[:, keep[0]].abs().max() == 0  # clean image tokens untouched
    assert out[1].abs().max() == 0  # dropped sample
    assert out[0, ~keep[0]].abs().max() > 0  # kept sample still gets the condition


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
