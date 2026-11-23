# Trajectory motion control for LTX-Video

Image + drawn path -> video in which one object follows the path. A small trainable adapter (5.9M parameters) is injected
into a **frozen** LTX-Video 2B (0.9.5); LTX's own weights are never modified (no LoRA).

```
object mask + path -> footprint / displacement / object feature on the latent grid
  -> conv encoder -> 2 small transformer blocks -> zero-init projections
  -> added (forward hooks) after every 2nd frozen LTX block, only near the object
```

Training is LTX's flow-matching loss with the first frame as clean conditioning. `pip install -r requirements.txt`;
all settings are in `configs/config.yaml` (CLI flags override single values). Colab: `notebooks/train_colab.ipynb`.

## Design choices

- **Relative scaling and cap.** The adapter's output is multiplied by the RMS of the hidden state it is added to and capped at
  `max_inject` (5%) of it. `inject_cap: global` bounds the RMS over all tokens; with a small footprint a token inside it can then
  receive many times the cap, `inject_cap: token` bounds every token on its own. At the trained strength the cap rarely binds.
- **Object-only injection (`local_inject`).** The residual is multiplied by a Gaussian footprint (2 x the object's radius) along
  the path, so background tokens receive nothing.
- **Entity condition.** The object's SAM silhouette moved along the path, its displacement and its own latent feature.
- **Focus loss.** Object tokens (SAM 2 per-frame masks) and background tokens are averaged separately,
  `(focus * L_obj + L_bg) / (focus + 1)`; `time_ramp` weighs later frames more; `--focus_loss 0` is the plain mean.
- **Per-clip prompts (`data.captions`).** One caption per clip (`scripts/caption_clips.py`, Qwen2-VL); LTX-2B breaks up on a generic prompt.
- **Condition dropout** trains some samples without the trajectory, so `--motion_guidance g` can amplify it at inference.
- **Training-free drag video (`src/reference.py`).** The object is cut out, the hole inpainted and the cut-out moved along the
  path. `--ref_init 0.8` starts denoising from that video (SDEdit), `--bg_comp 2.5` puts the object-free scene back outside the footprint.

## Data (GOT-10k, one moving target per clip)

```bash
python scripts/stream_got10k.py --out_dir data/dataset_got10k --max_clips 900
python scripts/camera_motion.py --data_root data/dataset_got10k --max_score 0.25 --move_rejected   # drop moving-camera clips
python scripts/make_masks.py --data_root data/dataset_got10k --use_box --overwrite                 # SAM: frame-0 mask
python scripts/make_video_masks.py --data_root data/dataset_got10k                                 # SAM 2: per-frame masks
python scripts/caption_clips.py --models qwen --write --data_root data/dataset_got10k              # per-clip prompts
```

## Train, generate, evaluate

| Step | Command |
|---|---|
| train | `python scripts/train.py [--resume checkpoints/last.pt] [--lr 1e-4] [--focus_loss 0]` |
| generate | `python scripts/inference.py --checkpoint best.pt --sample_dir data/dataset_got10k/val/<clip> --adapter_scale 3` |
| visual check | `python scripts/demo_grid.py --checkpoint best.pt --num 3 --variants base,adapter,adapter+as3,ref0.8+comp2.5` |
| benchmark | `python scripts/benchmark.py --checkpoint best.pt --num_images 8 --adapter_scale 3 [--seeds 0,1,2] [--ood_dir imgs]` |
| one traced example | `python scripts/trace_example.py --checkpoint best.pt --direction right` |
| demo app | `python app/app.py --checkpoint best.pt` (adapter-strength slider included) |
| tests | `python -m pytest tests` |

Inference dials: `--adapter_scale` (1 = as trained, 0 = off; the injection is small, 3 was benchmarked), `--adapter_min_sigma`
(adapter only at high noise), `--motion_guidance`, `--ref_init`, `--bg_comp`. `demo_grid` variants: `ref0.8`, `comp2.5`, `as3`, `until0.4`, `mg3`, `rel0`.

## Metrics (`scripts/benchmark.py`)

Straight left / right / up / down / diagonal paths from each val image's object (and from clean out-of-domain images with
`--ood_dir`: `name.jpg`, optional `name.txt` prompt and `name_mask.png`), adapter vs. plain LTX with the same seed and prompt.

| group | metrics |
|---|---|
| path following (CoTracker3 point) | `ate`, `epe_final`, `pck_5/10`, `direction_cos`, `progress` |
| object level (SAM 2 mask) | `obj_ate`, `obj_direction_cos`, `obj_progress`, `obj_clip_sim`, `obj_area_ratio` |
| unintended background motion | `bg_drift` (change vs. frame 0), `bg_flow` (optical flow outside the object) |
| image preservation, consistency | `clip_image_sim`, `first_frame_psnr/ssim`, `color_drift`, `clip_temporal`, `warp_error`, `sharpness_ratio` |

Training log (`log.jsonl`): `val_obj_gain` / `val_bg_gain` = validation loss without minus with the trajectory, on the object's
tokens and on the background (above 0 = the adapter uses the trajectory), measured at three noise levels (`gain_by_sigma`).
`best.pt` is the epoch with the lowest validation loss on the object's tokens. Benchmark outputs: `results.json`, `summary.md`, `montage.png`, videos.
