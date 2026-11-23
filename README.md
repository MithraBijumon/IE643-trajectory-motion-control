# Trajectory motion control for LTX-Video

Image + drawn trajectory -> video. A small trainable adapter injects the trajectory of one object into a
**frozen** LTX-Video (2B 0.9.5 or 13B 0.9.7); LTX's own weights and modules are never modified.

```
trajectory + object mask -> footprint / displacement / object feature (latent grid)   [+ drag video -> VAE latents]
  -> conv encoder -> cond_blocks small transformer blocks -> zero-init projections
  -> added (forward hooks) to the hidden states after every 2nd LTX block, only near the object
```

Training is LTX's flow-matching objective with the first frame as clean image conditioning. Trainable: condition
encoder, `adapter.cond_blocks`, the projections (about 15M parameters). Settings: `configs/config.yaml` (13B),
`configs/ltx2b.yaml` (2B, A100), `configs/t4.yaml` (2B, fp16, small frames). `pip install -r requirements.txt`.
**Run it: `notebooks/train_colab.ipynb`** (set `CFG = '--config configs/ltx2b.yaml'`).

## Design choices

- **Relative scaling.** A residual of fixed size is either drowned by LTX's activations (layers differ a lot in scale, and the
  2B and 13B models differ again) or blows them up. The adapter's output `r` is therefore multiplied by the RMS of the
  hidden state it is added to, and capped at `max_inject` (0.05) of that RMS. The learned weights then mean "this fraction
  of the layer's own scale", which is comparable across layers and models, and a high-lr step cannot wreck the activations.
- **Per-token cap (`inject_cap: token`).** `max_inject` bounds every token's own residual relative to the hidden RMS. The older
  `global` cap bounded the RMS over all tokens; with a footprint covering ~4% of the tokens that allows ~5x the cap on each token inside it, at
  every injection point, which is enough to leave LTX's activation range (the output then shows mosaic noise).
- **Object-only injection (`local_inject`).** The residual is multiplied by a Gaussian footprint (2 x the object radius)
  along the path, so background tokens get none. Attention can still carry a change to the background;
  `--bg_lock` removes even that at inference (see below).
- **Entity condition.** Footprint, displacement and the object's own latent feature painted on the latent grid; with
  `mask_shape` the SAM silhouette is moved rigidly along the path instead of a blob.
- **Drag-video condition (`adapter.reference`).** `src/reference.py` cuts the object out with its mask, inpaints the hole and moves
  the cut-out along the path. The video goes through the VAE and is added to the encoded condition: a pixel-aligned hint,
  like a ControlNet's depth image.
- **Loss.** `focus_loss`: object tokens (true SAM 2 per-frame masks) and background tokens are averaged separately,
  `(focus * L_obj + L_bg) / (focus + 1)`; `time_ramp` weighs later frames more. `--focus_loss 0` = plain mean (the ablation).
- **Per-clip prompts (`data.captions`).** Each clip has its own text prompt (`caption.txt`, written by
  `scripts/caption_clips.py` with Qwen2-VL: appearance and setting only; the motion sentence is a fixed template, because the
  trajectory decides the motion). LTX-2B breaks up with a generic prompt and holds together with a scene-specific one.
- **Condition dropout** trains some samples without the trajectory; `--motion_guidance g` then amplifies it at inference.

## Data (GOT-10k, one labelled moving target per clip)

```bash
python scripts/stream_got10k.py --out_dir data/dataset_got10k --max_clips 900       # streams the 69 GB archive, keeps filtered clips
python scripts/camera_motion.py --data_root data/dataset_got10k --max_score 0.25 --move_rejected   # drop moving-camera clips
python scripts/make_masks.py --data_root data/dataset_got10k --use_box --overwrite  # SAM: frame-0 mask
python scripts/make_video_masks.py --data_root data/dataset_got10k                  # SAM 2: per-frame masks
python scripts/caption_clips.py --models qwen --write --data_root data/dataset_got10k   # per-clip prompts
```

## Train, generate, evaluate

| Step | Command |
|---|---|
| train | `python scripts/train.py --config configs/ltx2b.yaml [--resume checkpoints/last.pt]` |
| quick visual check | `python scripts/demo_grid.py --config configs/ltx2b.yaml --num 3 --variants base,adapter,ref0.8,ref0.8+lock2.5` |
| generate | `python scripts/inference.py --checkpoint best.pt --sample_dir data/dataset_got10k/val/<clip> [--prompt "..."]` |
| benchmark | `python scripts/benchmark.py --config configs/ltx2b.yaml --checkpoint best.pt --num_images 6 --seeds 0,1,2 [--ood_dir imgs]` |
| traced example (for the slides) | `python scripts/trace_example.py --config configs/ltx2b.yaml --checkpoint best.pt --direction right --ref_init 0.8` |
| demo app | `python app/app.py --checkpoint best.pt` (config prompt only; per-image prompts not wired into the app) |
| tests | `python -m pytest tests` |

Inference switches: `--ref_init 0.8` starts denoising from the drag video noised to sigma 0.8 (training-free, works with an
untrained adapter); `--bg_comp 2.5` replaces the tokens outside 2.5 object radii of the path by the drag video's object-free scene at every step
(until sigma 0.25), which removes the ghost of the original object that `--ref_init` alone leaves behind; `--bg_lock 2.5` instead forces the
velocity there to be plain LTX's (one extra forward pass per step): it keeps the background but also keeps that ghost; `--motion_guidance g` amplifies the trajectory.
`--adapter_scale 0.5` scales the adapter's residual (1 = as trained, 0 = off but the drag video / composite still run: the training-free
path with the same checkpoint loaded) and `--adapter_min_sigma 0.4` lets the adapter act only at high noise, where the motion is decided, so plain
LTX finishes the fine detail. Use them to check a checkpoint without retraining (`demo_grid` variants `as0`, `as0.3`, `until0.4`).
Ablation switches for `train.py`: `--focus_loss 0`, `--no_reference`, `--no_captions`, `--local_inject 0`.

## Metrics (`scripts/benchmark.py`) and what each one captures

Straight left / right / up / down paths from each val image's object (and from clean out-of-domain images with `--ood_dir`:
`name.jpg`, optional `name.txt` prompt and `name_mask.png`; SAM from a click at the centre otherwise), adapter vs. plain LTX
with the same seed and prompt. `--seeds 0,1,2` repeats everything and reports the spread over seeds.

| group | metrics | what it tells you |
|---|---|---|
| followed the path? (CoTracker3 point) | `ate`, `epe_final`, `pck_5/10`, `direction_cos`, `progress` | error of the tracked point against the drawn path; right direction (1 = yes, 0 = no motion); fraction of the path travelled |
| object level (SAM 2 mask) | `obj_ate`, `obj_direction_cos`, `obj_progress`, `obj_clip_sim`, `obj_area_ratio` | same on the object's mask centroid; whether the object still looks like itself and keeps its size |
| **unintended background motion** | `bg_drift`, `bg_flow` | change of the scene outside the object: mean difference to frame 0, and mean optical flow (fraction of frame width per frame). Compare with the baseline on the same image |
| kept the input image | `clip_image_sim`, `first_frame_psnr/ssim`, `color_drift`, `lpips_to_input` | the video still shows the input |
| temporal consistency | `clip_temporal`, `warp_error`, `lpips_temporal` | flicker / popping |
| video quality | `sharpness_ratio`, `aesthetic`, FVD-style distance (VideoMAE features) | blur, looks, distance to real clips (not comparable with published FVD) |

With `--sde_prob` the log also has `val_sde_obj_gain` / `val_sde_obj_loss`: the same with / without comparison on samples that start from the
noised drag video (the situation of `--ref_init` generation), and `best.pt` is selected on `val_sde_obj_loss`. The plain `val_obj_gain` is measured
on noised real clips, where LTX already sees the object in the noisy input, so it can stay near 0 or below even when the adapter helps at generation time.

Training log: `cond_gain` = validation loss without minus with the trajectory (above 0 = the adapter uses it), measured at three
noise levels (`gain_by_sigma`), on object tokens (`val_obj_gain`) and on background tokens (`val_bg_gain`).

**Focus-loss experiment** (does the focus loss reduce unintended background movement while keeping path following?): train
twice with the same settings, once as is and once with `--focus_loss 0`; run the benchmark for both with the same images,
`--motions left,right,up,down` (opposite directions show forward / backward asymmetry), `--seeds 0,1,2` and `--ood_dir`; compare
`obj_ate` and `bg_flow` / `bg_drift` (they are reported separately, per motion, per group and as std over seeds). The notebook has these cells.

Outputs in `--out`: `results.json`, `summary.md`, `montage.png`, videos.
