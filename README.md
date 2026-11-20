# Trajectory motion control for LTX-Video

Image + drawn trajectory -> video. A small trainable adapter injects the trajectory of one object into a
**frozen** LTX-Video 0.9.7 (13B); LTX's own weights and modules are never modified.

```
trajectory + object mask -> footprint / displacement / object feature (latent grid)
  -> conv encoder -> cond_blocks small transformer blocks -> 24 zero-init projections
  -> added (forward hooks) to the hidden states after every 2nd LTX block
```

Training is LTX's flow-matching objective with the first frame as clean image conditioning. Trainable:
condition encoder, `adapter.cond_blocks` small transformer blocks, the zero-init projections.
Settings: `configs/config.yaml`. Dependencies: `pip install -r requirements.txt`. Needs a large GPU
(~26 GB bf16 weights; A100-class), not an 8 GB laptop card.

## Data (GOT-10k, one labelled moving target per clip)

```bash
python scripts/stream_got10k.py --out_dir data/dataset_got10k --max_clips 300      # streams the 69 GB archive, keeps filtered train clips
python scripts/camera_motion.py --data_root data/dataset_got10k                    # score camera motion (clips with a moving camera can be dropped)
python scripts/make_masks.py --data_root data/dataset_got10k --use_box --overwrite  # SAM, box-prompted frame-0 mask (mask.png, box.npy)
python scripts/make_video_masks.py --data_root data/dataset_got10k                  # SAM 2 tracks it through the clip -> masks_lowres.npy
```

## Train, generate, evaluate

| Step | Command |
|---|---|
| train | `python scripts/train.py [--resume checkpoints/last.pt]` (Colab: `notebooks/train_colab.ipynb`) |
| generate | `python scripts/inference.py --checkpoint checkpoints/best.pt --sample_dir data/dataset_got10k/val/<clip>` |
| benchmark | `python scripts/benchmark.py --checkpoint checkpoints/best.pt --num_images 14` |
| tables / figures | `scripts/report_tables.py`, `scripts/plot_training.py` |
| demo | `python app/app.py --checkpoint checkpoints/best.pt` |
| tests | `python -m pytest tests` |

`adapter.reference: true` also feeds a rendered "drag video" (`src/reference.py`: the object's SAM cut-out moved along the path over the inpainted scene) through the VAE into the condition: a pixel-aligned hint, like a ControlNet's depth image.
`--ref_init 0.8` (benchmark / inference) additionally starts denoising from that video noised to sigma 0.8, a training-free boost.
`adapter.local_inject: 2.0` adds the trajectory residual only near the object (a footprint 2x its radius); the background gets none. `0` = everywhere.
Models: `--config configs/ltx2b.yaml` (2B, bf16, A100) or `configs/t4.yaml` (2B, fp16, small frames) instead of the default 13B.

`train.cond_dropout` trains some samples without the trajectory; at inference `--motion_guidance 2` then
amplifies the trajectory: `v = v_without + g * (v_with - v_without)` (one extra forward pass per step).

## SAM

- **Training:** SAM 2 per-frame masks (`masks_lowres.npy`) give (1) the conditioning frame's mask -> object size, appearance feature and
  silhouette (`adapter.mask_shape`, moved rigidly along the path), and (2) the loss weighting: `train.focus_loss`, `split_loss`
  (object and background errors averaged separately), `time_ramp` (later frames count more). `data.mask_loss: false` turns (2) off.
  Validation logs `val_obj_gain` / `val_bg_gain` (the trajectory's benefit on object vs. background tokens).
- **Inference:** only the first-frame mask is needed (`mask.png`, `--mask`, or selected in the app); without one a disk is used.
- **Evaluation:** SAM 2 tracks the frame-0 mask through each generated video (`obj_*` metrics below).
- **App:** select the object with a box (two corners) and refine with + / - clicks; the result is tracked with SAM 2 and drawn over the target path.

## Benchmark metrics (`scripts/benchmark.py`)

Straight left / right / up / down / diagonal paths from each val image's object; adapter vs. plain-LTX baseline.

- trajectory adherence (CoTracker3): ATE, final-point error, PCK@5/10%, direction cosine, progress
- object level (SAM 2): `obj_ate`, `obj_direction_cos`, `obj_progress`, `obj_clip_sim` (object crop vs. input object), `obj_area_ratio`
- image preservation: CLIP similarity, first-frame PSNR/SSIM, colour drift, LPIPS to the input
- temporal consistency: CLIP neighbour similarity, flow warping error, temporal LPIPS
- video quality: sharpness ratio, LAION aesthetic score, FVD-style distance (VideoMAE features, not comparable with published FVD)

Outputs in `--out`: `results.json`, `summary.md`, `montage.png`, videos.
