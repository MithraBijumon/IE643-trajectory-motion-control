# Trajectory motion control for LTX-Video

Image + drawn trajectory -> video. A small trainable adapter injects a rasterized point trajectory
(heatmap + dx + dy per frame) into a **frozen** LTX-Video 0.9.7 (13B). Only the condition encoder and
the adapter (~1.2M params) are trained.

```
trajectory (F,2) -> rasterize (3,F,H,W) -> ConditionEncoder -> (128,F',H',W') -> tokens
noisy video tokens + scale * MLP(tokens)  ->  frozen LTX transformer  ->  velocity
```

Training uses LTX's flow-matching objective with the first frame as clean image conditioning
(timestep 0, excluded from the loss), matching `LTXConditionPipeline`'s image-to-video mode.

## Run everything

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --pickle data/tapvid_davis.pkl   # preprocess -> index -> train -> generate -> evaluate
```

Stages can be run alone (`--stages train,generate`) or by hand:

| Stage | Command |
|---|---|
| preprocess | `python scripts/preprocess_tapvid.py --pickle_path data/tapvid_davis.pkl` |
| index | `python scripts/build_index.py --dataset_dir data/dataset` |
| train | `python scripts/train.py [--resume checkpoints/last.pt]` |
| generate | `python scripts/inference.py --checkpoint checkpoints/best.pt --sample_dir data/dataset/val/goat` |
| evaluate | `python scripts/evaluate.py --video_path V.mp4 --trajectory_path T.npy` (CoTracker3 ATE) |
| demo | `python app/app.py --checkpoint checkpoints/best.pt` |

`run_pipeline.py` generates every val clip twice, with the adapter and with it switched off (plain LTX
image-to-video), and writes both ATE numbers to `results/metrics.json`.

All settings live in `configs/config.yaml`.

## Adapter variants

Set `adapter.mode` in `configs/config.yaml` (or `--variant` on `train.py` / `run_pipeline.py`):

| mode | what it does | trainable |
|---|---|---|
| `input` | adds `scale * MLP(trajectory tokens)` once to the video tokens | ~1.2M |
| `multilayer` | ControlNet-style: zero-initialised projections add the trajectory to the hidden states after every `inject_every`-th LTX block (forward hooks, LTX untouched) | ~7M |

Checkpoints record their variant; every script reads it from the checkpoint, so you do not have to
pass it again for inference.

`train.cond_dropout` trains a fraction of samples without the trajectory. At inference,
`--motion_guidance 2` (or the "Motion strength" slider) then amplifies the trajectory's effect:
`v = v_without + g * (v_with - v_without)`. It costs one extra forward pass per step.

## Benchmark and metrics

```bash
python scripts/benchmark.py --checkpoint checkpoints/best.pt --num_images 8 --out results/benchmark_input
```

Builds straight paths (left / right / up / down / diagonal) from each val image's object, generates
videos with the adapter and a plain-LTX baseline, tracks them with CoTracker3 and reports, per method:

- trajectory adherence: ATE, final-point error, direction cosine, progress (fraction of the intended distance travelled)
- image preservation: CLIP similarity to the input image (plus LPIPS if `pip install lpips`)
- temporal consistency: CLIP similarity between neighbouring frames

Outputs: `results.json`, `summary.md` (table), `montage.png` (baseline vs adapter), overlay videos.
Run it once per checkpoint/variant and compare the tables.

## Hardware

The full model needs a large GPU (~26 GB of bf16 weights plus activations; use an A100/H100-class card).
It will not fit an 8 GB laptop GPU.

## Check the pipeline without the big model

```bash
python scripts/run_pipeline.py --tiny --skip_eval
```

`--tiny` swaps in a miniature randomly initialised LTX (64x96, 17 frames) that runs on CPU in about a
minute. It verifies that the code runs end to end; the output videos are noise and the metrics mean nothing.
