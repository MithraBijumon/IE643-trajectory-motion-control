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

## Hardware

The full model needs a large GPU (~26 GB of bf16 weights plus activations; use an A100/H100-class card).
It will not fit an 8 GB laptop GPU.

## Check the pipeline without the big model

```bash
python scripts/run_pipeline.py --tiny --skip_eval
```

`--tiny` swaps in a miniature randomly initialised LTX (64x96, 17 frames) that runs on CPU in about a
minute. It verifies that the code runs end to end; the output videos are noise and the metrics mean nothing.
