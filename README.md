# Synthetic Pest Video Generator

Generates synthetic kitchen videos with animated pests (mice, cockroaches) for training computer vision models. Uses semantic segmentation to detect walkable floor areas, depth maps for perspective scaling, and procedural sprite animation.

---

## Scripts Overview

| Script | Purpose |
|---|---|
| `generate_depth_map.py` | Your own depth estimation script |
| `generate_floor_mask.py` | Segments floor from kitchen image |
| `generate_configs.py` | Randomly generates video config files |
| `add_pests_to_kitchen.py` | Renders a single video from one config |
| `batch_render.py` | Renders all configs in a directory |
| `run_pipeline.py` | All-in-one: mask → configs → videos |
| `benchmark.py` | Measures render speed and projects scale costs |

---

## Installation

```bash
pip install torch torchvision transformers pillow opencv-python numpy scipy
```

> **Note:** `torch` and `transformers` are only needed for `generate_floor_mask.py`. All other scripts only require `opencv-python` and `numpy`.

---

## Quick Start

### Option A — Full pipeline (recommended)

```bash
python run_pipeline.py \
  --image kitchen1.png \
  --n 20 \
  --mice 0 3 \
  --cockroaches 0 5 \
  --duration 15 30 \
  --output_dir out/
```

This runs the complete pipeline: depth map → floor mask → 20 random configs → 20 rendered videos. Everything is saved under `out/`.

### Option B — Single hand-crafted video

```bash
python run_pipeline.py \
  --image kitchen1.png \
  --config my_config.json \
  --output_dir out/
```

### Option C — Step by step

```bash
# 1. Generate floor mask (first run downloads ~110MB model)
python generate_floor_mask.py \
  --image kitchen1.png \
  --depth kitchen1_depth.png \
  --output kitchen1_mask.png \
  --debug

# 2. Generate random configs
python generate_configs.py \
  --image kitchen1.png \
  --mask kitchen1_mask.png \
  --depth kitchen1_depth.png \
  --output_dir configs/ \
  --n 20 \
  --mice 0 3 \
  --cockroaches 0 5

# 3. Render all configs
python batch_render.py \
  --config_dir configs/ \
  --output_dir videos/ \
  --jobs 4

# Or render one config directly
python add_pests_to_kitchen.py --config configs/config_0000.json
```

---

## Config File Format

```json
{
  "image":    "kitchen1.png",
  "mask":     "kitchen1_mask.png",
  "depth":    "kitchen1_depth.png",
  "output":   "output.mp4",
  "duration": 30,
  "fps":      25,
  "pests": [
    { "type": "mouse",     "count": 1, "size": 50, "speed": 6 },
    { "type": "cockroach", "count": 3, "size": 30, "speed": 9 }
  ]
}
```

| Field | Required | Description |
|---|---|---|
| `image` | ✅ | Kitchen background image |
| `mask` | Recommended | Floor mask PNG from `generate_floor_mask.py` |
| `depth` | Optional | Grayscale depth map for perspective scaling |
| `output` | Optional | Output video path (default: `output.mp4`) |
| `duration` | Optional | Video length in seconds (default: 10) |
| `fps` | Optional | Frames per second (default: 25) |
| `pests` | ✅ | List of pest entries |

### Pest entry fields

| Field | Description | Default |
|---|---|---|
| `type` | `"mouse"` or `"cockroach"` | required |
| `count` | Number of this pest in the video | 1 |
| `size` | Base sprite size in pixels | 50 |
| `speed` | Movement speed in pixels/frame | 6 |

---

## Script Reference

### `generate_floor_mask.py`

```bash
python generate_floor_mask.py --image kitchen1.png --output kitchen1_mask.png --debug
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Input kitchen image |
| `--output` | `floor_mask.png` | Output binary mask PNG |
| `--depth` | None | Depth map to AND with segmentation result |
| `--depth_thresh` | 40 | Depth cutoff 0–255 (pixels above = floor) |
| `--floor_labels` | `3` | ADE20K label indices for floor. Add `6` or `52` if floor is missed |
| `--smooth_px` | 5 | Boundary smoothing radius (0 to disable) |
| `--debug` | off | Save a colour-coded overlay PNG |

**ADE20K floor labels:** `3`=floor, `6`=road, `52`=rug

If the mask is empty or wrong, check the debug overlay and try:
```bash
# Floor not detected
--floor_labels 3 6

# Depth filter killing the mask
--depth_thresh 10
# or remove --depth entirely
```

---

### `generate_configs.py`

```bash
python generate_configs.py \
  --image kitchen1.png \
  --mask kitchen1_mask.png \
  --output_dir configs/ \
  --n 50 \
  --mice 0 3 \
  --cockroaches 0 5 \
  --duration 15 30 \
  --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Kitchen image path |
| `--mask` | None | Floor mask path |
| `--depth` | None | Depth map path |
| `--output_dir` | `configs/` | Directory to write JSON configs |
| `--n` | 10 | Number of configs to generate |
| `--mice` | `0 3` | Min and max mice per video |
| `--cockroaches` | `0 5` | Min and max cockroaches per video |
| `--duration` | `15 30` | Duration range in seconds |
| `--fps` | 25 | Frames per second |
| `--seed` | None | Random seed for reproducibility |

Each config is guaranteed to have at least one pest.

---

### `batch_render.py`

```bash
python batch_render.py \
  --config_dir configs/ \
  --output_dir videos/ \
  --jobs 4
```

| Flag | Default | Description |
|---|---|---|
| `--config_dir` | required | Directory with JSON config files |
| `--output_dir` | alongside configs | Where to save MP4 videos |
| `--jobs` | 1 | Parallel render processes |
| `--no_skip` | off | Re-render even if video already exists |
| `--fail_fast` | off | Stop on first render error |

---

### `run_pipeline.py`

```bash
python run_pipeline.py --image kitchen1.png --n 20 --output_dir out/
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Kitchen image |
| `--output_dir` | `pipeline_out/` | Root output directory |
| `--n` | 10 | Number of random videos to generate |
| `--config` | None | Single config JSON (skips random generation) |
| `--mice` | `0 3` | Mice count range |
| `--cockroaches` | `0 5` | Cockroach count range |
| `--duration` | `15 30` | Video duration range (seconds) |
| `--fps` | 25 | Frames per second |
| `--jobs` | 1 | Parallel render jobs |
| `--floor_labels` | `3` | ADE20K floor labels for mask generation |
| `--depth_thresh` | 40 | Depth threshold for mask generation |
| `--skip_depth` | off | Skip depth generation if file exists |
| `--skip_mask` | off | Skip mask generation if file exists |
| `--skip_configs` | off | Skip config generation if configs dir has files |
| `--debug_mask` | off | Save debug overlay for the floor mask |
| `--seed` | None | Random seed |

---

### `benchmark.py`

```bash
# Quick estimate
python benchmark.py --quick

# Full benchmark with your actual image and masks
python benchmark.py \
  --image kitchen1.png \
  --mask kitchen1_mask.png \
  --depth kitchen1_depth.png \
  --duration 30 \
  --fps 25 \
  --runs 3
```

| Flag | Default | Description |
|---|---|---|
| `--image` | synthetic | Kitchen image (uses 1920×1080 grey if not provided) |
| `--mask` | synthetic | Floor mask |
| `--depth` | synthetic | Depth map |
| `--duration` | 10 | Video duration per benchmark run (seconds) |
| `--fps` | 25 | Frames per second |
| `--runs` | 3 | Timing runs to average per scenario |
| `--quick` | off | 5-second single run for fast estimate |
| `--output` | None | Save a sample rendered video |

Benchmarks four scenarios: 1 mouse, 3 cockroaches, 2 mice + 3 cockroaches, 5 cockroaches. Reports ms/frame, achieved fps, and projects wall-clock time for 100 / 1,000 / 10,000 videos with parallelism estimates.

---

## Tips

**Speeding up large batches**
```bash
# Use all available cores
python batch_render.py --config_dir configs/ --output_dir videos/ --jobs $(nproc)
```

**Resuming interrupted batches**
```bash
# batch_render skips already-rendered videos by default
python batch_render.py --config_dir configs/ --output_dir videos/
```

**Tuning pest appearance**

In `add_pests_to_kitchen.py`, all sprite proportions are defined as multiples of `r` (body radius) inside `draw_mouse()` and `draw_cockroach()`. Key values:

| Parameter | Mouse | Cockroach |
|---|---|---|
| Body radius ratio | `c * 0.22` | `c * 0.20` |
| Body length | `r * 1.6` | `r * 1.8` |
| Head size | `r * 0.68` | `r * 0.45` |
| Antenna length | — | `r * 3.0` |

**Floor mask troubleshooting**

| Symptom | Fix |
|---|---|
| Mask is completely black | Remove `--depth` or lower `--depth_thresh 10` |
| Counters included in floor | Default labels are correct; this is a model limitation |
| Floor misses edges | Lower `--smooth_px 2` or set `--smooth_px 0` |
| Wrong region detected | Add `--floor_labels 3 6` or check debug overlay |

---

## Output Structure

```
out/
├── kitchen1_depth.png       # depth map
├── kitchen1_mask.png        # floor mask
├── kitchen1_mask_debug.png  # debug overlay (with --debug_mask)
├── configs/
│   ├── config_0000.json
│   ├── config_0001.json
│   └── ...
└── videos/
    ├── video_0000.mp4
    ├── video_0001.mp4
    └── ...
```