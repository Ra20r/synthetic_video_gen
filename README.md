# Synthetic Pest Video Generator

Generates synthetic kitchen videos with animated pests (mice, cockroaches) for training computer vision models. Produces MP4 videos alongside COCO JSON annotation files containing bounding boxes, instance segmentation masks, and stable tracking IDs. A frame extraction step assembles everything into a train/val/test dataset.

---

## Scripts Overview

| Script | Purpose |
|---|---|
| `generate_depth_map.py` | Depth estimation using MiDaS DPT_Large |
| `generate_surface_masks.py` | Segments walkable surfaces (floor by default) via SegFormer |
| `estimate_light.py` | Estimates lighting direction and intensity from the image |
| `generate_configs.py` | Randomly generates video config JSON files |
| `add_pests_to_kitchen.py` | Renders one video + COCO JSON from a config |
| `batch_render.py` | Renders all configs in a directory |
| `extract_frames.py` | Extracts frames and assembles COCO train/val/test dataset |
| `run_pipeline.py` | All-in-one: depth → masks → light → render → dataset |
| `benchmark.py` | Measures render speed and projects scale compute costs |

---

## Installation

```bash
pip install torch torchvision transformers timm pillow opencv-python numpy scipy
```

> `torch`, `transformers`, and `timm` are only needed for `generate_depth_map.py` and `generate_surface_masks.py`. All other scripts only require `opencv-python` and `numpy`.

---

## Quick Start

### Option A — Single config (recommended)

```bash
python run_pipeline.py --config config.json --output_dir out/
```

`--image` is optional when using `--config` — the pipeline reads it from the config file automatically.

### Option B — Random video generation

```bash
python run_pipeline.py \
  --image kitchen1.png \
  --n 20 \
  --mice 0 3 \
  --cockroaches 0 5 \
  --duration 15 30 \
  --output_dir out/
```

### Option C — Multiple images without collisions

Each image gets its own isolated subdirectory under `--output_dir`:

```bash
python run_pipeline.py --config config_k1.json --output_dir out/
python run_pipeline.py --config config_k3.json --output_dir out/
# Produces out/kitchen1/ and out/kitchen3/ fully independently
```

### Option D — Step by step

```bash
# 1. Depth map
python generate_depth_map.py --image kitchen1.png --output out/kitchen1/kitchen1_depth.png

# 2. Surface masks (floor only by default)
python generate_surface_masks.py \
  --image kitchen1.png \
  --depth out/kitchen1/kitchen1_depth.png \
  --output_dir out/kitchen1/ \
  --debug

# 3. Light estimation
python estimate_light.py \
  --image kitchen1.png \
  --mask out/kitchen1/kitchen1_floor.png \
  --output out/kitchen1/kitchen1_light.json \
  --debug

# 4. Generate random configs
python generate_configs.py \
  --image kitchen1.png \
  --surfaces_json '[{"name":"floor","mask":"out/kitchen1/kitchen1_floor.png"}]' \
  --depth out/kitchen1/kitchen1_depth.png \
  --light out/kitchen1/kitchen1_light.json \
  --output_dir configs/ --n 20 --mice 0 3 --cockroaches 0 5

# 5. Render all configs
python batch_render.py --config_dir configs/ --output_dir videos/ --jobs 4

# 6. Extract frames and build dataset
python extract_frames.py \
  --video_dir videos/ --output_dir dataset/ \
  --split 0.8 0.1 0.1 --every_n 3
```

---

## Output Structure

```
out/
└── kitchen1/                           ← scoped per image, never collides
    ├── kitchen1_depth.png
    ├── kitchen1_floor.png              ← floor mask
    ├── kitchen1_surfaces_debug.png     ← with --debug_surfaces
    ├── kitchen1_light.json
    ├── configs/
    │   ├── config_0000.json
    │   └── ...
    ├── videos/
    │   ├── kitchen1_output.mp4
    │   ├── kitchen1_output_coco.json   ← per-video COCO annotations
    │   └── ...
    └── dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── annotations/
        │   ├── train.json              ← merged COCO JSON
        │   ├── val.json
        │   └── test.json
        └── dataset_info.json
```

---

## Config File Format

When running via `run_pipeline.py`, leave out `surfaces`, `depth`, and `light` — the pipeline computes and injects them automatically. Only include them when running `add_pests_to_kitchen.py` directly.

```json
{
  "image":    "kitchen1.png",
  "output":   "output.mp4",
  "duration": 30,
  "fps":      25,
  "grain":    0.02,
  "pests": [
    {"type": "mouse",     "count": 1, "size": 50, "speed": 6},
    {"type": "cockroach", "count": 2, "size": 30, "speed": 9}
  ]
}
```

When running standalone with `add_pests_to_kitchen.py` directly:

```json
{
  "image": "kitchen1.png",
  "surfaces": [
    {"name": "floor", "mask": "out/kitchen1/kitchen1_floor.png"}
  ],
  "depth":  "out/kitchen1/kitchen1_depth.png",
  "light":  "out/kitchen1/kitchen1_light.json",
  "output": "output.mp4",
  "duration": 30,
  "fps":    25,
  "grain":  0.02,
  "pests": [
    {"type": "mouse",     "count": 1, "size": 50, "speed": 6},
    {"type": "cockroach", "count": 2, "size": 30, "speed": 9}
  ]
}
```

### Config fields

| Field | Required | Description |
|---|---|---|
| `image` | ✅ | Kitchen background image |
| `surfaces` | Auto-injected | List of `{name, mask}` objects. Injected by pipeline. |
| `depth` | Auto-injected | Depth map for perspective scaling |
| `light` | Auto-injected | Light params JSON from `estimate_light.py` |
| `output` | Optional | Output video filename |
| `duration` | Optional | Video length in seconds (default: 10) |
| `fps` | Optional | Frames per second (default: 25) |
| `grain` | Optional | Film grain strength 0–1 (default: 0.02, 0 = off) |
| `pests` | ✅ | List of pest entries |

### Pest entry fields

| Field | Default | Description |
|---|---|---|
| `type` | required | `"mouse"` or `"cockroach"` |
| `count` | 1 | Number of this pest |
| `size` | 50 | Base sprite size in pixels at full depth |
| `speed` | 6 | Movement speed in pixels/frame |

---

## Realism Features

All enabled by default when assets are provided:

| Feature | What it does |
|---|---|
| **Perspective scaling** | Pest shrinks as it moves toward the horizon using the depth map |
| **Physics-based shadow** | Soft Gaussian shadow from `estimate_light.py` — direction, size, and opacity driven by estimated scene lighting |
| **Motion blur** | Directional blur on the sprite proportional to speed and direction |
| **Colour matching** | Sprite colour-shifted to match the surface's LAB colour cast |
| **Film grain** | Integer noise on the final composited frame simulating camera sensor |

---

## COCO Annotations

Each rendered video automatically produces a `_coco.json` sidecar. `extract_frames.py` merges these into standard train/val/test COCO JSON files.

### Annotation fields

| Field | Description |
|---|---|
| `id` | Unique annotation ID |
| `image_id` | Links to the frame image record |
| `category_id` | `1` = mouse, `2` = cockroach |
| `bbox` | `[x, y, w, h]` — tight bounding box from sprite alpha channel |
| `area` | Pixel area of instance mask |
| `segmentation` | Polygon contours or RLE — pixel-perfect from sprite alpha |
| `iscrowd` | Always `0` |
| `track_id` | Stable integer per pest across all frames of a video |
| `surface` | Surface name the pest is on (`"floor"`, etc.) |

### Classifier labels

`frame_meta` in each per-video JSON provides frame-level labels:

```json
{"frame_idx": 42, "has_pest": true, "pest_count": 3, "file_name": "video_frame_000042.jpg"}
```

### Categories

```json
[
  {"id": 1, "name": "mouse",     "supercategory": "pest"},
  {"id": 2, "name": "cockroach", "supercategory": "pest"}
]
```

---

## Script Reference

### `generate_depth_map.py`

```bash
python generate_depth_map.py --image kitchen1.png --output kitchen1_depth.png
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Input image |
| `--output` | `<image>_depth.png` | Output depth map |
| `--model` | `DPT_Large` | MiDaS model: `DPT_Large`, `DPT_Hybrid`, `MiDaS_small` |

---

### `generate_surface_masks.py`

Replaces the old `generate_floor_mask.py`. Generates all surface masks in one SegFormer pass. Defaults to floor only — add other surfaces only if segmentation looks clean in the debug overlay.

```bash
python generate_surface_masks.py \
  --image kitchen1.png \
  --depth kitchen1_depth.png \
  --output_dir out/kitchen1/ \
  --debug
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Input image |
| `--output_dir` | image directory | Where to save masks |
| `--depth` | None | Depth map to AND with each mask |
| `--depth_thresh` | 40 | Depth cutoff 0–255 |
| `--surfaces` | `floor` | Which masks to generate. Options: `floor counter table shelf` |
| `--smooth_px` | 5 | Edge smoothing radius (0 to disable) |
| `--debug` | off | Save colour-coded overlay (green=floor, orange=counter, blue=table) |

**If mask is empty:** remove `--depth` or lower `--depth_thresh 10`.  
**ADE20K floor labels** (if floor missed): default is `3`. Add `--floor_labels 3 6` to include roads.

---

### `estimate_light.py`

Run once per image. Analyses highlight/shadow distribution to estimate dominant light direction and intensity.

```bash
python estimate_light.py \
  --image kitchen1.png \
  --mask out/kitchen1/kitchen1_floor.png \
  --output out/kitchen1/kitchen1_light.json \
  --debug
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Input image |
| `--mask` | None | Floor mask — restricts analysis to floor region |
| `--output` | `<image>_light.json` | Output JSON path |
| `--force_overhead` | off | Skip estimation, use pure overhead defaults |
| `--debug` | off | Save overlay showing detected highlights/shadows and light direction arrow |

---

### `generate_configs.py`

```bash
python generate_configs.py \
  --image kitchen1.png \
  --surfaces_json '[{"name":"floor","mask":"out/kitchen1/kitchen1_floor.png"}]' \
  --depth out/kitchen1/kitchen1_depth.png \
  --light out/kitchen1/kitchen1_light.json \
  --output_dir configs/ \
  --n 50 --mice 0 3 --cockroaches 0 5 --duration 15 30 --grain 0.02 --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Kitchen image path |
| `--surfaces_json` | None | JSON string of surfaces list |
| `--depth` | None | Depth map path |
| `--light` | None | Light params JSON path |
| `--output_dir` | `configs/` | Output directory |
| `--n` | 10 | Number of configs |
| `--mice` | `0 3` | Min/max mice per video |
| `--cockroaches` | `0 5` | Min/max cockroaches per video |
| `--duration` | `15 30` | Duration range in seconds |
| `--fps` | 25 | Frames per second |
| `--grain` | 0.02 | Film grain strength |
| `--seed` | None | Random seed |

---

### `batch_render.py`

```bash
python batch_render.py --config_dir configs/ --output_dir videos/ --jobs 4
```

| Flag | Default | Description |
|---|---|---|
| `--config_dir` | required | Directory with JSON configs |
| `--output_dir` | alongside configs | Output video directory |
| `--jobs` | 1 | Parallel render processes |
| `--no_skip` | off | Re-render existing videos |
| `--fail_fast` | off | Stop on first error |

---

### `extract_frames.py`

```bash
python extract_frames.py \
  --video_dir videos/ \
  --output_dir dataset/ \
  --split 0.8 0.1 0.1 \
  --every_n 3
```

| Flag | Default | Description |
|---|---|---|
| `--video_dir` | — | Directory with `.mp4` + `_coco.json` pairs |
| `--video` | — | Single video (alternative to `--video_dir`) |
| `--output_dir` | `dataset/` | Root dataset output directory |
| `--split` | `0.8 0.1 0.1` | Train/val/test fractions (must sum to 1) |
| `--quality` | 95 | JPEG quality 1–100 |
| `--every_n` | 1 | Extract every Nth frame |
| `--no_empty` | off | Skip frames with no pest annotations |
| `--seed` | 42 | Random seed for split |

---

### `run_pipeline.py`

```bash
# Single config — no --image needed
python run_pipeline.py --config config.json --output_dir out/

# Random generation
python run_pipeline.py --image kitchen1.png --n 20 --output_dir out/
```

| Flag | Default | Description |
|---|---|---|
| `--image` | from config | Kitchen image |
| `--output_dir` | `pipeline_out/` | Root output directory |
| `--config` | None | Single config JSON |
| `--n` | 10 | Number of random videos |
| `--mice` | `0 3` | Mice count range |
| `--cockroaches` | `0 5` | Cockroach count range |
| `--duration` | `15 30` | Video duration range (seconds) |
| `--fps` | 25 | Frames per second |
| `--grain` | 0.02 | Film grain strength |
| `--jobs` | 1 | Parallel render jobs |
| `--surfaces` | `floor` | Surface masks to generate |
| `--depth_thresh` | 40 | Depth threshold for masks |
| `--split` | `0.8 0.1 0.1` | Dataset train/val/test split |
| `--every_n` | 1 | Extract every Nth frame |
| `--no_empty_frames` | off | Skip frames with no pests |
| `--skip_depth` | off | Skip depth generation |
| `--skip_surfaces` | off | Skip surface mask generation |
| `--skip_light` | off | Skip light estimation |
| `--skip_configs` | off | Skip config generation |
| `--skip_extract` | off | Skip frame extraction |
| `--debug_surfaces` | off | Save surface debug overlay |
| `--seed` | None | Random seed |

---

### `benchmark.py`

```bash
# Quick estimate (5s, 1 run per scenario)
python benchmark.py --quick

# Full benchmark with real files
python benchmark.py \
  --image kitchen1.png \
  --mask out/kitchen1/kitchen1_floor.png \
  --depth out/kitchen1/kitchen1_depth.png \
  --duration 30 --fps 25 --runs 3
```

| Flag | Default | Description |
|---|---|---|
| `--image` | synthetic 1920×1080 | Kitchen image |
| `--mask` | synthetic bottom 40% | Floor mask |
| `--depth` | synthetic gradient | Depth map |
| `--duration` | 10 | Seconds per benchmark run |
| `--fps` | 25 | Frames per second |
| `--runs` | 3 | Timing runs to average |
| `--quick` | off | 5-second single run |
| `--output` | None | Save a sample rendered video |

Benchmarks four scenarios (1 mouse, 3 cockroaches, 2 mice + 3 cockroaches, 5 cockroaches). Reports per-step timing breakdown (sprite, blur, shadow, grain, etc.) and projects wall-clock time for 100/1,000/10,000 videos with parallelism estimates.

---

## Tips

**Skipping already-done steps**
```bash
python run_pipeline.py --config config.json --output_dir out/ \
  --skip_depth --skip_surfaces --skip_light
```

**Resuming interrupted batch renders**
`batch_render.py` skips existing videos by default. Just re-run the same command.

**Speeding up large batches**
```bash
python batch_render.py --config_dir configs/ --output_dir videos/ --jobs $(nproc)
```

**Reducing dataset size without losing diversity**
Use `--every_n 3` or `--every_n 5` — removes temporal redundancy (adjacent frames are nearly identical) while keeping the full motion range.

**Disabling film grain** (fastest render)
```bash
# In config:
"grain": 0.0

# Or pipeline flag:
python run_pipeline.py --config config.json --grain 0.0 --output_dir out/
```

**Training with Ultralytics YOLO**
```bash
yolo detect train \
  data=out/kitchen1/dataset/annotations/train.json \
  model=yolov8n.pt
```

**Training with Detectron2**
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("pests_train", {},
    "out/kitchen1/dataset/annotations/train.json",
    "out/kitchen1/dataset/images/train")
register_coco_instances("pests_val", {},
    "out/kitchen1/dataset/annotations/val.json",
    "out/kitchen1/dataset/images/val")
```

**Tuning pest appearance**
Sprite proportions are multiples of `r` (body radius) inside `draw_mouse()` and `draw_cockroach()` in `add_pests_to_kitchen.py`:

| Parameter | Mouse | Cockroach |
|---|---|---|
| Body radius ratio | `c * 0.22` | `c * 0.20` |
| Body length | `r * 1.6` | `r * 1.8` |
| Head size | `r * 0.68` | `r * 0.45` |
| Antenna length | — | `r * 3.0` |

---

## How Annotations Are Generated

Annotations are computed during the render pass — no post-processing required, zero extra cost.

For each frame and each pest:
1. The sprite is drawn to an off-screen BGRA canvas
2. The **alpha channel** is the pixel-perfect instance mask
3. A tight **bounding box** is computed from mask extents
4. **Polygon contours** are traced from the alpha mask
5. The **track ID** is a stable integer assigned at pest creation, unchanged across all frames

Annotations are ground-truth accurate — no labelling error, no annotation noise.