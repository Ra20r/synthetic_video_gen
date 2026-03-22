# Synthetic Pest Video Generator

Generates synthetic kitchen videos with animated pests (mice, cockroaches) for training computer vision models. Outputs MP4 videos alongside COCO JSON annotation files with bounding boxes, instance segmentation masks, and tracking IDs. A frame extraction step assembles everything into a train/val/test dataset.

---

## Scripts Overview

| Script | Purpose |
|---|---|
| `generate_depth_map.py` | Your own depth estimation script (bring your own) |
| `generate_floor_mask.py` | Segments floor from kitchen image via SegFormer |
| `generate_configs.py` | Randomly generates video config JSON files |
| `add_pests_to_kitchen.py` | Renders one video + `_coco.json` from a config |
| `batch_render.py` | Renders all configs in a directory |
| `extract_frames.py` | Extracts frames + assembles COCO train/val/test dataset |
| `run_pipeline.py` | All-in-one: mask → configs → videos → dataset |
| `benchmark.py` | Measures render speed and projects scale costs |

---

## Installation

```bash
pip install torch torchvision transformers pillow opencv-python numpy scipy
```

> `torch` and `transformers` are only needed for `generate_floor_mask.py`. Everything else only requires `opencv-python` and `numpy`.

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

Runs the complete pipeline: depth map → floor mask → 20 random configs → 20 videos with COCO annotations → frame extraction → train/val/test dataset. Everything lands in `out/`.

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

# 3. Render all configs (each produces a .mp4 + _coco.json sidecar)
python batch_render.py \
  --config_dir configs/ \
  --output_dir videos/ \
  --jobs 4

# 4. Extract frames and build dataset
python extract_frames.py \
  --video_dir videos/ \
  --output_dir dataset/ \
  --split 0.8 0.1 0.1 \
  --every_n 3

# Or render one config directly
python add_pests_to_kitchen.py --config configs/config_0000.json
```

---

## Output Structure

```
out/
├── kitchen1_depth.png          # depth map
├── kitchen1_mask.png           # floor mask
├── kitchen1_mask_debug.png     # debug overlay (with --debug_mask)
├── configs/
│   ├── config_0000.json
│   └── ...
├── videos/
│   ├── video_0000.mp4
│   ├── video_0000_coco.json    # per-video annotations (auto-generated)
│   └── ...
└── dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── annotations/
    │   ├── train.json          # COCO JSON
    │   ├── val.json
    │   └── test.json
    └── dataset_info.json       # summary stats
```

---

## COCO Annotations

Each rendered video automatically produces a `_coco.json` sidecar. `extract_frames.py` then merges these into standard train/val/test COCO JSON files.

### Annotation fields

Every annotation contains:

| Field | Description |
|---|---|
| `id` | Unique annotation ID |
| `image_id` | Links to the image record |
| `category_id` | `1` = mouse, `2` = cockroach |
| `bbox` | `[x, y, w, h]` tight bounding box (pixel-accurate from sprite alpha) |
| `area` | Pixel area of the instance mask |
| `segmentation` | Polygon contours or RLE — instance segmentation mask |
| `iscrowd` | Always `0` |
| `track_id` | Stable integer per pest across all frames of a video |

### Classifier labels

The `frame_meta` array in the per-video JSON provides frame-level classifier information:

```json
{
  "frame_idx": 42,
  "has_pest": true,
  "pest_count": 3,
  "file_name": "video_0000_frame_000042.jpg"
}
```

### Categories

```json
[
  {"id": 1, "name": "mouse",     "supercategory": "pest"},
  {"id": 2, "name": "cockroach", "supercategory": "pest"}
]
```

### Example annotation

```json
{
  "id": 17,
  "image_id": 5,
  "category_id": 2,
  "bbox": [312.0, 418.0, 28.0, 22.0],
  "area": 412.0,
  "segmentation": [[318, 418, 320, 419, ...]],
  "iscrowd": 0,
  "track_id": 2
}
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
| `count` | Number of this pest | 1 |
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

**If the mask is empty:** try `--depth_thresh 10` or remove `--depth`.  
**If counters are included:** this is a model limitation — the floor boundary is correct for training.  
**ADE20K floor labels:** `3`=floor, `6`=road, `52`=rug

---

### `generate_configs.py`

```bash
python generate_configs.py \
  --image kitchen1.png \
  --mask kitchen1_mask.png \
  --output_dir configs/ \
  --n 50 --mice 0 3 --cockroaches 0 5 --duration 15 30 --seed 42
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

---

### `batch_render.py`

```bash
python batch_render.py --config_dir configs/ --output_dir videos/ --jobs 4
```

| Flag | Default | Description |
|---|---|---|
| `--config_dir` | required | Directory with JSON config files |
| `--output_dir` | alongside configs | Where to save MP4 videos |
| `--jobs` | 1 | Parallel render processes |
| `--no_skip` | off | Re-render even if video already exists |
| `--fail_fast` | off | Stop on first render error |

---

### `extract_frames.py`

```bash
python extract_frames.py \
  --video_dir videos/ \
  --output_dir dataset/ \
  --split 0.8 0.1 0.1 \
  --every_n 3 \
  --no_empty
```

| Flag | Default | Description |
|---|---|---|
| `--video_dir` | — | Directory with `.mp4` + `_coco.json` pairs |
| `--video` | — | Single video (alternative to `--video_dir`) |
| `--output_dir` | `dataset/` | Root dataset output directory |
| `--split` | `0.8 0.1 0.1` | Train/val/test fractions (must sum to 1) |
| `--quality` | 95 | JPEG quality 1–100 |
| `--every_n` | 1 | Extract every Nth frame (reduces redundancy) |
| `--no_empty` | off | Skip frames with no pest annotations |
| `--seed` | 42 | Random seed for split |

---

### `run_pipeline.py`

```bash
python run_pipeline.py --image kitchen1.png --n 20 --output_dir out/
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Kitchen image |
| `--output_dir` | `pipeline_out/` | Root output directory |
| `--n` | 10 | Number of random videos |
| `--config` | None | Single config JSON (skips random generation) |
| `--mice` | `0 3` | Mice count range |
| `--cockroaches` | `0 5` | Cockroach count range |
| `--duration` | `15 30` | Video duration range (seconds) |
| `--fps` | 25 | Frames per second |
| `--jobs` | 1 | Parallel render jobs |
| `--floor_labels` | `3` | ADE20K floor labels for mask generation |
| `--depth_thresh` | 40 | Depth threshold for mask generation |
| `--split` | `0.8 0.1 0.1` | Dataset train/val/test split |
| `--every_n` | 1 | Extract every Nth frame for dataset |
| `--no_empty_frames` | off | Skip frames with no pests in dataset |
| `--skip_depth` | off | Skip depth generation if file exists |
| `--skip_mask` | off | Skip mask generation if file exists |
| `--skip_configs` | off | Skip config generation if configs dir has files |
| `--skip_extract` | off | Skip frame extraction and dataset assembly |
| `--debug_mask` | off | Save debug overlay for the floor mask |
| `--seed` | None | Random seed for configs and dataset split |

---

### `benchmark.py`

```bash
# Quick estimate
python benchmark.py --quick

# Full benchmark with your actual files
python benchmark.py \
  --image kitchen1.png \
  --mask kitchen1_mask.png \
  --depth kitchen1_depth.png \
  --duration 30 --fps 25 --runs 3
```

| Flag | Default | Description |
|---|---|---|
| `--image` | synthetic 1920×1080 | Kitchen image |
| `--mask` | synthetic | Floor mask |
| `--depth` | synthetic gradient | Depth map |
| `--duration` | 10 | Video duration per benchmark run (seconds) |
| `--fps` | 25 | Frames per second |
| `--runs` | 3 | Timing runs to average per scenario |
| `--quick` | off | 5-second single run for fast estimate |
| `--output` | None | Save a sample rendered video |

Benchmarks four scenarios (1 mouse, 3 cockroaches, 2 mice + 3 cockroaches, 5 cockroaches). Reports ms/frame, achieved fps, and projects wall-clock time for 100 / 1,000 / 10,000 videos with parallelism estimates based on your core count.

---

## Tips

**Speeding up large batches**
```bash
python batch_render.py --config_dir configs/ --output_dir videos/ --jobs $(nproc)
```

**Resuming interrupted batches**  
`batch_render.py` skips already-rendered videos by default. Just re-run the same command.

**Reducing dataset size without losing diversity**  
Use `--every_n 3` or `--every_n 5` in `extract_frames.py` or `run_pipeline.py`. This removes temporal redundancy (adjacent frames look nearly identical) while preserving coverage of the full motion range.

**Training a detector with Ultralytics YOLO**
```bash
# Convert COCO to YOLO format (Ultralytics does this automatically)
yolo detect train data=dataset/annotations/train.json model=yolov8n.pt
```

**Training with Detectron2**
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("pests_train", {}, "dataset/annotations/train.json", "dataset/images/train")
register_coco_instances("pests_val",   {}, "dataset/annotations/val.json",   "dataset/images/val")
```

**Tuning pest appearance**  
In `add_pests_to_kitchen.py`, sprite proportions are defined as multiples of `r` (body radius) inside `draw_mouse()` and `draw_cockroach()`:

| Parameter | Mouse | Cockroach |
|---|---|---|
| Body radius ratio | `c * 0.22` | `c * 0.20` |
| Body length multiplier | `r * 1.6` | `r * 1.8` |
| Head size | `r * 0.68` | `r * 0.45` |
| Antenna length | — | `r * 3.0` |

**Floor mask troubleshooting**

| Symptom | Fix |
|---|---|
| Mask is completely black | Remove `--depth` or set `--depth_thresh 10` |
| Mask misses floor near walls | Lower `--smooth_px 2` or set `--smooth_px 0` |
| Wrong region detected | Check debug overlay; add `--floor_labels 3 6` |
| Too little floor coverage | Model limitation — acceptable for training use |

---

## How Annotations Are Generated

Annotations are computed during the render pass with zero overhead — no post-processing step required.

For each frame and each pest:
1. The sprite is drawn to an off-screen BGRA canvas
2. The **alpha channel** of the sprite is the pixel-perfect instance mask
3. A tight **bounding box** is extracted from the mask extents
4. **Polygon contours** are traced from the alpha mask for segmentation
5. The **track ID** is a stable integer assigned at pest creation time, unchanged across all frames

This means annotations are ground-truth accurate — there is no labelling error, occlusion ambiguity, or annotation noise.