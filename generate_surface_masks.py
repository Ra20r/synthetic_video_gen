"""
generate_surface_masks.py
=========================
Generates binary masks for all flat walkable surfaces in a kitchen image
using a single SegFormer segmentation pass.

Replaces generate_floor_mask.py — produces one mask per surface type,
all from one model inference.

Output masks (all white=walkable, black=non-walkable):
  <stem>_floor.png     — floor / flooring
  <stem>_counter.png   — countertops, kitchen islands
  <stem>_table.png     — tables
  <stem>_shelf.png     — shelves

Install:
    pip install torch torchvision transformers pillow opencv-python numpy scipy

Usage:
    python generate_surface_masks.py --image kitchen1.png --output_dir out/kitchen1/

    # With depth map refinement:
    python generate_surface_masks.py --image kitchen1.png \
        --depth kitchen1_depth.png --output_dir out/kitchen1/ --debug

Optional flags:
    --output_dir     Directory to save masks (default: same dir as image)
    --model          HuggingFace model (default: nvidia/segformer-b2-finetuned-ade-512-512)
    --depth          Depth map to AND with each mask
    --depth_thresh   Depth cutoff 0-255 (default: 40)
    --smooth_px      Boundary smoothing (default: 5, 0 to disable)
    --surfaces       Which surfaces to generate (default: floor counter table shelf)
    --debug          Save colour-coded debug overlay showing all surfaces

ADE20K label mapping:
    floor   : 3 (floor/flooring)
    counter : 44 (counter), 69 (countertop), 72 (kitchen island)
    table   : 15 (table), 63 (coffee table)
    shelf   : 24 (shelf)
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path


def check_imports():
    missing = []
    for pkg, pip in [("torch","torch"),("transformers","transformers"),
                     ("PIL","pillow"),("scipy","scipy")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pip)
    if missing:
        print(f"[ERROR] Missing: pip install {' '.join(missing)}")
        sys.exit(1)


# Surface definitions: name → ADE20K label IDs
SURFACE_LABELS = {
    "floor":   [3],
    "counter": [44, 69, 72],
    "table":   [15, 63],
    "shelf":   [24],
}

# Colours for debug overlay (BGR)
SURFACE_COLOURS = {
    "floor":   (0,   200,   0),    # green
    "counter": (0,   180, 255),    # orange
    "table":   (255,  80,  80),    # blue
    "shelf":   (180,   0, 255),    # purple
}

ADE20K_NAMES = {
    0:"wall", 3:"floor", 5:"ceiling", 10:"cabinet", 15:"table",
    24:"shelf", 44:"counter", 46:"sink", 49:"refrigerator",
    63:"coffee table", 69:"countertop", 70:"stove", 72:"kitchen island",
}


# ─────────────────────────────────────────────
#  SEGMENTATION (single pass, reused for all surfaces)
# ─────────────────────────────────────────────

def run_segformer(image_bgr, model_name):
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    import torch
    from PIL import Image

    print(f"[INFO] Loading model: {model_name}")
    print("       (First run downloads ~110MB — subsequent runs instant)")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model     = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()

    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    print("[INFO] Running segmentation (CPU — 10-30s)…")
    inputs = processor(images=pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    h, w = image_bgr.shape[:2]
    up = torch.nn.functional.interpolate(
        outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
    seg = up.argmax(dim=1).squeeze().numpy().astype(np.int32)

    # Print top classes found
    unique, counts = np.unique(seg, return_counts=True)
    print("[INFO] Top classes detected:")
    for cnt, lbl in sorted(zip(counts, unique), reverse=True)[:8]:
        name = ADE20K_NAMES.get(int(lbl), f"class_{lbl}")
        print(f"       [{lbl:3d}] {name:<22} {100*cnt/seg.size:.1f}%")

    return seg


# ─────────────────────────────────────────────
#  MASK PROCESSING
#  Safe pipeline — never destroys detected pixels
# ─────────────────────────────────────────────

def labels_to_mask(seg_map, labels):
    mask = np.zeros(seg_map.shape, dtype=np.uint8)
    for lbl in labels:
        mask[seg_map == lbl] = 255
    return mask


def process_mask(raw_mask, image_bgr, smooth_px=5):
    """Fill holes, remove specks, smooth edges. Never shrinks below raw."""
    from scipy.ndimage import binary_fill_holes
    h, w = image_bgr.shape[:2]

    filled = binary_fill_holes(raw_mask > 0).astype(np.uint8) * 255

    # Remove specks smaller than 0.3% of image
    min_area = max(100, int(0.003 * h * w))
    n_lbls, cc, stats, _ = cv2.connectedComponentsWithStats(filled)
    clean = np.zeros_like(filled)
    for lbl in range(1, n_lbls):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            clean[cc == lbl] = 255

    if clean.sum() == 0:
        print("[WARN] Speck removal emptied mask — using raw")
        clean = raw_mask.copy()

    if smooth_px > 0:
        ksize    = smooth_px * 2 + 1
        blurred  = cv2.GaussianBlur(clean.astype(np.float32), (ksize, ksize), 0)
        smoothed = (blurred > (0.45 * 255)).astype(np.uint8) * 255
        if (smoothed > 0).sum() < (clean > 0).sum() * 0.80:
            smoothed = clean
        clean = smoothed

    return clean


def refine_with_depth(mask, depth, thresh_norm):
    """AND mask with depth threshold. Skips if result is empty."""
    refined = np.where((depth > thresh_norm), mask, 0).astype(np.uint8)
    if refined.sum() == 0:
        print("[WARN] Depth refinement emptied mask — skipping")
        return mask
    return refined


# ─────────────────────────────────────────────
#  DEBUG OVERLAY
# ─────────────────────────────────────────────

def save_debug_overlay(image_bgr, masks, seg_map, out_path):
    overlay = image_bgr.copy()
    for name, mask in masks.items():
        if mask is None or mask.sum() == 0:
            continue
        colour = SURFACE_COLOURS.get(name, (200, 200, 200))
        overlay[mask > 0] = (
            overlay[mask > 0] * 0.4 + np.array(colour) * 0.6
        ).astype(np.uint8)

    # Legend
    h = overlay.shape[0]
    y = 35
    for name, colour in SURFACE_COLOURS.items():
        if name in masks and masks[name] is not None:
            pct = 100 * (masks[name] > 0).sum() / seg_map.size
            text = f"{name:<10} {pct:.1f}%"
            cv2.putText(overlay, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(overlay, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, colour, 1, cv2.LINE_AA)
            y += 28

    cv2.imwrite(out_path, overlay)
    print(f"[DEBUG] Overlay: {out_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate surface masks from a kitchen image.")
    parser.add_argument("--image",        required=True)
    parser.add_argument("--output_dir",   default=None,
                        help="Directory to save masks (default: same dir as image)")
    parser.add_argument("--model",        default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--depth",        default=None)
    parser.add_argument("--depth_thresh", type=float, default=40)
    parser.add_argument("--smooth_px",    type=int,   default=5)
    parser.add_argument("--surfaces",     nargs="+",
                        default=["floor"],
                        choices=list(SURFACE_LABELS.keys()),
                        help="Which surfaces to generate masks for (default: floor only). "
                             "Add others carefully: --surfaces floor counter")
    parser.add_argument("--debug",        action="store_true")
    args = parser.parse_args()

    check_imports()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load: {args.image}")
    h, w = image_bgr.shape[:2]

    stem     = Path(args.image).stem
    out_dir  = args.output_dir or str(Path(args.image).parent)
    os.makedirs(out_dir, exist_ok=True)

    # Load depth map once if provided
    depth_norm = None
    if args.depth:
        d = cv2.imread(args.depth, cv2.IMREAD_GRAYSCALE)
        if d is not None:
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            depth_norm = d.astype(np.float32) / 255.0
            print(f"[INFO] Depth map loaded: {args.depth}")
        else:
            print(f"[WARN] Could not load depth: {args.depth}")

    # Single segmentation pass
    seg_map = run_segformer(image_bgr, args.model)

    # Generate and save each requested surface mask
    saved_masks = {}
    output_paths = {}

    for surface in args.surfaces:
        labels  = SURFACE_LABELS[surface]
        raw     = labels_to_mask(seg_map, labels)
        raw_pct = 100 * (raw > 0).sum() / (h * w)
        print(f"\n[INFO] Surface: {surface}  raw coverage: {raw_pct:.1f}%")

        if raw_pct < 0.1:
            print(f"       [SKIP] Too little detected — mask not saved")
            saved_masks[surface] = None
            continue

        mask = process_mask(raw, image_bgr, smooth_px=args.smooth_px)

        if depth_norm is not None:
            mask = refine_with_depth(mask, depth_norm, args.depth_thresh / 255.0)

        out_path = os.path.join(out_dir, f"{stem}_{surface}.png")
        cv2.imwrite(out_path, mask)
        pct = 100 * (mask > 0).sum() / (h * w)
        print(f"       Saved: {out_path}  ({pct:.1f}%)")
        saved_masks[surface]  = mask
        output_paths[surface] = out_path

    # Summary
    print(f"\n[DONE] Masks saved to: {out_dir}/")
    for s, p in output_paths.items():
        print(f"       {s:<10} → {os.path.basename(p)}")

    # Debug overlay
    if args.debug:
        dbg = os.path.join(out_dir, f"{stem}_surfaces_debug.png")
        save_debug_overlay(image_bgr, saved_masks, seg_map, dbg)

    # Print suggested config snippet
    print(f"\n[INFO] Config surfaces snippet:")
    print('  "surfaces": [')
    for s, p in output_paths.items():
        print(f'    {{"name": "{s}", "mask": "{p}"}},')
    print('  ]')


if __name__ == "__main__":
    main()