"""
estimate_light.py
=================
Estimates the dominant lighting parameters from a kitchen image and saves
them as a JSON file. The renderer reads this JSON to produce physically
plausible shadows for each pest at render time.

For diffuse/fluorescent ceiling lighting (the common commercial kitchen case)
the key parameters are:
  - elevation_deg : light angle above the floor plane (0=horizontal, 90=overhead)
                    Higher = shorter, rounder shadow directly below pest
                    Lower  = longer shadow stretched away from light
  - azimuth_deg   : horizontal direction light comes FROM (0=right, 90=up-in-image)
  - intensity     : shadow darkness (0=invisible, 1=fully black)
  - softness      : Gaussian blur radius for shadow edge (px, at base sprite size)
  - ambient       : fraction of light that is non-directional (raises shadow floor)

Algorithm:
  1. Convert image to LAB — L channel captures luminance cleanly
  2. Detect bright (highlight) and dark (shadow) regions on the floor mask
  3. Estimate light direction as the vector from shadow centroids → highlight centroids
  4. Estimate elevation from the ratio of vertical to horizontal light spread
  5. Estimate intensity from shadow darkness relative to floor mean brightness
  6. Save all parameters as JSON

Usage:
    python estimate_light.py --image kitchen1.png --output kitchen1_light.json

    # With floor mask for better accuracy (only analyses floor region):
    python estimate_light.py --image kitchen1.png --mask kitchen1_mask.png \
                             --output kitchen1_light.json --debug

Optional flags:
    --image     Input kitchen image (required)
    --mask      Floor mask PNG (recommended — restricts analysis to floor)
    --output    Output JSON path (default: <image>_light.json)
    --debug     Save a debug overlay showing detected highlights/shadows
    --force_overhead  Skip estimation and output a pure overhead light
                      (safe fallback for very uniform images)
"""

import cv2
import numpy as np
import argparse
import json
import os


# ─────────────────────────────────────────────
#  LIGHT ESTIMATION
# ─────────────────────────────────────────────

def estimate_lighting(image_bgr, floor_mask=None, debug=False):
    """
    Estimate dominant lighting parameters from the image.
    Returns a dict with: elevation_deg, azimuth_deg, intensity, softness, ambient.
    """
    h, w = image_bgr.shape[:2]

    # Work in LAB — L channel is perceptually uniform luminance
    lab   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lum   = lab[:, :, 0]   # 0-255

    # Restrict analysis to floor region if mask provided
    if floor_mask is not None and floor_mask.sum() > 100:
        analysis_mask = floor_mask.astype(np.uint8)
    else:
        # Fall back to bottom 50% of image as proxy for floor
        analysis_mask = np.zeros((h, w), dtype=np.uint8)
        analysis_mask[h//2:, :] = 1

    floor_lum = lum[analysis_mask > 0]
    if len(floor_lum) == 0:
        return _default_params()

    floor_mean = float(floor_lum.mean())
    floor_std  = float(floor_lum.std())

    # ── Highlight and shadow detection ───────────────────────────────
    # Highlights: pixels significantly brighter than floor mean
    # Shadows:    pixels significantly darker than floor mean
    highlight_thresh = floor_mean + 0.8 * floor_std
    shadow_thresh    = floor_mean - 0.8 * floor_std

    highlight_mask = ((lum > highlight_thresh) & (analysis_mask > 0)).astype(np.uint8)
    shadow_mask    = ((lum < shadow_thresh)    & (analysis_mask > 0)).astype(np.uint8)

    highlight_px = int(highlight_mask.sum())
    shadow_px    = int(shadow_mask.sum())
    floor_px     = int(analysis_mask.sum())

    # ── Light direction from highlight/shadow centroids ───────────────
    azimuth_deg = 270.0   # default: light from above-right (common in kitchens)

    if highlight_px > 50 and shadow_px > 50:
        hy, hx = np.where(highlight_mask > 0)
        sy, sx = np.where(shadow_mask    > 0)

        h_cx, h_cy = float(hx.mean()), float(hy.mean())
        s_cx, s_cy = float(sx.mean()), float(sy.mean())

        # Light direction = from shadow centroid toward highlight centroid
        dx = h_cx - s_cx
        dy = h_cy - s_cy   # image Y is flipped vs world Y

        if abs(dx) > 2 or abs(dy) > 2:
            # atan2 in image space: 0=right, 90=down, 180=left, 270=up
            azimuth_deg = float(np.degrees(np.arctan2(-dy, dx)) % 360)

    # ── Elevation from contrast ratio ────────────────────────────────
    # Strong contrast → low angle (harsh directional)
    # Weak contrast   → high angle (overhead/diffuse)
    contrast_ratio = floor_std / max(floor_mean, 1.0)

    # Map contrast ratio to elevation:
    # contrast near 0   → elevation ~85° (nearly overhead)
    # contrast near 0.3 → elevation ~45°
    # contrast near 0.5 → elevation ~20°
    elevation_deg = float(np.clip(85.0 - contrast_ratio * 200.0, 15.0, 88.0))

    # ── Shadow intensity from darkness of shadow regions ─────────────
    if shadow_px > 50:
        shadow_lum    = lum[shadow_mask > 0].mean()
        # How dark are shadows relative to floor mean (0=same, 1=fully black)
        raw_intensity = float(np.clip((floor_mean - shadow_lum) / max(floor_mean, 1.0), 0, 1))
        intensity     = float(np.clip(raw_intensity * 1.5, 0.15, 0.65))
    else:
        intensity = 0.30   # default for very uniform images

    # ── Softness from highlight spread ───────────────────────────────
    # Wide, gradual highlights → diffuse light → soft shadows
    # Tight, sharp highlights  → point source  → hard shadows
    if highlight_px > 50:
        hy, hx = np.where(highlight_mask > 0)
        spread = float(np.sqrt(hx.std()**2 + hy.std()**2)) / max(w, h)
        # Map spread to softness in pixels (at base 50px sprite)
        softness = float(np.clip(spread * 200, 4, 28))
    else:
        softness = 14.0   # default diffuse

    # ── Ambient (non-directional light fraction) ──────────────────────
    # For fluorescent ceilings: high ambient (0.6-0.8)
    # For directional spotlights: low ambient (0.2-0.4)
    ambient = float(np.clip(0.4 + (elevation_deg - 15) / 73 * 0.45, 0.4, 0.85))

    params = {
        "elevation_deg": round(elevation_deg, 1),
        "azimuth_deg":   round(azimuth_deg,   1),
        "intensity":     round(intensity,      3),
        "softness":      round(softness,       1),
        "ambient":       round(ambient,        3),
        # Derived shadow offset multiplier used by renderer
        # At elevation 90°: offset=0 (directly below)
        # At elevation 20°: offset large (long stretched shadow)
        "offset_scale":  round(float(np.cos(np.radians(elevation_deg))), 3),
    }

    debug_data = None
    if debug:
        debug_data = {
            "floor_mean":      round(floor_mean, 1),
            "floor_std":       round(floor_std,  1),
            "contrast_ratio":  round(contrast_ratio, 3),
            "highlight_px":    highlight_px,
            "shadow_px":       shadow_px,
            "floor_px":        floor_px,
            "highlight_mask":  highlight_mask,
            "shadow_mask":     shadow_mask,
        }

    return params, debug_data


def _default_params():
    """Safe fallback — pure overhead fluorescent."""
    return {
        "elevation_deg": 85.0,
        "azimuth_deg":   270.0,
        "intensity":     0.28,
        "softness":      12.0,
        "ambient":       0.82,
        "offset_scale":  0.087,
    }, None


# ─────────────────────────────────────────────
#  DEBUG OVERLAY
# ─────────────────────────────────────────────

def save_debug_overlay(image_bgr, params, debug_data, out_path):
    overlay = image_bgr.copy()

    if debug_data:
        hm = debug_data["highlight_mask"]
        sm = debug_data["shadow_mask"]
        overlay[hm > 0] = (overlay[hm > 0] * 0.4 + np.array([0, 255, 255]) * 0.6).astype(np.uint8)
        overlay[sm > 0] = (overlay[sm > 0] * 0.4 + np.array([0, 0, 255])   * 0.6).astype(np.uint8)

    h, w = overlay.shape[:2]

    # Draw estimated light direction arrow
    cx, cy   = w // 2, h // 2
    az_rad   = np.radians(params["azimuth_deg"])
    arr_len  = min(w, h) // 6
    ax = int(cx + arr_len * np.cos(az_rad))
    ay = int(cy - arr_len * np.sin(az_rad))
    cv2.arrowedLine(overlay, (cx, cy), (ax, ay), (0, 220, 255), 3, tipLength=0.25)

    # Text summary
    lines = [
        f"elevation : {params['elevation_deg']}deg",
        f"azimuth   : {params['azimuth_deg']}deg",
        f"intensity : {params['intensity']}",
        f"softness  : {params['softness']}px",
        f"ambient   : {params['ambient']}",
        f"offset_sc : {params['offset_scale']}",
    ]
    y = 35
    for line in lines:
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (220, 220, 220), 1, cv2.LINE_AA)
        y += 26

    cv2.putText(overlay, "YELLOW=highlights  RED=shadows  ARROW=light dir",
                (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(overlay, "YELLOW=highlights  RED=shadows  ARROW=light dir",
                (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200,200,200), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, overlay)
    print(f"[DEBUG] Overlay saved: {out_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Estimate lighting parameters from a kitchen image.")
    parser.add_argument("--image",          required=True)
    parser.add_argument("--mask",           default=None,
                        help="Floor mask PNG — restricts analysis to floor region")
    parser.add_argument("--output",         default=None,
                        help="Output JSON path (default: <image>_light.json)")
    parser.add_argument("--debug",          action="store_true")
    parser.add_argument("--force_overhead", action="store_true",
                        help="Skip estimation, output pure overhead light params")
    args = parser.parse_args()

    out_path = args.output or args.image.replace(".png", "_light.json").replace(".jpg", "_light.json")

    if args.force_overhead:
        params, _ = _default_params()
        print("[INFO] Using forced overhead lighting params")
    else:
        image_bgr = cv2.imread(args.image)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load: {args.image}")

        floor_mask = None
        if args.mask:
            m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                floor_mask = m > 127
                print(f"[INFO] Using floor mask: {args.mask}")
            else:
                print(f"[WARN] Could not load mask: {args.mask} — using bottom half")

        print("[INFO] Estimating lighting…")
        result = estimate_lighting(image_bgr, floor_mask, debug=args.debug)
        params, debug_data = result if isinstance(result, tuple) else (result, None)

        if args.debug and debug_data:
            dbg_path = out_path.replace(".json", "_debug.png")
            save_debug_overlay(image_bgr, params, debug_data, dbg_path)

    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"[DONE] Light params saved: {out_path}")
    for k, v in params.items():
        print(f"       {k:<16} {v}")


if __name__ == "__main__":
    main()