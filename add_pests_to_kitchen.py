"""
add_pests_to_kitchen.py
========================
Renders animated pests on a kitchen image and exports:
  - MP4 video
  - COCO JSON annotations (bbox, segmentation, track_id)

Realism features:
  - Multi-surface spawning (floor, counter, table, shelf)
  - Perspective scaling from depth map
  - Motion blur proportional to speed and direction
  - Sprite colour matching to surface colour cast
  - Film grain / sensor noise
  - Physics-based soft shadow from estimate_light.py

Requirements:
    pip install opencv-python numpy

Usage:
    python add_pests_to_kitchen.py --config config.json

Config format:
    {
      "image":    "kitchen1.png",
      "surfaces": [
        {"name": "floor",   "mask": "kitchen1_floor.png"},
        {"name": "counter", "mask": "kitchen1_counter.png"}
      ],
      "depth":    "kitchen1_depth.png",
      "light":    "kitchen1_light.json",
      "output":   "output.mp4",
      "duration": 30,
      "fps":      25,
      "grain":    0.02,
      "pests": [
        {"type": "mouse",     "count": 1, "size": 50, "speed": 6},
        {"type": "cockroach", "count": 2, "size": 30, "speed": 9}
      ]
    }

    Backward compatible: "mask" and "depth" at top level still work
    (treated as a single "floor" surface).

    grain: float 0-1, controls film grain strength (default 0.0 = off)
"""

import cv2
import numpy as np
import argparse
import math
import random
import json
import sys
import os


# ─────────────────────────────────────────────
#  CATEGORIES
# ─────────────────────────────────────────────

CATEGORIES = [
    {"id": 1, "name": "mouse",     "supercategory": "pest"},
    {"id": 2, "name": "cockroach", "supercategory": "pest"},
]
CAT_ID = {"mouse": 1, "cockroach": 2}


# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

def load_depth_map(path, target_hw):
    depth = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Could not load depth map: {path}")
    depth = cv2.resize(depth, (target_hw[1], target_hw[0]),
                       interpolation=cv2.INTER_LINEAR)
    return depth.astype(np.float32) / 255.0


def load_mask(path, target_hw):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {path}")
    mask = cv2.resize(mask, (target_hw[1], target_hw[0]),
                      interpolation=cv2.INTER_NEAREST)
    return mask > 127


def depth_at(depth_map, x, y):
    h, w = depth_map.shape
    return float(depth_map[int(np.clip(y, 0, h-1)),
                            int(np.clip(x, 0, w-1))])


def load_light_params(path):
    with open(path) as f:
        return json.load(f)


def default_light_params():
    return {
        "elevation_deg": 85.0,
        "azimuth_deg":   270.0,
        "intensity":     0.28,
        "softness":      12.0,
        "ambient":       0.82,
        "offset_scale":  0.087,
    }


# ─────────────────────────────────────────────
#  SURFACE COLOUR SAMPLING
#  Compute the mean LAB colour of a surface mask
#  region. Used to tint pest sprites so they
#  match the colour cast of the surface they're on.
# ─────────────────────────────────────────────

def sample_surface_colour(image_bgr, mask):
    """
    Returns mean LAB colour of the surface region as a (3,) float32 array,
    or None if mask is empty.
    """
    if mask is None or mask.sum() == 0:
        return None
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[mask]
    return pixels.mean(axis=0)   # [L, A, B]


def apply_colour_tint(sprite_bgra, surface_lab, strength=0.25):
    """
    Tint the pest sprite toward the surface's colour cast.

    Only shifts the A and B channels (colour, not lightness) so the pest
    retains its own luminance/contrast. Strength 0=no tint, 1=full match.
    """
    if surface_lab is None or strength <= 0:
        return sprite_bgra

    alpha = sprite_bgra[:, :, 3]
    visible = alpha > 32   # only tint opaque pixels

    if not visible.any():
        return sprite_bgra

    result = sprite_bgra.copy()
    bgr = result[:, :, :3].astype(np.float32)
    lab = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    # Neutral grey in LAB has A=128, B=128
    neutral_a, neutral_b = 128.0, 128.0
    shift_a = (surface_lab[1] - neutral_a) * strength
    shift_b = (surface_lab[2] - neutral_b) * strength

    lab[visible, 1] = np.clip(lab[visible, 1] + shift_a, 0, 255)
    lab[visible, 2] = np.clip(lab[visible, 2] + shift_b, 0, 255)

    tinted_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    result[:, :, :3] = tinted_bgr
    return result


# ─────────────────────────────────────────────
#  MOTION BLUR
#  Apply directional blur to sprite in the
#  direction of movement. Magnitude scales with
#  speed so fast pests are blurrier.
# ─────────────────────────────────────────────

def apply_motion_blur(sprite_bgra, dx, dy, speed, fps, base_size):
    """
    Directional motion blur on the sprite.

    dx, dy    : movement vector this frame (pixels)
    speed     : current movement speed (pixels/frame)
    fps       : video fps — faster fps = less blur per frame
    base_size : base sprite size — blur scales proportionally
    """
    # Blur length in pixels: how far the pest moved relative to its size
    # At 25fps a pest moving 8px/frame on a 50px sprite gets ~3px blur
    blur_len = int(np.clip(speed * base_size / 50.0 * (25.0 / fps), 1, 12))

    if blur_len < 2:
        return sprite_bgra   # too slow to blur

    # Build directional kernel
    mag = math.sqrt(dx*dx + dy*dy)
    if mag < 0.01:
        return sprite_bgra

    ndx, ndy = dx / mag, dy / mag
    ksize = blur_len * 2 + 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    cx, cy = ksize // 2, ksize // 2

    # Draw line through kernel centre in direction of movement
    for t in range(-blur_len, blur_len + 1):
        kx = int(round(cx + t * ndx))
        ky = int(round(cy + t * ndy))
        if 0 <= kx < ksize and 0 <= ky < ksize:
            kernel[ky, kx] = 1.0

    if kernel.sum() == 0:
        kernel[cy, cx] = 1.0
    kernel /= kernel.sum()

    # Apply to RGB channels only (preserve alpha shape)
    result = sprite_bgra.copy()
    blurred_rgb = cv2.filter2D(result[:, :, :3], -1, kernel)

    # Also blur alpha slightly so edges feather in motion direction
    blurred_a = cv2.filter2D(result[:, :, 3], -1, kernel)

    result[:, :, :3] = blurred_rgb
    result[:, :, 3]  = blurred_a
    return result


# ─────────────────────────────────────────────
#  FILM GRAIN
#  Add sensor noise to the final composited frame.
#  Applied once per frame, after all pests drawn.
# ─────────────────────────────────────────────

def add_film_grain(frame, strength=0.02):
    """
    Add noise to simulate camera sensor grain.
    strength: 0=off, 0.01=subtle, 0.04=visible, 0.08=heavy

    Uses randint on uint8 directly — ~10x faster than np.random.normal
    because it avoids float32 allocation and stays in integer arithmetic.
    """
    if strength <= 0:
        return frame
    amplitude = max(1, int(strength * 128))
    # Random signed integer noise in [-amplitude, +amplitude]
    noise = np.random.randint(-amplitude, amplitude + 1,
                              frame.shape, dtype=np.int16)
    result = frame.astype(np.int16) + noise
    np.clip(result, 0, 255, out=result)
    return result.astype(np.uint8)


# ─────────────────────────────────────────────
#  SHADOW
# ─────────────────────────────────────────────

def add_shadow(frame, cx, cy, sprite_size, depth_value, light):
    h_img, w_img = frame.shape[:2]

    elevation  = light.get("elevation_deg",  85.0)
    azimuth    = light.get("azimuth_deg",   270.0)
    intensity  = light.get("intensity",      0.28)
    softness   = light.get("softness",       12.0)
    ambient    = light.get("ambient",        0.82)
    offset_sc  = light.get("offset_scale",   0.087)

    depth_scale = float(np.clip(0.4 + 0.6 * depth_value, 0.2, 1.0))
    base_r      = max(3, int(sprite_size * 0.22 * depth_scale))
    w_r         = max(2, int(base_r * 1.3))
    h_r         = max(1, int(base_r * 0.45))

    offset_px = int(sprite_size * offset_sc * depth_scale * 0.8)
    az_rad    = np.radians(azimuth)
    off_x     = int(-offset_px * np.cos(az_rad))
    off_y     = int( offset_px * np.sin(az_rad))

    scx = int(np.clip(cx + off_x,                   w_r+1, w_img-w_r-1))
    scy = int(np.clip(cy + int(sprite_size*0.12) + off_y, h_r+1, h_img-h_r-1))

    blur_r  = max(3, int(softness * depth_scale * sprite_size / 50.0))
    blur_r  = blur_r if blur_r % 2 == 1 else blur_r + 1
    opacity = float(np.clip(intensity * depth_scale * (1.0 - ambient*0.5),
                             0.05, 0.55))

    # Work on a small crop around the shadow — much faster than full-frame blur
    pad  = blur_r + w_r + 4
    rx0  = max(0, scx - pad);  ry0 = max(0, scy - pad)
    rx1  = min(w_img, scx + pad); ry1 = min(h_img, scy + pad)
    if rx1 <= rx0 or ry1 <= ry0:
        return frame

    crop_shadow = np.zeros((ry1-ry0, rx1-rx0), dtype=np.float32)
    # Draw ellipse in crop coords
    cv2.ellipse(crop_shadow, (scx-rx0, scy-ry0), (w_r, h_r),
                0, 0, 360, 1.0, -1, cv2.LINE_AA)
    crop_shadow = cv2.GaussianBlur(crop_shadow, (blur_r, blur_r), 0)

    # Apply darkening only to the crop region
    crop       = frame[ry0:ry1, rx0:rx1].astype(np.float32)
    mask_3ch   = crop_shadow[:, :, np.newaxis]
    crop      *= (1.0 - mask_3ch * opacity)
    np.clip(crop, 0, 255, out=crop)
    frame[ry0:ry1, rx0:rx1] = crop.astype(np.uint8)
    return frame


# ─────────────────────────────────────────────
#  SPRITE: MOUSE
# ─────────────────────────────────────────────

def draw_mouse(base_size, frame_idx, angle_deg, scale=1.0):
    c = max(16, int(base_size * scale))
    img = np.zeros((c, c, 4), dtype=np.uint8)
    cx, cy = c // 2, c // 2
    r = int(c * 0.22)
    if r < 2:
        return img

    FUR   = (80,  80, 100, 255)
    BELLY = (130, 130, 160, 255)
    EAR   = (100,  60, 120, 255)
    PINK  = (140,  90, 200, 255)
    EYE   = (10,   10,  10, 255)
    NOSE  = (100,  80, 200, 255)
    TAIL  = (90,   70, 110, 255)

    def fc(fx, fy, rad, col):
        if rad < 1: return
        cv2.circle(img, (int(fx), int(fy)), int(rad), col[:3], -1, cv2.LINE_AA)
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(m, (int(fx), int(fy)), int(rad), 255, -1, cv2.LINE_AA)
        img[:, :, 3] = np.where(m > 0, col[3], img[:, :, 3])

    def la(p1, p2, col, t):
        cv2.line(img, p1, p2, col[:3], max(1, t), cv2.LINE_AA)

    ar = math.radians(angle_deg); ca, sa = math.cos(ar), math.sin(ar)
    def rot(dx, dy): return int(cx+dx*ca-dy*sa), int(cy+dx*sa+dy*ca)

    sway = math.sin(frame_idx*0.25)*r*0.8
    tp = [rot(-r*1.1-(i/12)*r*2.0, sway*(i/12)**2) for i in range(13)]
    for i in range(12): la(tp[i], tp[i+1], TAIL, max(1, int((1-i/12)*r*0.18)))

    for t in np.linspace(-0.45, 0.45, 10):
        bx, by = rot(t*r*1.6, 0); fc(bx, by, int(r*(1.0-0.3*abs(t/0.45))), FUR)
    for t in np.linspace(-0.2, 0.2, 5):
        bx, by = rot(t*r*0.9, 0); fc(bx, by, int(r*0.45*(1-0.4*abs(t/0.2))), BELLY)

    for s in (-1, 1):
        ex, ey = rot(r*0.55, s*r*0.85)
        fc(ex, ey, int(r*0.38), EAR); fc(ex, ey, int(r*0.22), PINK)

    fc(*rot(r*0.95, 0), int(r*0.68), FUR)

    for s in (-1, 1):
        ex, ey = rot(r*1.05, s*r*0.32)
        fc(ex, ey, int(r*0.13), EYE)
        fc(ex-1, ey-1, max(1, int(r*0.05)), (255,255,255,255))

    fc(*rot(r*1.55, 0), int(r*0.12), NOSE)

    wig = math.sin(frame_idx*0.5)*3
    for lx, ly, w_ in [(0.3,0.9,wig),(0.3,-0.9,-wig),(-0.3,0.9,-wig),(-0.3,-0.9,wig)]:
        la(rot(lx*r,ly*r), rot(lx*r,(ly+w_*0.1)*r*1.45), FUR, max(1,int(r*0.18)))

    return img


# ─────────────────────────────────────────────
#  SPRITE: COCKROACH
# ─────────────────────────────────────────────

def draw_cockroach(base_size, frame_idx, angle_deg, scale=1.0):
    c = max(16, int(base_size * scale))
    img = np.zeros((c, c, 4), dtype=np.uint8)
    cx, cy = c // 2, c // 2
    r = int(c * 0.20)
    if r < 2:
        return img

    SHELL  = (20,  45,  60, 255)
    LEGS   = (15,  35,  50, 255)
    HEAD   = (10,  30,  45, 255)
    STRIPE = (30,  60,  80, 255)
    EYE    = (200, 220, 255, 255)
    ANT    = (10,  30,  45, 255)

    def fc(fx, fy, rad, col):
        if rad < 1: return
        cv2.circle(img, (int(fx), int(fy)), int(rad), col[:3], -1, cv2.LINE_AA)
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(m, (int(fx), int(fy)), int(rad), 255, -1, cv2.LINE_AA)
        img[:, :, 3] = np.where(m > 0, col[3], img[:, :, 3])

    def la(p1, p2, col, t):
        cv2.line(img, p1, p2, col[:3], max(1, t), cv2.LINE_AA)

    ar = math.radians(angle_deg); ca, sa = math.cos(ar), math.sin(ar)
    def rot(dx, dy): return int(cx+dx*ca-dy*sa), int(cy+dx*sa+dy*ca)

    lp = frame_idx*0.6
    for side in (-1, 1):
        for lx, ly, po in [(0.1,1.0,0.2),(-0.2,0.9,0.0),(-0.5,0.8,-0.2)]:
            sw = math.sin(lp+po)*0.25*side
            p1 = rot(lx*r, side*ly*r); p2 = rot((lx-0.5)*r, side*(ly+0.6+sw)*r)
            la(p1, p2, LEGS, max(1, int(r*0.12)))
            la(p2, rot((lx-0.8)*r, side*(ly+0.9+sw)*r), LEGS, max(1,int(r*0.09)))

    for t in np.linspace(-0.5, 0.5, 12):
        bx, by = rot(t*r*1.8, 0); fc(bx, by, int(r*(1.0-0.35*abs(t/0.5))), SHELL)
    for t in np.linspace(-0.35, 0.35, 5):
        bx, by = rot(t*r*1.4, 0); fc(bx, by, int(r*0.18*(1-0.5*abs(t/0.35))), STRIPE)

    fc(*rot(r*1.1, 0), int(r*0.45), HEAD)
    for s in (-1, 1): fc(*rot(r*1.25, s*r*0.28), max(1,int(r*0.10)), EYE)

    asw = math.sin(frame_idx*0.3)*r*0.3
    for s in (-1, 1):
        la(rot(r*1.4, s*r*0.2), rot(r*2.2, s*(r*0.4+asw*0.5)), ANT, max(1,int(r*0.08)))
        la(rot(r*2.2, s*(r*0.4+asw*0.5)), rot(r*3.0, s*(r*0.5+asw)), ANT, max(1,int(r*0.06)))

    return img


SPRITE_FNS = {"mouse": draw_mouse, "cockroach": draw_cockroach}


# ─────────────────────────────────────────────
#  ANNOTATION HELPERS
# ─────────────────────────────────────────────

def sprite_to_bbox_and_mask(sprite_bgra, cx, cy, img_w, img_h):
    sh, sw = sprite_bgra.shape[:2]
    alpha  = sprite_bgra[:, :, 3]
    x0, y0 = cx - sw//2, cy - sh//2
    ix0 = max(0, x0);  iy0 = max(0, y0)
    ix1 = min(img_w, x0+sw); iy1 = min(img_h, y0+sh)
    if ix1 <= ix0 or iy1 <= iy0:
        return None, None, 0.0
    sx0, sy0 = ix0-x0, iy0-y0
    sx1, sy1 = sx0+(ix1-ix0), sy0+(iy1-iy0)
    img_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    img_mask[iy0:iy1, ix0:ix1] = (alpha[sy0:sy1, sx0:sx1] > 64).astype(np.uint8)
    area = float(img_mask.sum())
    if area == 0:
        return None, None, 0.0
    rows = np.any(img_mask, axis=1); cols = np.any(img_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0,-1]]
    cmin, cmax = np.where(cols)[0][[0,-1]]
    return [float(cmin), float(rmin), float(cmax-cmin+1), float(rmax-rmin+1)], \
           img_mask, area


def mask_to_rle(binary_mask):
    h, w = binary_mask.shape
    flat = binary_mask.flatten(order="F").tolist()
    counts, current, count = [], 0, 0
    for px in flat:
        if px == current: count += 1
        else: counts.append(count); count = 1; current = px
    counts.append(count)
    if flat[0] != 0: counts = [0] + counts
    return {"counts": counts, "size": [h, w]}


def mask_to_polygon(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        c = c.squeeze()
        if c.ndim != 2 or len(c) < 3: continue
        flat = c.flatten().tolist()
        if len(flat) >= 6: polys.append(flat)
    return polys if polys else None


# ─────────────────────────────────────────────
#  PATH GENERATION
# ─────────────────────────────────────────────

def generate_path(width, height, n_frames, speed,
                  floor_mask=None, margin=30, seed=None):
    if seed is not None:
        random.seed(seed)

    if floor_mask is not None:
        valid_ys, valid_xs = np.where(floor_mask)
        if len(valid_xs) == 0:
            raise ValueError("Mask has no walkable pixels.")
        valid_set = set(zip(valid_xs.tolist(), valid_ys.tolist()))

        def is_valid(x, y):
            return (int(np.clip(round(x),0,width-1)),
                    int(np.clip(round(y),0,height-1))) in valid_set

        def nearest_valid(x, y):
            dists = (valid_xs-int(round(x)))**2 + (valid_ys-int(round(y)))**2
            idx = np.argmin(dists)
            return float(valid_xs[idx]), float(valid_ys[idx])

        idx = random.randint(0, len(valid_xs)-1)
        x, y = float(valid_xs[idx]), float(valid_ys[idx])
    else:
        def is_valid(x, y):
            return margin<=x<=width-margin and margin<=y<=height-margin
        def nearest_valid(x, y):
            return float(np.clip(x,margin,width-margin)), \
                   float(np.clip(y,margin,height-margin))
        x = float(random.randint(margin, width-margin))
        y = float(random.randint(margin, height-margin))

    current_angle = random.uniform(0, 2*math.pi)
    target_angle  = current_angle
    current_speed = speed
    target_speed  = speed
    pause_timer = steer_timer = stuck_counter = 0
    positions = []

    for _ in range(n_frames):
        positions.append((x, y))
        if pause_timer > 0:
            pause_timer -= 1; continue

        if steer_timer <= 0:
            target_angle = random.uniform(0, 2*math.pi)
            target_speed = speed * random.uniform(0.5, 1.5)
            steer_timer  = random.randint(20, 60)
        steer_timer -= 1

        diff = (target_angle - current_angle + math.pi) % (2*math.pi) - math.pi
        current_angle += max(-math.radians(8), min(math.radians(8), diff))
        current_speed += (target_speed - current_speed) * 0.08

        if random.random() < 0.008:
            pause_timer = random.randint(12, 35)
            target_speed = speed * 1.4; continue

        vx = math.cos(current_angle) * current_speed
        vy = math.sin(current_angle) * current_speed
        nx, ny = x+vx, y+vy

        if nx < margin or nx > width-margin:
            vx *= -1; nx = float(np.clip(nx,margin,width-margin))
            current_angle = math.atan2(vy,vx); target_angle = current_angle
        if ny < margin or ny > height-margin:
            vy *= -1; ny = float(np.clip(ny,margin,height-margin))
            current_angle = math.atan2(vy,vx); target_angle = current_angle

        if is_valid(nx, ny):
            x, y = nx, ny; stuck_counter = 0
        else:
            escaped = False
            for attempt in range(32):
                spread = 0.3 + attempt*0.18
                ta = current_angle + math.pi + random.uniform(-spread, spread)
                tnx = float(np.clip(x+math.cos(ta)*current_speed, margin, width-margin))
                tny = float(np.clip(y+math.sin(ta)*current_speed, margin, height-margin))
                if is_valid(tnx, tny):
                    x, y = tnx, tny
                    current_angle = ta
                    target_angle  = ta + random.uniform(-0.4, 0.4)
                    escaped = True; stuck_counter = 0; break
            if not escaped:
                stuck_counter += 1
                if stuck_counter >= 10:
                    x, y = nearest_valid(x, y)
                    current_angle = random.uniform(0, 2*math.pi)
                    target_angle  = current_angle; stuck_counter = 0

    return positions


# ─────────────────────────────────────────────
#  OVERLAY
# ─────────────────────────────────────────────

def overlay_sprite(background, sprite_bgra, cx, cy):
    h, w   = sprite_bgra.shape[:2]
    bh, bw = background.shape[:2]
    x0, y0 = cx-w//2, cy-h//2; x1, y1 = x0+w, y0+h
    sx0=max(0,-x0); sy0=max(0,-y0)
    sx1=w-max(0,x1-bw); sy1=h-max(0,y1-bh)
    bx0=max(0,x0); by0=max(0,y0)
    bx1=bx0+(sx1-sx0); by1=by0+(sy1-sy0)
    if bx1<=bx0 or by1<=by0: return background
    roi    = background[by0:by1,bx0:bx1].astype(np.float32)
    sp     = sprite_bgra[sy0:sy1,sx0:sx1]
    alpha  = sp[:,:,3:4].astype(np.float32)/255.0
    color  = sp[:,:,:3].astype(np.float32)
    background[by0:by1,bx0:bx1] = np.clip(
        roi*(1-alpha)+color*alpha, 0, 255).astype(np.uint8)
    return background


# ─────────────────────────────────────────────
#  PEST AGENT
# ─────────────────────────────────────────────

class PestAgent:
    def __init__(self, pest_cfg, surface_name, width, height, n_frames,
                 surface_mask, depth_map, surface_colour, light, fps, agent_idx):
        ptype = pest_cfg.get("type", "mouse").lower()
        if ptype not in SPRITE_FNS:
            raise ValueError(f"Unknown pest type: {ptype}")

        self.ptype          = ptype
        self.cat_id         = CAT_ID[ptype]
        self.track_id       = agent_idx + 1
        self.surface_name   = surface_name
        self.draw_fn        = SPRITE_FNS[ptype]
        self.base_size      = int(pest_cfg.get("size", 50))
        self.speed          = float(pest_cfg.get("speed", 6))
        self.depth_map      = depth_map
        self.surface_colour = surface_colour   # mean LAB of surface, or None
        self.light          = light
        self.fps            = fps

        seed = agent_idx * 1337 + random.randint(0, 9999)
        self.path       = generate_path(width, height, n_frames, self.speed,
                                        floor_mask=surface_mask, seed=seed)
        self.prev_angle = 0.0

    def get_frame_data(self, frame_idx, img_h, img_w):
        """
        Returns everything needed to render this pest for one frame:
        (sprite_bgra, cx, cy, persp_scale, dx, dy, depth_value)
        """
        px, py = self.path[frame_idx]

        # Smoothed facing angle (3-frame window)
        if frame_idx >= 3:
            dx = self.path[frame_idx][0] - self.path[frame_idx-3][0]
            dy = self.path[frame_idx][1] - self.path[frame_idx-3][1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self.prev_angle = math.degrees(math.atan2(dy, dx))
        angle = self.prev_angle

        # Per-frame velocity for motion blur
        if frame_idx > 0:
            fdx = self.path[frame_idx][0] - self.path[frame_idx-1][0]
            fdy = self.path[frame_idx][1] - self.path[frame_idx-1][1]
        else:
            fdx, fdy = 0.0, 0.0

        # Depth-driven perspective scale
        if self.depth_map is not None:
            d = depth_at(self.depth_map, int(px), int(py))
        else:
            d = py / img_h
        persp_scale = float(np.clip(0.35 + 0.65 * d, 0.15, 1.0))

        # Draw base sprite
        sprite = self.draw_fn(self.base_size, frame_idx, angle, scale=persp_scale)

        # Colour tint to match surface
        sprite = apply_colour_tint(sprite, self.surface_colour, strength=0.25)

        # Motion blur
        current_speed = math.sqrt(fdx**2 + fdy**2)
        sprite = apply_motion_blur(sprite, fdx, fdy, current_speed,
                                   self.fps, self.base_size * persp_scale)

        return sprite, int(px), int(py), persp_scale, fdx, fdy, d

    def render_onto(self, frame, frame_idx, img_h, img_w):
        sprite, cx, cy, scale, _, _, d = self.get_frame_data(frame_idx, img_h, img_w)
        actual_size = max(8, int(self.base_size * scale))
        frame = add_shadow(frame, cx, cy, actual_size, d, self.light)
        frame = overlay_sprite(frame, sprite, cx, cy)
        return frame


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render pest video with COCO annotations.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}"); sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    image_path  = cfg.get("image")
    output_path = cfg.get("output",   "output.mp4")
    duration    = float(cfg.get("duration", 10))
    fps         = int(cfg.get("fps",        25))
    grain       = float(cfg.get("grain",    0.0))
    pest_cfgs   = cfg.get("pests", [{"type": "mouse", "count": 1}])

    if not image_path:
        print("[ERROR] Config must include 'image'"); sys.exit(1)

    bg = cv2.imread(image_path)
    if bg is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = bg.shape[:2]
    print(f"[INFO] Image: {w}×{h}")

    # ── Load depth map ────────────────────────────────────────────────
    depth_map  = None
    depth_path = cfg.get("depth", None)
    if depth_path:
        print(f"[INFO] Depth map: {depth_path}")
        depth_map = load_depth_map(depth_path, (h, w))

    # ── Load light params ─────────────────────────────────────────────
    light_path   = cfg.get("light", None)
    light_params = load_light_params(light_path) if (light_path and os.path.exists(light_path)) \
                   else default_light_params()
    if light_path and os.path.exists(light_path):
        print(f"[INFO] Light params: {light_path}")
    else:
        print("[INFO] Light params: default (overhead fluorescent)")

    # ── Load surfaces ─────────────────────────────────────────────────
    # Support new "surfaces" list format AND old "mask" flat format
    surfaces_cfg = cfg.get("surfaces", None)
    if surfaces_cfg is None:
        # Backward compat: single mask at top level → one "floor" surface
        old_mask = cfg.get("mask", None)
        if old_mask:
            surfaces_cfg = [{"name": "floor", "mask": old_mask}]
        else:
            surfaces_cfg = [{"name": "floor", "mask": None}]

    surfaces = []   # list of dicts: name, mask (bool array or None), colour
    for s in surfaces_cfg:
        sname     = s.get("name", "floor")
        smask_path = s.get("mask", None)
        smask      = None
        scolour    = None

        if smask_path:
            if os.path.exists(smask_path):
                smask = load_mask(smask_path, (h, w))
                n_px  = int(smask.sum())
                if n_px == 0:
                    print(f"[WARN] Surface '{sname}' mask is empty — skipping")
                    continue
                print(f"[INFO] Surface '{sname}': {n_px} px ({100*n_px/(h*w):.1f}%)")
                scolour = sample_surface_colour(bg, smask)
            else:
                print(f"[WARN] Surface '{sname}' mask not found: {smask_path}")
                continue

        surfaces.append({"name": sname, "mask": smask, "colour": scolour})

    if not surfaces:
        print("[WARN] No valid surfaces — using full image as walkable area")
        surfaces = [{"name": "floor", "mask": None, "colour": None}]

    print(f"[INFO] Active surfaces: {[s['name'] for s in surfaces]}")

    # ── Build pest agents ─────────────────────────────────────────────
    # Each pest is randomly assigned a surface and stays there.
    n_frames = int(duration * fps)
    agents   = []

    for pest_cfg_entry in pest_cfgs:
        count = int(pest_cfg_entry.get("count", 1))
        ptype = pest_cfg_entry.get("type", "mouse")
        print(f"[INFO] Generating paths for {count}× {ptype}…")
        for _ in range(count):
            # Random surface assignment
            surface = random.choice(surfaces)
            agent   = PestAgent(
                pest_cfg_entry,
                surface["name"],
                w, h, n_frames,
                surface["mask"],
                depth_map,
                surface["colour"],
                light_params,
                fps,
                len(agents)
            )
            agents.append(agent)
            print(f"       agent {len(agents)}: {ptype} → surface '{surface['name']}'")

    print(f"[INFO] Total pests: {len(agents)}")

    # ── COCO skeleton ─────────────────────────────────────────────────
    coco = {
        "info": {
            "description": "Synthetic pest video annotations",
            "version":      "1.0",
            "source_image": os.path.basename(image_path),
            "video":        os.path.basename(output_path),
            "fps":          fps,
            "duration":     duration,
            "width":        w,
            "height":       h,
            "surfaces":     [s["name"] for s in surfaces],
            "grain":        grain,
        },
        "categories": CATEGORIES,
        "images":      [],
        "annotations": [],
        "frame_meta":  [],
    }

    ann_id     = 1
    video_stem = os.path.splitext(os.path.basename(output_path))[0]

    # ── Video writer ──────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter.")

    print(f"[INFO] Rendering {n_frames} frames → {output_path}")
    if grain > 0:
        print(f"[INFO] Film grain: {grain}")

    for i in range(n_frames):
        frame      = bg.copy()
        frame_anns = []
        has_pest   = False

        for agent in agents:
            sprite, cx, cy, scale, _, _, d = agent.get_frame_data(i, h, w)

            # Shadow + composite
            actual_size = max(8, int(agent.base_size * scale))
            frame = add_shadow(frame, cx, cy, actual_size, d, light_params)
            frame = overlay_sprite(frame, sprite, cx, cy)

            # Annotations
            bbox, mask_arr, area = sprite_to_bbox_and_mask(sprite, cx, cy, w, h)
            if bbox is None:
                continue

            has_pest = True
            polys    = mask_to_polygon(mask_arr)
            seg      = polys if polys else mask_to_rle(mask_arr)

            frame_anns.append({
                "id":           ann_id,
                "image_id":     i,
                "category_id":  agent.cat_id,
                "bbox":         [round(v, 2) for v in bbox],
                "area":         round(area, 2),
                "segmentation": seg,
                "iscrowd":      0,
                "track_id":     agent.track_id,
                "surface":      agent.surface_name,
            })
            ann_id += 1

        # Film grain applied to the full composited frame
        if grain > 0:
            frame = add_film_grain(frame, strength=grain)

        frame_filename = f"{video_stem}_frame_{i:06d}.jpg"
        coco["images"].append({
            "id":        i,
            "file_name": frame_filename,
            "width":     w,
            "height":    h,
            "frame_idx": i,
            "timestamp": round(i / fps, 4),
            "video":     os.path.basename(output_path),
        })
        coco["annotations"].extend(frame_anns)
        coco["frame_meta"].append({
            "frame_idx":  i,
            "has_pest":   has_pest,
            "pest_count": len(frame_anns),
            "file_name":  frame_filename,
        })

        writer.write(frame)
        if i % fps == 0:
            print(f"  frame {i:4d}/{n_frames}", end="\r")

    writer.release()

    ann_path = os.path.splitext(output_path)[0] + "_coco.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    pest_frames = sum(1 for fm in coco["frame_meta"] if fm["has_pest"])
    print(f"\n[DONE] Video   : {output_path}")
    print(f"[DONE] COCO    : {ann_path}")
    print(f"       frames  : {n_frames}  |  pest frames: {pest_frames}")
    print(f"       annotations: {len(coco['annotations'])}  |  pests: {len(agents)}")


if __name__ == "__main__":
    main()