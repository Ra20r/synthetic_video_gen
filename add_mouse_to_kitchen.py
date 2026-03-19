"""
add_mouse_to_kitchen.py
========================
Overlays an animated mouse/rat that scurries randomly across a 2D kitchen image
and exports the result as an MP4 video.

Supports a depth map image for:
  - Floor-only movement (mouse stays on walkable areas)
  - Perspective scaling (mouse shrinks near the horizon)
  - Depth-aware shadow (fainter and smaller when far away)

Requirements:
    pip install opencv-python pillow numpy

Usage:
    python add_mouse_to_kitchen.py --image kitchen1.png --mask kitchen1_mask.png \
        --depth kitchen1_depth.png --output output.mp4

Optional flags:
    --duration      Video length in seconds            (default: 6)
    --fps           Frames per second                  (default: 30)
    --mouse_size    Base pixel size of mouse sprite    (default: 60)
    --speed         Mouse movement speed (px/frame)    (default: 6)
    --mask          Binary floor mask PNG (white=walkable). Takes priority over depth.
    --floor_thresh  Depth value 0-255 above which pixel is floor (default: 30)
"""

import cv2
import numpy as np
import argparse
import math
import random

# ─────────────────────────────────────────────
#  DEPTH MAP UTILITIES
# ─────────────────────────────────────────────

def load_depth_map(path, target_size):
    depth = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Could not load depth map: {path}")
    depth = cv2.resize(depth, (target_size[1], target_size[0]),
                       interpolation=cv2.INTER_LINEAR)
    return depth.astype(np.float32) / 255.0


def load_floor_mask(path, target_size):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load floor mask: {path}")
    mask = cv2.resize(mask, (target_size[1], target_size[0]),
                      interpolation=cv2.INTER_NEAREST)
    return mask > 127


def depth_at(depth_map, x, y):
    h, w = depth_map.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    return float(depth_map[y, x])


def build_floor_mask(depth_map, threshold):
    return depth_map > threshold


# ─────────────────────────────────────────────
#  MOUSE SPRITE DRAWING
#  FIX 1: canvas scales with `scale` so the
#  mouse genuinely appears larger/smaller in
#  the scene rather than just drawing a tiny
#  sprite inside a fixed-size canvas.
# ─────────────────────────────────────────────

def draw_mouse(base_size, frame_idx, angle_deg, scale=1.0):
    """
    Returns a BGRA sprite whose canvas is base_size*scale pixels square.
    Scaling the canvas (not just r) means the overlaid sprite physically
    occupies more/fewer pixels — giving true perspective size change.
    """
    c = max(16, int(base_size * scale))
    img = np.zeros((c, c, 4), dtype=np.uint8)
    cx, cy = c // 2, c // 2
    r = int(c * 0.22)          # fixed ratio inside the scaled canvas
    if r < 2:
        return img

    FUR   = (80,  80, 100, 255)
    BELLY = (130, 130, 160, 255)
    EAR   = (100,  60, 120, 255)
    PINK  = (140,  90, 200, 255)
    EYE   = (10,   10,  10, 255)
    NOSE  = (100,  80, 200, 255)
    TAIL  = (90,   70, 110, 255)

    def filled_circle(img, fx, fy, radius, color):
        if radius < 1:
            return
        cv2.circle(img, (int(fx), int(fy)), int(radius), color[:3], -1, cv2.LINE_AA)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(fx), int(fy)), int(radius), 255, -1, cv2.LINE_AA)
        img[:, :, 3] = np.where(mask > 0, color[3], img[:, :, 3])

    def line_aa(img, p1, p2, color, thickness):
        cv2.line(img, p1, p2, color[:3], max(1, thickness), cv2.LINE_AA)

    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    def rotate(dx, dy):
        x = cx + dx * cos_a - dy * sin_a
        y = cy + dx * sin_a + dy * cos_a
        return int(x), int(y)

    # Tail
    sway = math.sin(frame_idx * 0.25) * r * 0.8
    tail_pts = []
    segs = 12
    for i in range(segs + 1):
        t = i / segs
        tail_pts.append(rotate(-r * 1.1 - t * r * 2.0, sway * t * t))
    for i in range(len(tail_pts) - 1):
        line_aa(img, tail_pts[i], tail_pts[i + 1], TAIL,
                max(1, int((1 - i / segs) * r * 0.18)))

    # Body
    for t in np.linspace(-0.45, 0.45, 10):
        bx, by = rotate(t * r * 1.6, 0)
        filled_circle(img, bx, by, int(r * (1.0 - 0.3 * abs(t / 0.45))), FUR)
    for t in np.linspace(-0.2, 0.2, 5):
        bx, by = rotate(t * r * 0.9, 0)
        filled_circle(img, bx, by, int(r * 0.45 * (1 - 0.4 * abs(t / 0.2))), BELLY)

    # Ears
    for side in (-1, 1):
        ex, ey = rotate(r * 0.55, side * r * 0.85)
        filled_circle(img, ex, ey, int(r * 0.38), EAR)
        filled_circle(img, ex, ey, int(r * 0.22), PINK)

    # Head
    hx, hy = rotate(r * 0.95, 0)
    filled_circle(img, hx, hy, int(r * 0.68), FUR)

    # Eyes
    for side in (-1, 1):
        ex, ey = rotate(r * 1.05, side * r * 0.32)
        filled_circle(img, ex, ey, int(r * 0.13), EYE)
        filled_circle(img, ex - 1, ey - 1, max(1, int(r * 0.05)), (255, 255, 255, 255))

    # Nose
    nx, ny = rotate(r * 1.55, 0)
    filled_circle(img, nx, ny, int(r * 0.12), NOSE)

    # Legs — speed up wiggle proportionally so fast mouse looks fast
    wiggle = math.sin(frame_idx * 0.5) * 3
    for lx_off, ly_off, wig in [(0.3, 0.9, wiggle), (0.3, -0.9, -wiggle),
                                  (-0.3, 0.9, -wiggle), (-0.3, -0.9, wiggle)]:
        lx1, ly1 = rotate(lx_off * r, ly_off * r)
        lx2, ly2 = rotate(lx_off * r, (ly_off + wig * 0.1) * r * 1.45)
        line_aa(img, (lx1, ly1), (lx2, ly2), FUR, max(1, int(r * 0.18)))

    return img


# ─────────────────────────────────────────────
#  SCURRY PATH
#  FIX 2: smooth steering — instead of
#  snapping to a new random angle, the mouse
#  gradually steers toward a target angle.
#  This removes the jerky direction changes.
# ─────────────────────────────────────────────

def generate_scurry_path(width, height, n_frames, speed,
                         floor_mask=None, margin=40):
    def is_valid(x, y):
        xi = int(np.clip(x, 0, width - 1))
        yi = int(np.clip(y, 0, height - 1))
        if floor_mask is not None:
            return bool(floor_mask[yi, xi])
        return margin <= x <= width - margin and margin <= y <= height - margin

    # Valid start position
    if floor_mask is not None:
        ys, xs = np.where(floor_mask)
        if len(xs) == 0:
            raise ValueError("No walkable floor pixels found.")
        idx = random.randint(0, len(xs) - 1)
        x, y = float(xs[idx]), float(ys[idx])
    else:
        x = float(random.randint(margin, width - margin))
        y = float(random.randint(margin, height - margin))

    # Steering state
    current_angle  = random.uniform(0, 2 * math.pi)
    target_angle   = current_angle
    current_speed  = speed
    target_speed   = speed
    pause_timer    = 0
    steer_timer    = 0          # frames until next target-angle change
    positions      = []

    for _ in range(n_frames):
        positions.append((x, y))

        if pause_timer > 0:
            pause_timer -= 1
            continue

        # Pick a new target angle every 20-60 frames (smooth wandering)
        if steer_timer <= 0:
            target_angle  = random.uniform(0, 2 * math.pi)
            target_speed  = speed * random.uniform(0.5, 1.5)
            steer_timer   = random.randint(20, 60)

        steer_timer -= 1

        # Smoothly interpolate current angle toward target (max 8° per frame)
        max_turn = math.radians(8)
        angle_diff = (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
        current_angle += max(-max_turn, min(max_turn, angle_diff))

        # Smoothly interpolate speed
        current_speed += (target_speed - current_speed) * 0.08

        vx = math.cos(current_angle) * current_speed
        vy = math.sin(current_angle) * current_speed

        # Random sniff pause
        if random.random() < 0.008:
            pause_timer  = random.randint(12, 35)
            target_speed = speed * 1.4   # burst after pause
            continue

        nx, ny = x + vx, y + vy

        # Hard screen-edge clamp — always enforced, reflects angle off walls
        if nx < margin:
            nx = float(margin)
            current_angle = math.pi - current_angle
            target_angle = current_angle
        elif nx > width - margin:
            nx = float(width - margin)
            current_angle = math.pi - current_angle
            target_angle = current_angle
        if ny < margin:
            ny = float(margin)
            current_angle = -current_angle
            target_angle = current_angle
        elif ny > height - margin:
            ny = float(height - margin)
            current_angle = -current_angle
            target_angle = current_angle

        # If new position is off-floor, steer away with increasing spread
        if not is_valid(nx, ny):
            escaped = False
            for attempt in range(24):
                spread = 0.4 + attempt * 0.2
                test_angle = current_angle + math.pi + random.uniform(-spread, spread)
                tvx = math.cos(test_angle) * current_speed
                tvy = math.sin(test_angle) * current_speed
                tnx = float(np.clip(x + tvx, margin, width - margin))
                tny = float(np.clip(y + tvy, margin, height - margin))
                if is_valid(tnx, tny):
                    nx, ny = tnx, tny
                    current_angle = test_angle
                    target_angle  = test_angle + random.uniform(-0.5, 0.5)
                    escaped = True
                    break
            if not escaped:
                # Last resort: point toward centre of floor
                nx, ny = x, y
                current_angle = math.atan2(height / 2 - y, width / 2 - x)
                target_angle  = current_angle

        x, y = nx, ny

    return positions


# ─────────────────────────────────────────────
#  OVERLAY HELPER
# ─────────────────────────────────────────────

def overlay_sprite(background, sprite_bgra, cx, cy):
    h, w = sprite_bgra.shape[:2]
    bh, bw = background.shape[:2]
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h
    sx0 = max(0, -x0);          sy0 = max(0, -y0)
    sx1 = w - max(0, x1 - bw); sy1 = h - max(0, y1 - bh)
    bx0 = max(0, x0);           by0 = max(0, y0)
    bx1 = bx0 + (sx1 - sx0);   by1 = by0 + (sy1 - sy0)
    if bx1 <= bx0 or by1 <= by0:
        return background
    roi    = background[by0:by1, bx0:bx1].astype(np.float32)
    sprite = sprite_bgra[sy0:sy1, sx0:sx1]
    alpha  = sprite[:, :, 3:4].astype(np.float32) / 255.0
    color  = sprite[:, :, :3].astype(np.float32)
    background[by0:by1, bx0:bx1] = np.clip(
        roi * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    return background


# ─────────────────────────────────────────────
#  DEPTH-AWARE SHADOW
# ─────────────────────────────────────────────

def add_shadow(frame, cx, cy, sprite_size, depth_value):
    """Shadow scales with the actual rendered sprite size and depth."""
    radius = sprite_size // 2
    if radius < 2:
        return frame
    opacity  = 0.12 + 0.38 * depth_value
    w_radius = max(1, int(radius * (0.45 + 0.35 * depth_value)))
    h_radius = max(1, int(radius * (0.10 + 0.12 * depth_value)))
    overlay  = frame.copy()
    cv2.ellipse(overlay, (int(cx), int(cy) + int(radius * 0.6)),
                (w_radius, h_radius), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    return frame


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Add a depth-aware scurrying mouse to a kitchen image.")
    parser.add_argument("--image",        required=True)
    parser.add_argument("--mask",         default=None,
                        help="Binary floor mask PNG (white=floor). "
                             "Overrides depth-based floor detection.")
    parser.add_argument("--depth",        default=None,
                        help="Grayscale depth map. White=near, Black=far.")
    parser.add_argument("--output",       default="output.mp4")
    parser.add_argument("--duration",     type=float, default=6)
    parser.add_argument("--fps",          type=int,   default=30)
    parser.add_argument("--mouse_size",   type=int,   default=60,
                        help="Base sprite size in px at full depth (default: 60)")
    parser.add_argument("--speed",        type=float, default=6,
                        help="Movement speed px/frame (default: 6)")
    parser.add_argument("--floor_thresh", type=float, default=30,
                        help="Depth threshold 0-255 for walkable floor (default: 30)")
    args = parser.parse_args()

    floor_thresh = args.floor_thresh / 255.0 if args.floor_thresh > 1 else args.floor_thresh

    bg = cv2.imread(args.image)
    if bg is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    h, w = bg.shape[:2]

    depth_map  = None
    floor_mask = None

    if args.mask:
        print(f"[INFO] Loading floor mask: {args.mask}")
        floor_mask = load_floor_mask(args.mask, (h, w))
        n_floor    = int(floor_mask.sum())
        print(f"[INFO] Walkable pixels: {n_floor} ({100 * n_floor / (h * w):.1f}%)")
        if n_floor == 0:
            print("[WARN] Mask has no white pixels — check mask file")

    if args.depth:
        print(f"[INFO] Loading depth map: {args.depth}")
        depth_map = load_depth_map(args.depth, (h, w))
        if floor_mask is None:
            floor_mask = build_floor_mask(depth_map, floor_thresh)
            n_floor    = int(floor_mask.sum())
            print(f"[INFO] Walkable pixels from depth: {n_floor} ({100 * n_floor / (h * w):.1f}%)")

    if floor_mask is None and depth_map is None:
        print("[INFO] No mask or depth — using Y-position as depth proxy")

    n_frames = int(args.duration * args.fps)
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter.")

    print(f"[INFO] Generating path ({n_frames} frames)…")
    path = generate_scurry_path(w, h, n_frames, args.speed, floor_mask=floor_mask)

    print(f"[INFO] Rendering → {args.output}")
    prev_angle = 0.0

    for i, (px, py) in enumerate(path):
        frame = bg.copy()

        # Facing angle — use smoothed velocity over last 3 frames for stability
        if i >= 2:
            dx = path[i][0] - path[i - 2][0]
            dy = path[i][1] - path[i - 2][1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                prev_angle = math.degrees(math.atan2(dy, dx))
        angle = prev_angle

        # Depth value at current position
        if depth_map is not None:
            d = depth_at(depth_map, int(px), int(py))
        else:
            d = py / h   # fallback: lower = nearer

        # FIX 3: perspective scale applied to the canvas size directly.
        # Near (d=1) → full mouse_size canvas, Far (d=0) → 35% canvas.
        # The sprite itself then occupies the right number of pixels in the scene.
        persp_scale = float(np.clip(0.35 + 0.65 * d, 0.15, 1.0))
        actual_size = max(8, int(args.mouse_size * persp_scale))

        # Shadow — pass actual rendered size so it matches
        frame = add_shadow(frame, int(px), int(py), actual_size, d)

        # Mouse sprite at correct scale
        sprite = draw_mouse(args.mouse_size, i, angle, scale=persp_scale)
        frame  = overlay_sprite(frame, sprite, int(px), int(py))

        writer.write(frame)
        if i % args.fps == 0:
            print(f"  frame {i:4d}/{n_frames}", end="\r")

    writer.release()
    print(f"\n[DONE] Saved to: {args.output}")


if __name__ == "__main__":
    main()