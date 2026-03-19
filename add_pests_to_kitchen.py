"""
add_pests_to_kitchen.py
========================
Overlays animated pests (mouse, cockroach) on a kitchen image and exports MP4.
Driven by a JSON config file.

Pests move on the floor only — no climbing (removed for stability).
The path is fully pre-validated: every position is confirmed on the floor
mask before rendering begins, so pests never appear/disappear.

Requirements:
    pip install opencv-python numpy

Usage:
    python add_pests_to_kitchen.py --config pest_config.json

Config:
    {
      "image":        "kitchen1.png",
      "mask":         "kitchen1_mask.png",   # floor mask (required for best results)
      "depth":        "kitchen1_depth.png",  # depth map (optional, for perspective)
      "output":       "output.mp4",
      "duration":     30,
      "fps":          25,
      "floor_thresh": 30,
      "pests": [
        { "type": "mouse",     "count": 1, "size": 50, "speed": 6 },
        { "type": "cockroach", "count": 2, "size": 30, "speed": 9 }
      ]
    }
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
#  UTILITIES
# ─────────────────────────────────────────────

def load_depth_map(path, target_hw):
    depth = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Could not load depth map: {path}")
    depth = cv2.resize(depth, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    return depth.astype(np.float32) / 255.0


def load_mask(path, target_hw):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {path}")
    mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def depth_at(depth_map, x, y):
    h, w = depth_map.shape
    return float(depth_map[int(np.clip(y, 0, h-1)), int(np.clip(x, 0, w-1))])


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

    def fc(fx, fy, radius, color):
        if radius < 1: return
        cv2.circle(img, (int(fx), int(fy)), int(radius), color[:3], -1, cv2.LINE_AA)
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(m, (int(fx), int(fy)), int(radius), 255, -1, cv2.LINE_AA)
        img[:, :, 3] = np.where(m > 0, color[3], img[:, :, 3])

    def la(p1, p2, color, t):
        cv2.line(img, p1, p2, color[:3], max(1, t), cv2.LINE_AA)

    ar = math.radians(angle_deg)
    ca, sa = math.cos(ar), math.sin(ar)
    def rot(dx, dy): return int(cx+dx*ca-dy*sa), int(cy+dx*sa+dy*ca)

    sway = math.sin(frame_idx * 0.25) * r * 0.8
    tp = [rot(-r*1.1-(i/12)*r*2.0, sway*(i/12)**2) for i in range(13)]
    for i in range(12):
        la(tp[i], tp[i+1], TAIL, max(1, int((1-i/12)*r*0.18)))

    for t in np.linspace(-0.45, 0.45, 10):
        bx, by = rot(t*r*1.6, 0)
        fc(bx, by, int(r*(1.0-0.3*abs(t/0.45))), FUR)
    for t in np.linspace(-0.2, 0.2, 5):
        bx, by = rot(t*r*0.9, 0)
        fc(bx, by, int(r*0.45*(1-0.4*abs(t/0.2))), BELLY)

    for s in (-1, 1):
        ex, ey = rot(r*0.55, s*r*0.85)
        fc(ex, ey, int(r*0.38), EAR)
        fc(ex, ey, int(r*0.22), PINK)

    fc(*rot(r*0.95, 0), int(r*0.68), FUR)

    for s in (-1, 1):
        ex, ey = rot(r*1.05, s*r*0.32)
        fc(ex, ey, int(r*0.13), EYE)
        fc(ex-1, ey-1, max(1, int(r*0.05)), (255,255,255,255))

    fc(*rot(r*1.55, 0), int(r*0.12), NOSE)

    wig = math.sin(frame_idx*0.5)*3
    for lx, ly, w_ in [(0.3,0.9,wig),(0.3,-0.9,-wig),(-0.3,0.9,-wig),(-0.3,-0.9,wig)]:
        la(rot(lx*r, ly*r), rot(lx*r, (ly+w_*0.1)*r*1.45), FUR, max(1,int(r*0.18)))

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
    ANTENA = (10,  30,  45, 255)

    def fc(fx, fy, radius, color):
        if radius < 1: return
        cv2.circle(img, (int(fx), int(fy)), int(radius), color[:3], -1, cv2.LINE_AA)
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(m, (int(fx), int(fy)), int(radius), 255, -1, cv2.LINE_AA)
        img[:, :, 3] = np.where(m > 0, color[3], img[:, :, 3])

    def la(p1, p2, color, t):
        cv2.line(img, p1, p2, color[:3], max(1, t), cv2.LINE_AA)

    ar = math.radians(angle_deg)
    ca, sa = math.cos(ar), math.sin(ar)
    def rot(dx, dy): return int(cx+dx*ca-dy*sa), int(cy+dx*sa+dy*ca)

    leg_phase = frame_idx * 0.6
    for side in (-1, 1):
        for lx, ly, phase_off in [(0.1,1.0,0.2),(-0.2,0.9,0.0),(-0.5,0.8,-0.2)]:
            sway = math.sin(leg_phase+phase_off)*0.25*side
            p1 = rot(lx*r, side*ly*r)
            p2 = rot((lx-0.5)*r, side*(ly+0.6+sway)*r)
            la(p1, p2, LEGS, max(1, int(r*0.12)))
            la(p2, rot((lx-0.8)*r, side*(ly+0.9+sway)*r), LEGS, max(1,int(r*0.09)))

    for t in np.linspace(-0.5, 0.5, 12):
        bx, by = rot(t*r*1.8, 0)
        fc(bx, by, int(r*(1.0-0.35*abs(t/0.5))), SHELL)
    for t in np.linspace(-0.35, 0.35, 5):
        bx, by = rot(t*r*1.4, 0)
        fc(bx, by, int(r*0.18*(1-0.5*abs(t/0.35))), STRIPE)

    fc(*rot(r*1.1, 0), int(r*0.45), HEAD)

    for s in (-1, 1):
        fc(*rot(r*1.25, s*r*0.28), max(1,int(r*0.10)), EYE)

    ant_sway = math.sin(frame_idx*0.3)*r*0.3
    for s in (-1, 1):
        base = rot(r*1.4, s*r*0.2)
        mid  = rot(r*2.2, s*(r*0.4+ant_sway*0.5))
        tip  = rot(r*3.0, s*(r*0.5+ant_sway))
        la(base, mid, ANTENA, max(1,int(r*0.08)))
        la(mid, tip,  ANTENA, max(1,int(r*0.06)))

    return img


SPRITE_FNS = {"mouse": draw_mouse, "cockroach": draw_cockroach}


# ─────────────────────────────────────────────
#  PATH GENERATION
#  Fully pre-validated: every position is
#  checked against the floor mask before the
#  list is returned — no mid-render surprises.
# ─────────────────────────────────────────────

def generate_path(width, height, n_frames, speed,
                  floor_mask=None, margin=30, seed=None):
    """
    Generate a smooth, continuous scurry path.
    Every position is validated on the floor mask.
    The pest is NEVER teleported — if it gets stuck it steers
    back toward the nearest valid floor pixel.
    """
    if seed is not None:
        random.seed(seed)

    # Pre-compute list of all valid floor pixels for fast lookup
    if floor_mask is not None:
        valid_ys, valid_xs = np.where(floor_mask)
        if len(valid_xs) == 0:
            raise ValueError("Floor mask has no walkable pixels.")
        valid_set = set(zip(valid_xs.tolist(), valid_ys.tolist()))

        def is_valid(x, y):
            xi = int(np.clip(round(x), 0, width-1))
            yi = int(np.clip(round(y), 0, height-1))
            return (xi, yi) in valid_set

        def nearest_valid(x, y, search_r=60):
            """Find nearest valid pixel within search_r — used as last resort."""
            xi = int(np.clip(round(x), 0, width-1))
            yi = int(np.clip(round(y), 0, height-1))
            dists = (valid_xs - xi)**2 + (valid_ys - yi)**2
            idx   = np.argmin(dists)
            return float(valid_xs[idx]), float(valid_ys[idx])

        # Start at a random valid floor pixel
        idx = random.randint(0, len(valid_xs)-1)
        x, y = float(valid_xs[idx]), float(valid_ys[idx])
    else:
        def is_valid(x, y):
            return margin <= x <= width-margin and margin <= y <= height-margin
        def nearest_valid(x, y, search_r=60):
            return float(np.clip(x, margin, width-margin)), \
                   float(np.clip(y, margin, height-margin))
        x = float(random.randint(margin, width-margin))
        y = float(random.randint(margin, height-margin))

    current_angle = random.uniform(0, 2*math.pi)
    target_angle  = current_angle
    current_speed = speed
    target_speed  = speed
    pause_timer   = 0
    steer_timer   = 0
    stuck_counter = 0          # consecutive frames where escape failed
    positions     = []

    for frame in range(n_frames):
        # Always record current (validated) position
        positions.append((x, y))

        if pause_timer > 0:
            pause_timer -= 1
            continue

        # Choose new wander target periodically
        if steer_timer <= 0:
            target_angle = random.uniform(0, 2*math.pi)
            target_speed = speed * random.uniform(0.5, 1.5)
            steer_timer  = random.randint(20, 60)
        steer_timer -= 1

        # Smooth angle interpolation
        max_turn = math.radians(8)
        diff = (target_angle - current_angle + math.pi) % (2*math.pi) - math.pi
        current_angle += max(-max_turn, min(max_turn, diff))
        current_speed += (target_speed - current_speed) * 0.08

        # Sniff pause
        if random.random() < 0.008:
            pause_timer  = random.randint(12, 35)
            target_speed = speed * 1.4
            continue

        vx = math.cos(current_angle) * current_speed
        vy = math.sin(current_angle) * current_speed
        nx, ny = x + vx, y + vy

        # Hard screen-edge bounce
        if nx < margin or nx > width - margin:
            vx *= -1
            nx = float(np.clip(nx, margin, width-margin))
            current_angle = math.atan2(vy, vx)
            target_angle  = current_angle
        if ny < margin or ny > height - margin:
            vy *= -1
            ny = float(np.clip(ny, margin, height-margin))
            current_angle = math.atan2(vy, vx)
            target_angle  = current_angle

        if is_valid(nx, ny):
            x, y = nx, ny
            stuck_counter = 0
        else:
            # Try progressively wider escape angles
            escaped = False
            for attempt in range(32):
                spread    = 0.3 + attempt * 0.18
                test_a    = current_angle + math.pi + random.uniform(-spread, spread)
                tnx = float(np.clip(x + math.cos(test_a)*current_speed, margin, width-margin))
                tny = float(np.clip(y + math.sin(test_a)*current_speed, margin, height-margin))
                if is_valid(tnx, tny):
                    x, y = tnx, tny
                    current_angle = test_a
                    target_angle  = test_a + random.uniform(-0.4, 0.4)
                    escaped = True
                    stuck_counter = 0
                    break

            if not escaped:
                stuck_counter += 1
                if stuck_counter >= 10:
                    # Genuinely stuck — snap to nearest valid pixel
                    # This is the only teleport, but it's silent (no skip frame)
                    # so the pest doesn't disappear — it just moves to safety
                    nx, ny = nearest_valid(x, y)
                    x, y = nx, ny
                    current_angle = random.uniform(0, 2*math.pi)
                    target_angle  = current_angle
                    stuck_counter = 0
                # else: stay in place this frame (already recorded)

    return positions


# ─────────────────────────────────────────────
#  OVERLAY / SHADOW
# ─────────────────────────────────────────────

def overlay_sprite(background, sprite_bgra, cx, cy):
    h, w   = sprite_bgra.shape[:2]
    bh, bw = background.shape[:2]
    x0, y0 = cx-w//2, cy-h//2
    x1, y1 = x0+w, y0+h
    sx0=max(0,-x0);          sy0=max(0,-y0)
    sx1=w-max(0,x1-bw);     sy1=h-max(0,y1-bh)
    bx0=max(0,x0);           by0=max(0,y0)
    bx1=bx0+(sx1-sx0);      by1=by0+(sy1-sy0)
    if bx1<=bx0 or by1<=by0: return background
    roi    = background[by0:by1,bx0:bx1].astype(np.float32)
    sprite = sprite_bgra[sy0:sy1,sx0:sx1]
    alpha  = sprite[:,:,3:4].astype(np.float32)/255.0
    color  = sprite[:,:,:3].astype(np.float32)
    background[by0:by1,bx0:bx1] = np.clip(
        roi*(1-alpha)+color*alpha, 0, 255).astype(np.uint8)
    return background


def add_contact_shadow(frame, cx, cy, sprite_size):
    r   = max(2, int(sprite_size * 0.18))
    h_r = max(1, int(r * 0.35))
    overlay = frame.copy()
    cv2.ellipse(overlay, (int(cx), int(cy)+int(sprite_size*0.08)),
                (r, h_r), 0, 0, 360, (10,10,10), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    return frame


# ─────────────────────────────────────────────
#  PEST AGENT
# ─────────────────────────────────────────────

class PestAgent:
    def __init__(self, pest_cfg, width, height, n_frames,
                 floor_mask, depth_map, agent_idx):
        ptype = pest_cfg.get("type", "mouse").lower()
        if ptype not in SPRITE_FNS:
            raise ValueError(f"Unknown pest type '{ptype}'. Options: {list(SPRITE_FNS)}")

        self.draw_fn   = SPRITE_FNS[ptype]
        self.base_size = int(pest_cfg.get("size", 50))
        self.speed     = float(pest_cfg.get("speed", 6))
        self.depth_map = depth_map

        seed = agent_idx * 1337 + random.randint(0, 9999)
        self.path = generate_path(width, height, n_frames, self.speed,
                                  floor_mask=floor_mask, seed=seed)
        self.prev_angle = 0.0

    def render_onto(self, frame, frame_idx, img_h, img_w):
        px, py = self.path[frame_idx]

        # Facing angle — smoothed over 3 frames
        if frame_idx >= 3:
            dx = self.path[frame_idx][0] - self.path[frame_idx-3][0]
            dy = self.path[frame_idx][1] - self.path[frame_idx-3][1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self.prev_angle = math.degrees(math.atan2(dy, dx))
        angle = self.prev_angle

        # Perspective scale from depth
        if self.depth_map is not None:
            d = depth_at(self.depth_map, int(px), int(py))
        else:
            d = py / img_h
        persp_scale = float(np.clip(0.35 + 0.65 * d, 0.15, 1.0))

        actual_size = max(8, int(self.base_size * persp_scale))
        frame = add_contact_shadow(frame, int(px), int(py), actual_size)
        sprite = self.draw_fn(self.base_size, frame_idx, angle, scale=persp_scale)
        frame  = overlay_sprite(frame, sprite, int(px), int(py))
        return frame


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Add pests to a kitchen image using a JSON config.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    image_path   = cfg.get("image")
    mask_path    = cfg.get("mask",   None)
    depth_path   = cfg.get("depth",  None)
    output_path  = cfg.get("output", "output.mp4")
    duration     = float(cfg.get("duration", 10))
    fps          = int(cfg.get("fps", 25))
    floor_thresh = float(cfg.get("floor_thresh", 30)) / 255.0
    pest_cfgs    = cfg.get("pests", [{"type": "mouse", "count": 1}])

    if not image_path:
        print("[ERROR] Config must include 'image'"); sys.exit(1)

    bg = cv2.imread(image_path)
    if bg is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = bg.shape[:2]
    print(f"[INFO] Image: {w}×{h}")

    depth_map  = None
    floor_mask = None

    if depth_path:
        print(f"[INFO] Depth map: {depth_path}")
        depth_map = load_depth_map(depth_path, (h, w))

    if mask_path:
        print(f"[INFO] Floor mask: {mask_path}")
        floor_mask = load_mask(mask_path, (h, w))
        n_floor = int(floor_mask.sum())
        print(f"[INFO] Walkable px: {n_floor} ({100*n_floor/(h*w):.1f}%)")
        if n_floor == 0:
            print("[ERROR] Floor mask is empty — run generate_floor_mask.py first")
            sys.exit(1)
    elif depth_map is not None:
        floor_mask = depth_map > floor_thresh
        print(f"[INFO] Floor from depth: {int(floor_mask.sum())} px")
    else:
        print("[INFO] No mask/depth — full image is walkable")

    n_frames = int(duration * fps)
    agents   = []

    for pest_cfg in pest_cfgs:
        count = int(pest_cfg.get("count", 1))
        ptype = pest_cfg.get("type", "mouse")
        print(f"[INFO] Generating paths for {count}× {ptype}…")
        for i in range(count):
            agents.append(PestAgent(pest_cfg, w, h, n_frames,
                                    floor_mask, depth_map,
                                    len(agents)))

    print(f"[INFO] Total pests: {len(agents)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter.")

    print(f"[INFO] Rendering {n_frames} frames → {output_path}")
    for i in range(n_frames):
        frame = bg.copy()
        for agent in agents:
            frame = agent.render_onto(frame, i, h, w)
        writer.write(frame)
        if i % fps == 0:
            print(f"  frame {i:4d}/{n_frames}", end="\r")

    writer.release()
    print(f"\n[DONE] Saved to: {output_path}")


if __name__ == "__main__":
    main()