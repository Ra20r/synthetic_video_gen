"""
run_pipeline.py
===============
All-in-one pipeline: depth map → floor mask → generate configs → render videos.

Usage:
    # Generate 20 random videos:
    python run_pipeline.py --image kitchen1.png --n 20 --output_dir out/

    # Use a single hand-crafted config instead of random generation:
    python run_pipeline.py --image kitchen1.png --config my_config.json --output_dir out/

    # Skip steps you've already run:
    python run_pipeline.py --image kitchen1.png --n 10 --output_dir out/ \
                           --skip_depth --skip_mask

Full arguments:
    --image         Kitchen image path (required)
    --output_dir    Root output directory (default: pipeline_out/)
    --n             Number of random configs/videos to generate (default: 10)
    --config        Path to a single config JSON — skips random generation
    --mice          Min max mice range (default: 0 3)
    --cockroaches   Min max cockroach range (default: 0 5)
    --duration      Video duration range seconds (default: 15 30)
    --fps           Frames per second (default: 25)
    --jobs          Parallel render jobs (default: 1)
    --floor_labels  ADE20K floor label indices (default: 3)
    --depth_thresh  Depth threshold 0-255 (default: 40)
    --skip_depth    Skip depth map generation (if already exists)
    --skip_mask     Skip floor mask generation (if already exists)
    --skip_configs  Skip config generation (if already exists in configs/)
    --debug_mask    Save debug overlay for the floor mask
    --seed          Random seed for config generation
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path


HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = {
    "depth":   os.path.join(HERE, "generate_depth_map.py"),
    "mask":    os.path.join(HERE, "generate_floor_mask.py"),
    "configs": os.path.join(HERE, "generate_configs.py"),
    "render":  os.path.join(HERE, "batch_render.py"),
    "single":  os.path.join(HERE, "add_pests_to_kitchen.py"),
}


def run(cmd, step_name):
    print(f"\n{'═'*60}")
    print(f"  STEP: {step_name}")
    print(f"{'═'*60}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed (exit {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Full pest video generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image",         required=True)
    parser.add_argument("--output_dir",    default="pipeline_out")
    parser.add_argument("--n",             type=int, default=10)
    parser.add_argument("--config",        default=None,
                        help="Single config JSON — skips random config generation")
    parser.add_argument("--mice",          type=int, nargs=2, default=[0, 3])
    parser.add_argument("--cockroaches",   type=int, nargs=2, default=[0, 5])
    parser.add_argument("--duration",      type=float, nargs=2, default=[15, 30])
    parser.add_argument("--fps",           type=int, default=25)
    parser.add_argument("--jobs",          type=int, default=1)
    parser.add_argument("--floor_labels",  type=int, nargs="+", default=[3])
    parser.add_argument("--depth_thresh",  type=float, default=40)
    parser.add_argument("--skip_depth",    action="store_true")
    parser.add_argument("--skip_mask",     action="store_true")
    parser.add_argument("--skip_configs",  action="store_true")
    parser.add_argument("--debug_mask",    action="store_true")
    parser.add_argument("--seed",          type=int, default=None)
    args = parser.parse_args()

    # ── Validate image ───────────────────────────────────────────────
    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    image_stem = Path(args.image).stem
    out        = args.output_dir
    os.makedirs(out, exist_ok=True)

    depth_path  = os.path.join(out, f"{image_stem}_depth.png")
    mask_path   = os.path.join(out, f"{image_stem}_mask.png")
    config_dir  = os.path.join(out, "configs")
    video_dir   = os.path.join(out, "videos")
    os.makedirs(video_dir, exist_ok=True)

    print(f"\n[PIPELINE] Image : {args.image}")
    print(f"[PIPELINE] Output: {out}/")

    # ── Step 1: Depth map ────────────────────────────────────────────
    if args.skip_depth and os.path.exists(depth_path):
        print(f"\n[SKIP] Depth map exists: {depth_path}")
    else:
        if not os.path.exists(SCRIPTS["depth"]):
            print(f"[WARN] Depth script not found: {SCRIPTS['depth']}")
            print("       Skipping depth generation. "
                  "Run your own depth script and place output at:")
            print(f"       {depth_path}")
            depth_path = None
        else:
            run([sys.executable, SCRIPTS["depth"],
                 "--image", args.image,
                 "--output", depth_path],
                "Generate depth map")

    # ── Step 2: Floor mask ───────────────────────────────────────────
    if args.skip_mask and os.path.exists(mask_path):
        print(f"\n[SKIP] Floor mask exists: {mask_path}")
    else:
        mask_cmd = [
            sys.executable, SCRIPTS["mask"],
            "--image",        args.image,
            "--output",       mask_path,
            "--floor_labels", *[str(l) for l in args.floor_labels],
            "--depth_thresh", str(args.depth_thresh),
        ]
        if depth_path and os.path.exists(depth_path):
            mask_cmd += ["--depth", depth_path]
        if args.debug_mask:
            mask_cmd += ["--debug"]
        run(mask_cmd, "Generate floor mask")

    # ── Step 3: Configs or single config ─────────────────────────────
    if args.config:
        # Single config mode — render just this one
        print(f"\n[INFO] Single config mode: {args.config}")

        # Patch the config to use our computed mask/depth if not already set
        with open(args.config) as f:
            cfg = json.load(f)
        if "mask" not in cfg and os.path.exists(mask_path):
            cfg["mask"] = mask_path
        if "depth" not in cfg and depth_path and os.path.exists(depth_path):
            cfg["depth"] = depth_path
        if "output" not in cfg:
            cfg["output"] = os.path.join(video_dir, "output_0000.mp4")
        else:
            cfg["output"] = os.path.join(video_dir, os.path.basename(cfg["output"]))

        patched = os.path.join(out, "_single_config.json")
        with open(patched, "w") as f:
            json.dump(cfg, f, indent=2)

        run([sys.executable, SCRIPTS["single"], "--config", patched],
            "Render single video")
        print(f"\n[DONE] Video: {cfg['output']}")
        return

    # Random config generation
    if args.skip_configs and os.path.isdir(config_dir) and \
       any(Path(config_dir).glob("*.json")):
        print(f"\n[SKIP] Configs exist in: {config_dir}")
    else:
        os.makedirs(config_dir, exist_ok=True)
        cfg_cmd = [
            sys.executable, SCRIPTS["configs"],
            "--image",       args.image,
            "--output_dir",  config_dir,
            "--n",           str(args.n),
            "--mice",        str(args.mice[0]),        str(args.mice[1]),
            "--cockroaches", str(args.cockroaches[0]), str(args.cockroaches[1]),
            "--duration",    str(args.duration[0]),    str(args.duration[1]),
            "--fps",         str(args.fps),
            "--output_prefix", "video",
        ]
        if os.path.exists(mask_path):
            cfg_cmd += ["--mask", mask_path]
        if depth_path and os.path.exists(depth_path):
            cfg_cmd += ["--depth", depth_path]
        if args.seed is not None:
            cfg_cmd += ["--seed", str(args.seed)]
        run(cfg_cmd, f"Generate {args.n} random configs")

    # ── Step 4: Batch render ─────────────────────────────────────────
    run([
        sys.executable, SCRIPTS["render"],
        "--config_dir",  config_dir,
        "--output_dir",  video_dir,
        "--jobs",        str(args.jobs),
    ], f"Batch render (jobs={args.jobs})")

    print(f"\n{'═'*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Videos → {video_dir}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()