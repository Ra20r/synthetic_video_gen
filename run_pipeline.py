"""
run_pipeline.py
===============
All-in-one pipeline:
  depth map → surface masks → light estimate → configs → render → dataset

Usage:
    # Single config (image read from config automatically):
    python run_pipeline.py --config config.json --output_dir out/

    # Random generation:
    python run_pipeline.py --image kitchen1.png --n 20 --output_dir out/

    # Multiple images — fully isolated output per image:
    python run_pipeline.py --config config_k1.json --output_dir out/
    python run_pipeline.py --config config_k3.json --output_dir out/

    # Skip steps already done:
    python run_pipeline.py --config config.json --output_dir out/ \
        --skip_depth --skip_surfaces --skip_light

Full arguments:
    --image             Kitchen image (read from config if not provided)
    --output_dir        Root output dir (default: pipeline_out/)
    --config            Single config JSON — skips random generation
    --n                 Number of random videos (default: 10)
    --mice              Min max mice per video (default: 0 3)
    --cockroaches       Min max cockroaches per video (default: 0 5)
    --duration          Video duration range seconds (default: 15 30)
    --fps               Frames per second (default: 25)
    --grain             Film grain strength 0-1 (default: 0.02)
    --jobs              Parallel render jobs (default: 1)
    --surfaces          Which surfaces to generate masks for
                        (default: floor counter table shelf)
    --floor_labels      ADE20K floor label override (default: 3)
    --depth_thresh      Depth threshold 0-255 (default: 40)
    --split             Train/val/test fractions (default: 0.8 0.1 0.1)
    --every_n           Extract every Nth frame (default: 1)
    --no_empty_frames   Skip frames with no pest annotations
    --skip_depth        Skip depth map generation
    --skip_surfaces     Skip surface mask generation
    --skip_light        Skip light estimation
    --skip_configs      Skip config generation
    --skip_extract      Skip frame extraction and dataset assembly
    --debug_surfaces    Save colour-coded surface debug overlay
    --seed              Random seed
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path


HERE = os.path.dirname(os.path.abspath(__file__))


def _find_script(here, *names):
    for name in names:
        p = os.path.join(here, name)
        if os.path.exists(p):
            return p
    return os.path.join(here, names[0])


SCRIPTS = {
    "depth":    _find_script(HERE, "generate_depth_map.py"),
    "surfaces": _find_script(HERE, "generate_surface_masks.py"),
    "light":    _find_script(HERE, "estimate_light.py"),
    "configs":  _find_script(HERE, "generate_configs.py"),
    "render":   _find_script(HERE, "batch_render.py"),
    "single":   _find_script(HERE, "add_pests_to_kitchen.py"),
    "extract":  _find_script(HERE, "extract_frames.py", "extract_frame.py"),
}


def run(cmd, step_name):
    print(f"\n{'═'*62}")
    print(f"  STEP: {step_name}")
    print(f"{'═'*62}")
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed (exit {result.returncode})")
        sys.exit(result.returncode)


def build_surfaces_config(out, image_stem, requested_surfaces):
    """
    Return list of surface dicts for the config, using mask paths
    that were (or will be) generated in the output directory.
    Only includes surfaces whose mask file exists.
    """
    surfaces = []
    for name in requested_surfaces:
        mask_path = os.path.join(out, f"{image_stem}_{name}.png")
        if os.path.exists(mask_path):
            surfaces.append({"name": name, "mask": mask_path})
    return surfaces


def main():
    parser = argparse.ArgumentParser(
        description="Full pest video pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--image",           default=None)
    parser.add_argument("--output_dir",      default="pipeline_out")
    parser.add_argument("--config",          default=None)
    parser.add_argument("--n",               type=int,   default=10)
    parser.add_argument("--mice",            type=int,   nargs=2, default=[0, 3])
    parser.add_argument("--cockroaches",     type=int,   nargs=2, default=[0, 5])
    parser.add_argument("--duration",        type=float, nargs=2, default=[15, 30])
    parser.add_argument("--fps",             type=int,   default=25)
    parser.add_argument("--grain",           type=float, default=0.02,
                        help="Film grain strength 0-1 (default: 0.02)")
    parser.add_argument("--jobs",            type=int,   default=1)
    parser.add_argument("--surfaces",        nargs="+",
                        default=["floor"],
                        choices=["floor", "counter", "table", "shelf"],
                        help="Which surface masks to generate (default: floor only). "
                             "Add others if segmentation is reliable: "
                             "--surfaces floor counter")
    parser.add_argument("--floor_labels",    type=int,   nargs="+", default=[3])
    parser.add_argument("--depth_thresh",    type=float, default=40)
    parser.add_argument("--split",           type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--every_n",         type=int,   default=1)
    parser.add_argument("--no_empty_frames", action="store_true")
    parser.add_argument("--skip_depth",      action="store_true")
    parser.add_argument("--skip_surfaces",   action="store_true")
    parser.add_argument("--skip_light",      action="store_true")
    parser.add_argument("--skip_configs",    action="store_true")
    parser.add_argument("--skip_extract",    action="store_true")
    parser.add_argument("--debug_surfaces",  action="store_true")
    parser.add_argument("--seed",            type=int,   default=None)
    args = parser.parse_args()

    # ── Resolve image ─────────────────────────────────────────────────
    if args.image is None:
        if args.config is None:
            print("[ERROR] Provide --image, or --config with an 'image' field")
            sys.exit(1)
        try:
            with open(args.config) as f:
                _peek = json.load(f)
            args.image = _peek.get("image")
            if not args.image:
                print("[ERROR] Config has no 'image' field")
                sys.exit(1)
            print(f"[INFO] Image from config: {args.image}")
        except Exception as e:
            print(f"[ERROR] Could not read config: {e}"); sys.exit(1)

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}"); sys.exit(1)

    image_stem = Path(args.image).stem
    out        = os.path.join(args.output_dir, image_stem)
    os.makedirs(out, exist_ok=True)

    depth_path  = os.path.join(out, f"{image_stem}_depth.png")
    light_path  = os.path.join(out, f"{image_stem}_light.json")
    config_dir  = os.path.join(out, "configs")
    video_dir   = os.path.join(out, "videos")
    dataset_dir = os.path.join(out, "dataset")
    os.makedirs(video_dir, exist_ok=True)

    print(f"\n[PIPELINE] Image  : {args.image}")
    print(f"[PIPELINE] Output : {out}/")
    print(f"[PIPELINE] Surfaces: {args.surfaces}")

    # ── Step 1: Depth map ─────────────────────────────────────────────
    if args.skip_depth and os.path.exists(depth_path):
        print(f"\n[SKIP] Depth map exists: {depth_path}")
    elif not os.path.exists(SCRIPTS["depth"]):
        print(f"[WARN] Depth script not found: {SCRIPTS['depth']}")
        print(f"       Place depth map at: {depth_path}")
        depth_path = None
    else:
        run([sys.executable, SCRIPTS["depth"],
             "--image", args.image, "--output", depth_path],
            "Generate depth map")
        if not Path(depth_path).exists():
            print(f"[WARN] Depth map not produced"); depth_path = None

    # ── Step 2: Surface masks ─────────────────────────────────────────
    if args.skip_surfaces and any(
        os.path.exists(os.path.join(out, f"{image_stem}_{s}.png"))
        for s in args.surfaces
    ):
        print(f"\n[SKIP] Surface masks exist in: {out}/")
    else:
        surf_cmd = [
            sys.executable, SCRIPTS["surfaces"],
            "--image",       args.image,
            "--output_dir",  out,
            "--surfaces",    *args.surfaces,
            "--depth_thresh", str(args.depth_thresh),
        ]
        if depth_path and os.path.exists(depth_path):
            surf_cmd += ["--depth", depth_path]
        if args.debug_surfaces:
            surf_cmd += ["--debug"]
        run(surf_cmd, f"Generate surface masks: {args.surfaces}")

    # ── Step 3: Light estimation ──────────────────────────────────────
    floor_mask_path = os.path.join(out, f"{image_stem}_floor.png")
    if args.skip_light and os.path.exists(light_path):
        print(f"\n[SKIP] Light params exist: {light_path}")
    else:
        light_cmd = [
            sys.executable, SCRIPTS["light"],
            "--image",  args.image,
            "--output", light_path,
        ]
        if os.path.exists(floor_mask_path):
            light_cmd += ["--mask", floor_mask_path]
        run(light_cmd, "Estimate lighting")

    # Build surfaces config list from what was generated
    surfaces_list = build_surfaces_config(out, image_stem, args.surfaces)
    if not surfaces_list:
        print("[WARN] No surface masks found — pests will use full image")

    # ── Step 4: Single config or random generation ────────────────────
    if args.config:
        print(f"\n[INFO] Single config mode: {args.config}")

        with open(args.config) as f:
            cfg = json.load(f)

        # Always inject pipeline-computed paths
        if surfaces_list:
            cfg["surfaces"] = surfaces_list
        if depth_path and os.path.exists(depth_path):
            cfg["depth"] = depth_path
        if os.path.exists(light_path):
            cfg["light"] = light_path
        if "grain" not in cfg:
            cfg["grain"] = args.grain

        # Output video path
        if "output" not in cfg:
            cfg["output"] = os.path.join(video_dir, f"{image_stem}_output.mp4")
        else:
            base = os.path.basename(cfg["output"])
            if not base.startswith(image_stem):
                base = f"{image_stem}_{base}"
            cfg["output"] = os.path.join(video_dir, base)

        patched = os.path.join(out, "_single_config.json")
        with open(patched, "w") as f:
            json.dump(cfg, f, indent=2)

        run([sys.executable, SCRIPTS["single"], "--config", patched],
            "Render single video")

        if not args.skip_extract:
            run([
                sys.executable, SCRIPTS["extract"],
                "--video",      cfg["output"],
                "--output_dir", dataset_dir,
                "--split",      str(args.split[0]), str(args.split[1]), str(args.split[2]),
                "--every_n",    str(args.every_n),
                "--quality",    "95",
                *( ["--no_empty"] if args.no_empty_frames else []),
                *( ["--seed", str(args.seed)] if args.seed is not None else []),
            ], "Extract frames + COCO dataset")

        print(f"\n{'═'*62}")
        print(f"  PIPELINE COMPLETE")
        print(f"  Video   → {cfg['output']}")
        if not args.skip_extract:
            print(f"  Dataset → {dataset_dir}/")
        print(f"{'═'*62}\n")
        return

    # ── Random config generation ──────────────────────────────────────
    if args.skip_configs and os.path.isdir(config_dir) and \
       any(Path(config_dir).glob("*.json")):
        print(f"\n[SKIP] Configs exist in: {config_dir}")
    else:
        os.makedirs(config_dir, exist_ok=True)
        cfg_cmd = [
            sys.executable, SCRIPTS["configs"],
            "--image",         args.image,
            "--output_dir",    config_dir,
            "--n",             str(args.n),
            "--mice",          str(args.mice[0]),        str(args.mice[1]),
            "--cockroaches",   str(args.cockroaches[0]), str(args.cockroaches[1]),
            "--duration",      str(args.duration[0]),    str(args.duration[1]),
            "--fps",           str(args.fps),
            "--grain",         str(args.grain),
            "--output_prefix", f"{image_stem}_video",
        ]
        if surfaces_list:
            # Pass surfaces as JSON string to generate_configs
            cfg_cmd += ["--surfaces_json", json.dumps(surfaces_list)]
        if depth_path and os.path.exists(depth_path):
            cfg_cmd += ["--depth", depth_path]
        if os.path.exists(light_path):
            cfg_cmd += ["--light", light_path]
        if args.seed is not None:
            cfg_cmd += ["--seed", str(args.seed)]
        run(cfg_cmd, f"Generate {args.n} random configs")

    # ── Step 5: Batch render ──────────────────────────────────────────
    run([
        sys.executable, SCRIPTS["render"],
        "--config_dir", config_dir,
        "--output_dir", video_dir,
        "--jobs",       str(args.jobs),
    ], f"Batch render (jobs={args.jobs})")

    # ── Step 6: Extract frames ────────────────────────────────────────
    if args.skip_extract:
        print(f"\n[SKIP] Frame extraction skipped")
    else:
        run([
            sys.executable, SCRIPTS["extract"],
            "--video_dir",  video_dir,
            "--output_dir", dataset_dir,
            "--split",      str(args.split[0]), str(args.split[1]), str(args.split[2]),
            "--every_n",    str(args.every_n),
            "--quality",    "95",
            *( ["--no_empty"] if args.no_empty_frames else []),
            *( ["--seed", str(args.seed)] if args.seed is not None else []),
        ], "Extract frames + COCO dataset")

    print(f"\n{'═'*62}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Videos  → {video_dir}/")
    if not args.skip_extract:
        print(f"  Dataset → {dataset_dir}/")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()