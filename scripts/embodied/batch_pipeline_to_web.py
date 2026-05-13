#!/usr/bin/env python3
"""Batch pipeline: convert multiple HyMotion eval NPZ files → ProtoMotions cache → JSON for web.

Usage:
    # Process all NPZ files, output caches + JSON
    python scripts/embodied/batch_pipeline_to_web.py \
        --npz-dir work_dirs/all_tasks_after_fix_20260421/uncond_local/E2_B/npz/ \
        --output-dir output/embodied_comparison/data/motions/ \
        --max-motions 20

    # Process specific files
    python scripts/embodied/batch_pipeline_to_web.py \
        --npz-files 00000.npz 00003.npz 00010.npz \
        --output-dir output/embodied_comparison/data/motions/

    # Skip existing, only process new
    python scripts/embodied/batch_pipeline_to_web.py \
        --npz-dir ... --output-dir ... --skip-existing
"""
import argparse
import glob
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Import cache-to-json converter
sys.path.insert(0, str(SCRIPT_DIR))
from convert_cache_to_json import convert_cache_to_json


def quality_filter_npz(npz_path, min_height=1.2, max_height=2.0):
    """Quick quality check: reject degenerate motions based on estimated body height.

    Loads the positions array from the NPZ and checks that the median
    head-to-foot Y-axis distance is within a reasonable range.
    Degenerate motions (sitting, crawling, negative heights) are rejected.

    Args:
        npz_path: path to motion_135 NPZ file
        min_height: minimum acceptable height in meters
        max_height: maximum acceptable height in meters

    Returns:
        (ok, height, reason): tuple of (pass/fail, estimated height, reason string)
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        if 'positions' in data:
            # positions: (T, 22, 3) — joint positions
            pos = data['positions']
            # Head joint = 15, foot joints = 10, 11 (SMPL convention)
            head_y = pos[:, 15, 1]
            foot_y = np.minimum(pos[:, 10, 1], pos[:, 11, 1])
            heights = head_y - foot_y
            median_h = float(np.median(heights))
            if median_h < min_height:
                return False, median_h, f"height {median_h:.2f}m < {min_height}m (degenerate)"
            if median_h > max_height:
                return False, median_h, f"height {median_h:.2f}m > {max_height}m (degenerate)"
            return True, median_h, "ok"
        else:
            # No positions available, accept by default
            return True, None, "no_positions_key"
    except Exception as e:
        return False, None, f"load_error: {e}"


def run_pipeline(npz_path: str, cache_output: str, extra_args: list = None) -> bool:
    """Run pipeline_motion_to_robot.py on a single NPZ file.

    Returns True on success, False on failure.
    """
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "pipeline_motion_to_robot.py"),
        "--input", str(npz_path),
        "--output", str(cache_output),
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours per motion (PyRoki CPU retarget takes ~60-70 min)
        )
        if result.returncode != 0:
            print(f"    PIPELINE FAILED (exit {result.returncode})")
            if result.stderr:
                # Print last 20 lines of stderr
                lines = result.stderr.strip().split('\n')
                for line in lines[-20:]:
                    print(f"      {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    PIPELINE TIMEOUT (>7200s)")
        return False
    except Exception as e:
        print(f"    PIPELINE ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch pipeline: NPZ → cache → JSON for web visualization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz-dir", help="Directory containing NPZ files")
    group.add_argument("--npz-files", nargs="+", help="Specific NPZ file paths")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSON files")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory for intermediate .pt caches (default: output_dir/../caches/)")
    parser.add_argument("--max-motions", type=int, default=None, help="Max motions to process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if JSON already exists")
    parser.add_argument("--pipeline-args", nargs="*", default=[],
                        help="Extra args to pass to pipeline_motion_to_robot.py")
    parser.add_argument("--pattern", default="*.npz", help="Glob pattern for --npz-dir mode")
    parser.add_argument("--name-prefix", default="pipeline_",
                        help="Prefix for output filenames (default: pipeline_)")
    parser.add_argument("--quality-filter", action="store_true", default=True,
                        help="Enable quality filtering of input NPZs (default: True)")
    parser.add_argument("--no-quality-filter", dest="quality_filter", action="store_false",
                        help="Disable quality filtering")
    parser.add_argument("--min-height", type=float, default=1.2,
                        help="Min body height for quality filter (default: 1.2m)")
    parser.add_argument("--max-height", type=float, default=2.0,
                        help="Max body height for quality filter (default: 2.0m)")
    args = parser.parse_args()

    # Collect NPZ files
    if args.npz_dir:
        npz_files = sorted(glob.glob(os.path.join(args.npz_dir, args.pattern)))
    else:
        npz_files = [f for f in args.npz_files if os.path.isfile(f)]

    if not npz_files:
        print("No NPZ files found!")
        sys.exit(1)

    # Quality filtering
    if args.quality_filter:
        print(f"\nQuality filtering {len(npz_files)} NPZ files (height range [{args.min_height}, {args.max_height}]m)...")
        good_files = []
        rejected = 0
        for npz_path in npz_files:
            ok, height, reason = quality_filter_npz(npz_path, args.min_height, args.max_height)
            if ok:
                good_files.append(npz_path)
            else:
                rejected += 1
        print(f"  {len(good_files)} passed, {rejected} rejected")
        npz_files = good_files

    if args.max_motions:
        npz_files = npz_files[:args.max_motions]

    print(f"Found {len(npz_files)} NPZ files to process")

    # Setup directories
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = pathlib.Path(args.cache_dir) if args.cache_dir else output_dir.parent / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = []
    successes = 0
    failures = 0
    skipped = 0

    for i, npz_path in enumerate(npz_files):
        npz_name = pathlib.Path(npz_path).stem  # e.g., "00000"
        motion_id = f"{args.name_prefix}{npz_name}"
        json_path = output_dir / f"{motion_id}.json"
        cache_path = cache_dir / f"{motion_id}.pt"

        print(f"\n[{i+1}/{len(npz_files)}] {npz_name}")
        print(f"  NPZ:   {npz_path}")
        print(f"  Cache: {cache_path}")
        print(f"  JSON:  {json_path}")

        # Skip if exists
        if args.skip_existing and json_path.exists():
            print(f"  SKIPPED (JSON exists)")
            skipped += 1
            # Still add to results for manifest
            try:
                with open(json_path) as f:
                    data = json.load(f)
                results.append({
                    "id": motion_id,
                    "num_frames": data["num_frames"],
                    "fps": data["fps"],
                    "status": "existing",
                })
            except Exception:
                pass
            continue

        t0 = time.time()

        # Step 1: Run pipeline (NPZ → cache .pt)
        if not cache_path.exists():
            print(f"  Running pipeline...")
            ok = run_pipeline(npz_path, str(cache_path), args.pipeline_args)
            if not ok:
                failures += 1
                results.append({"id": motion_id, "status": "pipeline_failed"})
                continue
        else:
            print(f"  Cache exists, skipping pipeline")

        # Step 2: Convert cache → JSON
        try:
            info = convert_cache_to_json(str(cache_path), str(json_path))
            dt = time.time() - t0
            print(f"  OK ({dt:.1f}s)")
            successes += 1
            results.append({
                "id": motion_id,
                "num_frames": info["num_frames"],
                "fps": info["fps"],
                "status": "success",
                "source_npz": str(npz_path),
            })
        except Exception as e:
            print(f"  JSON CONVERT FAILED: {e}")
            traceback.print_exc()
            failures += 1
            results.append({"id": motion_id, "status": "json_failed"})

    # Write/update manifest
    manifest_path = output_dir / "manifest.json"

    # Merge with existing manifest
    existing_motions = {}
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                existing = json.load(f)
            for m in existing.get("motions", []):
                existing_motions[m["id"]] = m
        except Exception:
            pass

    # Update with new results
    for r in results:
        if r.get("status") in ("success", "existing") and "num_frames" in r:
            existing_motions[r["id"]] = {
                "id": r["id"],
                "num_frames": r["num_frames"],
                "fps": r["fps"],
            }

    manifest = {"motions": sorted(existing_motions.values(), key=lambda x: x["id"])}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Batch complete!")
    print(f"{'='*60}")
    print(f"  Total:    {len(npz_files)}")
    print(f"  Success:  {successes}")
    print(f"  Failed:   {failures}")
    print(f"  Skipped:  {skipped}")
    print(f"  Manifest: {manifest_path} ({len(manifest['motions'])} motions)")

    # Write detailed report
    report_path = output_dir.parent / "batch_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(npz_files),
            "success": successes,
            "failed": failures,
            "skipped": skipped,
            "results": results,
        }, f, indent=2)
    print(f"  Report:   {report_path}")


if __name__ == "__main__":
    main()
