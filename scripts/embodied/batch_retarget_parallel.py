#!/usr/bin/env python3
"""Parallel batch retargeting: Run PyRoki pipeline on multiple NPZ files concurrently.

Since PyRoki retargeting uses CPU-only JAX (~60 min per motion), running motions
in parallel is essential. Each motion is independent, so we can process N motions
concurrently using multiprocessing.

This script handles ONLY the retarget step (NPZ -> .motion), not T2M inference
or SMPL mesh generation. It's designed for the case where NPZ files already exist
(e.g., re-processing V4 output).

Usage:
    # 4 parallel workers (default)
    python3 scripts/embodied/batch_retarget_parallel.py \
        --npz-dir output/embodied_t2m_v4/data/npz \
        --output-dir output/embodied_t2m_v4/data/retarget \
        --workers 4

    # Process specific NPZ files
    python3 scripts/embodied/batch_retarget_parallel.py \
        --npz-files v4_turn_001.npz v4_arm_002.npz \
        --output-dir output/embodied_t2m_v4/data/retarget \
        --workers 2

    # Skip already-processed motions
    python3 scripts/embodied/batch_retarget_parallel.py \
        --npz-dir output/embodied_t2m_v4/data/npz \
        --output-dir output/embodied_t2m_v4/data/retarget \
        --skip-existing --workers 4
"""
import argparse
import glob
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PIPELINE_SCRIPT = SCRIPT_DIR / "pipeline_motion_to_robot.py"


def process_single_npz(npz_path: str, output_dir: str, keep_intermediates: bool = True,
                        fps: int = 30) -> dict:
    """Process a single NPZ file through the PyRoki pipeline.

    Returns dict with status info.
    """
    npz_path = pathlib.Path(npz_path)
    stem = npz_path.stem
    motion_output_dir = pathlib.Path(output_dir) / f"motion_{stem}"

    t0 = time.time()
    result = {
        "npz": str(npz_path),
        "stem": stem,
        "output_dir": str(motion_output_dir),
        "status": "unknown",
    }

    try:
        cmd = [
            sys.executable, str(PIPELINE_SCRIPT),
            "--input", str(npz_path),
            "--output", str(motion_output_dir),
            "--fps", str(fps),
        ]
        if keep_intermediates:
            cmd.append("--keep-intermediates")

        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per motion
        )

        dt = time.time() - t0

        if proc.returncode == 0:
            # Check for .motion file
            motion_files = list(motion_output_dir.glob("*.motion"))
            if motion_files:
                result["status"] = "success"
                result["motion_file"] = str(motion_files[0])
                result["time_s"] = round(dt, 1)
                print(f"  OK  [{stem}] {dt:.0f}s -> {motion_files[0].name}")
            else:
                result["status"] = "no_motion_file"
                result["time_s"] = round(dt, 1)
                print(f"  WARN [{stem}] {dt:.0f}s - no .motion file")
        else:
            result["status"] = "failed"
            result["returncode"] = proc.returncode
            result["stderr"] = proc.stderr[-500:] if proc.stderr else ""
            result["time_s"] = round(dt, 1)
            print(f"  FAIL [{stem}] exit={proc.returncode} ({dt:.0f}s)")
            if proc.stderr:
                for line in proc.stderr.strip().split("\n")[-5:]:
                    print(f"       {line}")

    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        result["status"] = "timeout"
        result["time_s"] = round(dt, 1)
        print(f"  TIMEOUT [{stem}] >{dt:.0f}s")
    except Exception as e:
        dt = time.time() - t0
        result["status"] = "error"
        result["error"] = str(e)
        result["time_s"] = round(dt, 1)
        print(f"  ERROR [{stem}] {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parallel batch retargeting: NPZ -> .motion via PyRoki"
    )
    parser.add_argument("--npz-dir", type=str,
                        help="Directory containing NPZ files")
    parser.add_argument("--npz-files", nargs="+",
                        help="Specific NPZ files to process")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for retarget results")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip motions that already have .motion files")
    parser.add_argument("--keep-intermediates", action="store_true", default=True,
                        help="Keep intermediate files (default: True)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Motion FPS (default: 30)")
    parser.add_argument("--max-motions", type=int, default=None,
                        help="Max number of motions to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without executing")
    args = parser.parse_args()

    # Collect NPZ files
    if args.npz_files:
        npz_files = [str(pathlib.Path(f).resolve()) for f in args.npz_files]
    elif args.npz_dir:
        npz_files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    else:
        print("ERROR: Must specify --npz-dir or --npz-files")
        sys.exit(1)

    if not npz_files:
        print("ERROR: No NPZ files found")
        sys.exit(1)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out already-processed if requested
    if args.skip_existing:
        to_process = []
        for npz_path in npz_files:
            stem = pathlib.Path(npz_path).stem
            motion_dir = output_dir / f"motion_{stem}"
            motion_files = list(motion_dir.glob("*.motion")) if motion_dir.exists() else []
            if motion_files:
                print(f"  SKIP [{stem}] already has {motion_files[0].name}")
            else:
                to_process.append(npz_path)
        npz_files = to_process

    if args.max_motions:
        npz_files = npz_files[:args.max_motions]

    total = len(npz_files)
    print(f"\n{'='*60}")
    print(f"  Parallel Batch Retargeting (PyRoki V6)")
    print(f"{'='*60}")
    print(f"  NPZ files:     {total}")
    print(f"  Workers:        {args.workers}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Skip existing:  {args.skip_existing}")
    print(f"  FPS:            {args.fps}")
    print(f"{'='*60}\n")

    if args.dry_run:
        for npz_path in npz_files:
            stem = pathlib.Path(npz_path).stem
            print(f"  Would process: {stem}")
        print(f"\n  Total: {total} motions with {args.workers} workers")
        print(f"  Estimated time: ~{total * 60 / args.workers:.0f} minutes")
        return

    t_start = time.time()

    # Run in parallel
    results = []
    successes = 0
    failures = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for npz_path in npz_files:
            future = executor.submit(
                process_single_npz,
                npz_path, str(output_dir),
                args.keep_intermediates, args.fps
            )
            futures[future] = npz_path

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["status"] == "success":
                successes += 1
            else:
                failures += 1

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  Batch Complete!")
    print(f"{'='*60}")
    print(f"  Total:     {total}")
    print(f"  Success:   {successes}")
    print(f"  Failed:    {failures}")
    print(f"  Time:      {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Avg/motion: {total_time/max(total,1):.0f}s")
    print(f"  Output:    {output_dir}")

    # List failures
    failed = [r for r in results if r["status"] != "success"]
    if failed:
        print(f"\n  Failed motions:")
        for r in failed:
            print(f"    {r['stem']}: {r['status']} - {r.get('error', r.get('stderr', '')[:100])}")

    # Save report
    import json
    report_path = output_dir / "batch_retarget_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": total,
            "success": successes,
            "failed": failures,
            "time_s": round(total_time, 1),
            "workers": args.workers,
            "results": results,
        }, f, indent=2)
    print(f"  Report:    {report_path}")


if __name__ == "__main__":
    main()
