#!/usr/bin/env python3
"""Post-process eval results: run MotionQualityChecker + phys_metrics on output.npz.

Writes quality.json with:
  - is_valid / failed_checks / borderline_checks (from MotionQualityChecker)
  - physical_metrics: FK-based physical plausibility metrics (from phys_metrics.py)

Uses multiprocessing for speed (~10x faster than serial).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMPL_MODEL_PATH = (
    "/apdcephfs_cq11/share_1467498/home/dkang/codes/"
    "MoreDiff-Data/motion_process/body_model/smplh/neutral/model.npz"
)
if not os.path.exists(SMPL_MODEL_PATH):
    SMPL_MODEL_PATH = None

PHYS_KEYS = [
    "jerk_with_rot", "local_pose_jerk", "pelvis_trans_jerk",
    "tremor_ratio", "snap_ratio", "joint_pop_ratio", "bone_length_cv_mean",
    "avg_skate", "frame_avg_skate", "skate_ratio",
    "avg_penetrate", "avg_float", "phys_err",
]


def process_one(case_dir_str: str) -> str | None:
    """Process a single case. Returns None on success, error string on failure."""
    case_dir = Path(case_dir_str)
    output_npz = case_dir / "output.npz"
    quality_json = case_dir / "quality.json"

    try:
        from hftrainer.evaluation.quality_check_rules.motion_quality_checker import (
            MotionQualityChecker,
        )
        from hftrainer.evaluation.motion.phys_metrics import compute_phys_metrics

        checker = MotionQualityChecker(device="cpu")
        result = checker.check_from_file(str(output_npz))

        phys = compute_phys_metrics(
            str(output_npz),
            smpl_model_path=SMPL_MODEL_PATH,
            device="cpu",
            use_cache=False,
            output_unit="cm",
        )

        physical = {}
        for key in PHYS_KEYS:
            v = phys.get(key)
            if v is not None and isinstance(v, (int, float)):
                physical[key] = round(float(v), 6)

        fs_details = result.all_results.get("foot_sliding", {}).get("details", {})
        sliding_frames = int(fs_details.get("total_sliding_frames", 0))
        physical["foot_sliding_frames"] = sliding_frames
        meta_path = case_dir / "meta.json"
        if meta_path.is_file():
            with open(meta_path) as f:
                meta = json.load(f)
            num_frames = meta.get("num_frames", 0)
            if num_frames > 0:
                physical["foot_sliding_ratio"] = round(sliding_frames / num_frames, 4)

        qd = {
            "is_valid": result.is_valid,
            "failed_checks": result.failed_checks,
            "borderline_checks": result.borderline_checks,
            "physical_metrics": physical,
        }
        with open(quality_json, "w") as f:
            json.dump(qd, f, indent=2)
        return None
    except Exception as e:
        return f"{case_dir.parent.parent.name}/{case_dir.parent.name}/{case_dir.name}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="eval_results/m2m")
    parser.add_argument("--tasks", nargs="*", default=None, help="Tasks to process (default: all)")
    parser.add_argument("--force", action="store_true", help="Recompute existing quality.json")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    tasks = args.tasks or sorted(
        d for d in os.listdir(eval_dir)
        if os.path.isdir(eval_dir / d) and d.startswith("T")
    )

    # Collect all case dirs to process
    jobs = []
    skipped = 0
    for task in tasks:
        task_dir = eval_dir / task
        if not task_dir.is_dir():
            continue
        for model in sorted(os.listdir(task_dir)):
            model_dir = task_dir / model
            if not model_dir.is_dir():
                continue
            for case_id in sorted(os.listdir(model_dir)):
                case_dir = model_dir / case_id
                output_npz = case_dir / "output.npz"
                quality_json = case_dir / "quality.json"
                if not output_npz.is_file():
                    continue
                if quality_json.is_file() and not args.force:
                    skipped += 1
                    continue
                jobs.append(str(case_dir))

    print(f"Jobs: {len(jobs)} to process, {skipped} skipped, workers={args.workers}")
    if not jobs:
        print("Nothing to do.")
        return

    completed = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, j): j for j in jobs}
        for future in as_completed(futures):
            completed += 1
            err = future.result()
            if err:
                errors += 1
                if errors <= 10:
                    print(f"  ERROR: {err}")
            if completed % 200 == 0 or completed == len(jobs):
                print(f"  Progress: {completed}/{len(jobs)} ({errors} errors)")

    print(f"Quality check done: {completed} computed, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
