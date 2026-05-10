#!/usr/bin/env python3
"""Batch quality check for M2M eval outputs.

Runs MotionQualityChecker on all eval_results/m2m/<task>/<model>/<case>/output.npz
and saves quality.json alongside each.

Usage:
    python3 scripts/run_quality_check_m2m.py [--force] [--device cuda]
"""

import argparse
import json
import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from hftrainer.evaluation.quality_check_rules.motion_quality_checker import MotionQualityChecker


def main():
    parser = argparse.ArgumentParser(description="Batch quality check for M2M eval outputs")
    parser.add_argument("--eval-dir", default=os.path.join(PROJECT_ROOT, "eval_results", "m2m"),
                        help="M2M eval results directory")
    parser.add_argument("--device", default="cuda", help="Device for checker (default: cuda)")
    parser.add_argument("--force", action="store_true", help="Re-check even if quality.json exists")
    args = parser.parse_args()

    eval_dir = args.eval_dir
    if not os.path.isdir(eval_dir):
        print(f"Error: {eval_dir} not found")
        sys.exit(1)

    # Collect all output.npz paths
    targets = []
    for task in sorted(os.listdir(eval_dir)):
        task_dir = os.path.join(eval_dir, task)
        if not os.path.isdir(task_dir) or not task.startswith("T"):
            continue
        for model in sorted(os.listdir(task_dir)):
            model_dir = os.path.join(task_dir, model)
            if not os.path.isdir(model_dir):
                continue
            for case_id in sorted(os.listdir(model_dir)):
                case_dir = os.path.join(model_dir, case_id)
                npz_path = os.path.join(case_dir, "output.npz")
                quality_path = os.path.join(case_dir, "quality.json")
                if not os.path.isfile(npz_path):
                    continue
                if os.path.isfile(quality_path) and not args.force:
                    continue
                targets.append((npz_path, quality_path, f"{task}/{model}/{case_id}"))

    print(f"Found {len(targets)} output.npz files to check")
    if not targets:
        print("Nothing to do.")
        return

    # Initialize checker
    print(f"Initializing MotionQualityChecker on {args.device}...")
    checker = MotionQualityChecker(device=args.device)

    done = 0
    errors = 0
    t0 = time.time()

    for npz_path, quality_path, label in targets:
        try:
            result = checker.check_from_file(npz_path)
            with open(quality_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            done += 1
        except Exception as e:
            print(f"  ERROR {label}: {e}")
            errors += 1

        total = done + errors
        if total % 100 == 0:
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            print(f"  [{total}/{len(targets)}] {rate:.1f} cases/s, {errors} errors")

    elapsed = time.time() - t0
    print(f"\nDone: {done} checked, {errors} errors, {elapsed:.1f}s total")


if __name__ == "__main__":
    main()
