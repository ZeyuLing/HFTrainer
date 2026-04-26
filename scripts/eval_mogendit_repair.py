#!/usr/bin/env python3
"""
MoGenDiT Baseline Repair Evaluation.

Runs MoGenDiT denoise and ada_denoise modes on the low-quality evaluation
samples, producing results in the same format as M2M repair eval so they
can be compared side-by-side in the web viewer.

Output goes into: output/<eval_dir>/mogendit_denoise/   (repaired/ + details_live.jsonl)
                  output/<eval_dir>/mogendit_ada_denoise/ (repaired/ + details_live.jsonl)

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_mogendit_repair.py \\
        --eval-dir m2m_repair_eval_latest_ckpt \\
        --max-samples 200 \\
        --denoise-step 50

The script:
1. Reads the sample list from an existing eval directory's adaptive_masks/
   or from the low_quality.json list
2. Runs MoGenDiT denoise + ada_denoise on each sample
3. Runs quality checker before/after
4. Writes results incrementally to details_live.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# MoGenDiT's import chain may pull in seaborn
try:
    import seaborn  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "seaborn"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="MoGenDiT Baseline Repair Evaluation")
    parser.add_argument("--eval-dir", type=str, required=True,
                        help="Eval directory name (e.g. m2m_repair_eval_latest_ckpt)")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--denoise-step", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-root", type=str,
                        default=str(PROJECT_ROOT / "data" / "hymotion_data"))
    parser.add_argument("--output-root", type=str,
                        default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--quality-list", type=str,
                        default=str(PROJECT_ROOT / "data" / "hymotion_m2m_refine_data"
                                    / "data_quality_list" / "low_quality.json"))
    parser.add_argument("--modes", type=str, nargs="+",
                        default=["denoise", "ada_denoise"],
                        choices=["denoise", "ada_denoise", "trans_regen"])
    parser.add_argument("--model-name", type=str, default="MoreDiff-0.1B")
    parser.add_argument("--window-size", type=int, default=224)
    parser.add_argument("--prev-padding", type=int, default=20)
    return parser.parse_args()


# ====================================================================
# Quality checker
# ====================================================================

_CHECKER_INSTANCE = None


def get_checker():
    global _CHECKER_INSTANCE
    if _CHECKER_INSTANCE is None:
        from hftrainer.evaluation.quality_check_rules import MotionQualityChecker
        _CHECKER_INSTANCE = MotionQualityChecker(device="cpu")
    return _CHECKER_INSTANCE


def check_npz(npz_path: str):
    """Run quality checker on NPZ. Returns (is_valid, failed_checks_list)."""
    try:
        checker = get_checker()
        result = checker.check_from_file(npz_path)
        result_dict = result.to_dict()
        return result_dict.get("is_valid", True), result_dict.get("failed_checks", [])
    except Exception as e:
        return False, [f"checker_error:{str(e)[:50]}"]


# ====================================================================
# Sample collection
# ====================================================================

def collect_samples(eval_dir_path: Path, quality_list_path: str,
                    data_root: str, max_samples: int) -> list[dict]:
    """Collect samples from either adaptive_masks dir or quality list."""
    samples = []

    # Try to collect from an existing config's details
    for sub in sorted(eval_dir_path.iterdir()):
        if not sub.is_dir() or sub.name.startswith("mogendit"):
            continue
        jsonl = sub / "details_live.jsonl"
        stats = sub / "repair_stats.json"
        if stats.is_file():
            with open(stats) as f:
                data = json.load(f)
            for detail in data.get("details", []):
                path = detail.get("path", "")
                if path:
                    samples.append({
                        "path": path,
                        "failed_checks": detail.get("before_failed", []),
                    })
            break
        elif jsonl.is_file():
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        detail = json.loads(line)
                        path = detail.get("path", "")
                        if path:
                            samples.append({
                                "path": path,
                                "failed_checks": detail.get("before_failed", []),
                            })
                    except json.JSONDecodeError:
                        continue
            break

    # Fallback: quality list
    if not samples and os.path.isfile(quality_list_path):
        with open(quality_list_path) as f:
            data = json.load(f)
        items = data.get("items", data) if isinstance(data, dict) else data
        for item in items:
            path = item.get("path", "")
            if path and os.path.isfile(os.path.join(data_root, path)):
                samples.append({
                    "path": path,
                    "failed_checks": item.get("failed_checks", []),
                })

    # Deduplicate
    seen = set()
    unique = []
    for s in samples:
        if s["path"] not in seen:
            seen.add(s["path"])
            unique.append(s)
    samples = unique

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    return samples


# ====================================================================
# Main
# ====================================================================

def main():
    args = parse_args()

    eval_dir_path = Path(args.output_root) / args.eval_dir
    if not eval_dir_path.is_dir():
        print(f"[ERROR] Eval directory not found: {eval_dir_path}")
        sys.exit(1)

    data_root = args.data_root

    print(f"[INFO] Collecting samples from {eval_dir_path}...")
    samples = collect_samples(eval_dir_path, args.quality_list, data_root, args.max_samples)
    print(f"[INFO] Found {len(samples)} samples")
    if not samples:
        print("[WARN] No samples found. Exiting.")
        return

    # Initialize MoGenDiT pipeline
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline

    print(f"[INFO] Initializing MoGenDiT pipeline: model={args.model_name}, device={args.device}")
    t0 = time.time()
    pipeline = MoGenDITRepairPipeline(
        model_name=args.model_name,
        device=args.device,
        use_ema=True,
    )
    print(f"[INFO] Pipeline initialized in {time.time() - t0:.1f}s")

    for mode in args.modes:
        print(f"\n{'='*60}")
        print(f"[INFO] Running MoGenDiT mode: {mode}")
        print(f"{'='*60}")

        config_name = f"mogendit_{mode}"
        config_dir = eval_dir_path / config_name
        repaired_dir = config_dir / "repaired"
        repaired_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = config_dir / "details_live.jsonl"

        # Check for already-processed samples (resume support)
        done_paths = set()
        if jsonl_path.is_file():
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        d = json.loads(line.strip())
                        if d.get("path"):
                            done_paths.add(d["path"])
                    except (json.JSONDecodeError, KeyError):
                        pass
            print(f"[INFO] Resuming: {len(done_paths)} already done")

        success = 0
        failed = 0

        for i, sample in enumerate(samples):
            rel_path = sample["path"]

            if rel_path in done_paths:
                success += 1
                continue

            input_path = os.path.join(data_root, rel_path)
            output_path = str(repaired_dir / rel_path)

            print(f"[{i+1}/{len(samples)}] {rel_path} ...", end=" ", flush=True)
            t_start = time.time()

            try:
                # Get frame count
                orig_data = np.load(input_path, allow_pickle=True)
                num_frames = int(orig_data.get("num_frames", orig_data["poses"].shape[0]))

                # Run quality check on original (before)
                before_valid, before_failed = check_npz(input_path)

                # Run MoGenDiT repair
                pipeline.repair_npz(
                    input_path=input_path,
                    output_path=output_path,
                    mode=mode,
                    step=args.denoise_step,
                    use_windowed=True,
                    window_size=args.window_size,
                    prev_padding=args.prev_padding,
                )

                # Run quality check on repaired (after)
                after_valid, after_failed = check_npz(output_path)

                elapsed = time.time() - t_start
                improved = (not before_valid) and after_valid

                detail = {
                    "path": rel_path,
                    "num_frames": num_frames,
                    "mask_ratio": 1.0,  # MoGenDiT processes entire motion
                    "mask_source": f"mogendit_{mode}",
                    "improved": improved,
                    "before_failed": before_failed,
                    "after_valid": after_valid,
                    "after_failed": after_failed,
                    "mpjpe_unmasked": None,  # not applicable for whole-motion repair
                    "elapsed_s": round(elapsed, 2),
                }

                # Write incrementally
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(detail) + "\n")

                status_str = "IMPROVED" if improved else ("STILL BAD" if not after_valid else "WAS OK")
                print(f"{status_str} ({elapsed:.1f}s) before={before_failed} after={after_failed}")
                success += 1

            except Exception as e:
                elapsed = time.time() - t_start
                print(f"FAILED ({elapsed:.1f}s): {e}")
                traceback.print_exc()
                failed += 1

        print(f"\n[INFO] Mode {mode} complete: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
