#!/usr/bin/env python3
"""
Run MoGenDIT ada_denoise repair on the same 100 T7 eval cases for comparison
with M2M repair results.

Output goes into: output/eval_results/m2m/T7/mogendit_ada_denoise/case_XXXX/
  - output.npz: repaired motion
  - gt.npz: symlink to original gt
  - meta.json: metrics + comparison info

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_mogendit_t7.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_ROOT = PROJECT_ROOT.parent / "output" / "eval_results" / "m2m" / "T7"
REF_CONFIG = "uncond_fm_man"  # use this config's cases as reference
DATA_ROOT = PROJECT_ROOT / "data" / "hymotion_data"


def compute_metrics(gt_npz_path: str, output_npz_path: str):
    """Compute trans_err and rot_err between gt and output."""
    gt = np.load(gt_npz_path, allow_pickle=True)
    out = np.load(output_npz_path, allow_pickle=True)

    gt_trans = gt["trans"].astype(np.float64)
    out_trans = out["trans"].astype(np.float64)
    T = min(len(gt_trans), len(out_trans))
    gt_trans, out_trans = gt_trans[:T], out_trans[:T]

    trans_err_mm = float(np.mean(np.linalg.norm(gt_trans - out_trans, axis=-1)) * 1000)

    gt_poses = gt["poses"].astype(np.float64)[:T]
    out_poses = out["poses"].astype(np.float64)[:T]
    rot_err = float(np.mean(np.abs(gt_poses - out_poses)))

    return {
        "trans_err_mm": round(trans_err_mm, 2),
        "rot_err": round(rot_err, 6),
    }


def main():
    # Collect case list from reference config
    ref_dir = EVAL_ROOT / REF_CONFIG
    case_dirs = sorted([d for d in ref_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    print(f"[INFO] Found {len(case_dirs)} T7 cases from {REF_CONFIG}")

    # Collect motion paths
    cases = []
    for case_dir in case_dirs:
        meta_path = case_dir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        gt_path = case_dir / "gt.npz"
        if not gt_path.exists():
            continue
        cases.append({
            "case_id": case_dir.name,
            "gt_path": str(gt_path),
            "motion_path": meta.get("motion_path", ""),
            "num_frames": meta.get("num_frames", 0),
            "fps": meta.get("fps", 30),
        })

    print(f"[INFO] {len(cases)} cases with gt.npz")

    # Init MoGenDIT
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline

    print("[INFO] Initializing MoGenDIT pipeline...")
    t0 = time.time()
    pipeline = MoGenDITRepairPipeline(
        model_name="MoreDiff-0.1B",
        device="cuda:0",
        use_ema=True,
    )
    print(f"[INFO] Pipeline ready in {time.time() - t0:.1f}s")

    # Output directory
    out_config = "mogendit_ada_denoise"
    out_base = EVAL_ROOT / out_config

    success = 0
    errors = 0

    for i, case in enumerate(cases):
        case_id = case["case_id"]
        case_out_dir = out_base / case_id
        output_npz = case_out_dir / "output.npz"
        gt_link = case_out_dir / "gt.npz"
        meta_out = case_out_dir / "meta.json"

        # Skip if already done
        if meta_out.exists():
            success += 1
            continue

        case_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{len(cases)}] {case_id} ({case['num_frames']} frames)...", end=" ", flush=True)
        t_start = time.time()

        try:
            # Symlink gt
            if not gt_link.exists():
                os.symlink(case["gt_path"], str(gt_link))

            # Run ada_denoise repair
            pipeline.repair_npz(
                input_path=case["gt_path"],
                output_path=str(output_npz),
                mode="ada_denoise",
                step=50,
                use_windowed=True,
                window_size=224,
                prev_padding=20,
            )

            # Compute metrics
            metrics = compute_metrics(case["gt_path"], str(output_npz))
            elapsed = time.time() - t_start

            meta = {
                "task": "T7",
                "setting": "T7-A",
                "config": out_config,
                "motion_path": case["motion_path"],
                "num_frames": case["num_frames"],
                "fps": case["fps"],
                "num_steps": 50,
                "mask_ratio": 1.0,  # MoGenDIT processes entire motion
                "metrics": metrics,
                "elapsed_sec": round(elapsed, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            with open(meta_out, "w") as f:
                json.dump(meta, f, indent=2)

            success += 1
            print(f"trans_err={metrics['trans_err_mm']:.1f}mm  ({elapsed:.1f}s)")

        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"[DONE] {success} success, {errors} errors out of {len(cases)} cases")
    print(f"Output: {out_base}")


if __name__ == "__main__":
    main()
