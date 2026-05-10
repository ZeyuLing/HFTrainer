#!/usr/bin/env python3
"""Single-GPU worker for M2M repair eval. Called by eval_m2m_repair_parallel.py.

Computes MoGenDiT adaptive mask and M2M repair inline per sample (no separate phase).
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import seaborn  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "seaborn"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--mode", type=str, required=True, choices=["inpaint", "edit"])
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--mogendit-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quality-list", type=str)
    p.add_argument("--data-root", type=str)
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


# Import shared utilities from eval_m2m_repair
from scripts.eval_m2m_repair import (
    CONFIG_PATHS, WORK_DIR_NAMES,
    load_npz_as_motion, motion_135_to_npz_format, save_repaired_npz,
    adaptive_mask_to_dense,
    build_model, build_mogendit,
    repair_single,
    get_checker, check_npz,
    compute_mpjpe_unmasked,
)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    config_name = args.config
    edit_mode = args.mode == "edit"
    is_man = "man" in config_name
    mode_label = f"{config_name}_{args.mode}" + ("_impute" if is_man else "")

    output_dir = Path(args.output_dir)
    mode_output_dir = output_dir / mode_label
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{mode_label}] Starting on {device}")

    # Load quality list
    with open(args.quality_list) as f:
        quality_data = json.load(f)
    data_root = Path(args.data_root)
    items = quality_data.get("items", [])
    if args.max_samples > 0:
        items = items[:args.max_samples]

    # Build MoGenDiT (for adaptive mask)
    print(f"[{mode_label}] Loading MoGenDiT...")
    mogendit = build_mogendit(device)

    # Build M2M model
    print(f"[{mode_label}] Loading M2M model: {config_name}...")
    pipeline, bundle, ckpt_path, _ = build_model(config_name, device, args.num_steps)

    stats = {
        "config": config_name, "mode": args.mode, "is_man": is_man,
        "checkpoint": ckpt_path,
        "replacement_guidance": pipeline.replacement_guidance,
        "num_steps": args.num_steps,
        "total": 0, "processed": 0, "skipped": 0, "errors": [],
        "before_pass": 0, "after_pass": 0,
        "improved": 0, "degraded": 0, "unchanged_pass": 0, "unchanged_fail": 0,
        "per_failure_type": defaultdict(lambda: {"total": 0, "fixed": 0, "still_fail": 0}),
        "mpjpe_unmasked_list": [], "details": [],
    }

    for idx, item in enumerate(items):
        rel_path = item["path"]
        npz_path = str(data_root / rel_path)
        stats["total"] += 1

        if not os.path.isfile(npz_path):
            stats["skipped"] += 1
            stats["errors"].append({"path": rel_path, "error": "file not found"})
            continue

        try:
            t0 = time.time()

            # 1. Load motion
            motion_135, num_frames, fps, abs_trans_frame0 = load_npz_as_motion(npz_path)

            # 2. Compute adaptive mask inline
            try:
                ada = mogendit.compute_adaptive_mask(
                    npz_path, step=args.mogendit_steps,
                    joint_threshold=0.15, trans_threshold=0.05,
                    max_mask_ratio=0.15,
                )
            except Exception as e:
                stats["skipped"] += 1
                stats["errors"].append({"path": rel_path, "error": f"adaptive mask: {str(e)[:100]}"})
                continue

            mask_135 = adaptive_mask_to_dense(
                ada['joint_mask'], ada['trans_mask'],
                num_frames, temporal_dilate=5,
            )
            mask_ratio = mask_135.sum().item() / max(mask_135.numel(), 1)

            if mask_ratio < 0.001:
                stats["skipped"] += 1
                continue

            # Save adaptive mask for visualization tools
            ada_mask_out = output_dir / "adaptive_masks" / rel_path
            os.makedirs(os.path.dirname(str(ada_mask_out)) or ".", exist_ok=True)
            if not ada_mask_out.is_file():
                np.savez_compressed(
                    str(ada_mask_out),
                    joint_mask=ada["joint_mask"],
                    trans_mask=ada["trans_mask"],
                )

            # 3. Repair
            repaired_motion, repaired_raw = repair_single(
                pipeline, motion_135, mask_135, device, edit_mode=edit_mode,
            )

            # 4. Sanity check
            if torch.isnan(repaired_motion).any():
                stats["errors"].append({"path": rel_path, "error": "NaN in output"})
                stats["skipped"] += 1
                continue

            # 5. Save repaired NPZ
            repaired_aa, repaired_trans = motion_135_to_npz_format(repaired_motion, abs_trans_frame0)
            if np.isnan(repaired_trans).any() or np.abs(repaired_trans).max() > 20.0:
                stats["errors"].append({"path": rel_path, "error": f"trans extreme ({np.abs(repaired_trans).max():.1f})"})
                stats["skipped"] += 1
                continue

            out_npz = str(mode_output_dir / "repaired" / rel_path)
            orig_data = dict(np.load(npz_path, allow_pickle=True))
            save_repaired_npz(out_npz, repaired_aa, repaired_trans, orig_data, fps)

            # 6. Quality check
            before_failed = item.get("failed_checks", [])
            before_valid = len(before_failed) == 0
            after_valid, after_failed = check_npz(out_npz)

            elapsed = time.time() - t0
            stats["processed"] += 1

            if before_valid: stats["before_pass"] += 1
            if after_valid: stats["after_pass"] += 1
            if not before_valid and after_valid: stats["improved"] += 1
            elif before_valid and not after_valid: stats["degraded"] += 1
            elif after_valid: stats["unchanged_pass"] += 1
            else: stats["unchanged_fail"] += 1

            for fc in before_failed:
                stats["per_failure_type"][fc]["total"] += 1
                if after_valid: stats["per_failure_type"][fc]["fixed"] += 1
                else: stats["per_failure_type"][fc]["still_fail"] += 1

            mpjpe_um = compute_mpjpe_unmasked(motion_135, repaired_raw, mask_135)
            if mpjpe_um is not None:
                stats["mpjpe_unmasked_list"].append(mpjpe_um)

            detail = {
                "path": rel_path, "num_frames": num_frames,
                "mask_ratio": round(mask_ratio, 4), "mask_source": "adaptive",
                "before_failed": before_failed, "after_valid": after_valid,
                "after_failed": after_failed,
                "improved": not before_valid and after_valid,
                "mpjpe_unmasked": round(mpjpe_um, 6) if mpjpe_um is not None else None,
                "elapsed_s": round(elapsed, 2),
            }
            stats["details"].append(detail)

            # Incremental: append to JSONL for live viewing
            jsonl_path = mode_output_dir / "details_live.jsonl"
            with open(jsonl_path, "a") as jf:
                jf.write(json.dumps(detail, ensure_ascii=False) + "\n")

            status = "✓ FIXED" if detail["improved"] else ("✗ STILL BAD" if not after_valid else "= OK")
            if (idx + 1) % 10 == 0 or detail["improved"]:
                print(f"  [{idx+1}/{len(items)}] {status} | "
                      f"before={before_failed} after={after_failed} | "
                      f"mask={mask_ratio:.1%} | {elapsed:.1f}s")

        except Exception as e:
            stats["skipped"] += 1
            stats["errors"].append({"path": rel_path, "error": str(e)[:200]})
            continue

    # Summary
    processed = max(stats["processed"], 1)
    mpjpe_list = stats["mpjpe_unmasked_list"]
    mpjpe_mean = float(np.mean(mpjpe_list)) if mpjpe_list else None
    mpjpe_std = float(np.std(mpjpe_list)) if mpjpe_list else None

    print(f"\n{'='*60}")
    print(f"SUMMARY — {mode_label}")
    print(f"{'='*60}")
    print(f"Total:        {stats['total']}")
    print(f"Processed:    {stats['processed']}")
    print(f"Skipped:      {stats['skipped']}")
    print(f"Improved:     {stats['improved']} ({stats['improved']/processed*100:.1f}%)")
    print(f"Degraded:     {stats['degraded']}")
    if mpjpe_mean is not None:
        print(f"MPJPE (unmasked): {mpjpe_mean:.6f} ± {mpjpe_std:.6f}")

    stats["per_failure_type"] = dict(stats["per_failure_type"])
    stats["mpjpe_unmasked_mean"] = mpjpe_mean
    stats["mpjpe_unmasked_std"] = mpjpe_std

    stats_path = mode_output_dir / "repair_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
