#!/usr/bin/env python3
"""Parallel globalrot repair: 2 configs × 4 GPUs each = 8 workers on 8 GPUs.

Each config's data is split into 4 equal shards, one per GPU.
After all workers finish, merges per-shard stats into a unified report.

Usage:
    python3 scripts/eval_globalrot_repair_parallel.py
    python3 scripts/eval_globalrot_repair_parallel.py --num-steps 50
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--mogendit-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-list", type=str,
                        default="data/hymotion_m2m_refine_data/data_quality_list/low_quality.json")
    parser.add_argument("--data-root", type=str, default="data/hymotion_data")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or str(PROJECT_ROOT / "output" / f"globalrot_repair_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",")]
    assert len(gpus) >= 8, f"Need 8 GPUs, got {len(gpus)}"

    # Count total items
    with open(args.quality_list) as f:
        quality_data = json.load(f)
    total_items = len(quality_data.get("items", []))

    configs = ["uncond_fm_man_globalrot", "uncond_jit_man_globalrot"]
    shards_per_config = 4  # 4 GPUs per config

    shard_size = (total_items + shards_per_config - 1) // shards_per_config

    print(f"Output: {output_dir}")
    print(f"Total samples: {total_items}")
    print(f"Configs: {configs}")
    print(f"Shards per config: {shards_per_config}, shard size: ~{shard_size}")
    print()

    # Launch workers: config0 on GPU 0-3, config1 on GPU 4-7
    procs = []
    for ci, cfg_name in enumerate(configs):
        for si in range(shards_per_config):
            gpu_idx = ci * shards_per_config + si
            gpu_id = gpus[gpu_idx]

            start_idx = si * shard_size
            end_idx = min((si + 1) * shard_size, total_items)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "_eval_globalrot_single.py"),
                "--config", cfg_name,
                "--num-steps", str(args.num_steps),
                "--mogendit-steps", str(args.mogendit_steps),
                "--seed", str(args.seed),
                "--quality-list", args.quality_list,
                "--data-root", args.data_root,
                "--output-dir", output_dir,
                "--start-idx", str(start_idx),
                "--end-idx", str(end_idx),
            ]

            log_path = os.path.join(output_dir, f"{cfg_name}_shard{si}_gpu{gpu_id}.log")
            log_f = open(log_path, "w")
            print(f"  GPU {gpu_id}: {cfg_name} [{start_idx}:{end_idx}] -> {log_path}")
            p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            procs.append((cfg_name, si, gpu_id, start_idx, end_idx, p, log_f, log_path))

    print(f"\nWaiting for {len(procs)} workers to finish...")
    print(f"Monitor logs: tail -f {output_dir}/*.log\n")

    # Wait for all workers
    for cfg_name, si, gpu_id, start_idx, end_idx, p, log_f, log_path in procs:
        retcode = p.wait()
        log_f.close()
        status = "✓" if retcode == 0 else "✗"
        print(f"  {status} GPU {gpu_id}: {cfg_name} shard{si} [{start_idx}:{end_idx}] (exit={retcode})")

    # Merge shard stats into per-config summaries
    print(f"\n{'='*70}")
    print("MERGING RESULTS")
    print(f"{'='*70}")

    for cfg_name in configs:
        mode_label = f"{cfg_name}_inpaint_impute"
        mode_dir = Path(output_dir) / mode_label

        # Find all shard stats
        shard_stats = []
        for si in range(shards_per_config):
            start_idx = si * shard_size
            end_idx = min((si + 1) * shard_size, total_items)
            stats_path = mode_dir / f"repair_stats_{start_idx}_{end_idx}.json"
            if stats_path.is_file():
                with open(stats_path) as f:
                    shard_stats.append(json.load(f))

        if not shard_stats:
            print(f"  {cfg_name}: NO STATS FOUND")
            continue

        # Merge
        merged = {
            "config": cfg_name,
            "mode": "inpaint",
            "rotation_space": "global",
            "num_shards": len(shard_stats),
            "total": sum(s["total"] for s in shard_stats),
            "processed": sum(s["processed"] for s in shard_stats),
            "skipped": sum(s["skipped"] for s in shard_stats),
            "before_pass": sum(s["before_pass"] for s in shard_stats),
            "after_pass": sum(s["after_pass"] for s in shard_stats),
            "improved": sum(s["improved"] for s in shard_stats),
            "degraded": sum(s["degraded"] for s in shard_stats),
            "unchanged_pass": sum(s["unchanged_pass"] for s in shard_stats),
            "unchanged_fail": sum(s["unchanged_fail"] for s in shard_stats),
        }

        # Merge per_failure_type
        pft = defaultdict(lambda: {"total": 0, "fixed": 0, "still_fail": 0})
        for s in shard_stats:
            for fc, fstats in s.get("per_failure_type", {}).items():
                pft[fc]["total"] += fstats.get("total", 0)
                pft[fc]["fixed"] += fstats.get("fixed", 0)
                pft[fc]["still_fail"] += fstats.get("still_fail", 0)
        merged["per_failure_type"] = dict(pft)

        # Merge MPJPE
        all_mpjpe = []
        for s in shard_stats:
            all_mpjpe.extend(s.get("mpjpe_unmasked_list", []))
        merged["mpjpe_unmasked_mean"] = float(np.mean(all_mpjpe)) if all_mpjpe else None
        merged["mpjpe_unmasked_std"] = float(np.std(all_mpjpe)) if all_mpjpe else None

        # Merge errors
        all_errors = []
        for s in shard_stats:
            all_errors.extend(s.get("errors", []))
        merged["errors"] = all_errors[:50]  # cap

        # Save merged
        merged_path = mode_dir / "repair_stats_merged.json"
        with open(merged_path, "w") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2, default=str)

        # Print summary
        processed = max(merged["processed"], 1)
        improved = merged["improved"]
        degraded = merged["degraded"]
        mpjpe = merged["mpjpe_unmasked_mean"]

        print(f"\n  {cfg_name}:")
        print(f"    Processed: {merged['processed']}/{merged['total']} (skipped {merged['skipped']})")
        print(f"    Improved:  {improved} ({improved/processed*100:.1f}%)")
        print(f"    Degraded:  {degraded}")
        print(f"    After pass: {merged['after_pass']} ({merged['after_pass']/processed*100:.1f}%)")
        if mpjpe:
            print(f"    MPJPE (unmasked): {mpjpe:.6f} ± {merged['mpjpe_unmasked_std']:.6f}")
        print(f"    Per failure type:")
        for fc, fstats in sorted(merged["per_failure_type"].items()):
            t = fstats["total"]
            fx = fstats["fixed"]
            print(f"      {fc}: {fx}/{t} fixed ({fx/max(t,1)*100:.1f}%)")
        print(f"    Merged stats: {merged_path}")

    print(f"\n{'='*70}")
    print(f"ALL RESULTS: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
