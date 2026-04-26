#!/usr/bin/env python3
"""Parallel globalrot repair v2: MoGenDIT-aligned adaptive denoise.

2 configs × 4 GPUs each = 8 workers on 8 GPUs.
No MoGenDIT dependency — mask computed in M2M normalized space.

Usage:
    python3 scripts/eval_globalrot_repair_parallel_v2.py
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
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--denoise-strength", type=float, default=0.02)
    parser.add_argument("--change-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-list", type=str,
                        default="data/hymotion_m2m_refine_data/data_quality_list/low_quality.json")
    parser.add_argument("--data-root", type=str, default="data/hymotion_data")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or str(PROJECT_ROOT / "output" / f"globalrot_ada_repair_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",")]
    assert len(gpus) >= 8, f"Need 8 GPUs, got {len(gpus)}"

    with open(args.quality_list) as f:
        total_items = len(json.load(f).get("items", []))

    configs = ["uncond_fm_man_globalrot", "uncond_jit_man_globalrot"]
    shards_per_config = 4
    shard_size = (total_items + shards_per_config - 1) // shards_per_config

    print(f"Output: {output_dir}")
    print(f"Total samples: {total_items}, shard size: ~{shard_size}")
    print(f"denoise_strength={args.denoise_strength}, denoise_steps={args.denoise_steps}, "
          f"change_threshold={args.change_threshold}")
    print()

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
                str(PROJECT_ROOT / "scripts" / "_eval_globalrot_single_v2.py"),
                "--config", cfg_name,
                "--denoise-steps", str(args.denoise_steps),
                "--denoise-strength", str(args.denoise_strength),
                "--change-threshold", str(args.change_threshold),
                "--seed", str(args.seed),
                "--quality-list", args.quality_list,
                "--data-root", args.data_root,
                "--output-dir", output_dir,
                "--start-idx", str(start_idx),
                "--end-idx", str(end_idx),
            ]

            log_path = os.path.join(output_dir, f"{cfg_name}_shard{si}_gpu{gpu_id}.log")
            log_f = open(log_path, "w")
            print(f"  GPU {gpu_id}: {cfg_name} [{start_idx}:{end_idx}]")
            p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            procs.append((cfg_name, si, gpu_id, start_idx, end_idx, p, log_f, log_path))

    print(f"\nWaiting for {len(procs)} workers...")
    print(f"Monitor: tail -f {output_dir}/*.log\n")

    for cfg_name, si, gpu_id, start_idx, end_idx, p, log_f, log_path in procs:
        retcode = p.wait()
        log_f.close()
        status = "✓" if retcode == 0 else "✗"
        print(f"  {status} GPU {gpu_id}: {cfg_name} shard{si} [{start_idx}:{end_idx}] (exit={retcode})")

    # Merge stats
    print(f"\n{'='*70}")
    print("MERGING RESULTS")
    print(f"{'='*70}")

    for cfg_name in configs:
        mode_label = f"{cfg_name}_ada_denoise"
        mode_dir = Path(output_dir) / mode_label

        shard_stats = []
        for si in range(shards_per_config):
            s_start = si * shard_size
            s_end = min((si + 1) * shard_size, total_items)
            sp = mode_dir / f"repair_stats_{s_start}_{s_end}.json"
            if sp.is_file():
                with open(sp) as f:
                    shard_stats.append(json.load(f))

        if not shard_stats:
            print(f"  {cfg_name}: NO STATS")
            continue

        merged = {
            "config": cfg_name, "mode": "ada_denoise",
            "num_shards": len(shard_stats),
            "total": sum(s["total"] for s in shard_stats),
            "processed": sum(s["processed"] for s in shard_stats),
            "skipped": sum(s["skipped"] for s in shard_stats),
            "before_pass": sum(s["before_pass"] for s in shard_stats),
            "after_pass": sum(s["after_pass"] for s in shard_stats),
            "improved": sum(s["improved"] for s in shard_stats),
            "degraded": sum(s["degraded"] for s in shard_stats),
            "unchanged_pass": sum(s.get("unchanged_pass", 0) for s in shard_stats),
            "unchanged_fail": sum(s.get("unchanged_fail", 0) for s in shard_stats),
        }

        pft = defaultdict(lambda: {"total": 0, "fixed": 0, "still_fail": 0})
        for s in shard_stats:
            for fc, fs in s.get("per_failure_type", {}).items():
                pft[fc]["total"] += fs.get("total", 0)
                pft[fc]["fixed"] += fs.get("fixed", 0)
                pft[fc]["still_fail"] += fs.get("still_fail", 0)
        merged["per_failure_type"] = dict(pft)

        merged_path = mode_dir / "repair_stats_merged.json"
        with open(merged_path, "w") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2, default=str)

        processed = max(merged["processed"], 1)
        improved = merged["improved"]
        degraded = merged["degraded"]

        print(f"\n  {cfg_name}:")
        print(f"    Processed: {merged['processed']}/{merged['total']}")
        print(f"    Improved:  {improved} ({improved/processed*100:.1f}%)")
        print(f"    Degraded:  {degraded}")
        print(f"    After pass: {merged['after_pass']} ({merged['after_pass']/processed*100:.1f}%)")
        print(f"    Per failure type:")
        for fc, fs in sorted(merged["per_failure_type"].items()):
            t = fs["total"]; fx = fs["fixed"]
            print(f"      {fc}: {fx}/{t} fixed ({fx/max(t,1)*100:.1f}%)")

    print(f"\n{'='*70}")
    print(f"ALL RESULTS: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
