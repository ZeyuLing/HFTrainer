#!/usr/bin/env python3
"""Multi-GPU parallel M2M repair evaluation.

Launches one process per (config, mode) pair, each on a separate GPU.
MoGenDiT adaptive mask is computed per-sample inline (no separate phase).

Usage:
    python3 scripts/eval_m2m_repair_parallel.py --max-samples 200 --num-steps 50
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--mogendit-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-list", type=str,
                        default="data/hymotion_m2m_refine_data/data_quality_list/low_quality.json")
    parser.add_argument("--data-root", type=str, default="data/hymotion_data")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=["uncond_fm", "uncond_fm_man",
                                 "uncond_jit", "uncond_jit_man",
                                 "caption_fm", "caption_fm_man",
                                 "caption_jit", "caption_jit_man"])
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs to use")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or str(PROJECT_ROOT / "output" / f"m2m_repair_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",")]
    # Jobs: (config, mode) pairs
    jobs = []
    for cfg_name in args.configs:
        for mode in ["inpaint", "edit"]:
            jobs.append((cfg_name, mode))

    print(f"Output: {output_dir}")
    print(f"Jobs: {len(jobs)}, GPUs: {len(gpus)}")
    print(f"Samples: {args.max_samples}, Steps: {args.num_steps}")

    # Launch one subprocess per job
    procs = []
    for i, (cfg_name, mode) in enumerate(jobs):
        gpu_id = gpus[i % len(gpus)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "_eval_m2m_single.py"),
            "--config", cfg_name,
            "--mode", mode,
            "--max-samples", str(args.max_samples),
            "--num-steps", str(args.num_steps),
            "--mogendit-steps", str(args.mogendit_steps),
            "--seed", str(args.seed),
            "--quality-list", args.quality_list,
            "--data-root", args.data_root,
            "--output-dir", output_dir,
        ]
        print(f"  GPU {gpu_id}: {cfg_name}_{mode}")
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((cfg_name, mode, gpu_id, p))

    # Wait and collect
    for cfg_name, mode, gpu_id, p in procs:
        retcode = p.wait()
        output = p.stdout.read().decode(errors="replace")
        label = f"{cfg_name}_{mode}"
        log_path = os.path.join(output_dir, f"{label}_gpu{gpu_id}.log")
        with open(log_path, "w") as f:
            f.write(output)
        status = "✓" if retcode == 0 else "✗"
        print(f"  {status} GPU {gpu_id}: {label} (exit={retcode}, log={log_path})")

    # Print summary from stats files
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for cfg_name, mode, _, _ in procs:
        is_man = "man" in cfg_name
        label = f"{cfg_name}_{mode}" + ("_impute" if is_man else "")
        stats_path = os.path.join(output_dir, label, "repair_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                s = json.load(f)
            processed = max(s.get("processed", 0), 1)
            improved = s.get("improved", 0)
            degraded = s.get("degraded", 0)
            mpjpe = s.get("mpjpe_unmasked_mean")
            print(f"  {label}: processed={s.get('processed',0)}/{s.get('total',0)}, "
                  f"improved={improved} ({improved/processed*100:.1f}%), "
                  f"degraded={degraded}, "
                  f"keep_MAE={mpjpe:.4f}" if mpjpe else f"  {label}: no stats")
        else:
            print(f"  {label}: stats not found")

    print(f"\nAll results: {output_dir}")


if __name__ == "__main__":
    main()
