#!/usr/bin/env python3
"""Parallel globalrot repair v3: MoGenDIT mask + M2M denoise-impute."""

import argparse, json, os, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--denoise-strength", type=float, default=0.5)
    parser.add_argument("--mogendit-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-list", type=str,
                        default="data/hymotion_m2m_refine_data/data_quality_list/low_quality.json")
    parser.add_argument("--data-root", type=str, default="data/hymotion_data")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=["uncond_fm_man_globalrot", "uncond_jit_man_globalrot"])
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or str(PROJECT_ROOT / "output" / f"globalrot_repair_v3_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",")]
    with open(args.quality_list) as f:
        total_items = len(json.load(f).get("items", []))

    configs = args.configs
    shards_per_config = len(gpus) // len(configs)
    shard_size = (total_items + shards_per_config - 1) // shards_per_config

    print(f"Output: {output_dir}")
    print(f"Total: {total_items}, Configs: {configs}, Shards/config: {shards_per_config}")
    print(f"denoise_strength={args.denoise_strength}, steps={args.denoise_steps}")
    print()

    procs = []
    for ci, cfg in enumerate(configs):
        for si in range(shards_per_config):
            gi = ci * shards_per_config + si
            gpu_id = gpus[gi]
            s = si * shard_size
            e = min((si + 1) * shard_size, total_items)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "_eval_globalrot_single_v3.py"),
                   "--config", cfg,
                   "--denoise-steps", str(args.denoise_steps),
                   "--denoise-strength", str(args.denoise_strength),
                   "--mogendit-steps", str(args.mogendit_steps),
                   "--seed", str(args.seed),
                   "--quality-list", args.quality_list,
                   "--data-root", args.data_root,
                   "--output-dir", output_dir,
                   "--start-idx", str(s), "--end-idx", str(e)]
            log = os.path.join(output_dir, f"{cfg}_shard{si}_gpu{gpu_id}.log")
            lf = open(log, "w")
            print(f"  GPU {gpu_id}: {cfg} [{s}:{e}]")
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            procs.append((cfg, si, gpu_id, s, e, p, lf))

    print(f"\nWaiting for {len(procs)} workers...\n")
    for cfg, si, gpu_id, s, e, p, lf in procs:
        rc = p.wait(); lf.close()
        print(f"  {'✓' if rc==0 else '✗'} GPU {gpu_id}: {cfg} shard{si} (exit={rc})")

    # Merge
    print(f"\n{'='*70}\nMERGING\n{'='*70}")
    for cfg in configs:
        ml = f"{cfg}_denoise_impute"
        md = Path(output_dir) / ml
        ss = []
        for si in range(shards_per_config):
            sp = md / f"repair_stats_{si*shard_size}_{min((si+1)*shard_size,total_items)}.json"
            if sp.is_file():
                with open(sp) as f: ss.append(json.load(f))
        if not ss: print(f"  {cfg}: NO STATS"); continue
        m = {k: sum(s.get(k,0) for s in ss) for k in
             ["total","processed","skipped","before_pass","after_pass","improved","degraded","unchanged_pass","unchanged_fail"]}
        m["config"] = cfg
        pft = defaultdict(lambda: {"total":0,"fixed":0,"still_fail":0})
        for s in ss:
            for fc, fs in s.get("per_failure_type",{}).items():
                for k in pft[fc]: pft[fc][k] += fs.get(k,0)
        m["per_failure_type"] = dict(pft)
        mp = md / "repair_stats_merged.json"
        with open(mp,"w") as f: json.dump(m,f,indent=2,default=str)
        p = max(m["processed"],1)
        print(f"\n  {cfg}:")
        print(f"    Processed: {m['processed']}/{m['total']}")
        print(f"    Improved: {m['improved']} ({m['improved']/p*100:.1f}%)")
        print(f"    Degraded: {m['degraded']}")
        print(f"    After pass: {m['after_pass']} ({m['after_pass']/p*100:.1f}%)")
        for fc, fs in sorted(m["per_failure_type"].items()):
            t=fs["total"]; fx=fs["fixed"]
            print(f"      {fc}: {fx}/{t} ({fx/max(t,1)*100:.1f}%)")
    print(f"\nALL RESULTS: {output_dir}")

if __name__ == "__main__":
    main()
