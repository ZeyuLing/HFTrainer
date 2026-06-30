#!/usr/bin/env python3
"""Launch one detached SONIC Table-2 evaluation shard on a Taiji worker."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_ROOT = Path("/apdcephfs_zwfy7/share_305994131/home/zeyuling/hf_trainer")
DEFAULT_SPLITS = "lafan1_fixed600 amass_test_fixed600 wild_clean_fixed600"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--total-shards", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--force-eval", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    script = root / "scripts/embodied/run_table2_sonic_eval_shards.sh"
    if not script.exists():
        print(f"missing runner: {script}", file=sys.stderr)
        return 2

    logs = root / "outputs/evaluation/physflow/table2_tracker/unified_protocol_v1/runs/sonic/logs"
    logs.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "GPU_ID": str(args.gpu_id),
            "SPLITS": args.splits,
            "TOTAL_SHARDS": str(args.total_shards),
            "SHARD_ID": str(args.shard_id),
            "FORCE_EVAL": str(args.force_eval),
        }
    )

    log_path = logs / f"launch_{args.host_label}_shard{args.shard_id}.out"
    pid_path = logs / f"pid_{args.host_label}_shard{args.shard_id}.txt"
    log = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        ["bash", str(script)],
        cwd=root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(f"{proc.pid}\n")
    print(f"launched {args.host_label} shard={args.shard_id}/{args.total_shards} pid={proc.pid} log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
