#!/usr/bin/env python3
"""Launch the four phase-1 HYMotion-TMR runs on an existing H20 Taiji task."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List


DEFAULT_TASK_FLAG = "task_zeyuling_20260701154254_2d380cde"
DEFAULT_INSTANCE_ID = "8b1d80079f17c734019f1cea93130715"
DEFAULT_REMOTE_ROOT = "/apdcephfs_zwfy7/share_305994131/home/zeyuling/hf_trainer"


RUNS: List[Dict[str, str]] = [
    {
        "run_name": "tmr_hymotion_g1_scene_clean_main",
        "representation": "g1_38d",
        "input_format": "g1",
        "anno": "data/annotation/train_g1_t2m_emb_minus_heldout_scene_clean.json",
        "max_items": "0",
        "max_epochs": "120",
    },
    {
        "run_name": "tmr_hymotion_g1_full_clean_ablation",
        "representation": "g1_38d",
        "input_format": "g1",
        "anno": "data/annotation/train_g1_t2m_emb_scene_clean.json",
        "max_items": "0",
        "max_epochs": "120",
    },
    {
        "run_name": "tmr_hymotion_smpl_or_kimodo_bridge",
        "representation": "smplx_pose159",
        "input_format": "raw_hymotion",
        "anno": "data/annotation/train_hymotion_400h_hq_20260403.json",
        "caption_source": "json",
        "max_items": "0",
        "max_epochs": "80",
    },
    {
        "run_name": "tmr_hymotion_g1_small_debug",
        "representation": "g1_38d",
        "input_format": "g1",
        "anno": "data/annotation/train_g1_t2m_emb_minus_heldout_scene_clean.json",
        "max_items": "4096",
        "max_epochs": "8",
    },
]


def shell_quote_env(env: Dict[str, str]) -> str:
    return " ".join("%s=%s" % (key, shlex.quote(value)) for key, value in env.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-flag", default=DEFAULT_TASK_FLAG)
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE_ID)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--hosts", default="9,10,11,12")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated run_name filter, e.g. tmr_hymotion_g1_small_debug.",
    )
    parser.add_argument("--num-gpus", default="8")
    parser.add_argument("--batch-size", default="128")
    parser.add_argument("--num-workers", default="8")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--taiji-exec-host", default="tools/taiji_exec_host.py")
    args = parser.parse_args()

    if not os.environ.get("TOKEN"):
        raise SystemExit("TOKEN is not set; export the Taiji token before launching.")

    runs = RUNS
    if args.only.strip():
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        runs = [run for run in RUNS if run["run_name"] in wanted]
        missing = wanted - {run["run_name"] for run in runs}
        if missing:
            raise SystemExit("unknown run_name(s): %s" % ", ".join(sorted(missing)))

    hosts = [int(x) for x in args.hosts.split(",") if x.strip()]
    if len(hosts) < len(runs):
        raise SystemExit("need at least %d hosts for %d runs" % (len(runs), len(runs)))

    root = Path(args.remote_root)
    for host, spec in zip(hosts, runs):
        run_root = root / "outputs/evaluation/physflow/tmr_hymotion" / spec["representation"] / spec["run_name"]
        env = {
            "PROJECT_ROOT": str(root),
            "RUN_NAME": spec["run_name"],
            "REPRESENTATION": spec["representation"],
            "INPUT_FORMAT": spec["input_format"],
            "ANNO": spec["anno"],
            "MAX_ITEMS": spec["max_items"],
            "MAX_EPOCHS": spec["max_epochs"],
            "NUM_GPUS": args.num_gpus,
            "BATCH_SIZE": args.batch_size,
            "NUM_WORKERS": args.num_workers,
            "REOCCUPY_AFTER": "1",
            "RUN_ROOT": str(run_root),
            "CAPTION_SOURCE": spec.get("caption_source", "embedding"),
        }
        session = spec["run_name"].replace("-", "_")[:80]
        inner = (
            "cd {root} && env {envs} bash scripts/embodied/run_hymotion_tmr_training.sh "
            "> {log_file} 2>&1"
        ).format(
            root=shlex.quote(str(root)),
            envs=shell_quote_env(env),
            log_file=shlex.quote(str(run_root / "logs" / "launcher.log")),
        )
        launch = (
            "set -euo pipefail; "
            "cd {root}; "
            "mkdir -p {log_dir}; "
            "if ps -eo pid,cmd | grep -E '[t]orchrun|tools/[t]rain.py' >/dev/null; then "
            "echo BUSY_HOST_HAS_TRAINING_PROCESS; exit 42; fi; "
            "if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits "
            "| awk -F, '{{gsub(/ /,\"\",$1); gsub(/ /,\"\",$2); if ($1 > 4096 || $2 > 20) busy=1}} END {{exit busy ? 0 : 1}}'; then "
            "echo BUSY_HOST_HAS_ACTIVE_GPUS; exit 43; fi; "
            "pkill -f '[o]ccupy.py' || true; "
            "tmux has-session -t {session} 2>/dev/null && tmux kill-session -t {session} || true; "
            "if tmux ls 2>/dev/null | grep -E '^tmr_hymotion_' >/dev/null; then "
            "echo BUSY_HOST_HAS_TMR_SESSION; exit 44; fi; "
            "tmux new-session -d -s {session} {inner}; "
            "echo launched:{run_name}:tmux:{session}"
        ).format(
            root=shlex.quote(str(root)),
            log_dir=shlex.quote(str(run_root / "logs")),
            session=shlex.quote(session),
            inner=shlex.quote(inner),
            run_name=spec["run_name"],
        )
        cmd = [
            "python3",
            args.taiji_exec_host,
            args.task_flag,
            args.instance_id,
            str(host),
            launch,
            "120",
        ]
        print("HOST", host, spec["run_name"])
        print(" ".join(shlex.quote(x) for x in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
