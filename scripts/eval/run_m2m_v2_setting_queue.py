#!/usr/bin/env python3
"""Run eval_m2m_v2_all_tasks.py as a per-(model, task, setting) GPU queue."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.eval_m2m_v2_all_tasks import ALL_MODELS
from scripts.eval.run_m2m_v2_latest_queue import runnable_settings


def build_jobs(args: argparse.Namespace) -> List[Dict[str, str]]:
    jobs: List[Dict[str, str]] = []
    for model in args.models:
        model_info = {**ALL_MODELS[model], "name": model}
        for task in args.tasks:
            settings = runnable_settings(
                task,
                model_info,
                run_caption_nonaware=args.run_caption_nonaware,
                allow_uncond_caption_required=args.allow_uncond_caption_required,
                include_routine_skipped=args.include_routine_skipped,
                include_disabled_settings=args.include_disabled_settings,
            )
            if args.settings:
                settings = [s for s in settings if s in set(args.settings)]
            for setting in settings:
                jobs.append({"model": model, "task": task, "setting": setting})
    return jobs


def worker(gpu: str, q: Queue, args: argparse.Namespace, failures: List[str]) -> None:
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        model = job["model"]
        task = job["task"]
        setting = job["setting"]
        out_dir = Path(args.output_root) / f"{model}_{task}_{setting}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"
        cmd = [
            sys.executable,
            "scripts/eval/eval_m2m_v2_all_tasks.py",
            "--models",
            model,
            "--tasks",
            task,
            "--settings",
            setting,
            "--max-samples",
            str(args.max_samples),
            "--num-steps",
            str(args.num_steps),
            "--replacement-guidance",
            args.replacement_guidance,
            "--output-dir",
            str(out_dir),
            "--text-guidance-scale",
            str(args.text_guidance_scale),
        ]
        if args.use_rewritten:
            cmd.append("--use-rewritten")
        if args.save_npz:
            cmd.append("--save-npz")
        if args.run_caption_nonaware:
            cmd.append("--run-caption-nonaware")
        if args.allow_uncond_caption_required:
            cmd.append("--allow-uncond-caption-required")
        if args.include_disabled_settings:
            cmd.append("--include-disabled-settings")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("PYTHONPATH", ".")
        print(f"[setting-queue] gpu={gpu} start {model} {task}/{setting}", flush=True)
        start = time.time()
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, cwd=args.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        if proc.returncode != 0:
            msg = f"{model} {task}/{setting} failed rc={proc.returncode} log={log_path}"
            failures.append(msg)
            print(f"[setting-queue] gpu={gpu} FAIL {msg}", flush=True)
        else:
            print(f"[setting-queue] gpu={gpu} done {model} {task}/{setting} elapsed={elapsed:.1f}s", flush=True)
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--models", nargs="+", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--settings", nargs="*", default=None)
    parser.add_argument("--gpus", nargs="+", default=[str(i) for i in range(8)])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=1000000)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--replacement-guidance", default="skip_last")
    parser.add_argument("--text-guidance-scale", type=float, default=1.0)
    parser.add_argument("--use-rewritten", action="store_true")
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--run-caption-nonaware", action="store_true")
    parser.add_argument("--allow-uncond-caption-required", action="store_true")
    parser.add_argument("--include-routine-skipped", action="store_true")
    parser.add_argument("--include-disabled-settings", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    manifest = Path(args.output_root) / "manifest.txt"
    with open(manifest, "w") as f:
        for job in jobs:
            f.write(f"{job['model']} {job['task']} {job['setting']}\n")
    print(f"[setting-queue] jobs={len(jobs)} gpus={','.join(args.gpus)} output={args.output_root}", flush=True)

    q: Queue = Queue()
    failures: List[str] = []
    threads = []
    for gpu in args.gpus:
        t = threading.Thread(target=worker, args=(gpu, q, args, failures), daemon=True)
        t.start()
        threads.append(t)
    for job in jobs:
        q.put(job)
    for _ in threads:
        q.put(None)
    q.join()

    if failures:
        failure_path = Path(args.output_root) / "failures.txt"
        failure_path.write_text("\n".join(failures) + "\n")
        print(f"[setting-queue] failures={len(failures)} written={failure_path}", flush=True)
        sys.exit(1)
    print("[setting-queue] all jobs finished", flush=True)


if __name__ == "__main__":
    main()
