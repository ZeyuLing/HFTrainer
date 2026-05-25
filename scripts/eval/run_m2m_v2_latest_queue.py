#!/usr/bin/env python3
"""Run eval_m2m_v2_all_tasks.py as a per-GPU job queue.

The all-tasks evaluator loads one model per process. This wrapper splits work
by (model, task) and keeps one subprocess per visible GPU, so completed task
JSONs appear incrementally and can be imported into the dashboard while the
rest of the queue continues.
"""
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

from hftrainer.evaluation.motion.m2m_eval_tasks import EVAL_TASKS
from scripts.eval.eval_m2m_v2_all_tasks import ALL_MODELS


def runnable_settings(
    task_id: str,
    model_info: Dict[str, object],
    run_caption_nonaware: bool = False,
    allow_uncond_caption_required: bool = False,
    include_routine_skipped: bool = False,
    include_disabled_settings: bool = False,
) -> List[str]:
    model_name = str(model_info.get("name", ""))
    if task_id == "E6" and not include_routine_skipped:
        return []
    if (
        model_name == "smpl_uncond_E1"
        and task_id == "E1"
        and not include_routine_skipped
    ):
        return []

    task = EVAL_TASKS[task_id]
    has_caption = bool(model_info.get("has_caption", False))
    if (
        (not getattr(task, "caption_aware", True))
        and has_caption
        and not run_caption_nonaware
    ):
        return []

    out: List[str] = []
    for setting_name, setting in task.settings.items():
        if setting.mask_kwargs.get("_disabled", False) and not include_disabled_settings:
            continue
        setting_uc = getattr(setting, "use_caption", None)
        if setting_uc is True:
            caption_policy = "require"
        elif setting_uc is False:
            caption_policy = "blank"
        elif task.needs_caption:
            caption_policy = "require"
        else:
            caption_policy = "neutral"
        if (
            caption_policy == "require"
            and not has_caption
            and not allow_uncond_caption_required
        ):
            continue
        out.append(setting_name)
    return out


def build_jobs(args: argparse.Namespace) -> List[Dict[str, object]]:
    task_ids = list(EVAL_TASKS.keys()) if args.all_tasks else args.tasks
    jobs: List[Dict[str, object]] = []
    for model in args.models:
        model_info = ALL_MODELS[model]
        model_info = {**model_info, "name": model}
        for task_id in task_ids:
            settings = runnable_settings(
                task_id,
                model_info,
                run_caption_nonaware=args.run_caption_nonaware,
                allow_uncond_caption_required=args.allow_uncond_caption_required,
                include_routine_skipped=args.include_routine_skipped,
                include_disabled_settings=args.include_disabled_settings,
            )
            if not settings:
                print(f"[queue] skip {model} {task_id}: no runnable settings", flush=True)
                continue
            jobs.append({"model": model, "task": task_id, "settings": settings})
    return jobs


def worker(gpu: str, q: Queue, args: argparse.Namespace, failures: List[str]) -> None:
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        model = str(job["model"])
        task = str(job["task"])
        settings = list(job["settings"])
        out_dir = Path(args.output_root) / f"{model}_{task}"
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
            *settings,
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
        print(
            f"[queue] gpu={gpu} start {model} {task} settings={','.join(settings)}",
            flush=True,
        )
        start = time.time()
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, cwd=args.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        if proc.returncode != 0:
            msg = f"{model} {task} failed rc={proc.returncode} log={log_path}"
            failures.append(msg)
            print(f"[queue] gpu={gpu} FAIL {msg}", flush=True)
        else:
            print(f"[queue] gpu={gpu} done {model} {task} elapsed={elapsed:.1f}s", flush=True)
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--models", nargs="+", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--tasks", nargs="+", default=list(EVAL_TASKS.keys()))
    parser.add_argument("--all-tasks", action="store_true")
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
            f.write(f"{job['model']} {job['task']} {' '.join(job['settings'])}\n")
    print(f"[queue] jobs={len(jobs)} gpus={','.join(args.gpus)} output={args.output_root}", flush=True)

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
        print(f"[queue] failures={len(failures)} written={failure_path}", flush=True)
        sys.exit(1)
    print("[queue] all jobs finished", flush=True)


if __name__ == "__main__":
    main()
