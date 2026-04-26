#!/usr/bin/env python3
"""Submit M2M v2 evaluation jobs to Taiji cluster.

Usage:
    python tools/submit_v2_eval.py                          # All 4 v2 models
    python tools/submit_v2_eval.py --models caption_local   # Single model
    python tools/submit_v2_eval.py --kimodo                 # KIMODO baseline
"""
import argparse
import json
import os
import sys
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(SCRIPT_DIR)
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "taiji_template.json")

sys.path.insert(0, SCRIPT_DIR)
from taiji_submit import get_token, submit_task


def main():
    parser = argparse.ArgumentParser(description='Submit M2M v2 eval to Taiji')
    parser.add_argument('--models', nargs='+',
                        default=['caption_local', 'caption_global', 'uncond_local', 'uncond_global'],
                        help='V2 model names to evaluate')
    parser.add_argument('--kimodo', action='store_true',
                        help='Also submit KIMODO all-tasks eval')
    parser.add_argument('--max-samples', type=int, default=80)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without submitting')
    args = parser.parse_args()

    token = get_token()
    if not token and not args.dry_run:
        print("ERROR: No Taiji token found")
        sys.exit(1)

    with open(TEMPLATE_PATH) as f:
        template = json.load(f)

    jobs = []

    # V2 model eval jobs (1 GPU each, all tasks)
    for model in args.models:
        job_name = f"m2m_v2_eval_{model}_latest"
        cmd = (
            f"cd {PROJ_ROOT} && "
            f"export PYTHONPATH={PROJ_ROOT}:${{PYTHONPATH:-}} && "
            f"python3 tools/eval_m2m_v2_all_tasks.py "
            f"--models {model} "
            f"--all-tasks "
            f"--max-samples {args.max_samples} "
            f"--num-steps 50 "
            f"--replacement-guidance skip_last "
            f"--text-guidance-scale 5.0 "
            f"--save-npz "
            f"--output-dir work_dirs/m2m_v2_eval_latest/{model} "
            f"--device cuda"
        )
        jobs.append((job_name, cmd))

    # KIMODO eval job
    if args.kimodo:
        job_name = "m2m_v2_eval_kimodo_all"
        cmd = (
            f"cd {PROJ_ROOT} && "
            f"export PYTHONPATH={PROJ_ROOT}:${{PYTHONPATH:-}} && "
            f"python3 tools/run_kimodo_all_tasks.py "
            f"--all-tasks "
            f"--max-samples {args.max_samples} "
            f"--output-dir work_dirs/m2m_v2_eval_latest/kimodo "
            f"--device cuda"
        )
        jobs.append((job_name, cmd))

    for job_name, cmd in jobs:
        print(f"\n{'='*60}")
        print(f"Job: {job_name}")
        print(f"Cmd: {cmd[:120]}...")

        if args.dry_run:
            print("[DRY RUN] Skipping submission")
            continue

        cfg = copy.deepcopy(template)
        cfg['common']['readable_name'] = job_name
        cfg['common']['task_flag'] = job_name
        cfg['designated_resource']['host_num'] = 1
        cfg['designated_resource']['host_gpu_num'] = 1
        # Set start command
        if 'start_cmd' in cfg.get('running', {}):
            cfg['running']['start_cmd'] = cmd
        elif 'running' in cfg:
            cfg['running']['start_cmd'] = cmd
        else:
            cfg['running'] = {'start_cmd': cmd}

        try:
            submit_task(cfg, token)
            print(f"  ✅ Submitted: {job_name}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")


if __name__ == "__main__":
    main()
