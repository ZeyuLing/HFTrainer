#!/usr/bin/env python3
"""Submit M2M v2 evaluation jobs to Taiji (1 GPU each).

Usage:
    python tools/submit_v2_eval_taiji.py                  # All 4 v2 models + KIMODO
    python tools/submit_v2_eval_taiji.py --models caption_local  # Single model
    python tools/submit_v2_eval_taiji.py --kimodo-only     # KIMODO only
    python tools/submit_v2_eval_taiji.py --dry-run         # Preview commands
"""
import argparse
import copy
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from taiji_submit import get_token, TEMPLATE_PATH, API_URL

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)


def submit_eval_job(task_flag: str, start_cmd: str, token: str, gpu_name: str = "V100"):
    """Submit a 1-GPU eval job to Taiji via HTTP API."""
    with open(TEMPLATE_PATH) as f:
        tmpl = json.load(f)

    # Override for 1-GPU eval
    tmpl["designated_resource"]["host_num"] = 1
    tmpl["designated_resource"]["host_gpu_num"] = 1
    tmpl["designated_resource"]["GPUName"] = gpu_name

    headers = {
        "Content-Type": "application/json",
        "Iplus-Task-Server-Api-Token": token,
    }

    payload = {
        "id": uuid.uuid4().hex[:32],
        "jsonrpc": "1.0",
        "method": "TASK_CREATE",
        "params": {
            "req_module_id": "YG_00000000000000000000000000000000_00",
            "event": "TASK_CREATE",
            "task": {
                "task_type": tmpl.get("task_type", "general_gpu_type"),
                "common": {
                    "business_flag": tmpl["common"]["business_flag"],
                    "readable_name": task_flag,
                    "template_flag": tmpl["common"].get("template_flag", "kubeflow_job_learning"),
                    "task_flag": task_flag,
                    "dataset_id": tmpl["common"]["dataset_id"],
                    "dataset_params": {
                        "dataset_name": task_flag,
                        "dataset_source": "plat_ceph",
                        "path_info": {"path": "/"},
                    },
                    "model_id": tmpl["common"]["model_id"],
                    "running_type": "manual",
                    "rtx": tmpl["common"].get("rtx", "zeyuling"),
                    "permission": tmpl["common"].get("permission", {}),
                    "network_path_info": tmpl["common"].get("network_path_info", {}),
                },
                "project_id": None,
                "task_config": {
                    "designated_resource": tmpl["designated_resource"],
                    "job_config": {
                        **tmpl.get("job_config", {}),
                        "start_cmd": start_cmd,
                    },
                },
            },
        },
        "ver": "3.0.10",
    }

    resp = requests.post(f"{API_URL}/create/", json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"CREATE failed: {resp.status_code} {resp.text[:200]}")

    data = resp.json()
    if data.get("code") not in (0, None):
        raise RuntimeError(f"CREATE error: {data}")

    # ENABLE (start the task)
    enable_payload = {
        "id": uuid.uuid4().hex[:32],
        "jsonrpc": "1.0",
        "method": "TASK_ENABLE",
        "params": {
            "req_module_id": "YG_00000000000000000000000000000000_00",
            "event": "TASK_ENABLE",
            "task_flag": task_flag,
            "one_off": "true",
        },
        "ver": "3.0.10",
    }
    resp2 = requests.post(f"{API_URL}/enable/", json=enable_payload, headers=headers, timeout=30)
    if resp2.status_code != 200:
        raise RuntimeError(f"ENABLE failed: {resp2.status_code} {resp2.text[:200]}")
    result2 = resp2.json()
    if result2.get("result", {}).get("code") != 0:
        raise RuntimeError(f"ENABLE error: {result2.get('result', {}).get('message', '')}")

    instance_id = result2.get("result", {}).get("data", {}).get("instance_id", "?")
    return instance_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['caption_local', 'caption_global', 'uncond_local', 'uncond_global'])
    parser.add_argument('--kimodo-only', action='store_true')
    parser.add_argument('--no-kimodo', action='store_true')
    parser.add_argument('--max-samples', type=int, default=80)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--gpu', default='V100')
    args = parser.parse_args()

    token = get_token()
    if not token and not args.dry_run:
        print("ERROR: No Taiji token")
        sys.exit(1)

    ts = datetime.now().strftime("%m%d")
    proj = str(PROJECT_ROOT)
    jobs = []

    if not args.kimodo_only:
        for model in args.models:
            name = f"eval_v2_{model}_{ts}"
            cmd = (
                f"cd {proj} && "
                f"export PYTHONPATH={proj}:${{PYTHONPATH:-}} && "
                f"python3 scripts/eval/eval_m2m_v2_all_tasks.py "
                f"--models {model} --all-tasks "
                f"--max-samples {args.max_samples} --num-steps 50 "
                f"--replacement-guidance skip_last --text-guidance-scale 1.0 "
                f"--save-npz --output-dir work_dirs/m2m_v2_eval_latest/{model}"
            )
            jobs.append((name, cmd))

    if not args.no_kimodo:
        name = f"eval_kimodo_all_{ts}"
        cmd = (
            f"cd {proj} && "
            f"export PYTHONPATH={proj}:${{PYTHONPATH:-}} && "
            f"python3 tools/run_kimodo_all_tasks.py "
            f"--all-tasks --max-samples {args.max_samples} "
            f"--output-dir work_dirs/m2m_v2_eval_latest/kimodo"
        )
        jobs.append((name, cmd))

    for name, cmd in jobs:
        print(f"\n{'='*50}")
        print(f"Job: {name}")
        print(f"Cmd: {cmd[:100]}...")
        if args.dry_run:
            print("  [DRY RUN]")
            continue
        try:
            submit_eval_job(name, cmd, token, args.gpu)
            print(f"  ✅ Submitted")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    print(f"\n{'='*50}")
    print(f"{'DRY RUN complete' if args.dry_run else f'Submitted {len(jobs)} jobs'}")
    print(f"Monitor: taijirun trl")


if __name__ == "__main__":
    main()
