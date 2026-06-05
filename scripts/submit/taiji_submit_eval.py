#!/usr/bin/env python3
"""Submit M2M v2 evaluation tasks to Taiji cluster.

Each job runs on 1 host × 1 GPU (V100). The eval script handles one model
at a time, running all missing tasks for that model.

Usage:
    # Submit all 4 M2M v2 models + KIMODO re-eval (5 jobs total)
    python tools/taiji_submit_eval.py --all

    # Submit specific models
    python tools/taiji_submit_eval.py --models uncond_local caption_local

    # Submit KIMODO only
    python tools/taiji_submit_eval.py --kimodo

    # Dry run (print commands without submitting)
    python tools/taiji_submit_eval.py --all --dry-run
"""
import argparse
import copy
import json
import os
import sys

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "taiji_template.json")
API_URL = "http://taijiapi.oa.com/taskmanagement/task_server/task_management/api/training_task"

M2M_MODELS = ['uncond_local', 'uncond_global', 'caption_local', 'caption_global']
EVAL_SCRIPT = "scripts/eval/eval_m2m_v2_all_tasks.py"
KIMODO_SCRIPT = "scripts/kimodo/run_kimodo_all_tasks.py"
EVAL_OUTPUT_DIR = "work_dirs/m2m_v2_eval_latest"


def get_token():
    """Get Taiji API token."""
    token = os.environ.get("TOKEN", "")
    if not token:
        for path in [
            os.path.expanduser("~/.claude-dashboard/taiji_token"),
            os.path.expanduser("~/.taijiconfig"),
        ]:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    token = data.get("user", {}).get("token", "")
                except (json.JSONDecodeError, KeyError):
                    try:
                        token = open(path).read().strip()
                    except Exception:
                        pass
            if token:
                break
    return token


def build_m2m_eval_cmd(model_name: str) -> str:
    """Build command to evaluate a single M2M model on all applicable tasks."""
    cmd_parts = [
        f"cd {PROJECT_ROOT}",
        f"python3 {EVAL_SCRIPT}",
        f"  --models {model_name}",
        f"  --all-tasks",
        f"  --save-npz",
        f"  --output-dir {EVAL_OUTPUT_DIR}/{model_name}",
        f"  --max-samples 80",
        f"  --num-steps 50",
    ]
    return " && ".join([cmd_parts[0], " ".join(cmd_parts[1:])])


def build_kimodo_eval_cmd() -> str:
    """Build command to re-run KIMODO eval with rotation-based retarget."""
    return (
        f"cd {PROJECT_ROOT} && "
        f"python3 {KIMODO_SCRIPT} "
        f"  --all-tasks "
        f"  --max-samples 80 "
        f"  --output-dir {EVAL_OUTPUT_DIR}/kimodo"
    )


def submit_job(task_flag: str, start_cmd: str, gpu_name: str = "V100",
               host_num: int = 1, host_gpu_num: int = 1,
               dry_run: bool = False):
    """Submit a single eval job to Taiji."""
    print(f"\n{'='*60}")
    print(f"Job: {task_flag}")
    print(f"GPU: {host_num}×{host_gpu_num} {gpu_name}")
    print(f"Cmd: {start_cmd[:120]}...")

    if dry_run:
        print("  [DRY RUN] Skipping submission")
        return None

    token = get_token()
    if not token:
        print("ERROR: No Taiji token found")
        return None

    with open(TEMPLATE_PATH) as f:
        tmpl = json.load(f)

    # Override for eval job (single GPU, lighter resources)
    tmpl["common"]["task_flag"] = task_flag
    tmpl["common"]["readable_name"] = task_flag
    tmpl["designated_resource"]["host_num"] = host_num
    tmpl["designated_resource"]["host_gpu_num"] = float(host_gpu_num)
    tmpl["designated_resource"]["GPUName"] = gpu_name
    tmpl["designated_resource"]["is_enable_rdma"] = False
    tmpl["designated_resource"]["rdma_in_same_module"] = False
    tmpl["designated_resource"]["keep_running_after_trainer_finish"] = False
    tmpl["designated_resource"]["keep_alive_time"] = 0
    tmpl["designated_resource"]["priority_level"] = "HIGH"

    # Eval-specific start command
    tmpl["job_config"]["start_cmd"] = start_cmd
    tmpl["job_config"]["exec_start_in_all_mpi_pods"] = False

    headers = {
        "Content-Type": "application/json",
        "Cookie": f"TOKEN={token}",
    }

    # Create task
    resp = requests.post(f"{API_URL}/create/", json=tmpl, headers=headers, timeout=30)
    if resp.status_code != 200:
        print(f"  ERROR: CREATE failed: {resp.text[:200]}")
        return None
    result = resp.json()
    if result.get("result", {}).get("code") != 0:
        print(f"  ERROR: CREATE failed: {result.get('result', {}).get('message', '')[:200]}")
        return None
    print(f"  Created OK")

    # Enable (start) task
    enable_payload = {
        "common": {"task_flag": task_flag, "rtx": tmpl["common"]["rtx"]},
        "designated_resource": tmpl["designated_resource"],
    }
    resp2 = requests.post(f"{API_URL}/enable/", json=enable_payload, headers=headers, timeout=30)
    if resp2.status_code != 200:
        print(f"  ERROR: ENABLE failed: {resp2.text[:200]}")
        return None
    result2 = resp2.json()
    if result2.get("result", {}).get("code") != 0:
        print(f"  ERROR: ENABLE failed: {result2.get('result', {}).get('message', '')[:200]}")
        return None

    instance_id = result2.get("result", {}).get("data", {}).get("instance_id", "?")
    print(f"  Started OK — instance_id: {instance_id}")
    print(f"  Monitor: taiji_client il {task_flag}")
    return task_flag


def main():
    parser = argparse.ArgumentParser(description="Submit M2M v2 eval jobs to Taiji")
    parser.add_argument("--all", action="store_true",
                        help="Submit all 4 M2M models + KIMODO (5 jobs)")
    parser.add_argument("--models", nargs="+", choices=M2M_MODELS,
                        help="Specific M2M models to evaluate")
    parser.add_argument("--kimodo", action="store_true",
                        help="Re-run KIMODO evaluation with rotation-based retarget")
    parser.add_argument("--gpu", default="V100", choices=["V100", "A100"],
                        help="GPU type (default: V100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
    args = parser.parse_args()

    if not args.all and not args.models and not args.kimodo:
        parser.print_help()
        print("\nError: specify --all, --models, or --kimodo")
        sys.exit(1)

    models_to_run = []
    run_kimodo = False

    if args.all:
        models_to_run = M2M_MODELS
        run_kimodo = True
    else:
        if args.models:
            models_to_run = args.models
        if args.kimodo:
            run_kimodo = True

    submitted = []

    # Submit M2M model eval jobs
    for model in models_to_run:
        task_flag = f"eval_m2m_v2_{model}_0417"
        cmd = build_m2m_eval_cmd(model)
        result = submit_job(
            task_flag=task_flag,
            start_cmd=cmd,
            gpu_name=args.gpu,
            host_num=1,
            host_gpu_num=1,
            dry_run=args.dry_run,
        )
        if result:
            submitted.append(result)

    # Submit KIMODO eval job (needs GPU for diffusion)
    if run_kimodo:
        task_flag = "eval_kimodo_retarget_0417"
        cmd = build_kimodo_eval_cmd()
        result = submit_job(
            task_flag=task_flag,
            start_cmd=cmd,
            gpu_name=args.gpu,
            host_num=1,
            host_gpu_num=1,
            dry_run=args.dry_run,
        )
        if result:
            submitted.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Submitted {len(submitted)} jobs:")
    for tf in submitted:
        print(f"  • {tf}")
    if submitted:
        print(f"\nMonitor all: taiji_client trl")


if __name__ == "__main__":
    main()
