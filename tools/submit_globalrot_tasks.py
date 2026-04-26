#!/usr/bin/env python3
"""Batch submit 4 global rotation HyMotion M2M tasks to Taiji.

This script submits all 4 variants of global rotation models:
  1. caption_fm_man_globalrot
  2. caption_jit_man_globalrot
  3. uncond_fm_man_globalrot
  4. uncond_jit_man_globalrot

Each task uses 6 hosts = 48 V100/A100 GPUs.

Usage:
    python tools/submit_globalrot_tasks.py [--location qingyuan|chongqing]
"""

import argparse
import json
import os
import sys
import uuid
import time

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "taiji_template.json")
API_URL = "http://taijiapi.oa.com/taskmanagement/task_server/task_management/api/training_task"
TAIJI_TOKEN_FILE = os.path.expanduser("~/.claude-dashboard/taiji_token")

# 4 configurations to submit
CONFIGS = [
    "configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_man_globalrot_046b.py",
    "configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_man_globalrot_046b.py",
    "configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_globalrot_046b.py",
    "configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_man_globalrot_046b.py",
]

TASK_NAMES = [
    "hymotion_m2m_caption_fm_man_globalrot_046b",
    "hymotion_m2m_caption_jit_man_globalrot_046b",
    "hymotion_m2m_uncond_fm_man_globalrot_046b",
    "hymotion_m2m_uncond_jit_man_globalrot_046b",
]


def get_token():
    """Get Taiji API token from env or file."""
    token = os.environ.get("TOKEN", "")
    if not token:
        try:
            if os.path.exists(TAIJI_TOKEN_FILE):
                token = open(TAIJI_TOKEN_FILE).read().strip()
        except Exception:
            pass
    if not token:
        try:
            import configparser
            cfg_file = os.path.expanduser("~/.taijiconfig")
            if os.path.exists(cfg_file):
                with open(cfg_file) as f:
                    data = json.load(f)
                token = data.get("user", {}).get("token", "")
        except Exception:
            pass
    return token


def submit_task(task_flag, config_path, host_num=6, location="qingyuan"):
    """Submit a single training task to Taiji."""
    token = get_token()
    if not token:
        print("ERROR: No Taiji token found. Set TOKEN env var or write to ~/.claude-dashboard/taiji_token")
        return False

    # Load template
    with open(TEMPLATE_PATH) as f:
        tmpl = json.load(f)

    # Override fields
    tmpl["common"]["task_flag"] = task_flag
    tmpl["common"]["readable_name"] = task_flag
    tmpl["designated_resource"]["host_num"] = host_num

    # Update location based on parameter
    if location == "qingyuan":
        tmpl["designated_resource"]["location"] = "yz"
        tmpl["designated_resource"]["GPUName"] = "A100"
    elif location == "chongqing":
        tmpl["designated_resource"]["location"] = "cq"
        tmpl["designated_resource"]["GPUName"] = "A100"
    else:
        print(f"Unknown location: {location}")
        return False

    # Build start_cmd
    start_cmd = tmpl["job_config"]["start_cmd"]
    start_cmd = start_cmd.replace("__TASK_FLAG__", task_flag)
    start_cmd = start_cmd.replace("__CONFIG_PATH__", config_path)
    tmpl["job_config"]["start_cmd"] = start_cmd

    headers = {
        "Content-Type": "application/json",
        "Iplus-Task-Server-Api-Token": token,
    }

    # Step 1: CREATE
    create_payload = {
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
                    "job_config": tmpl["job_config"],
                },
            },
        },
        "ver": "3.0.10",
    }

    gpu_type = tmpl['designated_resource']['GPUName']
    print(f"\n{'='*70}")
    print(f"Submitting task '{task_flag}'...")
    print(f"  Config: {config_path}")
    print(f"  GPUs: {host_num}x{int(tmpl['designated_resource']['host_gpu_num'])} = {host_num * int(tmpl['designated_resource']['host_gpu_num'])} {gpu_type}")
    print(f"  Location: {location}")
    print(f"  Business: {tmpl['common']['business_flag']}")
    print(f"  RDMA: {tmpl['designated_resource'].get('is_enable_rdma', False)}")

    try:
        resp = requests.post(f"{API_URL}/create/", json=create_payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"ERROR: CREATE failed with status {resp.status_code}: {resp.text[:300]}")
            return False

        result = resp.json()
        if result.get("result", {}).get("code") != 0:
            msg = result.get("result", {}).get("message", resp.text[:300])
            print(f"ERROR: CREATE failed: {msg}")
            return False

        print(f"  ✓ Created OK")

        # Step 2: ENABLE (start)
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
            print(f"ERROR: ENABLE failed with status {resp2.status_code}: {resp2.text[:300]}")
            return False

        result2 = resp2.json()
        if result2.get("result", {}).get("code") != 0:
            msg = result2.get("result", {}).get("message", resp2.text[:300])
            print(f"ERROR: ENABLE failed: {msg}")
            return False

        instance_id = result2.get("result", {}).get("data", {}).get("instance_id", "?")
        print(f"  ✓ Started OK")
        print(f"  task_flag: {task_flag}")
        print(f"  instance_id: {instance_id}")
        print(f"  URL: http://taiji.oa.com/#/project-list/jizhi/task-inst-detail/{instance_id}")

        return True

    except Exception as e:
        print(f"ERROR: Request failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch submit 4 global rotation HyMotion M2M tasks")
    parser.add_argument(
        "--location",
        choices=["qingyuan", "chongqing"],
        default="qingyuan",
        help="GPU location (qingyuan A100 has 8 hosts available, chongqing A100 has 5)"
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("BATCH SUBMIT: 4 HyMotion M2M Global Rotation Variants")
    print(f"{'='*70}")
    print(f"Location: {args.location}")
    print(f"Resources per task: 6 hosts × 8 {args.location.upper()} = 48 GPUs")
    print(f"Total tasks: {len(CONFIGS)}")
    print(f"\nConfigs to submit:")
    for i, cfg in enumerate(CONFIGS, 1):
        print(f"  {i}. {cfg}")

    # Check resource availability
    print(f"\nNote: Current available resources:")
    print(f"  - Qingyuan (A100): 8 hosts available")
    print(f"  - Chongqing (A100): 5 hosts available")
    print(f"  - Shenzhen (V100): 0 hosts available (0~20 range)")

    confirmed = input(f"\nProceed with submitting 4 tasks to {args.location}? [y/N]: ")
    if confirmed.lower() != 'y':
        print("Cancelled.")
        return

    results = []
    for task_name, config_path in zip(TASK_NAMES, CONFIGS):
        success = submit_task(task_name, config_path, host_num=6, location=args.location)
        results.append((task_name, success))
        time.sleep(1)  # Brief delay between submissions

    # Summary
    print(f"\n{'='*70}")
    print("SUBMISSION SUMMARY")
    print(f"{'='*70}")

    success_count = sum(1 for _, s in results if s)
    print(f"\nSuccessful: {success_count}/{len(results)}")

    for task_name, success in results:
        status = "✓ OK" if success else "✗ FAILED"
        print(f"  {status} - {task_name}")

    if success_count == len(results):
        print(f"\n✓ All {len(results)} tasks submitted successfully!")
        print(f"\nMonitor all tasks:")
        print(f"  taiji_client trl | grep -i 'globalrot\\|uncond_fm\\|uncond_jit\\|caption_fm\\|caption_jit'")
        print(f"\nCheck individual task:")
        for task_name in TASK_NAMES:
            print(f"  taiji_client td {task_name}")
    else:
        print(f"\n✗ {len(results) - success_count} task(s) failed to submit")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
