#!/usr/bin/env python3
"""Submit a custom command task to Taiji. Adapted from tools/taiji_submit.py."""
import json
import os
import sys
import uuid

try:
    import requests
except ImportError:
    print("ERROR: requests not installed"); sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "tools", "taiji_template.json")
API_URL = "http://taijiapi.oa.com/taskmanagement/task_server/task_management/api/training_task"

def get_token():
    token = os.environ.get("TOKEN", "")
    if not token:
        try:
            tf = os.path.expanduser("~/.claude-dashboard/taiji_token")
            if os.path.exists(tf):
                token = open(tf).read().strip()
        except Exception:
            pass
    return token

def submit(task_flag, cmd, host_num=1):
    token = get_token()
    if not token:
        print("ERROR: No TOKEN"); sys.exit(1)

    with open(TEMPLATE_PATH) as f:
        tmpl = json.load(f)

    tmpl["common"]["task_flag"] = task_flag
    tmpl["common"]["readable_name"] = task_flag
    tmpl["designated_resource"]["host_num"] = host_num
    tmpl["designated_resource"]["is_enable_rdma"] = False
    tmpl["designated_resource"]["rdma_in_same_module"] = False
    tmpl["designated_resource"]["keep_running_after_trainer_finish"] = True
    tmpl["job_config"]["start_cmd"] = f"cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/; {cmd}"

    headers = {
        "Content-Type": "application/json",
        "Iplus-Task-Server-Api-Token": token,
    }

    create_payload = {
        "id": uuid.uuid4().hex[:32], "jsonrpc": "1.0", "method": "TASK_CREATE",
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
                    "dataset_params": {"dataset_name": task_flag, "dataset_source": "plat_ceph", "path_info": {"path": "/"}},
                    "model_id": tmpl["common"]["model_id"],
                    "running_type": "manual",
                    "rtx": tmpl["common"].get("rtx", "zeyuling"),
                    "permission": tmpl["common"].get("permission", {}),
                    "network_path_info": tmpl["common"].get("network_path_info", {}),
                },
                "project_id": None,
                "task_config": {"designated_resource": tmpl["designated_resource"], "job_config": tmpl["job_config"]},
            },
        },
        "ver": "3.0.10",
    }

    print(f"Creating task '{task_flag}'...")
    print(f"  CMD: {cmd}")
    resp = requests.post(f"{API_URL}/create/", json=create_payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        print(f"ERROR: CREATE failed: {resp.text[:300]}"); sys.exit(1)
    result = resp.json()
    if result.get("result", {}).get("code") != 0:
        print(f"ERROR: {result.get('result', {}).get('message', '')}"); sys.exit(1)
    print("  Created OK")

    enable_payload = {
        "id": uuid.uuid4().hex[:32], "jsonrpc": "1.0", "method": "TASK_ENABLE",
        "params": {"req_module_id": "YG_00000000000000000000000000000000_00", "event": "TASK_ENABLE",
                   "task_flag": task_flag, "one_off": "true"},
        "ver": "3.0.10",
    }
    resp2 = requests.post(f"{API_URL}/enable/", json=enable_payload, headers=headers, timeout=30)
    if resp2.status_code != 200:
        print(f"ERROR: ENABLE failed: {resp2.text[:300]}"); sys.exit(1)
    result2 = resp2.json()
    if result2.get("result", {}).get("code") != 0:
        print(f"ERROR: {result2.get('result', {}).get('message', '')}"); sys.exit(1)

    instance_id = result2.get("result", {}).get("data", {}).get("instance_id", "?")
    print(f"  Started OK: task_flag={task_flag}  instance_id={instance_id}")
    print(f"  Monitor: taiji_client il {task_flag}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("task_flag")
    p.add_argument("--cmd", required=True)
    p.add_argument("--host_num", type=int, default=1)
    args = p.parse_args()
    submit(args.task_flag, args.cmd, args.host_num)
