#!/usr/bin/env python3
"""Submit M2M repair eval to Taiji.

Usage:
    python3 scripts/submit_eval.py
"""
import json
import os
import sys
import uuid

try:
    import requests
except ImportError:
    print("ERROR: requests not installed")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "tools")
TEMPLATE_PATH = os.path.join(TOOLS_DIR, "taiji_template.json")
API_URL = "http://taijiapi.oa.com/taskmanagement/task_server/task_management/api/training_task"


def get_token():
    token = os.environ.get("TOKEN", "")
    if not token:
        try:
            cfg_file = os.path.expanduser("~/.taijiconfig")
            if os.path.exists(cfg_file):
                with open(cfg_file) as f:
                    data = json.load(f)
                token = data.get("user", {}).get("token", "")
        except Exception:
            pass
    return token


def main():
    task_flag = "lzy_m2m_repair_eval_v2"
    token = get_token()
    if not token:
        print("ERROR: No Taiji token found")
        sys.exit(1)

    with open(TEMPLATE_PATH) as f:
        tmpl = json.load(f)

    tmpl["common"]["task_flag"] = task_flag
    tmpl["common"]["readable_name"] = task_flag
    tmpl["designated_resource"]["host_num"] = 1
    tmpl["designated_resource"]["GPUName"] = "A100"
    tmpl["designated_resource"]["rdma_in_same_module"] = False
    tmpl["designated_resource"]["is_enable_rdma"] = False

    start_cmd = "cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ && bash scripts/run_eval_m2m_repair.sh"
    tmpl["job_config"]["start_cmd"] = start_cmd
    tmpl["job_config"]["exec_start_in_all_mpi_pods"] = False

    headers = {
        "Content-Type": "application/json",
        "Iplus-Task-Server-Api-Token": token,
    }

    # CREATE
    create_payload = {
        "id": uuid.uuid4().hex[:32],
        "jsonrpc": "1.0",
        "method": "TASK_CREATE",
        "params": {
            "req_module_id": "YG_00000000000000000000000000000000_00",
            "event": "TASK_CREATE",
            "task": tmpl,
        },
    }

    resp = requests.post(API_URL, json=create_payload, headers=headers, timeout=60)
    print(f"CREATE: {resp.status_code}")
    print(resp.text[:500])

    if resp.status_code != 200:
        sys.exit(1)

    # START
    start_payload = {
        "id": uuid.uuid4().hex[:32],
        "jsonrpc": "1.0",
        "method": "TASK_START",
        "params": {
            "req_module_id": "YG_00000000000000000000000000000000_00",
            "event": "TASK_START",
            "task_flag": task_flag,
        },
    }

    resp = requests.post(API_URL, json=start_payload, headers=headers, timeout=60)
    print(f"START: {resp.status_code}")
    print(resp.text[:500])
    print(f"\nTask submitted: {task_flag}")
    print(f"Monitor: taiji_client il {task_flag}")
    print(f"Logs: taiji_client logs {task_flag}")


if __name__ == "__main__":
    main()
