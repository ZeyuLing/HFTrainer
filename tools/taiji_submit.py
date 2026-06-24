#!/usr/bin/env python3
"""Submit a training task to Taiji via HTTP API.

Uses the full nested config format (taiji_template.json) to ensure all fields
(template_flag, task_queuing_priority, RDMA, etc.) are properly set.

Usage:
    python tools/taiji_submit.py <task_flag> <config_path> [--host_num N]

Example:
    python tools/taiji_submit.py my_train_v1 configs/hymotion_umo/hymotion_umo_201dim_046b.py --host_num 2
"""
import argparse
import json
import os
import sys
import uuid

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "taiji_template.json")
API_URL = "http://taijiapi.oa.com/taskmanagement/task_server/task_management/api/training_task"
TAIJI_TOKEN_FILE = os.path.expanduser("~/.claude-dashboard/taiji_token")


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


def submit(task_flag, config_path, host_num=4, business_flag=None, elastic=False,
           start_cmd_override=None, host_gpu_num=None, gpu_name=None,
           cuda_version=None):
    """Submit a training task to Taiji.

    Args:
        start_cmd_override: when provided, replaces the rendered training
            start_cmd entirely.  Used for inference / eval jobs that don't
            go through ``tools/taiji_dist_train.sh`` — pass any single-host
            shell command.  ``config_path`` is then optional and ignored
            unless interpolated by the caller.
        host_gpu_num: override per-host GPU count (default from template).
    """
    token = get_token()
    if not token:
        print("ERROR: No Taiji token found. Set TOKEN env var or write to ~/.claude-dashboard/taiji_token")
        sys.exit(1)

    # Load template
    with open(TEMPLATE_PATH) as f:
        tmpl = json.load(f)

    # Override fields
    tmpl["common"]["task_flag"] = task_flag
    tmpl["common"]["readable_name"] = task_flag
    # Some newer Hunyuan/Taiji business pools require task metadata even for
    # general_gpu_type jobs. Older pools ignore these extra fields.
    task_description = os.environ.get(
        "TAIJI_TASK_DESCRIPTION",
        "BeyondMimic official reproduction experiment",
    )
    task_category = os.environ.get("TAIJI_TASK_CATEGORY", os.environ.get("HUNYUAN_TASK_CATEGORY", "OTHERS"))
    task_detail_category = os.environ.get(
        "TAIJI_TASK_DETAIL_CATEGORY",
        os.environ.get("HUNYUAN_TASK_DETAIL_CATEGORY", "beyondmimic_reproduction"),
    )
    resource_usage = os.environ.get("TAIJI_RESOURCE_USAGE", os.environ.get("HUNYUAN_RESOURCE_USAGE", "experiment"))
    base_model_id = os.environ.get("TAIJI_BASE_MODEL_ID", tmpl["common"].get("model_id", ""))
    output_model_id = os.environ.get("TAIJI_OUTPUT_MODEL_ID", tmpl["common"].get("model_id", ""))
    output_base_model = os.environ.get(
        "TAIJI_OUTPUT_BASE_MODEL",
        os.environ.get("HUNYUAN_OUTPUT_BASE_MODEL", "200M-dense-a2hv0.1"),
    )
    description_info = {
        "application_scenario": task_detail_category,
        "subdivision": task_detail_category,
        "description": task_description,
        "hyper_parameters": "",
        "resource_usage": resource_usage,
    }
    input_model_item = {
        "model_id": base_model_id,
        "model_name": tmpl["common"].get("model_name", task_flag),
        "model_source": "local_upload",
        "model_type": "base_model",
        "entity_type": "input_base_model",
    }
    output_model_item = {
        "model_id": output_model_id,
        "model_name": output_base_model,
        "model_source": "local_upload",
        "model_type": "base_model",
        "entity_type": "output_base_model",
    }
    model_items = []
    if base_model_id:
        model_items.append(input_model_item)
    if output_model_id or output_base_model:
        model_items.append(output_model_item)
    tmpl["common"].update({
        "task_description": task_description,
        "task_desc": task_description,
        "description": task_description,
        "task_category": task_category,
        "category": task_category,
        "application_scenario": task_detail_category,
        "subdivision": task_detail_category,
        "subdivision_list": [task_detail_category],
        "resource_usage": resource_usage,
        "resource_use": resource_usage,
        "resource_purpose": resource_usage,
        "input_base_model_id": base_model_id,
        "base_model_id": base_model_id,
        "using_base_model_id": base_model_id,
        "output_base_model_id": output_model_id,
        "output_model_id": output_model_id,
        "output_base_model": output_base_model,
        "output_base_model_name": output_base_model,
        "description_info": task_description,
        "resource_usage": resource_usage,
        "model_items": model_items,
    })
    tmpl["designated_resource"]["host_num"] = host_num
    tmpl["designated_resource"]["is_elasticity"] = elastic
    if host_gpu_num is not None:
        tmpl["designated_resource"]["host_gpu_num"] = float(host_gpu_num)
    if gpu_name is not None:
        tmpl["designated_resource"]["GPUName"] = gpu_name
    if cuda_version is not None:
        tmpl["designated_resource"]["cuda_version"] = cuda_version

    # Force RDMA for multi-node training (guard against CephFS caching stale template)
    if host_num > 1:
        tmpl["designated_resource"]["is_enable_rdma"] = True
        tmpl["designated_resource"]["rdma_in_same_module"] = True

    if business_flag:
        tmpl["common"]["business_flag"] = business_flag

    # Build start_cmd: either custom (eval/inference) or templated (training).
    if start_cmd_override is not None:
        start_cmd = start_cmd_override.replace("__TASK_FLAG__", task_flag)
    else:
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
            "description_info": description_info,
            "resource_usage": resource_usage,
            "application_scenario": task_detail_category,
            "subdivision": task_detail_category,
            "subdivision_list": [task_detail_category],
            "model_items": model_items,
            "task": {
                "task_type": tmpl.get("task_type", "general_gpu_type"),
                "description_info": description_info,
                "resource_usage": resource_usage,
                "application_scenario": task_detail_category,
                "subdivision": task_detail_category,
                "subdivision_list": [task_detail_category],
                "model_items": model_items,
                "common": {
                    "business_flag": tmpl["common"]["business_flag"],
                    "readable_name": task_flag,
                    "template_flag": tmpl["common"].get("template_flag", "kubeflow_job_learning"),
                    "task_flag": task_flag,
                    "task_description": task_description,
                    "task_desc": task_description,
                    "description": task_description,
                    "task_category": task_category,
                    "category": task_category,
                    "application_scenario": task_detail_category,
                    "subdivision": task_detail_category,
                    "subdivision_list": [task_detail_category],
                    "resource_usage": resource_usage,
                    "resource_use": resource_usage,
                    "resource_purpose": resource_usage,
                    "description_info": task_description,
                    "model_items": model_items,
                    "dataset_id": tmpl["common"]["dataset_id"],
                    "dataset_params": {
                        "dataset_name": task_flag,
                        "dataset_source": "plat_ceph",
                        "path_info": {"path": "/"},
                    },
                    "model_id": tmpl["common"]["model_id"],
                    "input_base_model_id": base_model_id,
                    "base_model_id": base_model_id,
                    "using_base_model_id": base_model_id,
                    "output_base_model_id": output_model_id,
                    "output_model_id": output_model_id,
                    "output_base_model": output_base_model,
                    "output_base_model_name": output_base_model,
                    "running_type": "manual",
                    "rtx": tmpl["common"].get("rtx", "zeyuling"),
                    "permission": tmpl["common"].get("permission", {}),
                    "network_path_info": tmpl["common"].get("network_path_info", {}),
                },
                "project_id": None,
                "task_config": {
                    "description_info": description_info,
                    "resource_usage": resource_usage,
                    "application_scenario": task_detail_category,
                    "subdivision": task_detail_category,
                    "subdivision_list": [task_detail_category],
                    "model_items": model_items,
                    "designated_resource": tmpl["designated_resource"],
                    "job_config": tmpl["job_config"],
                },
            },
        },
        "ver": "3.0.10",
    }

    print(f"Creating task '{task_flag}'...")
    print(f"  Config: {config_path}")
    print(f"  GPUs: {host_num}x{int(tmpl['designated_resource']['host_gpu_num'])} = {host_num * int(tmpl['designated_resource']['host_gpu_num'])} {tmpl['designated_resource']['GPUName']}")
    print(f"  Business: {tmpl['common']['business_flag']}")
    print(f"  RDMA: {tmpl['designated_resource'].get('is_enable_rdma', False)}")
    print(f"  Elastic: {elastic}")
    print(f"  Elastic: {elastic}")

    resp = requests.post(f"{API_URL}/create/", json=create_payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        print(f"ERROR: CREATE failed with status {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    result = resp.json()
    if result.get("result", {}).get("code") != 0:
        msg = result.get("result", {}).get("message", resp.text[:300])
        print(f"ERROR: CREATE failed: {msg}")
        sys.exit(1)

    print(f"  Created OK")

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
        sys.exit(1)

    result2 = resp2.json()
    if result2.get("result", {}).get("code") != 0:
        msg = result2.get("result", {}).get("message", resp2.text[:300])
        print(f"ERROR: ENABLE failed: {msg}")
        sys.exit(1)

    instance_id = result2.get("result", {}).get("data", {}).get("instance_id", "?")
    print(f"  Started OK")
    print(f"  task_flag: {task_flag}")
    print(f"  instance_id: {instance_id}")
    print(f"  URL: http://taiji.oa.com/#/project-list/jizhi/task-inst-detail/{instance_id}")
    print()
    print(f"Monitor: taiji_client il {task_flag}")
    print(f"Detail:  taiji_client td {task_flag}")
    print(f"Stop:    taiji_client stop {task_flag}")

    return task_flag, instance_id


def main():
    parser = argparse.ArgumentParser(description="Submit training task to Taiji")
    parser.add_argument("task_flag", help="Task identifier (e.g. my_train_v1)")
    parser.add_argument("config_path", nargs='?', default='__UNUSED__',
                        help="Config file path relative to hf_trainer "
                             "(omitted when --start-cmd is used)")
    parser.add_argument("--host_num", type=int, default=4, help="Number of hosts (default: 4, each with 8 GPUs)")
    parser.add_argument("--host_gpu_num", type=int, default=None,
                        help="Override per-host GPU count (default from template)")
    parser.add_argument("--gpu_name", default=None,
                        help="Override GPU type, e.g. V100 or A100")
    parser.add_argument("--cuda_version", default=None,
                        help="Override requested CUDA version, e.g. 11.4")
    parser.add_argument("--elastic", action="store_true", help="Use elastic (preemptible) GPUs")
    parser.add_argument("--business_flag", "-b", default=None, help="Override business flag")
    parser.add_argument("--start-cmd", default=None,
                        help="Replace the templated training start_cmd with a "
                             "custom one (e.g. for inference/eval jobs).  When "
                             "set, config_path is ignored.")
    args = parser.parse_args()

    submit(args.task_flag, args.config_path, args.host_num, args.business_flag,
           args.elastic, start_cmd_override=args.start_cmd,
           host_gpu_num=args.host_gpu_num, gpu_name=args.gpu_name,
           cuda_version=args.cuda_version)


if __name__ == "__main__":
    main()
