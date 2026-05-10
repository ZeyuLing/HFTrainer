#!/usr/bin/env bash
set -euo pipefail

# =========================
# submit_ablation_taiji.sh — Submit all M2M ablation experiments to Taiji cluster
#
# Usage:
#   bash scripts/submit_ablation_taiji.sh [experiment_name|all]
#
# Examples:
#   bash scripts/submit_ablation_taiji.sh all          # Submit all 9 experiments
#   bash scripts/submit_ablation_taiji.sh m2_baseline  # Submit only baseline
# =========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="${1:-all}"

# Shared Taiji config values (from existing tasks)
BUSINESS_FLAG="AILab_DHA"
DATASET_ID="8b1d81349b30baad019b5032e89126cc"
MODEL_ID="8b1d80659b30baa9019b5032ea7e271a"
IMAGE="mirrors.tencent.com/zeyuling_mirrors/vermo:latest"
GPU_NAME="A100"
HOST_GPU_NUM=8
HOST_NUM=2  # 2 nodes × 8 GPUs = 16 GPUs total
LOCATION="cq"  # chongqing — data on cq ceph, cross-region mount not allowed

# Experiment definitions: name -> config file
declare -A EXPERIMENTS=(
    ["m2_baseline"]="ablation_m2_baseline.py"
    ["m1_t2m_only"]="ablation_m1_t2m_only.py"
    ["m3_t2m_heavy"]="ablation_m3_t2m_heavy.py"
    ["l1_fk_loss"]="ablation_l1_fk_loss.py"
    ["l3a_trans_w1"]="ablation_l3a_trans_w1.py"
    ["l3b_trans_w10"]="ablation_l3b_trans_w10.py"
    ["l4_velocity_loss"]="ablation_l4_velocity_loss.py"
    ["t1_ema"]="ablation_t1_ema.py"
    ["t2_curriculum_p1"]="ablation_t2_curriculum_p1.py"
)

submit_experiment() {
    local exp_name="$1"
    local config_file="${EXPERIMENTS[$exp_name]}"
    local task_flag="ablation_m2m_${exp_name}"
    local config_path="configs/hymotion_m2m/ablation/${config_file}"

    local start_cmd="cd ${PROJ_ROOT}; bash tools/taiji_dist_train.sh ${config_path}"

    echo "Submitting: ${task_flag}"
    echo "  Config: ${config_path}"
    echo "  GPUs: ${HOST_NUM} × ${HOST_GPU_NUM} = $((HOST_NUM * HOST_GPU_NUM))"

    # Create Taiji simple config JSON
    local cfg_json="/tmp/taiji_ablation_${exp_name}.json"
    cat > "${cfg_json}" <<CFGEOF
{
    "business_flag": "${BUSINESS_FLAG}",
    "mount_ceph_business_flag": "${BUSINESS_FLAG}",
    "dataset_id": "${DATASET_ID}",
    "model_id": "${MODEL_ID}",
    "host_num": ${HOST_NUM},
    "host_gpu_num": ${HOST_GPU_NUM},
    "GPUName": "${GPU_NAME}",
    "image_full_name": "${IMAGE}",
    "task_flag": "${task_flag}",
    "start_cmd": "${start_cmd}",
    "enable_fault_tolerance": true,
    "is_resource_waiting": true,
    "location": "${LOCATION}",
    "extra_plat_business": "AILab_DHC_DD,AILab_DHC_private_data",
    "is_enable_ssh_without_password": true,
    "priority_level": "HIGH"
}
CFGEOF

    echo "  Config JSON: ${cfg_json}"

    # Submit to Taiji
    taiji_client create -t task --scfg "${cfg_json}" 2>&1 || {
        echo "  WARNING: create failed, trying start..."
    }
    taiji_client start --tf "${task_flag}" --scfg "${cfg_json}" 2>&1 || {
        echo "  ERROR: Failed to start ${task_flag}"
        return 1
    }

    echo "  ✓ Submitted: ${task_flag}"
    echo ""
}

echo "============================================="
echo "HyMotion M2M Ablation — Taiji Submission"
echo "============================================="
echo "Location: ${LOCATION} (${GPU_NAME})"
echo "Per-experiment: ${HOST_NUM} nodes × ${HOST_GPU_NUM} GPUs = $((HOST_NUM * HOST_GPU_NUM)) GPUs"
echo ""

if [[ "${TARGET}" == "all" ]]; then
    echo "Submitting all ${#EXPERIMENTS[@]} experiments..."
    echo ""
    for exp_name in m2_baseline m1_t2m_only m3_t2m_heavy l1_fk_loss l3a_trans_w1 l3b_trans_w10 l4_velocity_loss t1_ema t2_curriculum_p1; do
        submit_experiment "${exp_name}"
        sleep 3
    done
    echo "All experiments submitted."
else
    if [[ -n "${EXPERIMENTS[$TARGET]+x}" ]]; then
        submit_experiment "${TARGET}"
    else
        echo "Unknown experiment: ${TARGET}"
        echo "Available: ${!EXPERIMENTS[*]}"
        exit 1
    fi
fi

echo ""
echo "Monitor with: taiji_client task_running_list"
echo "Check logs:   taiji_client logs --tf ablation_m2m_<name>"
