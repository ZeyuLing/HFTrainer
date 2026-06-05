#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "${PROJECT_ROOT}"

if [[ -z "${TOKEN:-}" ]]; then
    for token_file in /root/.claude-dashboard/taiji_token /root/.codex/skills/taiji/.token; do
        if [[ -r "${token_file}" ]]; then
            TOKEN="$(<"${token_file}")"
            export TOKEN
            break
        fi
    done
fi
if [[ -z "${TOKEN:-}" ]]; then
    echo "ERROR: TOKEN is not set and no readable Taiji token file was found." >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TASK_FLAG="${TASK_FLAG:-amass_g1_proto_baseline_eval_${TIMESTAMP}}"
HOST_NUM="${HOST_NUM:-1}"
HOST_GPU_NUM="${HOST_GPU_NUM:-8}"
BUSINESS_FLAG="${BUSINESS_FLAG:-AILab_DHC_DD}"

START_CMD=$(cat <<'EOF'
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer && \
bash scripts/embodied/run_amass_g1_proto_baseline_eval.sh
EOF
)

python3 tools/taiji_submit.py "${TASK_FLAG}" __UNUSED__ \
    --host_num "${HOST_NUM}" \
    --host_gpu_num "${HOST_GPU_NUM}" \
    --business_flag "${BUSINESS_FLAG}" \
    --start-cmd "${START_CMD}"
