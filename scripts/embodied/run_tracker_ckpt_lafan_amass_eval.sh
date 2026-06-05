#!/usr/bin/env bash
# Evaluate one tracker checkpoint against the released G1 ProtoMotions tracker
# on the local LAFAN1-G1 and AMASS-G1 benchmark pipelines.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
CKPT_NAME="${CKPT_NAME:-candidate}"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the tracker checkpoint}"
OUT_TAG="${OUT_TAG:-$(date +%Y%m%d_%H%M%S)}"

LAFAN_BASELINE_OUT="${LAFAN_BASELINE_OUT:-${PROJECT_ROOT}/output/lafan1_g1_proto_baseline_eval/lafan1_g1_proto_baseline_eval_20260605_001420}"
AMASS_BASELINE_OUT="${AMASS_BASELINE_OUT:-${PROJECT_ROOT}/output/amass_g1_proto_baseline_eval/debug2_20260604_1904_wxyz_4gpu}"
AMASS_NUM_SHARDS_EFFECTIVE="${AMASS_NUM_SHARDS:-4}"
AMASS_NUM_ENVS_EFFECTIVE="${AMASS_NUM_ENVS:-256}"

LAFAN_OUT="${LAFAN_OUT:-${PROJECT_ROOT}/output/lafan1_g1_proto_baseline_eval/${OUT_TAG}}"
AMASS_OUT="${AMASS_OUT:-${PROJECT_ROOT}/output/amass_g1_proto_baseline_eval/${OUT_TAG}}"

cd "${PROJECT_ROOT}"
mkdir -p "${LAFAN_OUT}" "${AMASS_OUT}"
PREFLIGHT_LOG="${LAFAN_OUT}/preflight.log"
{
    echo "[tracker-eval] preflight $(date)"
    echo "[tracker-eval] host=$(hostname)"
    echo "[tracker-eval] ckpt_name=${CKPT_NAME}"
    echo "[tracker-eval] ckpt_path=${CKPT_PATH}"
    echo "[tracker-eval] out_tag=${OUT_TAG}"
} | tee -a "${PREFLIGHT_LOG}"

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[tracker-eval] ERROR: checkpoint missing: ${CKPT_PATH}" | tee -a "${PREFLIGHT_LOG}" >&2
    exit 2
fi
if [[ ! -f "$(dirname "${CKPT_PATH}")/resolved_configs_inference.pt" ]]; then
    echo "[tracker-eval] ERROR: missing resolved_configs_inference.pt next to ${CKPT_PATH}" | tee -a "${PREFLIGHT_LOG}" >&2
    exit 3
fi
if [[ -n "${REQUIRE_MIN_DRIVER_MAJOR:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    driver_major="$(
        nvidia-smi --query-gpu=driver_version --format=csv,noheader |
            head -n 1 |
        tr -d '[:space:]' |
            cut -d. -f1
    )"
    echo "[tracker-eval] driver_major=${driver_major} require>=${REQUIRE_MIN_DRIVER_MAJOR}" | tee -a "${PREFLIGHT_LOG}"
    if [[ "${driver_major}" =~ ^[0-9]+$ ]] && (( driver_major < REQUIRE_MIN_DRIVER_MAJOR )); then
        echo "[tracker-eval] ERROR: bad node driver ${driver_major}; require >= ${REQUIRE_MIN_DRIVER_MAJOR}" | tee -a "${PREFLIGHT_LOG}" >&2
        exit 42
    fi
fi

if [[ -d "${LAFAN_BASELINE_OUT}/motion_shards" ]]; then
    ln -sfn "${LAFAN_BASELINE_OUT}/motion_shards" "${LAFAN_OUT}/motion_shards"
fi
if [[ -d "${AMASS_BASELINE_OUT}/motion_shards" ]]; then
    amass_cache_count=$(find "${AMASS_BASELINE_OUT}/motion_shards" -maxdepth 1 -type f -name 'amass_g1_full_shard_*.pt' | wc -l)
    amass_cache_last="${AMASS_BASELINE_OUT}/motion_shards/amass_g1_full_shard_$((AMASS_NUM_SHARDS_EFFECTIVE - 1)).pt"
    if [[ "${amass_cache_count}" -eq "${AMASS_NUM_SHARDS_EFFECTIVE}" && -s "${amass_cache_last}" ]]; then
        ln -sfn "${AMASS_BASELINE_OUT}/motion_shards" "${AMASS_OUT}/motion_shards"
    else
        echo "[tracker-eval] AMASS cache shard count ${amass_cache_count} does not match requested ${AMASS_NUM_SHARDS_EFFECTIVE}; not linking cache"
    fi
fi

CHECKPOINT_SPECS="protomotions_g1_bones=data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt,${CKPT_NAME}=${CKPT_PATH}"
export CHECKPOINT_SPECS

echo "[tracker-eval] ckpt_name=${CKPT_NAME}"
echo "[tracker-eval] ckpt_path=${CKPT_PATH}"
echo "[tracker-eval] out_tag=${OUT_TAG}"
echo "[tracker-eval] lafan_out=${LAFAN_OUT}"
echo "[tracker-eval] amass_out=${AMASS_OUT}"

if [[ "${RUN_LAFAN:-1}" == "1" ]]; then
    OUT_ROOT="${LAFAN_OUT}" \
    NUM_SHARDS="${LAFAN_NUM_SHARDS:-8}" \
    NUM_ENVS="${LAFAN_NUM_ENVS:-64}" \
    MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-600}" \
    RUN_NODE_SETUP="${RUN_NODE_SETUP:-1}" \
    REBUILD_GYMTORCH="${REBUILD_GYMTORCH:-0}" \
    FORCE_EVAL="${FORCE_EVAL:-1}" \
    SAVE_PREDICTED_MOTION_LIB="${SAVE_PREDICTED_MOTION_LIB:-0}" \
    PREDICTED_MOTION_LIB_EVERY="${PREDICTED_MOTION_LIB_EVERY:-1}" \
    bash scripts/embodied/run_lafan1_g1_proto_baseline_eval.sh
else
    echo "[tracker-eval] run_lafan=0; skipping LAFAN"
fi

if [[ "${RUN_AMASS:-1}" != "1" ]]; then
    echo "[tracker-eval] run_amass=0; stopping after LAFAN"
    echo "[tracker-eval] lafan_summary=${LAFAN_OUT}/summary.md"
    exit 0
fi

if [[ "${RUN_LAFAN:-1}" == "1" ]]; then
    AMASS_RUN_NODE_SETUP=0
else
    AMASS_RUN_NODE_SETUP="${RUN_NODE_SETUP:-1}"
fi

OUT_ROOT="${AMASS_OUT}" \
NUM_SHARDS="${AMASS_NUM_SHARDS_EFFECTIVE}" \
NUM_ENVS="${AMASS_NUM_ENVS_EFFECTIVE}" \
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-600}" \
RUN_NODE_SETUP="${AMASS_RUN_NODE_SETUP}" \
REBUILD_GYMTORCH="${REBUILD_GYMTORCH:-0}" \
FORCE_EVAL="${FORCE_EVAL:-1}" \
SAVE_PREDICTED_MOTION_LIB="${SAVE_PREDICTED_MOTION_LIB:-0}" \
PREDICTED_MOTION_LIB_EVERY="${PREDICTED_MOTION_LIB_EVERY:-1}" \
bash scripts/embodied/run_amass_g1_proto_baseline_eval.sh

echo "[tracker-eval] lafan_summary=${LAFAN_OUT}/summary.md"
echo "[tracker-eval] amass_summary=${AMASS_OUT}/summary.md"
