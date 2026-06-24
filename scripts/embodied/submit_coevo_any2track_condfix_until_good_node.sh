#!/usr/bin/env bash
# Resubmit the Any2Track co-evolution run after the JAX callback cond fix.
set -uo pipefail

TOKEN="${TOKEN:-HzrPZC3djhwaU9HPdEA_Bg}"
REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
DOCKER="${DOCKER:-mirrors.tencent.com/zeyuling_mirrors/vermo:latest}"
BUSINESS_FLAG="${BUSINESS_FLAG:-AILab_DHA}"
GPU="${GPU:-V100}"
NUM_GPU="${NUM_GPU:-4}"
NUM_HOST="${NUM_HOST:-1}"
CUDA_VERSION="${CUDA_VERSION:-11.0}"
MAXATT="${MAXATT:-8}"
GATE_POLLS="${GATE_POLLS:-180}"
GATE_SLEEP_SEC="${GATE_SLEEP_SEC:-10}"

ROOT="${ROOT:-work_dirs/physflow_coevolve_any2track_hymotion130k_formal_modelmeta_cuda114_4g_0621_a2}"
ARM="${ARM:-any2track_hymotion130k_closedloop_formal_modelmeta_cuda114_4g_0621_a2}"
LOG_PREFIX="${LOG_PREFIX:-work_dirs/physflow_coevo_any2track_formal_modelmeta_cuda114_4g_0621_condfix}"
TASK_PREFIX="${TASK_PREFIX:-physflow-coevoa2t-condfix0621}"

cd "${REPO}"
export TOKEN

for att in $(seq 1 "${MAXATT}"); do
  log="${LOG_PREFIX}_a${att}.log"
  name="${TASK_PREFIX}-${att}"
  remote_cmd="cd ${REPO} && mkdir -p work_dirs && export MIN_CUDA_DRV=11.4 MAX_CUDA_DRV=11.4 OPENTRACK_CUDA_FLAVOR=cuda11 OPENTRACK_VENV_DIR=.venv_physflow_opentrack_cuda114_smoke_orbax016_0619 SKIP_UV_SYNC=1 NUM_GPUS=4 GENCKPT=work_dirs/physflow_verify_hymotion_g1_base/checkpoint-iter_130000 ROOT=${ROOT} ARM=${ARM} NROUNDS=10 GEN_ITERS=120 A2T_TIMESTEPS=2000000000 A2T_ADV_MAX_FILES=96 A2T_ADV_SELECTION_STRATEGY=evenly A2T_ADV_SELECTION_SEED=20260621 ADV_PROB=0.35; bash scripts/embodied/physflow_coevo_any2track_latest_node.sh > ${log} 2>&1; status=\$?; echo COEVO_A2TFORMAL_CONDFIX_EXIT=\$status >> ${log}; exit \$status"

  echo "[coevo-a2t-condfix-submit] === attempt ${att}: submit ${name} gpu=${GPU} num_gpu=${NUM_GPU} cuda=${CUDA_VERSION} ==="
  out=$(python3 .claude/skills/taiji/taiji_ops.py submit --token "${TOKEN}" -n "${name}" \
    --gpu "${GPU}" --num_gpu "${NUM_GPU}" --num_host "${NUM_HOST}" -b "${BUSINESS_FLAG}" \
    --docker "${DOCKER}" --cuda-version "${CUDA_VERSION}" --cmd "${remote_cmd}" --no-confirm 2>&1)
  tf=$(echo "${out}" | grep -oE "task_flag: +[^ ]+" | head -1 | awk '{print $2}')
  echo "[coevo-a2t-condfix-submit] task_flag=${tf}; polling driver gate..."

  good=""
  bad=""
  ended=""
  for _ in $(seq 1 "${GATE_POLLS}"); do
    sleep "${GATE_SLEEP_SEC}"
    if grep -q "FATAL_BAD_NODE" "${log}" 2>/dev/null; then
      bad=1
      break
    fi
    if grep -q "host CUDA driver version: 11.4" "${log}" 2>/dev/null; then
      good=1
      break
    fi
    if [[ -n "${tf}" ]] && taiji_client il "${tf}" 2>/dev/null | grep -qE '\|[[:space:]]*false[[:space:]]*\|[[:space:]]*END[[:space:]]*\|'; then
      bad=1
      ended=1
      break
    fi
  done

  if [[ -n "${good}" ]]; then
    echo "[coevo-a2t-condfix-submit] GOOD NODE on attempt ${att} (task ${tf}). Leaving job running."
    echo "${tf}" > "${LOG_PREFIX}_good_task.txt"
    echo "GOOD_NODE_FOUND attempt=${att} task=${tf} log=${log}"
    exit 0
  fi

  echo "[coevo-a2t-condfix-submit] attempt ${att} bad/timeout (bad=${bad:-0} ended=${ended:-0}); stopping ${tf}"
  if [[ -n "${tf}" ]]; then
    taiji_client stop "${tf}" >/dev/null 2>&1 || true
  fi
  sleep 5
done

echo "[coevo-a2t-condfix-submit] EXHAUSTED ${MAXATT} attempts without a CUDA 11.4 node"
exit 1
