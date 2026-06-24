#!/usr/bin/env bash
# Submit formal replay Any2Track/OpenTrack training until it lands on CUDA 11.4.
set -uo pipefail

TOKEN="${TOKEN:-HzrPZC3djhwaU9HPdEA_Bg}"
REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
DOCKER="${DOCKER:-mirrors.tencent.com/zeyuling_mirrors/vermo:latest}"
cd "${REPO}"

MAXATT="${MAXATT:-8}"
GATE_POLLS="${GATE_POLLS:-180}"
GATE_SLEEP_SEC="${GATE_SLEEP_SEC:-10}"
GPU="${GPU:-V100}"
BUSINESS_FLAG="${BUSINESS_FLAG:-AILab_DHA}"
CUDA_VERSION="${CUDA_VERSION:-11.4}"
TAG="${TAG:-physflow_gen_any2track_formal2k_fkmeta_cuda114_4g_20260621b}"
NUM_GPU="${NUM_GPU:-4}"
NUM_HOST="${NUM_HOST:-1}"
NUM_GPUS="${NUM_GPUS:-${NUM_GPU}}"
POOL="${POOL:-${REPO}/output/generator_tracker_replay/physflow_hg1_formal2k_20260617}"

export TOKEN
if [[ -z "${TOKEN}" ]]; then
  echo "[formal-replay-a2t-submit] ERROR: TOKEN is empty"
  exit 2
fi

for att in $(seq 1 "${MAXATT}"); do
  log="work_dirs/physflow_replay_any2track_formal_${TAG}_a${att}.log"
  : > "${log}"
  name="physflow-replaya2t-114-${att}"
  echo "[formal-replay-a2t-submit] === attempt ${att}: submit ${name} gpu=${GPU} num_gpu=${NUM_GPU} business=${BUSINESS_FLAG} cuda=${CUDA_VERSION} ==="
  out=$(python3 .claude/skills/taiji/taiji_ops.py submit --token "${TOKEN}" -n "${name}" \
    --gpu "${GPU}" --num_gpu "${NUM_GPU}" --num_host "${NUM_HOST}" -b "${BUSINESS_FLAG}" --docker "${DOCKER}" --cuda-version "${CUDA_VERSION}" \
    --cmd "cd ${REPO} && POOL='${POOL}' TAG='${TAG}_a${att}' NUM_GPUS='${NUM_GPUS}' OPENTRACK_VENV_DIR='.venv_physflow_any2track_formal2k_cuda114_fkmeta' bash scripts/embodied/physflow_formal_replay_any2track_node.sh > ${log} 2>&1; echo REPLAY_A2T_FORMAL_EXIT=\$? >> ${log}" \
    --no-confirm 2>&1)
  tf=$(echo "${out}" | grep -oE "task_flag: +[^ ]+" | head -1 | awk '{print $2}')
  echo "[formal-replay-a2t-submit] task_flag=${tf} ; polling driver gate (<= $((GATE_POLLS * GATE_SLEEP_SEC / 60)) min)..."
  good=""; bad=""; ended=""
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
    echo "[formal-replay-a2t-submit] GOOD NODE on attempt ${att} (task ${tf}). Leaving job running."
    echo "${tf}" > "work_dirs/physflow_replay_any2track_formal_${TAG}_good_task.txt"
    echo "GOOD_NODE_FOUND attempt=${att} task=${tf} log=${log}"
    exit 0
  fi
  echo "[formal-replay-a2t-submit] attempt ${att} bad/timeout (bad=${bad:-0} ended=${ended:-0}); stopping ${tf}"
  if [[ -n "${tf}" ]]; then
    taiji_client stop "${tf}" >/dev/null 2>&1 || true
  fi
  sleep 5
done

echo "[formal-replay-a2t-submit] EXHAUSTED ${MAXATT} attempts without a CUDA 11.4 node"
exit 1
