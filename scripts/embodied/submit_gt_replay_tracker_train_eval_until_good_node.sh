#!/usr/bin/env bash
# Submit GT-only tracker replay sanity check until Taiji lands on a usable node.
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
CUDA_VERSION="${CUDA_VERSION:-11.0}"
TAG="${TAG:-amass_sanity}"
NUM_GPU="${NUM_GPU:-1}"
NUM_HOST="${NUM_HOST:-1}"

export TOKEN
if [[ -z "${TOKEN}" ]]; then
  echo "[gt-replay-submit] ERROR: TOKEN is empty"
  exit 2
fi

for att in $(seq 1 "${MAXATT}"); do
  log="work_dirs/physflow_gt_replay_tracker_${TAG}_a${att}.log"
  : > "${log}"
  name="physflow_gtreplay_${TAG}_a${att}"
  echo "[gt-replay-submit] === attempt ${att}: submit ${name} gpu=${GPU} num_gpu=${NUM_GPU} business=${BUSINESS_FLAG} cuda=${CUDA_VERSION} ==="
  out=$(python3 .claude/skills/taiji/taiji_ops.py submit --token "${TOKEN}" -n "${name}" \
    --gpu "${GPU}" --num_gpu "${NUM_GPU}" --num_host "${NUM_HOST}" -b "${BUSINESS_FLAG}" --docker "${DOCKER}" --cuda-version "${CUDA_VERSION}" \
    --cmd "cd ${REPO} && RUN_TAG='${TAG}' NGPU='${NUM_GPU}' bash scripts/embodied/run_gt_replay_tracker_train_eval.sh > ${log} 2>&1; echo GT_REPLAY_EXIT=\$? >> ${log}" \
    --no-confirm 2>&1)
  tf=$(echo "${out}" | grep -oE "task_flag: +[^ ]+" | head -1 | awk '{print $2}')
  echo "[gt-replay-submit] task_flag=${tf} ; polling driver gate (<= $((GATE_POLLS * GATE_SLEEP_SEC / 60)) min)..."
  good=""; bad=""; ended=""
  for _ in $(seq 1 "${GATE_POLLS}"); do
    sleep "${GATE_SLEEP_SEC}"
    if grep -q "FATAL_BAD_NODE" "${log}" 2>/dev/null; then
      bad=1
      break
    fi
    if grep -q "driver gate OK" "${log}" 2>/dev/null; then
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
    drv=$(grep -oE "host CUDA driver version: [0-9.]+" "${log}" | head -1)
    echo "[gt-replay-submit] GOOD NODE on attempt ${att} (task ${tf}, ${drv}). Leaving job running."
    echo "${tf}" > "work_dirs/physflow_gt_replay_tracker_${TAG}_good_task.txt"
    echo "GOOD_NODE_FOUND attempt=${att} task=${tf} log=${log}"
    exit 0
  fi
  echo "[gt-replay-submit] attempt ${att} bad/timeout (bad=${bad:-0} ended=${ended:-0}); stopping ${tf}"
  if [[ -n "${tf}" ]]; then
    taiji_client stop "${tf}" >/dev/null 2>&1 || true
  fi
  sleep 5
done

echo "[gt-replay-submit] EXHAUSTED ${MAXATT} attempts without a good node"
exit 1
