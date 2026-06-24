#!/usr/bin/env bash
# Submit hardstable fixed-noise visualization until it lands on a CUDA 11.4 V100.
set -uo pipefail

TOKEN="${TOKEN:-HzrPZC3djhwaU9HPdEA_Bg}"
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
DOCKER=mirrors.tencent.com/zeyuling_mirrors/vermo:latest
cd "$REPO"

MAXATT="${MAXATT:-8}"
GATE_POLLS="${GATE_POLLS:-180}"
GATE_SLEEP_SEC="${GATE_SLEEP_SEC:-10}"
GPU="${GPU:-V100}"
NUM_GPU="${NUM_GPU:-1}"
NUM_HOST="${NUM_HOST:-1}"
BUSINESS_FLAG="${BUSINESS_FLAG:-AILab_DHA}"
DOCKER_IMAGE="${DOCKER_IMAGE:-$DOCKER}"
TAIJI_CUDA_VERSION="${TAIJI_CUDA_VERSION:-11.4}"
TAG="${TAG:-hardstable_fixed_noise_0621}"

if [ -z "$TOKEN" ]; then
  echo "[hardstable-fixed-noise-submit] ERROR: TOKEN is empty"
  exit 2
fi

for att in $(seq 1 "$MAXATT"); do
  log="work_dirs/physflow_hardstable_fixed_noise_${TAG}_a${att}.log"
  : > "$log"
  name="physflow-hardstable-fn-${TAG}-a${att}"
  echo "[hardstable-fixed-noise-submit] === attempt $att: submit $name gpu=${GPU} num_gpu=${NUM_GPU} cuda=$TAIJI_CUDA_VERSION ==="
  out=$(python3 .claude/skills/taiji/taiji_ops.py submit --token "$TOKEN" -n "$name" \
    --gpu "$GPU" --num_gpu "$NUM_GPU" --num_host "$NUM_HOST" -b "$BUSINESS_FLAG" --docker "$DOCKER_IMAGE" --cuda-version "$TAIJI_CUDA_VERSION" \
    --cmd "cd $REPO && HARDSTABLE_CONFIG='${HARDSTABLE_CONFIG:-configs/physflow/verify_hymotion_g1_any2track_130k_hardstable_0620.py}' HARDSTABLE_BASE_CKPT='${HARDSTABLE_BASE_CKPT:-work_dirs/physflow_verify_hymotion_g1_base/checkpoint-iter_130000}' HARDSTABLE_OPT_CKPT='${HARDSTABLE_OPT_CKPT:-work_dirs/physflow_verify_hymotion_g1_any2track_130k_hardstable_0620/checkpoint-iter_6000}' HARDSTABLE_RUN_ROOT='${HARDSTABLE_RUN_ROOT:-output/physflow_fixed_noise_hardstable_any2track_compare_0621}' HARDSTABLE_VIZ_DIR='${HARDSTABLE_VIZ_DIR:-output/physflow_visualizations/hardstable_any2track_fixed_noise}' HARDSTABLE_NUM_SAMPLES='${HARDSTABLE_NUM_SAMPLES:-24}' HARDSTABLE_MAX_ITEMS='${HARDSTABLE_MAX_ITEMS:-4096}' HARDSTABLE_SAMPLE_STEPS='${HARDSTABLE_SAMPLE_STEPS:-30}' HARDSTABLE_SEED='${HARDSTABLE_SEED:-20260615}' bash scripts/embodied/fixed_noise_hardstable_compare_node.sh > $log 2>&1; status=\$?; echo HARDSTABLE_FIXED_NOISE_EXIT=\$status >> $log; exit \$status" \
    --no-confirm 2>&1)
  tf=$(echo "$out" | grep -oE "task_flag: +[^ ]+" | head -1 | awk '{print $2}')
  echo "[hardstable-fixed-noise-submit] task_flag=$tf ; polling driver gate (<= $((GATE_POLLS * GATE_SLEEP_SEC / 60)) min)..."
  good=""; bad=""; ended=""
  for _ in $(seq 1 "$GATE_POLLS"); do
    sleep "$GATE_SLEEP_SEC"
    if grep -q "FATAL_BAD_NODE" "$log" 2>/dev/null; then
      bad=1
      break
    fi
    if grep -q "driver gate OK" "$log" 2>/dev/null; then
      good=1
      break
    fi
    if [ -n "$tf" ] && taiji_client il "$tf" 2>/dev/null | grep -qE '\|[[:space:]]*false[[:space:]]*\|[[:space:]]*END[[:space:]]*\|'; then
      bad=1
      ended=1
      break
    fi
  done
  if [ -n "$good" ]; then
    drv=$(grep -oE "host CUDA driver version: [0-9.]+" "$log" | head -1)
    echo "[hardstable-fixed-noise-submit] GOOD NODE on attempt $att (task $tf, $drv). Leaving eval job running."
    echo "$tf" > "work_dirs/physflow_hardstable_fixed_noise_${TAG}_good_task.txt"
    echo "GOOD_NODE_FOUND attempt=$att task=$tf log=$log"
    exit 0
  fi
  echo "[hardstable-fixed-noise-submit] attempt $att bad/timeout (bad=${bad:-0} ended=${ended:-0}); stopping $tf"
  [ -n "$tf" ] && taiji_client stop "$tf" >/dev/null 2>&1 || true
  sleep 5
done

echo "[hardstable-fixed-noise-submit] EXHAUSTED $MAXATT attempts without a good node"
exit 1
