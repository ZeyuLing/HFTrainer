#!/usr/bin/env bash
# Submit one HYMotion G1 checkpoint quick-eval until Taiji lands on a usable V100.
set -uo pipefail

TOKEN="${TOKEN:-HzrPZC3djhwaU9HPdEA_Bg}"
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
DOCKER=mirrors.tencent.com/zeyuling_mirrors/vermo:latest
cd "$REPO"

MAXATT="${MAXATT:-8}"
GATE_POLLS="${GATE_POLLS:-180}"
GATE_SLEEP_SEC="${GATE_SLEEP_SEC:-10}"
EVAL_GPU="${EVAL_GPU:-V100}"
EVAL_BUSINESS_FLAG="${EVAL_BUSINESS_FLAG:-AILab_DHA}"
EVAL_DOCKER="${EVAL_DOCKER:-$DOCKER}"
TAIJI_CUDA_VERSION="${TAIJI_CUDA_VERSION:-11.0}"
TAG="${TAG:-hymotion_g1_eval}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/physflow/hymotion_g1_t2m_38dim_long.py}"
EVAL_CKPT="${EVAL_CKPT:?set EVAL_CKPT}"
EVAL_OUT="${EVAL_OUT:?set EVAL_OUT}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-24}"
EVAL_MAX_ITEMS="${EVAL_MAX_ITEMS:-4096}"
EVAL_SAMPLE_STEPS="${EVAL_SAMPLE_STEPS:-30}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_SEED="${EVAL_SEED:-20260615}"
EVAL_SCORE_GT="${EVAL_SCORE_GT:---score-gt}"
export TOKEN

if [ -z "$TOKEN" ]; then
  echo "[hymotion-g1-eval-submit] ERROR: TOKEN is empty"
  exit 2
fi

for att in $(seq 1 "$MAXATT"); do
  log="work_dirs/physflow_hymotion_g1_eval_${TAG}_a${att}.log"
  : > "$log"
  name="physflow_hg1_eval_${TAG}_a${att}"
  echo "[hymotion-g1-eval-submit] === attempt $att: submit $name gpu=$EVAL_GPU business=$EVAL_BUSINESS_FLAG cuda=$TAIJI_CUDA_VERSION ==="
  out=$(python3 .claude/skills/taiji/taiji_ops.py submit --token "$TOKEN" -n "$name" \
    --gpu "$EVAL_GPU" --num_gpu 1 --num_host 1 -b "$EVAL_BUSINESS_FLAG" --docker "$EVAL_DOCKER" --cuda-version "$TAIJI_CUDA_VERSION" \
    --cmd "cd $REPO && EVAL_CONFIG='$EVAL_CONFIG' EVAL_CKPT='$EVAL_CKPT' EVAL_OUT='$EVAL_OUT' EVAL_NUM_SAMPLES='$EVAL_NUM_SAMPLES' EVAL_MAX_ITEMS='$EVAL_MAX_ITEMS' EVAL_SAMPLE_STEPS='$EVAL_SAMPLE_STEPS' EVAL_BATCH_SIZE='$EVAL_BATCH_SIZE' EVAL_SEED='$EVAL_SEED' EVAL_SCORE_GT='$EVAL_SCORE_GT' bash scripts/embodied/eval_hymotion_g1_checkpoint_node.sh > $log 2>&1; echo HYMOTION_G1_EVAL_EXIT=\$? >> $log" \
    --no-confirm 2>&1)
  tf=$(echo "$out" | grep -oE "task_flag: +[^ ]+" | head -1 | awk '{print $2}')
  echo "[hymotion-g1-eval-submit] task_flag=$tf ; polling driver gate (<= $((GATE_POLLS * GATE_SLEEP_SEC / 60)) min)..."
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
    if taiji_client il "$tf" 2>/dev/null | grep -qE '\|[[:space:]]*false[[:space:]]*\|[[:space:]]*END[[:space:]]*\|'; then
      bad=1
      ended=1
      break
    fi
  done
  if [ -n "$good" ]; then
    drv=$(grep -oE "host CUDA driver version: [0-9.]+" "$log" | head -1)
    echo "[hymotion-g1-eval-submit] GOOD NODE on attempt $att (task $tf, $drv). Leaving eval job running."
    echo "$tf" > "work_dirs/physflow_hymotion_g1_eval_${TAG}_good_task.txt"
    echo "GOOD_NODE_FOUND attempt=$att task=$tf log=$log"
    exit 0
  fi
  echo "[hymotion-g1-eval-submit] attempt $att bad/timeout (bad=${bad:-0} ended=${ended:-0}); stopping $tf"
  taiji_client stop "$tf" >/dev/null 2>&1 || true
  sleep 5
done

echo "[hymotion-g1-eval-submit] EXHAUSTED $MAXATT attempts without a good node"
exit 1
