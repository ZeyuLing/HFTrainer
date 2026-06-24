#!/usr/bin/env bash
# Submit the restart-sliding action-distill AGILE run until Taiji lands on a
# CUDA>=11.4 V100 node.
set -uo pipefail
TOKEN="${TOKEN:-HzrPZC3djhwaU9HPdEA_Bg}"
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
DOCKER=mirrors.tencent.com/zeyuling_mirrors/vermo:latest
cd "$REPO"
MAXATT="${MAXATT:-15}"
GATE_POLLS="${GATE_POLLS:-180}"
GATE_SLEEP_SEC="${GATE_SLEEP_SEC:-10}"
TAG="${TAG:-adistill1}"
COEFF="${COEFF:-1.0}"
START_ROUND="${START_ROUND:-2}"
NUM_ROUNDS="${NUM_ROUNDS:-3}"
BOOTSTRAP_FROM_SRC="${BOOTSTRAP_FROM_SRC:-1}"
BOOTSTRAP_UP_TO_ROUND="${BOOTSTRAP_UP_TO_ROUND:-}"
SRC_ARM="${SRC_ARM:-}"
TRAINEE_SNAPSHOT_MODE="${TRAINEE_SNAPSHOT_MODE:-base-plus-latest}"
TRAINEE_EPOCHS="${TRAINEE_EPOCHS:-20}"
EXTRA_TRAINEE_OVERRIDES="${EXTRA_TRAINEE_OVERRIDES:-}"
EXTRA_GEN_CFG_OPTIONS="${EXTRA_GEN_CFG_OPTIONS:-}"
EXTRA_GEN_CFG_OPTIONS_BY_ROUND="${EXTRA_GEN_CFG_OPTIONS_BY_ROUND:-}"
FIXED_REPLAY_ANNO="${FIXED_REPLAY_ANNO:-}"
FIXED_REPLAY_BANK="${FIXED_REPLAY_BANK:-}"
FIXED_REPLAY_PREFIX="${FIXED_REPLAY_PREFIX:-fixed_}"
export TOKEN

for att in $(seq 1 "$MAXATT"); do
  log="work_dirs/physflow_hardovf_restart_sliding_${TAG}_a${att}.log"
  : > "$log"
  name="physflow_hovf_${TAG}_a${att}"
  echo "[retry] === attempt $att: submit $name ==="
	  out=$(python3 .claude/skills/taiji/taiji_ops.py submit --token "$TOKEN" -n "$name" \
	    --gpu V100 --num_gpu 1 --num_host 1 -b AILab_DHA --docker "$DOCKER" \
	    --cmd "cd $REPO && ADISTILL_TAG=$TAG ACTION_DISTILL_COEFF=$COEFF START_ROUND=$START_ROUND NUM_ROUNDS=$NUM_ROUNDS BOOTSTRAP_FROM_SRC=$BOOTSTRAP_FROM_SRC BOOTSTRAP_UP_TO_ROUND='$BOOTSTRAP_UP_TO_ROUND' TRAINEE_SNAPSHOT_MODE=$TRAINEE_SNAPSHOT_MODE TRAINEE_EPOCHS=$TRAINEE_EPOCHS EXTRA_GEN_CFG_OPTIONS='$EXTRA_GEN_CFG_OPTIONS' EXTRA_GEN_CFG_OPTIONS_BY_ROUND='$EXTRA_GEN_CFG_OPTIONS_BY_ROUND' EXTRA_TRAINEE_OVERRIDES='$EXTRA_TRAINEE_OVERRIDES' FIXED_REPLAY_ANNO='$FIXED_REPLAY_ANNO' FIXED_REPLAY_BANK='$FIXED_REPLAY_BANK' FIXED_REPLAY_PREFIX='$FIXED_REPLAY_PREFIX' SRC_ARM='$SRC_ARM' bash scripts/embodied/physflow_coevo_hardovf_restart_sliding_adistill_node.sh > $log 2>&1; echo HRSADISTILL_EXIT=\$? >> $log" \
	    --no-confirm 2>&1)
  tf=$(echo "$out" | grep -oE "task_flag: +[^ ]+" | head -1 | awk '{print $2}')
  echo "[retry] task_flag=$tf ; polling driver gate (<= $((GATE_POLLS * GATE_SLEEP_SEC / 60)) min)..."
  good=""; bad=""
  for i in $(seq 1 "$GATE_POLLS"); do
    sleep "$GATE_SLEEP_SEC"
    if grep -q "FATAL_BAD_NODE" "$log" 2>/dev/null; then bad=1; break; fi
    if grep -q "driver gate OK" "$log" 2>/dev/null; then good=1; break; fi
  done
  if [ -n "$good" ]; then
    drv=$(grep -oE "host CUDA driver version: [0-9.]+" "$log" | head -1)
    echo "[retry] GOOD NODE on attempt $att (task $tf, $drv). Leaving job running."
    echo "$tf" > "work_dirs/physflow_hardovf_restart_sliding_${TAG}_good_task.txt"
    echo "GOOD_NODE_FOUND attempt=$att task=$tf log=$log"
    exit 0
  fi
  echo "[retry] attempt $att bad/timeout (bad=${bad:-0}); stopping $tf"
  taiji_client stop "$tf" >/dev/null 2>&1 || true
  sleep 5
done
echo "[retry] EXHAUSTED $MAXATT attempts without a good node"
exit 1
