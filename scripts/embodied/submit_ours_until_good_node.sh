#!/usr/bin/env bash
# Resubmit the single-arm PhysFlow co-evolution (ours) on AILab_DHA V100 until we
# land on a host whose CUDA driver is >= 11.4 (IsaacGym PhysX GPU pipeline).  The
# node script's top-of-file driver gate exits ~10s on a bad host; this loop stops
# that job and retries.  On a good host it leaves the job running and exits.
set -uo pipefail
TOKEN=HzrPZC3djhwaU9HPdEA_Bg
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
DOCKER=mirrors.tencent.com/zeyuling_mirrors/vermo:latest
cd "$REPO"
MAXATT="${MAXATT:-15}"
export TOKEN

for att in $(seq 1 "$MAXATT"); do
  log="work_dirs/physflow_ours_node_a${att}.log"
  : > "$log"
  name="physflow_ours_a${att}"
  echo "[retry] === attempt $att: submit $name ==="
  out=$(python3 .claude/skills/taiji/taiji_ops.py submit --token "$TOKEN" -n "$name" \
    --gpu V100 --num_gpu 1 --num_host 1 -b AILab_DHA --docker "$DOCKER" \
    --cmd "cd $REPO && RUN_ARMS=ours bash scripts/embodied/physflow_coevo_formal_node.sh > $log 2>&1; echo OURS_EXIT=\$? >> $log" \
    --no-confirm 2>&1)
  tf=$(echo "$out" | grep -oE "task_flag: +[^ ]+" | head -1 | awk '{print $2}')
  echo "[retry] task_flag=$tf ; polling driver gate (<=10min)..."
  good=""; bad=""
  for i in $(seq 1 60); do
    sleep 10
    if grep -q "FATAL_BAD_NODE" "$log" 2>/dev/null; then bad=1; break; fi
    if grep -q "driver gate OK" "$log" 2>/dev/null; then good=1; break; fi
  done
  if [ -n "$good" ]; then
    drv=$(grep -oE "host CUDA driver version: [0-9.]+" "$log" | head -1)
    echo "[retry] GOOD NODE on attempt $att (task $tf, $drv). Leaving job running."
    echo "$tf" > work_dirs/physflow_ours_good_task.txt
    echo "GOOD_NODE_FOUND attempt=$att task=$tf log=$log"
    exit 0
  fi
  echo "[retry] attempt $att bad/timeout (bad=${bad:-0}); stopping $tf"
  taiji_client stop "$tf" >/dev/null 2>&1 || true
  sleep 5
done
echo "[retry] EXHAUSTED $MAXATT attempts without a good node"
exit 1
