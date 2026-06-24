#!/usr/bin/env bash
# Watch the current fixed-hard24 co-evolution probes and submit fixed-24 scoring
# as soon as each arm completes. Designed to run from the repo root under nohup.
set -uo pipefail
TOKEN="${TOKEN:-HzrPZC3djhwaU9HPdEA_Bg}"
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$REPO"

SLEEP_SEC="${SLEEP_SEC:-60}"
MAX_LOOPS="${MAX_LOOPS:-360}"

declare -a ARMS=(
  "fix24_prn8|work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_prn8_1/hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_prn8_1|output/hardovf_frontier_fixed_score_rsadfull_e5c2_fix24_prn8_1"
  "fix24_hi98|work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98_1/hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98_1|output/hardovf_frontier_fixed_score_rsadfull_e5c2_fix24_hi98_1"
  "fix24_hi98g15|work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98g15_1/hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98g15_1|output/hardovf_frontier_fixed_score_rsadfull_e5c2_fix24_hi98g15_1"
  "fix24_hi98gt1|work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98gt1_1/hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98gt1_1|output/hardovf_frontier_fixed_score_rsadfull_e5c2_fix24_hi98gt1_1"
  "fix24_hi98gt2|work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98gt2_1/hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98gt2_1|output/hardovf_frontier_fixed_score_rsadfull_e5c2_fix24_hi98gt2_1"
)

echo "[hardovf-monitor] $(date) start sleep=${SLEEP_SEC}s loops=${MAX_LOOPS}"
for loop in $(seq 1 "$MAX_LOOPS"); do
  pending=0
  for item in "${ARMS[@]}"; do
    IFS='|' read -r tag root out_dir <<< "$item"
    state="$root/state.jsonl"
    score_task="work_dirs/physflow_hardovf_score_${tag}_good_task.txt"
    summary="$out_dir/r2/heldout_score.json"
    echo "[hardovf-monitor] loop=$loop tag=$tag $(date '+%F %T')"
    if [ -f "$summary" ]; then
      echo "[hardovf-monitor] tag=$tag scored summary=$summary"
      continue
    fi
    if [ ! -f "$state" ]; then
      echo "[hardovf-monitor] tag=$tag waiting: missing state"
      pending=1
      continue
    fi
    if grep -q '"event": "gen_failed"\|"event": "trainee_failed"\|"event": "judge_export_failed"' "$state"; then
      echo "[hardovf-monitor] tag=$tag FAILED; inspect $state"
      continue
    fi
    if ! grep -q '"event": "orchestrator_done"' "$state"; then
      tail -n 3 "$state" | sed 's/^/[hardovf-monitor] state-tail /'
      pending=1
      continue
    fi
    if [ -f "$score_task" ]; then
      echo "[hardovf-monitor] tag=$tag score already submitted task=$(cat "$score_task")"
      pending=1
      continue
    fi
    echo "[hardovf-monitor] tag=$tag complete; submitting score out=$out_dir"
    TOKEN="$TOKEN" TAG="$tag" SCORE_ROOT="$root" SCORE_OUT="$out_dir" \
      bash scripts/embodied/submit_score_hardovf_frontier_until_good_node.sh
    pending=1
  done
  if [ "$pending" = "0" ]; then
    echo "[hardovf-monitor] all arms scored or failed; exiting"
    exit 0
  fi
  sleep "$SLEEP_SEC"
done
echo "[hardovf-monitor] reached max loops"
