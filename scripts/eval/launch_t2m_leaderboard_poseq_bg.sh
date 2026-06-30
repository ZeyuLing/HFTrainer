#!/usr/bin/env bash
# Launch T2M HumanML3D leaderboard PoseQ computation in the background.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

OUT="${OUT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/t2m_leaderboard_poseq_20260630}"
mkdir -p "$OUT/logs"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HFTRAINER_SKIP_AUTOREGISTER=1
export PYTHONPATH="$ROOT:$ROOT/tools:$ROOT/scripts/eval:${PYTHONPATH:-}"

nohup python3 scripts/eval/run_t2m_leaderboard_poseq.py \
  --out-dir "$OUT" \
  --jobs "${JOBS:-8}" \
  --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
  ${FORCE:+--force} \
  > "$OUT/logs/run_all.log" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$OUT/run_all.pid"
echo "pid=$pid"
echo "out=$OUT"
echo "log=$OUT/logs/run_all.log"
