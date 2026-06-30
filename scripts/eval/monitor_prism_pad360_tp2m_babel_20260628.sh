#!/usr/bin/env bash
# Lightweight monitor for PRISM pad360_crop TP2M/BABEL reruns.
# It logs coverage and submits evaluator jobs once full generation coverage is
# present. Inference/eval work remains on Taiji; this script only polls and
# submits.
set -uo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "${ROOT}" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "${ROOT}"

TP2M_ROOT=${TP2M_ROOT:-outputs/evaluation/tp2m/humanml3d_official_test/motion135/prism_epoch43_pad360crop_selected_20260628}
BABEL_ROOT=${BABEL_ROOT:-outputs/evaluation/babel/official_val/msstyle_30fps_gt/prism_epoch43_pad360crop_arcond5_depth_driven}
TP2M_EXPECTED=${TP2M_EXPECTED:-4042}
BABEL_EXPECTED=${BABEL_EXPECTED:-1295}
INTERVAL=${INTERVAL:-300}
BUSINESS=${BUSINESS:-TaiJi_HYAide_NEO_PRI_CQ_V100}
LOG=${LOG:-outputs/evaluation/_monitors/prism_pad360_tp2m_babel_20260628.log}
STATE_LOG=${STATE_LOG:-outputs/evaluation/_monitors/prism_pad360_tp2m_babel_20260628_state.log}
mkdir -p "$(dirname "$LOG")"

tp2m_eval_marker="$TP2M_ROOT/_EVAL_SUBMITTED"
babel_eval_marker="$BABEL_ROOT/_EVAL_SUBMITTED"
tp2m_summary="outputs/evaluation/tp2m/humanml3d_official_test/_suites/table2_prism_epoch43_pad360crop_selected_20260628_ms272/summary.json"
babel_metrics="outputs/evaluation/babel/official_val/msstyle_30fps_gt/metrics/prism_epoch43_pad360crop_arcond5_depth_ms272_eval_20260628.json"

count_npz() {
  find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l
}

submit_eval_once() {
  local marker="$1"
  local flag="$2"
  local cmd="$3"
  if [ -f "$marker" ]; then
    return 0
  fi
  echo "[$(date)] submit_eval $flag" | tee -a "$LOG"
  if python3 tools/taiji_submit.py "$flag" \
      --host_num 1 \
      --host_gpu_num 1 \
      --gpu_name V100 \
      --business_flag "$BUSINESS" \
      --start-cmd "$cmd" >> "$LOG" 2>&1; then
    date > "$marker"
  else
    echo "[$(date)] submit_eval_failed $flag" | tee -a "$LOG"
  fi
}

while true; do
  c1=$(count_npz "$TP2M_ROOT/cond1_depth_driven" '*.npz')
  c5=$(count_npz "$TP2M_ROOT/cond5_depth_driven" '*.npz')
  c9=$(count_npz "$TP2M_ROOT/cond9_depth_driven" '*.npz')
  cb=$(count_npz "$BABEL_ROOT" 'val_*.npz')
  echo "[$(date)] tp2m_cond1=$c1/$TP2M_EXPECTED tp2m_cond5=$c5/$TP2M_EXPECTED tp2m_cond9=$c9/$TP2M_EXPECTED babel=$cb/$BABEL_EXPECTED" | tee -a "$LOG"

  taiji_client trl --view_nums 300 \
    | egrep 'p43tp2m_c[159]_s[0-9][0-9]d_0628|p43babel64_s[0-9][0-9]a_0628|p43babel64_g6[0-3]a_0628|p43tp2m_eval_pad360_0628|p43babel_eval_pad360_0628|TaskFlag|---' \
    > "$STATE_LOG" 2>&1 || true

  timeout 30s bash -c 'find "$0" "$1" -type f -mmin -30 -name "*.log" -print0 2>/dev/null | xargs -0 grep -hE "\[fail\]|Traceback|RuntimeError|ValueError|CUDA out of memory|length mismatch|requires every requested" 2>/dev/null | tail -80' \
    "$TP2M_ROOT/logs" "$BABEL_ROOT/logs" >> "$LOG" 2>/dev/null || true

  if [ "$c1" -ge "$TP2M_EXPECTED" ] && [ "$c5" -ge "$TP2M_EXPECTED" ] && [ "$c9" -ge "$TP2M_EXPECTED" ]; then
    submit_eval_once \
      "$tp2m_eval_marker" \
      "p43tp2m_eval_pad360_0628" \
      "cd ${ROOT} && bash scripts/eval/run_prism_tp2m_epoch43_eval_taiji.sh"
  fi

  if [ "$cb" -ge "$BABEL_EXPECTED" ]; then
    submit_eval_once \
      "$babel_eval_marker" \
      "p43babel_eval_pad360_0628" \
      "cd ${ROOT} && bash scripts/eval/run_prism_babel_epoch43_pad360crop_eval_taiji.sh"
  fi

  if [ -f "$tp2m_summary" ] && [ -f "$babel_metrics" ]; then
    echo "[$(date)] done tp2m_summary=$tp2m_summary babel_metrics=$babel_metrics" | tee -a "$LOG"
    exit 0
  fi

  sleep "$INTERVAL"
done
