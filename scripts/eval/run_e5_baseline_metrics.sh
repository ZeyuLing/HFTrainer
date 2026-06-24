#!/usr/bin/env bash
# Table 7 (tab:trajectory) baseline metrics on an E5 eval-npz dir.
#
# Given a self-contained E5 eval-npz dir (motion_135 / gt_motion_135 / src_mask /
# caption, built by build_e5_baseline_eval_npz.py) compute the SAME 3 metric
# families \ours reported, into <metrics-dir>/<TAG>__{ric,new,fid}.json:
#   ric  -> jitter (Jitter column), foot     [paper_npz_ric_mpjpe.py]
#   new  -> trajectory_err_m (Traj.Err), foot_skating_ratio (Foot) [collect_ours_posthoc]
#   fid  -> FID / R@3 / Diversity            [eval_editing_272_fid.py]
#
# Usage:  run_e5_baseline_metrics.sh <TAG> <EVAL_NPZ_DIR> [GPU]
#   EVAL_NPZ_DIR must be <base>/<setting>/  so collect_ours_posthoc can glob
#   --base <base> --settings <setting>.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TAG="$1"; EN="$2"; GPU="${3:-0}"
MD="$PWD/output/evaluation/table7_traj/_metrics"; LOG="$PWD/output/evaluation/table7_traj/logs"
mkdir -p "$MD" "$LOG"
BASE="$(dirname "$EN")"; SETTING="$(basename "$EN")"

echo "[e5-metrics] TAG=$TAG EN=$EN base=$BASE setting=$SETTING gpu=$GPU n_npz=$(ls "$EN"/*.npz 2>/dev/null | wc -l)"

python3 scripts/eval/paper_npz_ric_mpjpe.py --npz-dir "$EN" \
  --tag "$TAG" --out-json "$MD/${TAG}__ric.json" > "$LOG/${TAG}_ric.log" 2>&1 \
  && echo "[ric ] ok -> $MD/${TAG}__ric.json" || echo "[ric ] FAIL (see $LOG/${TAG}_ric.log)"

python3 scripts/eval/collect_ours_posthoc_metrics.py \
  --base "$BASE" --settings "$SETTING" \
  --out "$MD/${TAG}__new.json" > "$LOG/${TAG}_new.log" 2>&1 \
  && echo "[new ] ok -> $MD/${TAG}__new.json" || echo "[new ] FAIL (see $LOG/${TAG}_new.log)"

# FID needs the DEFAULT HF cache (272 TMR evaluator weights); do NOT set offline.
CUDA_VISIBLE_DEVICES="$GPU" env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  python3 scripts/eval/eval_editing_272_fid.py \
  --pred-npz-dir "$EN" --tag "$TAG" --out-json "$MD/${TAG}__fid.json" \
  > "$LOG/${TAG}_fid.log" 2>&1 \
  && echo "[fid ] ok -> $MD/${TAG}__fid.json" || echo "[fid ] FAIL (see $LOG/${TAG}_fid.log)"

echo "[e5-metrics] done $TAG"
