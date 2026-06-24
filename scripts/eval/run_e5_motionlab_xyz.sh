#!/usr/bin/env bash
# Table 7 (tab:trajectory) MotionLab XYZ rows. MotionLab's trajectory hint ALWAYS
# pins the FULL pelvis xyz (motionlab_infer _build_trajectory_hint, "all 3 coords"),
# so the XZ and XYZ generations are IDENTICAL -- only the eval mask differs. We thus
# REUSE the already-generated XZ predictions (motionlab_{dense,sparse}/smplx) and only
# re-package + re-score them against the \ours XYZ settings (E5_D / E5_E), whose
# src_mask observes root channels 0,1,2 (X,Y,Z) so Traj.Err scores all three axes.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
GPU=${GPU:-0}
OURS_ROOT="$PWD/output/evaluation/paper_ours_ep590"

run_one() {
  local SRC_MODE="$1" SETTING="$2"   # SRC_MODE=dense|sparse ; SETTING=E5_D_xyz_dense|E5_E_xyz_sparse
  local SM="$PWD/output/evaluation/table7_traj/motionlab_${SRC_MODE}/smplx"
  local OURS_NPZ="$OURS_ROOT/$SETTING/smpl_caption_editfix_latest/$SETTING/npz"
  local OUT="$PWD/output/evaluation/table7_traj/motionlab_xyz_${SRC_MODE}"
  local EN="$OUT/$SETTING" LOG="$OUT/logs"
  mkdir -p "$EN" "$LOG"
  echo "[mlab-xyz] $SETTING  src=$SM  ours=$OURS_NPZ"
  python3 scripts/eval/build_e5_baseline_eval_npz.py \
    --ours-npz-dir "$OURS_NPZ" --pred-sid-dir "$SM" --out-dir "$EN" \
    > "$LOG/build.log" 2>&1
  echo "[build] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)"
  bash scripts/eval/run_e5_baseline_metrics.sh "motionlab_${SETTING}" "$EN" "$GPU" \
    >> "$LOG/run.log" 2>&1
  echo "[done] $SETTING -> output/evaluation/table7_traj/_metrics/motionlab_${SETTING}__{ric,new,fid}.json"
}

run_one dense  E5_D_xyz_dense
run_one sparse E5_E_xyz_sparse
echo "[ALL DONE] $(date)"
