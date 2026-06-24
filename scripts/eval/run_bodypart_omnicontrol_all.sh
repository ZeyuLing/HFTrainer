#!/usr/bin/env bash
# Table-6 ExpB OmniControl: regenerate ALL (or a subset of) body-part position
# settings after the Mean/Std fix (TeSMo training norm). For each PART we first
# purge the stale (wrong-norm) joints/smplx/eval_npz/_DONE + the old _stage
# Mean/Std symlinks (which point at the T2M *eval* norm), then run the full
# single-part pipeline on this node's local GPUs.
#
# Split across cluster jobs via PARTS, e.g.
#   PARTS=A_upper,B_lower,C_spine_only NGPU=8 bash scripts/eval/run_bodypart_omnicontrol_all.sh
#
# Env: PARTS(csv, default all 11), NGPU, GPUS, BATCH, MAX_SAMPLES, GUIDANCE, OUT.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUT=${OUT:-output/evaluation/bodypart_table6_pos}
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
BATCH=${BATCH:-48}
MAX_SAMPLES=${MAX_SAMPLES:-500}
GUIDANCE=${GUIDANCE:-2.5}
ALL="A_upper B_lower C_spine_only D_arms_only E_legs_only F_left_arm G_right_arm H_left_leg I_right_leg J_feet_only K_no_feet"
PARTS=${PARTS:-$ALL}
PARTS=${PARTS//,/ }

LOG="$OUT/logs"; mkdir -p "$LOG"
echo "[all-omni-bodypart] $(date) PARTS=[$PARTS] NGPU=$NGPU BATCH=$BATCH" | tee -a "$LOG/run_all.log"

for PART in $PARTS; do
  BASE="$OUT/omnicontrol/$PART"
  echo "[clean] $PART : purge stale joints/smplx/eval_npz/_DONE + wrong-norm _stage" | tee -a "$LOG/run_all.log"
  rm -rf "$BASE/joints" "$BASE/smplx" "$BASE/eval_npz" "$BASE/_DONE"
  rm -f  "$BASE/_stage/dataset/HumanML3D/Mean.npy" "$BASE/_stage/dataset/HumanML3D/Std.npy"
  rm -f  "$OUT/_metrics/omnicontrol_${PART}__new.json" "$OUT/_metrics/omnicontrol_${PART}__fid.json"
  echo "[run] $PART $(date)" | tee -a "$LOG/run_all.log"
  PART="$PART" OUT="$OUT" NGPU="$NGPU" GPUS="$GPUS" BATCH="$BATCH" \
    MAX_SAMPLES="$MAX_SAMPLES" GUIDANCE="$GUIDANCE" PHASE=all \
    bash scripts/eval/run_bodypart_omnicontrol.sh \
    >> "$LOG/run_all.log" 2>&1 || echo "[warn] $PART exited nonzero" | tee -a "$LOG/run_all.log"
  echo "[part-done] $PART metrics: $(ls "$OUT/_metrics/omnicontrol_${PART}__"*.json 2>/dev/null | wc -l)/2" | tee -a "$LOG/run_all.log"
done
echo "[all-done-omni-bodypart] $(date) PARTS=[$PARTS]" | tee -a "$LOG/run_all.log"
touch "$OUT/omnicontrol/_ALL_DONE_$(echo "$PARTS" | tr ' ' '_' | cut -c1-40)"
