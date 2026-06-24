#!/usr/bin/env bash
# Table-6 ExpB MotionLab: generate ALL (or a subset of) body-part position settings.
# For each PART we first purge any stale hml263/smplx/eval_npz/_DONE + old metrics,
# then run the full single-part pipeline on local GPUs.
#
# Split across cluster jobs via PARTS, e.g.
#   PARTS=A_upper NGPU=4 GPUS=0,1,2,3 bash scripts/eval/run_bodypart_motionlab_all.sh
#
# Env: PARTS(csv, default all 10), NGPU, GPUS, BATCH, MAX_SAMPLES, STAGE, OUT.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUT=${OUT:-output/evaluation/bodypart_table6_pos}
NGPU=${NGPU:-4}
GPUS=${GPUS:-0,1,2,3}
BATCH=${BATCH:-16}
MAX_SAMPLES=${MAX_SAMPLES:-500}
STAGE=${STAGE:-demo}
# E_legs_only dropped: identical joint set to B_lower (Table-6 dedup).
ALL="A_upper B_lower C_spine_only D_arms_only F_left_arm G_right_arm H_left_leg I_right_leg J_feet_only K_no_feet"
PARTS=${PARTS:-$ALL}
PARTS=${PARTS//,/ }

LOG="$OUT/logs"; mkdir -p "$LOG"
echo "[all-motionlab-bodypart] $(date) PARTS=[$PARTS] NGPU=$NGPU BATCH=$BATCH" | tee -a "$LOG/run_all_motionlab.log"

for PART in $PARTS; do
  BASE="$OUT/motionlab/$PART"
  echo "[clean] $PART : purge stale hml263/smplx/eval_npz/_DONE + metrics" | tee -a "$LOG/run_all_motionlab.log"
  rm -rf "$BASE/hml263" "$BASE/smplx" "$BASE/eval_npz" "$BASE/_DONE"
  rm -f  "$OUT/_metrics/motionlab_${PART}__new.json" "$OUT/_metrics/motionlab_${PART}__fid.json"
  echo "[run] $PART $(date)" | tee -a "$LOG/run_all_motionlab.log"
  PART="$PART" OUT="$OUT" NGPU="$NGPU" GPUS="$GPUS" BATCH="$BATCH" \
    MAX_SAMPLES="$MAX_SAMPLES" STAGE="$STAGE" PHASE=all \
    bash scripts/eval/run_bodypart_motionlab.sh \
    >> "$LOG/run_all_motionlab.log" 2>&1 || echo "[warn] $PART exited nonzero" | tee -a "$LOG/run_all_motionlab.log"
  echo "[part-done] $PART metrics: $(ls "$OUT/_metrics/motionlab_${PART}__"*.json 2>/dev/null | wc -l)/2" | tee -a "$LOG/run_all_motionlab.log"
done
echo "[all-done-motionlab-bodypart] $(date) PARTS=[$PARTS]" | tee -a "$LOG/run_all_motionlab.log"
touch "$OUT/motionlab/_ALL_DONE_$(echo "$PARTS" | tr ' ' '_' | cut -c1-40)"
