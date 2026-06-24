#!/usr/bin/env bash
# Table-6 ExpB CondMDI: generate ALL (or a subset of) body-part position settings.
# For each PART we first purge any stale joints/smplx/eval_npz/_DONE + old metrics
# (partial earlier runs), then run the full single-part pipeline on local GPUs.
#
# Split across cluster jobs via PARTS, e.g.
#   PARTS=A_upper NGPU=4 GPUS=0,1,2,3 bash scripts/eval/run_bodypart_condmdi_all.sh
#
# Env: PARTS(csv, default all 10), NGPU, GPUS, BATCH, MAX_SAMPLES, GUIDANCE, OUT.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUT=${OUT:-output/evaluation/bodypart_table6_pos}
NGPU=${NGPU:-4}
GPUS=${GPUS:-0,1,2,3}
BATCH=${BATCH:-16}
MAX_SAMPLES=${MAX_SAMPLES:-500}
GUIDANCE=${GUIDANCE:-2.5}
# E_legs_only dropped: identical joint set to B_lower (Table-6 dedup).
ALL="A_upper B_lower C_spine_only D_arms_only F_left_arm G_right_arm H_left_leg I_right_leg J_feet_only K_no_feet"
PARTS=${PARTS:-$ALL}
PARTS=${PARTS//,/ }

LOG="$OUT/logs"; mkdir -p "$LOG"
echo "[all-condmdi-bodypart] $(date) PARTS=[$PARTS] NGPU=$NGPU BATCH=$BATCH" | tee -a "$LOG/run_all_condmdi.log"

for PART in $PARTS; do
  BASE="$OUT/condmdi/$PART"
  echo "[clean] $PART : purge stale joints/smplx/eval_npz/_DONE + metrics" | tee -a "$LOG/run_all_condmdi.log"
  rm -rf "$BASE/joints" "$BASE/smplx" "$BASE/eval_npz" "$BASE/_DONE"
  rm -f  "$OUT/_metrics/condmdi_${PART}__new.json" "$OUT/_metrics/condmdi_${PART}__fid.json"
  echo "[run] $PART $(date)" | tee -a "$LOG/run_all_condmdi.log"
  PART="$PART" OUT="$OUT" NGPU="$NGPU" GPUS="$GPUS" BATCH="$BATCH" \
    MAX_SAMPLES="$MAX_SAMPLES" GUIDANCE="$GUIDANCE" PHASE=all \
    bash scripts/eval/run_bodypart_condmdi.sh \
    >> "$LOG/run_all_condmdi.log" 2>&1 || echo "[warn] $PART exited nonzero" | tee -a "$LOG/run_all_condmdi.log"
  echo "[part-done] $PART metrics: $(ls "$OUT/_metrics/condmdi_${PART}__"*.json 2>/dev/null | wc -l)/2" | tee -a "$LOG/run_all_condmdi.log"
done
echo "[all-done-condmdi-bodypart] $(date) PARTS=[$PARTS]" | tee -a "$LOG/run_all_condmdi.log"
touch "$OUT/condmdi/_ALL_DONE_$(echo "$PARTS" | tr ' ' '_' | cut -c1-40)"
