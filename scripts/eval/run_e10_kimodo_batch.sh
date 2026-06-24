#!/usr/bin/env bash
# Batch: KIMODO E10 rotation-transfer -> build eval npz -> Ctrl.Err + ric (Foot/Jitter)
# for all part-control settings. FID (GPU) is run separately.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT=output/evaluation/bodypart_table6_rot/kimodo
RAW=output/evaluation/paper_baseline_kimodo/E10
DATA=data/eval/m2m_v2/eval_e10_part_control.json
FILT='Warning|deprecat|ds_accelerator|def (forward|backward)|from torch|body_model|mesh_dist|tbs_axes'

# dir<TAB>key   (A_upper already done; include all for idempotent reruns)
SETTINGS=(
  "A_upper:upper" "B_lower:lower" "C_spine_only:spine_only"
  "D_arms_only:arms_only" "E_legs_only:legs_only" "F_left_arm:left_arm"
  "G_right_arm:right_arm" "H_left_leg:left_leg" "I_right_leg:right_leg"
  "J_feet_only:feet_only" "K_no_feet:no_feet"
)

for pair in "${SETTINGS[@]}"; do
  d="${pair%%:*}"; key="${pair##*:}"
  echo "==================== $d (key=$key) ===================="
  mkdir -p "$ROOT/$d/metrics"
  if [ ! -d "$ROOT/$d/smplx" ] || [ "$(ls $ROOT/$d/smplx/*.npz 2>/dev/null | wc -l)" -lt 90 ]; then
    python3 scripts/eval/kimodo_soma_to_smpl_byid.py \
      --data-file "$DATA" --max-samples 5000 \
      --raw-npz-dir "$RAW/E10_$d/npz" \
      --out-dir "$ROOT/$d/smplx" \
      --mode rotation --device cpu --num-shards 1 --shard-index 0 2>&1 | grep -vE "$FILT" | tail -2
  else
    echo "[skip retarget] $ROOT/$d/smplx already has $(ls $ROOT/$d/smplx/*.npz|wc -l)"
  fi
  python3 scripts/eval/build_e10_kimodo_eval_npz.py \
    --smplx-dir "$ROOT/$d/smplx" --setting-key "$key" \
    --out-dir "$ROOT/$d/npz" --data-file "$DATA" 2>&1 | grep -vE "$FILT" | tail -1
  python3 scripts/eval/e10_ctrl_err.py \
    --npz-dir "$ROOT/$d/npz" --tag "kimodo_$d" \
    --out-json "$ROOT/$d/metrics/ctrlerr.json" 2>&1 | grep -vE "$FILT" | tail -2
  python3 scripts/eval/paper_npz_ric_mpjpe.py \
    --npz-dir "$ROOT/$d/npz" --tag "kimodo_$d" \
    --out-json "$ROOT/$d/metrics/ric.json" 2>&1 | grep -vE "$FILT" | tail -2
done
echo "ALL_DONE_BATCH"
