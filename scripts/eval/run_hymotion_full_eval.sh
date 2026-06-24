#!/usr/bin/env bash
# Convert FULL HY-Motion-1.0 (1.04B) raw135 -> motionclip135 (row2col + yaw),
# then evaluate with the ORIGINAL caption protocol (H3D + MotionHub).
# MH predictions were generated from REWRITTEN captions (HY-Motion pipeline),
# but scored against ORIGINAL captions to match the unified eval protocol.
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT=outputs/evaluation/hymotion_full_0611
MET=$ROOT/metrics; LOG=$ROOT/_logs
mkdir -p "$MET" "$LOG"
CKPT=checkpoints/motion_clip/motionclip_base_1p_aug_hq

convert() {
  local src="$1" out="$2" anno="$3" tag="$4"
  echo "[conv] $tag $(date)" | tee -a "$LOG/eval_run.log"
  python3 scripts/eval/convert_hylite135_to_motionclip_col.py \
    --src-dir "$src" --out-dir "$out" --anno-file "$anno" --data-dir data/motionhub \
    --align-to-gt-root --align-root-mode yaw --workers 16 \
    > "$LOG/conv_${tag}.log" 2>&1
  echo "[conv] $tag n=$(ls "$out" 2>/dev/null | wc -l) $(date)" | tee -a "$LOG/eval_run.log"
}

evalo() {
  local anno="$1" pred="$2" oj="$3" tag="$4"
  echo "[eval] $tag $(date)" | tee -a "$LOG/eval_run.log"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt "$CKPT" --anno_file "$anno" --data_dir data/motionhub \
    --pred_dir "$pred" --chunk_size 64 --n_repeats 20 --seed 42 --forward_batch_size 64 \
    --out_json "$oj" > "$LOG/eval_${tag}.log" 2>&1
}

convert "$ROOT/h3d/raw135"   "$ROOT/h3d/mc135"   data/annotation/test_hml3d.json        h3d
convert "$ROOT/mh_rw/raw135" "$ROOT/mh_rw/mc135" data/annotation/test_motionhub_t2m.json mh

evalo data/annotation/test_hml3d.json        "$ROOT/h3d/mc135"   "$MET/hymotion_full_h3d_orig_c64.json" h3d
evalo data/annotation/test_motionhub_t2m.json "$ROOT/mh_rw/mc135" "$MET/hymotion_full_mh_orig_c64.json" mh

python3 - <<'PY' | tee "$MET/summary.txt"
import json
from pathlib import Path
for nm in ["hymotion_full_h3d_orig_c64.json","hymotion_full_mh_orig_c64.json"]:
    p=Path("outputs/evaluation/hymotion_full_0611/metrics")/nm
    if not p.exists(): print(nm,"MISSING"); continue
    d=json.load(open(p))
    print(nm,"N",d.get("samples"),
          "R1",round(d.get("r_precision_pred_top1_mean",float('nan')),4),
          "R3",round(d.get("r_precision_pred_top3_mean",float('nan')),4),
          "FID",round(d.get("fid_mean",float('nan')),4),
          "MM",round(d.get("mm_dist_pred_mean",float('nan')),4),
          "Div",round(d.get("diversity_pred_mean",float('nan')),4))
PY
touch "$ROOT/_EVAL_DONE"
echo "[all] done $(date)" | tee -a "$LOG/eval_run.log"
