#!/usr/bin/env bash
# Go-To-Zero (MotionMillion 7B) MotionHub T2M -> MotionCLIP metrics.
# Converts raw vector_272 MH predictions to column-major motionclip135, then
# evaluates with the ALL-ORIGINAL caption protocol (GTZ generates from original
# MotionHub captions). Run AFTER the Taiji gtz_mh_gen job finishes.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

GTZ=outputs/evaluation/motionmillion_gtz
MC135=$GTZ/motionclip135_mh
MET=$GTZ/metrics
LOG=$GTZ/_logs
mkdir -p "$MET" "$LOG"
CKPT=checkpoints/motion_clip/motionclip_base_1p_aug_hq

echo "[gtz-mh] convert 272->mc135 $(date)" | tee "$LOG/recompute_mh.log"
python3 scripts/eval/convert_ms272_dir_for_t2m_eval.py \
  --src-dir "$GTZ/raw272_mh" \
  --anno-file data/annotation/test_motionhub_t2m.json --data-dir data/motionhub \
  --motionclip-dir "$MC135" \
  --align-to-gt-root --align-root-mode yaw --rot6d-convention column \
  --overwrite --workers 32 \
  > "$LOG/convert_mh.log" 2>&1
echo "[gtz-mh] mc135 n=$(ls "$MC135" 2>/dev/null | wc -l)" | tee -a "$LOG/recompute_mh.log"

echo "[gtz-mh] eval all-original $(date)" | tee -a "$LOG/recompute_mh.log"
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt "$CKPT" --anno_file data/annotation/test_motionhub_t2m.json \
  --data_dir data/motionhub --pred_dir "$MC135" \
  --chunk_size 64 --n_repeats 20 --seed 42 --forward_batch_size 64 \
  --out_json "$MET/gtz_mh_orig_c64.json" > "$LOG/eval_mh_orig.log" 2>&1

python3 - <<'PY' | tee -a "$MET/summary.txt"
import json
from pathlib import Path
p = Path("outputs/evaluation/motionmillion_gtz/metrics/gtz_mh_orig_c64.json")
if not p.exists():
    print("gtz_mh_orig_c64.json MISSING")
else:
    d = json.load(open(p))
    print("gtz_mh_orig_c64.json", "N", d.get("samples"),
          "R1", round(d.get("r_precision_pred_top1_mean", float('nan')), 4),
          "R3", round(d.get("r_precision_pred_top3_mean", float('nan')), 4),
          "FID", round(d.get("fid_mean", float('nan')), 4),
          "MM", round(d.get("mm_dist_pred_mean", float('nan')), 4),
          "Div", round(d.get("diversity_pred_mean", float('nan')), 4))
PY
touch "$MET/_DONE_MH"
echo "[gtz-mh] done $(date)" | tee -a "$LOG/recompute_mh.log"
