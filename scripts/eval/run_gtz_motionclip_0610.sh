#!/usr/bin/env bash
# Go-To-Zero (MotionMillion 7B) HumanML3D T2M -> MotionCLIP metrics.
# Converts raw vector_272 predictions to column-major motionclip135, then
# evaluates with both original and rewritten captions (Go-To-Zero generates
# from ORIGINAL HumanML3D captions, so *_orig is the matched protocol).
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

GTZ=outputs/evaluation/motionmillion_gtz
MC135=$GTZ/motionclip135_h3d
MET=$GTZ/metrics
LOG=$GTZ/_logs
mkdir -p "$MET" "$LOG"
CKPT=checkpoints/motion_clip/motionclip_base_1p_aug_hq

echo "[gtz] convert 272->mc135 $(date)" | tee "$LOG/recompute.log"
python3 scripts/eval/convert_ms272_dir_for_t2m_eval.py \
  --src-dir "$GTZ/raw272" \
  --anno-file data/annotation/test_hml3d.json --data-dir data/motionhub \
  --motionclip-dir "$MC135" \
  --align-to-gt-root --align-root-mode yaw --rot6d-convention column \
  --overwrite --workers 32 \
  > "$LOG/convert_h3d.log" 2>&1
echo "[gtz] mc135 n=$(ls "$MC135" 2>/dev/null | wc -l)" | tee -a "$LOG/recompute.log"

echo "[gtz] eval orig + rw $(date)" | tee -a "$LOG/recompute.log"
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt "$CKPT" --anno_file data/annotation/test_hml3d.json \
  --data_dir data/motionhub --pred_dir "$MC135" \
  --chunk_size 64 --n_repeats 20 --seed 42 --forward_batch_size 64 \
  --out_json "$MET/gtz_h3d_orig_c64.json" > "$LOG/eval_h3d_orig.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt "$CKPT" --anno_file data/annotation/test_hml3d.json \
  --data_dir data/motionhub --pred_dir "$MC135" \
  --rewritten_caption_file data/annotation/test_hml3d_rewritten.json \
  --chunk_size 64 --n_repeats 20 --seed 42 --forward_batch_size 64 \
  --out_json "$MET/gtz_h3d_rw_c64.json" > "$LOG/eval_h3d_rw.log" 2>&1 &
wait

python3 - <<'PY' | tee "$MET/summary.txt"
import json, math
from pathlib import Path
for name in ["gtz_h3d_orig_c64.json", "gtz_h3d_rw_c64.json"]:
    p = Path("outputs/evaluation/motionmillion_gtz/metrics") / name
    if not p.exists():
        print(name, "MISSING"); continue
    d = json.load(open(p))
    print(name, "N", d.get("samples"),
          "R1", round(d.get("r_precision_pred_top1_mean", float('nan')), 4),
          "R3", round(d.get("r_precision_pred_top3_mean", float('nan')), 4),
          "FID", round(d.get("fid_mean", float('nan')), 4),
          "MM", round(d.get("mm_dist_pred_mean", float('nan')), 4),
          "Div", round(d.get("diversity_pred_mean", float('nan')), 4))
PY
touch "$MET/_DONE"
echo "[gtz] done $(date)" | tee -a "$LOG/recompute.log"
