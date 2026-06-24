#!/usr/bin/env bash
# VerMo (VersatileMotion) MotionHub T2M -> MotionCLIP metrics.
# Source predictions are row-major 135D; convert to column-major + yaw root
# alignment (matching the trusted baseline protocol), then evaluate with both
# original and rewritten captions to identify the matched generation protocol.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CKPT=checkpoints/motion_clip/motionclip_base_1p_aug_hq
SRC=output/evaluation/vermo_paper_t2m_m2t_ckpt22000_20260605_full/paper_t2m_pred_135d
OUT=outputs/evaluation/vermo_mh_motionclip_col_0610
PRED="$OUT/pred_col"; MET="$OUT/metrics"; LOG="$OUT/_logs"
mkdir -p "$PRED" "$MET" "$LOG"

echo "[convert] $(date)" | tee "$LOG/recompute.log"
python3 scripts/eval/convert_hylite135_to_motionclip_col.py \
  --src-dir "$SRC" --out-dir "$PRED" \
  --anno-file data/annotation/test_motionhub_t2m.json --data-dir data/motionhub \
  --align-to-gt-root --align-root-mode yaw --overwrite --workers 16 \
  > "$LOG/convert.log" 2>&1
echo "[convert done] n=$(ls "$PRED"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/recompute.log"

echo "[eval orig + rw] $(date)" | tee -a "$LOG/recompute.log"
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt "$CKPT" --anno_file data/annotation/test_motionhub_t2m.json \
  --data_dir data/motionhub --pred_dir "$PRED" \
  --chunk_size 64 --n_repeats 20 --seed 42 --forward_batch_size 64 \
  --out_json "$MET/vermo_mh_orig_c64.json" > "$LOG/eval_orig.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt "$CKPT" --anno_file data/annotation/test_motionhub_t2m.json \
  --data_dir data/motionhub --pred_dir "$PRED" \
  --rewritten_caption_file data/annotation/test_motionhub_t2m_rewritten.json \
  --chunk_size 64 --n_repeats 20 --seed 42 --forward_batch_size 64 \
  --out_json "$MET/vermo_mh_rw_c64.json" > "$LOG/eval_rw.log" 2>&1 &
wait

python3 - <<'PY' | tee "$MET/summary.txt"
import json, math
from pathlib import Path
for name in ["vermo_mh_orig_c64.json", "vermo_mh_rw_c64.json"]:
    p = Path("outputs/evaluation/vermo_mh_motionclip_col_0610/metrics") / name
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
echo "[done] $(date)" | tee -a "$LOG/recompute.log"
