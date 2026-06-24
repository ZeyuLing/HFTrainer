#!/usr/bin/env bash
# LoM SMPL-X NPZ -> 135-dim -> MotionCLIP135 (col + yaw) -> evaluate on
# ORIGINAL captions (no rewritten file).  Single-GPU friendly.
set -euo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1

SRC=outputs/evaluation/lom_0611
D135=outputs/evaluation/lom_0611/d135
MC=outputs/evaluation/lom_0611/mc135
MET=outputs/evaluation/lom_0611/metrics
LOG=outputs/evaluation/lom_0611/_logs
GPU=${GPU:-0}
mkdir -p "${D135}/h3d" "${D135}/mh" "${MC}/h3d" "${MC}/mh" "${MET}" "${LOG}"

conv135() { # split
  echo "[conv135 $1] $(date)" | tee -a "${LOG}/eval.log"
  python3 scripts/eval/convert_smplx_npz_dir_to_135d.py \
    --input-dir "${SRC}/$1/smplx" --output-dir "${D135}/$1" --skip-existing \
    > "${LOG}/conv135_$1.log" 2>&1
}

col() { # split anno
  echo "[col $1] $(date)" | tee -a "${LOG}/eval.log"
  python3 scripts/eval/convert_hylite135_to_motionclip_col.py \
    --src-dir "${D135}/$1" --out-dir "${MC}/$1" --anno-file "$2" --data-dir data/motionhub \
    --align-to-gt-root --align-root-mode yaw --overwrite --workers 16 \
    > "${LOG}/col_$1.log" 2>&1
}

evalo() { # split anno
  echo "[eval $1 ORIG] $(date)" | tee -a "${LOG}/eval.log"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "$2" --data_dir data/motionhub --pred_dir "${MC}/$1" \
    --chunk_size 64 --n_repeats 20 --seed 42 \
    --out_json "${MET}/lom_$1_orig_c64.json" \
    > "${LOG}/eval_$1.log" 2>&1
}

conv135 h3d
conv135 mh
col h3d data/annotation/test_hml3d.json
col mh  data/annotation/test_motionhub_t2m.json
evalo h3d data/annotation/test_hml3d.json
evalo mh  data/annotation/test_motionhub_t2m.json

python3 - <<PY | tee "${MET}/summary.txt"
import json
from pathlib import Path
for s in ("h3d","mh"):
    p=Path("${MET}")/f"lom_{s}_orig_c64.json"
    if not p.exists(): print(s,"missing"); continue
    d=json.load(open(p))
    print(s,"N",d.get("samples"),
          "R1",f"{d.get('r_precision_pred_top1_mean',float('nan')):.4f}",
          "R3",f"{d.get('r_precision_pred_top3_mean',float('nan')):.4f}",
          "FID",f"{d.get('fid_mean',float('nan')):.4f}",
          "MM",f"{d.get('mm_dist_pred_mean',float('nan')):.4f}",
          "Div",f"{d.get('diversity_pred_mean',float('nan')):.4f}",
          "REAL_R1",f"{d.get('r_precision_real_top1_mean',float('nan')):.4f}")
PY
touch "${MET}/_DONE"
echo "[done] $(date)" | tee -a "${LOG}/eval.log"
