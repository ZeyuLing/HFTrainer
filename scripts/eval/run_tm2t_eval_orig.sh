#!/usr/bin/env bash
# TM2T (HML263) -> SMPL135 (IK) -> MotionCLIP135 (remap+yaw) -> evaluate on
# ORIGINAL captions (no rewritten file).  Single-GPU friendly.
set -euo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1

SRC=outputs/evaluation/tm2t_0611
SMPL=outputs/evaluation/tm2t_0611/smpl135
MC=outputs/evaluation/tm2t_0611/mc135
MET=outputs/evaluation/tm2t_0611/metrics
LOG=outputs/evaluation/tm2t_0611/_logs
GPU=${GPU:-0}
LIMIT_ARG=""
[ "${LIMIT:-0}" != "0" ] && LIMIT_ARG="--limit ${LIMIT}"
mkdir -p "${SMPL}/h3d" "${SMPL}/mh" "${MC}/h3d" "${MC}/mh" "${MET}" "${LOG}"

ik() { # split src
  echo "[ik $1] $(date)" | tee -a "${LOG}/eval.log"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$2" --out-dir "${SMPL}/$1" --model-dir ref_repo/MDM/body_models \
    --source-fps 20 --target-fps 30 --batch-size 256 --device cuda \
    --floor-align --refine-iters 0 --skip-existing ${LIMIT_ARG} \
    > "${LOG}/ik_$1.log" 2>&1
}

remap() { # split anno
  echo "[remap $1] $(date)" | tee -a "${LOG}/eval.log"
  python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
    --anno-file "$2" --data-dir data/motionhub \
    --src-dir "${SMPL}/$1" --out-dir "${MC}/$1" \
    --include-mirrors --key-fallback --align-to-gt-root --align-root-mode yaw \
    --overwrite --workers 16 \
    > "${LOG}/remap_$1.log" 2>&1
}

evalo() { # split anno
  echo "[eval $1 ORIG] $(date)" | tee -a "${LOG}/eval.log"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "$2" --data_dir data/motionhub --pred_dir "${MC}/$1" \
    --chunk_size 64 --n_repeats 20 --seed 42 \
    --out_json "${MET}/tm2t_$1_orig_c64.json" \
    > "${LOG}/eval_$1.log" 2>&1
}

ik   h3d "${SRC}/h3d/raw263"
ik   mh  "${SRC}/mh/raw263"
remap h3d data/annotation/test_hml3d.json
remap mh  data/annotation/test_motionhub_t2m.json
evalo h3d data/annotation/test_hml3d.json
evalo mh  data/annotation/test_motionhub_t2m.json

python3 - <<PY | tee "${MET}/summary.txt"
import json
from pathlib import Path
for s in ("h3d","mh"):
    p=Path("${MET}")/f"tm2t_{s}_orig_c64.json"
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
