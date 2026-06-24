#!/usr/bin/env bash
# Small MotionLab prefix-conditioning smoke test for debugging Table 2.
set -euo pipefail

ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python3 - <<'PY' || python3 -m pip install --user -q --no-build-isolation rotary-embedding-torch roma chumpy
import numpy as np
for name, value in {
    "bool": np.bool_,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if name not in np.__dict__:
        setattr(np, name, value)
import rotary_embedding_torch  # noqa: F401
import roma  # noqa: F401
import chumpy  # noqa: F401
PY

OUT=${OUT:-outputs/evaluation/motionlab_prefix_smoke_0606/ckpt_cfg}
COND=${COND:-5}
MAX_SAMPLES=${MAX_SAMPLES:-256}
CFG_MODE=${CFG_MODE:-ckpt}  # ckpt | official
GPU=${GPU:-0}
GT263=${GT263:-outputs/evaluation/humanml3d/gt_smpl135_to_hml263/humanml3d}

mkdir -p "${OUT}/hml263" "${OUT}/smpl_npz" "${OUT}/motionclip135" "${OUT}/logs" "${OUT}/metrics"

extra_args=()
if [ "${CFG_MODE}" = "official" ]; then
  extra_args+=(--no-cfg-from-checkpoint --cfg configs/config_rfmotion.yaml)
elif [ "${CFG_MODE}" != "ckpt" ]; then
  echo "Unknown CFG_MODE=${CFG_MODE}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/motionlab_infer_hml3d263.py \
  --anno-file data/annotation/test_hml3d.json \
  --data-dir data/motionhub \
  --gt-hml263-dir "${GT263}" \
  --out-dir "${OUT}/hml263" \
  --condition-num-frames "${COND}" \
  --batch-size 32 \
  --stage eval \
  --max-samples "${MAX_SAMPLES}" \
  --skip-existing \
  "${extra_args[@]}" \
  > "${OUT}/logs/infer.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/hml263_to_smpl_ik.py \
  --in-dir "${OUT}/hml263" \
  --out-dir "${OUT}/smpl_npz" \
  --model-dir ref_repo/MDM/body_models \
  --source-fps 20 \
  --target-fps 30 \
  --device cuda \
  --batch-size 512 \
  --floor-align \
  --rotation-init hml263 \
  --rot6d-convention column \
  --refine-iters 0 \
  --skip-existing \
  > "${OUT}/logs/ik.log" 2>&1

python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
  --anno-file data/annotation/test_hml3d.json \
  --data-dir data/motionhub \
  --src-dir "${OUT}/smpl_npz" \
  --out-dir "${OUT}/motionclip135" \
  --include-mirrors \
  --key-fallback \
  --align-to-gt-root \
  --overwrite \
  --workers 8 \
  > "${OUT}/logs/remap.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file data/annotation/test_hml3d.json \
  --data_dir data/motionhub \
  --pred_dir "${OUT}/motionclip135" \
  --rot6d_convention column \
  --chunk_size 64 \
  --out_json "${OUT}/metrics/h3d_cond${COND}_motionlab_${CFG_MODE}_smoke_c64.json" \
  --n_repeats 3 \
  --seed 42 \
  > "${OUT}/logs/eval.log" 2>&1

python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("OUT", "outputs/evaluation/motionlab_prefix_smoke_0606/ckpt_cfg")) / "metrics"
for path in root.glob("*.json"):
    d = json.loads(path.read_text())
    print(path.name, {
        "samples": d.get("samples"),
        "r3": d.get("r_precision_pred_top3_mean"),
        "fid": d.get("fid_mean"),
        "mm": d.get("mm_dist_pred_mean"),
        "div": d.get("diversity_pred_mean"),
    })
PY
