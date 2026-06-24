#!/usr/bin/env bash
# MotionLab prefix-pose-conditioned generation for Table 2.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [ ! -d "${ROOT}" ]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python3 - <<'PY' || python3 -m pip install --user -q --no-build-isolation \
  'numpy<2' rotary-embedding-torch roma chumpy mmengine addict yapf rich termcolor smplx
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
import mmengine  # noqa: F401
import smplx  # noqa: F401
PY

OUT=${OUT:-outputs/evaluation/motionlab_tp2m_table2_0606}
GT263_ROOT=${GT263_ROOT:-outputs/evaluation/humanml3d/gt_smpl135_to_hml263}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
NUM_SHARDS=${NUM_SHARDS:-8}
WORKERS=${WORKERS:-16}
BATCH_SIZE=${BATCH_SIZE:-32}
STAGE=${STAGE:-eval}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
EVAL_GPU_H3D=${EVAL_GPU_H3D:-0}
EVAL_GPU_MH=${EVAL_GPU_MH:-1}
EXTRA_INFER_ARGS=${EXTRA_INFER_ARGS:---no-cfg-from-checkpoint --cfg configs/config_rfmotion.yaml}
MAX_SAMPLES=${MAX_SAMPLES:-}
SPLITS=${SPLITS:-"h3d mh"}
CONDS=${CONDS:-"1 5 9"}
STAGES=${STAGES:-"build infer retarget eval summary"}

mkdir -p "${OUT}/logs" "${OUT}/metrics" "${GT263_ROOT}/humanml3d" "${GT263_ROOT}/motionhub"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -lt "${NUM_SHARDS}" ]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

echo "[start] MotionLab TP2M out=${OUT} shards=${NUM_SHARDS}"

want_item() {
  local needle="$1"
  local haystack="$2"
  for item in ${haystack}; do
    if [ "${item}" = "${needle}" ]; then
      return 0
    fi
  done
  return 1
}

build_gt_hml263() {
  local split="$1"
  local anno="$2"
  local out_dir="$3"
  echo "[${split}-build-gt-hml263] $(date)"
  python3 scripts/eval/build_gt_smpl135_to_hml263.py \
    --anno-file "${anno}" \
    --data-dir data/motionhub \
    --out-dir "${out_dir}" \
    --workers "${WORKERS}" \
    --skip-existing \
    > "${OUT}/logs/build_gt_${split}.log" 2>&1
}

infer_split_cond() {
  local split="$1"
  local anno="$2"
  local gt263="$3"
  local cond="$4"
  local out_dir="${OUT}/hml263/${split}_cond${cond}"
  mkdir -p "${out_dir}" "${OUT}/logs/${split}_cond${cond}"
  echo "[${split}-cond${cond}-infer] $(date)"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$shard]}"
    local args=(
      --anno-file "${anno}"
      --data-dir data/motionhub
      --gt-hml263-dir "${gt263}"
      --out-dir "${out_dir}"
      --condition-num-frames "${cond}"
      --batch-size "${BATCH_SIZE}"
      --stage "${STAGE}"
      --num-shards "${NUM_SHARDS}"
      --shard-index "${shard}"
      --skip-existing
    )
    if [ -n "${MAX_SAMPLES}" ]; then
      args+=(--max-samples "${MAX_SAMPLES}")
    fi
    if [ "${split}" = "mh" ] && [ -f data/annotation/test_motionhub_t2m_rewritten.json ]; then
      args+=(--caption-file data/annotation/test_motionhub_t2m_rewritten.json)
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/motionlab_infer_hml3d263.py \
      "${args[@]}" \
      ${EXTRA_INFER_ARGS} \
      > "${OUT}/logs/${split}_cond${cond}/infer_s${shard}_gpu${gpu}.log" 2>&1 &
  done
  wait
  echo "[${split}-cond${cond}-infer-done] npy=$(find "${out_dir}" -maxdepth 1 -name '*.npy' | wc -l)"
}

retarget_split_cond() {
  local split="$1"
  local cond="$2"
  local src="${OUT}/hml263/${split}_cond${cond}"
  local out_dir="${OUT}/smpl_npz/${split}_cond${cond}"
  mkdir -p "${out_dir}" "${OUT}/logs/${split}_cond${cond}"
  echo "[${split}-cond${cond}-retarget] $(date)"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$shard]}"
    # Generated HML263 rotations can be inconsistent with recovered RIC
    # joints; position-based IK keeps the SMPL fit in the 20-30mm range.
    CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "${src}" \
      --out-dir "${out_dir}" \
      --model-dir ref_repo/MDM/body_models \
      --source-fps 20 \
      --target-fps 30 \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --device cuda \
      --batch-size 512 \
      --floor-align \
      --rotation-init position \
      --rot6d-convention column \
      --refine-iters 0 \
      --skip-existing \
      > "${OUT}/logs/${split}_cond${cond}/ik_s${shard}_of_${NUM_SHARDS}.log" 2>&1 &
  done
  wait
  echo "[${split}-cond${cond}-retarget-done] npz=$(find "${out_dir}" -maxdepth 1 -name '*.npz' | wc -l)"
}

remap_eval_split_cond() {
  local split="$1"
  local anno="$2"
  local cond="$3"
  local gpu="$4"
  local smpl_dir="${OUT}/smpl_npz/${split}_cond${cond}"
  local mc_dir="${OUT}/motionclip135/${split}_cond${cond}"
  mkdir -p "${mc_dir}" "${OUT}/logs/${split}_cond${cond}"
  echo "[${split}-cond${cond}-remap] $(date)"
  python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
    --anno-file "${anno}" \
    --data-dir data/motionhub \
    --src-dir "${smpl_dir}" \
    --out-dir "${mc_dir}" \
    --include-mirrors \
    --key-fallback \
    --align-to-gt-root \
    --overwrite \
    --workers "${WORKERS}" \
    > "${OUT}/logs/${split}_cond${cond}/remap.log" 2>&1

  echo "[${split}-cond${cond}-eval] $(date)"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${mc_dir}" \
    --rot6d_convention column \
    --chunk_size "${CHUNK_SIZE}" \
    --out_json "${OUT}/metrics/${split}_cond${cond}_motionlab_c64.json" \
    --n_repeats "${N_REPEATS}" \
    --seed 42 \
    > "${OUT}/logs/${split}_cond${cond}/eval.log" 2>&1
}

if want_item build "${STAGES}" && want_item h3d "${SPLITS}"; then
  build_gt_hml263 h3d data/annotation/test_hml3d.json "${GT263_ROOT}/humanml3d"
fi
if want_item build "${STAGES}" && want_item mh "${SPLITS}"; then
  build_gt_hml263 mh data/annotation/test_motionhub_t2m.json "${GT263_ROOT}/motionhub"
fi

for cond in ${CONDS}; do
  if want_item h3d "${SPLITS}"; then
    if want_item infer "${STAGES}"; then
      infer_split_cond h3d data/annotation/test_hml3d.json "${GT263_ROOT}/humanml3d" "${cond}"
    fi
    if want_item retarget "${STAGES}"; then
      retarget_split_cond h3d "${cond}"
    fi
  fi
  if want_item mh "${SPLITS}"; then
    if want_item infer "${STAGES}"; then
      infer_split_cond mh data/annotation/test_motionhub_t2m.json "${GT263_ROOT}/motionhub" "${cond}"
    fi
    if want_item retarget "${STAGES}"; then
      retarget_split_cond mh "${cond}"
    fi
  fi
  pids=()
  if want_item eval "${STAGES}" && want_item h3d "${SPLITS}"; then
    remap_eval_split_cond h3d data/annotation/test_hml3d.json "${cond}" "${EVAL_GPU_H3D}" &
    pids+=("$!")
  fi
  if want_item eval "${STAGES}" && want_item mh "${SPLITS}"; then
    remap_eval_split_cond mh data/annotation/test_motionhub_t2m.json "${cond}" "${EVAL_GPU_MH}" &
    pids+=("$!")
  fi
  if [ "${#pids[@]}" -gt 0 ]; then
    wait "${pids[@]}"
  fi
done

if want_item summary "${STAGES}"; then
python3 - <<'PY'
import json
from pathlib import Path

import os
root = Path(os.environ.get("OUT", "outputs/evaluation/motionlab_tp2m_table2_0606")) / "metrics"
for path in sorted(root.glob("*.json")):
    d = json.loads(path.read_text())
    print(path.name, {
        "samples": d.get("samples"),
        "r3": d.get("r_precision_pred_top3_mean"),
        "fid": d.get("fid_mean"),
        "mm": d.get("mm_dist_pred_mean"),
        "div": d.get("diversity_pred_mean"),
    })
PY
fi

echo "[done] $(date)"
