#!/usr/bin/env bash
# Retarget and evaluate one MotionLab TP2M condition without rerunning inference.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [ ! -d "${ROOT}" ]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python3 - <<'PY' || python3 -m pip install --user -q --no-build-isolation chumpy
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
import chumpy  # noqa: F401
PY

OUT=${OUT:-outputs/evaluation/motionlab_tp2m_table2_0606_fix2}
COND=${COND:-1}
SPLITS=${SPLITS:-"h3d mh"}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
NUM_SHARDS=${NUM_SHARDS:-8}
WORKERS=${WORKERS:-16}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -lt "${NUM_SHARDS}" ]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

retarget_split_cond() {
  local split="$1"
  local src="${OUT}/hml263/${split}_cond${COND}"
  local out_dir="${OUT}/smpl_npz/${split}_cond${COND}"
  mkdir -p "${out_dir}" "${OUT}/logs/${split}_cond${COND}"
  echo "[${split}-cond${COND}-retarget] $(date) src=${src}"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$shard]}"
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
      --rotation-init hml263 \
      --rot6d-convention column \
      --refine-iters 0 \
      --skip-existing \
      > "${OUT}/logs/${split}_cond${COND}/ik_recover_s${shard}_of_${NUM_SHARDS}.log" 2>&1 &
  done
  wait
  echo "[${split}-cond${COND}-retarget-done] npz=$(find "${out_dir}" -maxdepth 1 -name '*.npz' | wc -l)"
}

eval_split_cond() {
  local split="$1"
  local anno
  if [ "${split}" = "h3d" ]; then
    anno=data/annotation/test_hml3d.json
  else
    anno=data/annotation/test_motionhub_t2m.json
  fi
  local smpl_dir="${OUT}/smpl_npz/${split}_cond${COND}"
  local mc_dir="${OUT}/motionclip135/${split}_cond${COND}"
  mkdir -p "${mc_dir}" "${OUT}/logs/${split}_cond${COND}"
  echo "[${split}-cond${COND}-remap] $(date)"
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
    > "${OUT}/logs/${split}_cond${COND}/remap_recover.log" 2>&1

  echo "[${split}-cond${COND}-eval] $(date)"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${mc_dir}" \
    --rot6d_convention column \
    --chunk_size "${CHUNK_SIZE}" \
    --out_json "${OUT}/metrics/${split}_cond${COND}_motionlab_c64.json" \
    --n_repeats "${N_REPEATS}" \
    --seed 42 \
    > "${OUT}/logs/${split}_cond${COND}/eval_recover.log" 2>&1
}

for split in ${SPLITS}; do
  retarget_split_cond "${split}"
done

for split in ${SPLITS}; do
  eval_split_cond "${split}" &
done
wait

python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("OUT", "outputs/evaluation/motionlab_tp2m_table2_0606_fix2")) / "metrics"
for path in sorted(root.glob("*_motionlab_c64.json")):
    d = json.loads(path.read_text())
    print(path.name, {
        "samples": d.get("samples"),
        "r3": d.get("r_precision_pred_top3_mean"),
        "fid": d.get("fid_mean"),
        "mm": d.get("mm_dist_pred_mean"),
        "div": d.get("diversity_pred_mean"),
    })
PY
