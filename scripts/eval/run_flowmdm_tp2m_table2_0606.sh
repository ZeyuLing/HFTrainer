#!/usr/bin/env bash
# FlowMDM prefix-pose-conditioned generation for Table 2.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [ ! -d "${ROOT}" ]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PY=${PY:-python3}
OUT=${OUT:-outputs/evaluation/flowmdm_tp2m_table2_0606}
GT263_ROOT=${GT263_ROOT:-outputs/evaluation/humanml3d/gt_smpl135_to_hml263}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
NUM_SHARDS=${NUM_SHARDS:-8}
TOTAL_SHARDS=${TOTAL_SHARDS:-${NUM_SHARDS}}
SHARD_OFFSET=${SHARD_OFFSET:-0}
SHARD_COUNT=${SHARD_COUNT:-${NUM_SHARDS}}
SHARD_INDICES=${SHARD_INDICES:-}
WORKERS=${WORKERS:-16}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
GUIDANCE=${GUIDANCE:-2.5}
BPE_STEP=${BPE_STEP:-60}
MAX_SAMPLES=${MAX_SAMPLES:-}
MEAN_PATH=${MEAN_PATH:-}
STD_PATH=${STD_PATH:-}
SPLITS=${SPLITS:-"h3d mh"}
CONDS=${CONDS:-"1 5 9"}
STAGES=${STAGES:-"build infer retarget eval summary"}
CLIP_DOWNLOAD_ROOT=${CLIP_DOWNLOAD_ROOT:-checkpoints/clip}
CLIP_DOWNLOAD_ROOT=$(realpath -m "${CLIP_DOWNLOAD_ROOT}")

mkdir -p "${OUT}/logs" "${OUT}/metrics" "${GT263_ROOT}/humanml3d" "${GT263_ROOT}/motionhub" "${CLIP_DOWNLOAD_ROOT}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ -n "${SHARD_INDICES}" ]; then
  read -r -a SELECTED_SHARDS <<< "${SHARD_INDICES}"
else
  SELECTED_SHARDS=()
  for shard in $(seq "${SHARD_OFFSET}" $((SHARD_OFFSET + SHARD_COUNT - 1))); do
    SELECTED_SHARDS+=("${shard}")
  done
fi
if [ "${#GPUS[@]}" -lt "${#SELECTED_SHARDS[@]}" ]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but selected shards=${#SELECTED_SHARDS[@]}" >&2
  exit 2
fi
for shard in "${SELECTED_SHARDS[@]}"; do
  if [ "${shard}" -lt 0 ] || [ "${shard}" -ge "${TOTAL_SHARDS}" ]; then
    echo "selected shard ${shard} is outside TOTAL_SHARDS=${TOTAL_SHARDS}" >&2
    exit 2
  fi
done

echo "[start] FlowMDM TP2M out=${OUT} total_shards=${TOTAL_SHARDS} selected=${SELECTED_SHARDS[*]} stages=${STAGES}"

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

(
  flock 9
  "${PY}" - <<'PY' > "${OUT}/logs/install_check.log" 2>&1
missing = []
for mod in ["einops", "mmengine", "smplx"]:
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
print("missing", missing)
PY
  if grep -q "missing \\[\\]" "${OUT}/logs/install_check.log"; then
    :
  else
    "${PY}" -m pip install -q --upgrade 'pip<25' setuptools wheel >> "${OUT}/logs/install_check.log" 2>&1
    "${PY}" -m pip install -q 'numpy<2' einops smplx addict yapf rich termcolor >> "${OUT}/logs/install_check.log" 2>&1
    "${PY}" -m pip install -q --no-deps mmengine >> "${OUT}/logs/install_check.log" 2>&1
  fi
) 9>/tmp/flowmdm_tp2m_pip_install.lock

"${PY}" - <<PY > "${OUT}/logs/clip_prefetch.log" 2>&1
import clip
clip.load("ViT-B/32", device="cpu", jit=False, download_root="${CLIP_DOWNLOAD_ROOT}")
print("clip_prefetch_done", "${CLIP_DOWNLOAD_ROOT}")
PY

build_gt_hml263() {
  local split="$1"
  local anno="$2"
  local out_dir="$3"
  echo "[${split}-build-gt-hml263] $(date)"
  "${PY}" scripts/eval/build_gt_smpl135_to_hml263.py \
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
  for local_idx in "${!SELECTED_SHARDS[@]}"; do
    local shard="${SELECTED_SHARDS[$local_idx]}"
    local gpu="${GPUS[$local_idx]}"
    local args=(
      --anno-file "${anno}"
      --data-dir data/motionhub
      --gt-hml263-dir "${gt263}"
      --out-dir "${out_dir}"
      --condition-num-frames "${cond}"
      --guidance-param "${GUIDANCE}"
      --bpe-denoising-step "${BPE_STEP}"
      --clip-download-root "${CLIP_DOWNLOAD_ROOT}"
      --num-shards "${TOTAL_SHARDS}"
      --shard-index "${shard}"
      --skip-existing
      --device 0
    )
    if [ -n "${MAX_SAMPLES}" ]; then
      args+=(--max-samples "${MAX_SAMPLES}")
    fi
    if [ -n "${MEAN_PATH}" ] && [ -n "${STD_PATH}" ]; then
      args+=(--mean-path "${MEAN_PATH}" --std-path "${STD_PATH}")
    fi
    if [ "${split}" = "mh" ] && [ -f data/annotation/test_motionhub_t2m_rewritten.json ]; then
      args+=(--caption-file data/annotation/test_motionhub_t2m_rewritten.json)
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/eval/flowmdm_infer_hml3d263.py \
      "${args[@]}" \
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
  for local_idx in "${!SELECTED_SHARDS[@]}"; do
    local shard="${SELECTED_SHARDS[$local_idx]}"
    local gpu="${GPUS[$local_idx]}"
    # Generated HML263 rotations can be inconsistent with recovered RIC
    # joints; position-based IK keeps the SMPL fit in the 20-30mm range.
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "${src}" \
      --out-dir "${out_dir}" \
      --model-dir ref_repo/MDM/body_models \
      --source-fps 20 \
      --target-fps 30 \
      --num-shards "${TOTAL_SHARDS}" \
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
  "${PY}" scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
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
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${mc_dir}" \
    --rot6d_convention column \
    --chunk_size "${CHUNK_SIZE}" \
    --out_json "${OUT}/metrics/${split}_cond${cond}_flowmdm_c64.json" \
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
    if want_item eval "${STAGES}"; then
      remap_eval_split_cond h3d data/annotation/test_hml3d.json "${cond}" 0
    fi
  fi
  if want_item mh "${SPLITS}"; then
    if want_item infer "${STAGES}"; then
      infer_split_cond mh data/annotation/test_motionhub_t2m.json "${GT263_ROOT}/motionhub" "${cond}"
    fi
    if want_item retarget "${STAGES}"; then
      retarget_split_cond mh "${cond}"
    fi
    if want_item eval "${STAGES}"; then
      remap_eval_split_cond mh data/annotation/test_motionhub_t2m.json "${cond}" 0
    fi
  fi
done

if want_item summary "${STAGES}"; then
"${PY}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("OUT", "outputs/evaluation/flowmdm_tp2m_table2_0606")) / "metrics"
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
