#!/usr/bin/env bash
# Single split/condition MotionStreamer latent-prefix TP2M rerun.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [ ! -d "${ROOT}" ]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PY=${PY:-python3}

OUT=${OUT:-outputs/evaluation/motionstreamer_tp2m_table2_0606_parallel}
SPLIT=${SPLIT:-h3d}  # h3d | motionhub
COND=${COND:-1}
NUM_GPUS=${NUM_GPUS:-8}
TOTAL_SHARDS=${TOTAL_SHARDS:-${NUM_GPUS}}
SHARD_OFFSET=${SHARD_OFFSET:-0}
SHARD_COUNT=${SHARD_COUNT:-${NUM_GPUS}}
SHARD_INDICES=${SHARD_INDICES:-}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
RUN_EVAL=${RUN_EVAL:-1}
LOG_SUFFIX=${LOG_SUFFIX:-}
PREFIX_LATENT_SOURCE=${PREFIX_LATENT_SOURCE:-sample}
SAMPLING_METHOD=${SAMPLING_METHOD:-new_demo}
MS_CFG=${MS_CFG:-4.5}
MS_TEMPERATURE=${MS_TEMPERATURE:-1.0}
CAPTION_PROTOCOL=${CAPTION_PROTOCOL:-original}
REWRITTEN_FILE=${REWRITTEN_FILE:-}
MAX_SAMPLES=${MAX_SAMPLES:-}
EXTRA_GEN_ARGS=${EXTRA_GEN_ARGS:-}
T5_MODEL=${T5_MODEL:-}
GT272_H3D=${GT272_H3D:-${OUT}/gt272_humanml3d}
GT272_MH=${GT272_MH:-${OUT}/gt272_motionhub}

mkdir -p "${OUT}/logs/${SPLIT}_cond${COND}" "${OUT}/metrics"
if [ -n "${SHARD_INDICES}" ]; then
  read -r -a SELECTED_SHARDS <<< "${SHARD_INDICES}"
else
  SELECTED_SHARDS=()
  for shard in $(seq "${SHARD_OFFSET}" $((SHARD_OFFSET + SHARD_COUNT - 1))); do
    SELECTED_SHARDS+=("${shard}")
  done
fi
if [ "${#SELECTED_SHARDS[@]}" -gt "${NUM_GPUS}" ]; then
  echo "selected shards=${#SELECTED_SHARDS[@]} exceeds NUM_GPUS=${NUM_GPUS}" >&2
  exit 2
fi
for shard in "${SELECTED_SHARDS[@]}"; do
  if [ "${shard}" -lt 0 ] || [ "${shard}" -ge "${TOTAL_SHARDS}" ]; then
    echo "selected shard ${shard} is outside TOTAL_SHARDS=${TOTAL_SHARDS}" >&2
    exit 2
  fi
done

(
  flock 9
  "$PY" - <<'PY' > "${OUT}/logs/install_check.log" 2>&1
missing = []
for mod in ["sentence_transformers", "mmengine", "smplx"]:
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
print("missing", missing)
PY
  if grep -q "missing \\[\\]" "${OUT}/logs/install_check.log"; then
    :
  else
    "$PY" -m pip install -q --upgrade 'pip<25' setuptools wheel >> "${OUT}/logs/install_check.log" 2>&1
    "$PY" -m pip install -q --ignore-installed PyYAML 'numpy<2' sentence-transformers smplx addict yapf rich termcolor >> "${OUT}/logs/install_check.log" 2>&1
    "$PY" -m pip install -q --no-deps mmengine >> "${OUT}/logs/install_check.log" 2>&1
  fi
) 9>/tmp/motionstreamer_tp2m_pip_install.lock

if [ "${SPLIT}" = "h3d" ]; then
  ANNO=data/annotation/test_hml3d.json
  DATASET=humanml3d
  GT272="${GT272_H3D}"
elif [ "${SPLIT}" = "motionhub" ]; then
  ANNO=data/annotation/test_motionhub_t2m.json
  DATASET=motionhub
  GT272="${GT272_MH}"
else
  echo "Unknown SPLIT=${SPLIT}" >&2
  exit 2
fi

if [ ! -d "${GT272}" ]; then
  echo "Missing GT272 dir: ${GT272}" >&2
  exit 2
fi

echo "[${SPLIT}-cond${COND}-gen] $(date) out=${OUT} gt272=${GT272} total_shards=${TOTAL_SHARDS} selected=${SELECTED_SHARDS[*]}"
gen_args=(
  --dataset "${DATASET}"
  --out-dir "${OUT}/${SPLIT}"
  --gt-272-dir "${GT272}"
  --condition-num-frames "${COND}"
  --anno-file "${ANNO}"
  --data-dir data/motionhub
  --caption-protocol "${CAPTION_PROTOCOL}"
  --prefix-latent-source "${PREFIX_LATENT_SOURCE}"
  --sampling-method "${SAMPLING_METHOD}"
  --cfg "${MS_CFG}"
  --temperature "${MS_TEMPERATURE}"
  --align-to-gt-root
  --align-root-mode yaw
  --skip-existing
)
if [ -n "${T5_MODEL}" ]; then
  gen_args+=(--t5-model "${T5_MODEL}")
fi
if [ -n "${REWRITTEN_FILE}" ]; then
  gen_args+=(--rewritten-file "${REWRITTEN_FILE}")
fi
if [ -n "${MAX_SAMPLES}" ]; then
  gen_args+=(--max-samples "${MAX_SAMPLES}")
fi
for local_idx in "${!SELECTED_SHARDS[@]}"; do
  shard="${SELECTED_SHARDS[$local_idx]}"
  gpu="${local_idx}"
  CUDA_VISIBLE_DEVICES="${gpu}" "$PY" scripts/eval/gen_motionstreamer_tp2m_smpl_npz.py \
    "${gen_args[@]}" \
    --num-shards "${TOTAL_SHARDS}" \
    --shard-index "${shard}" \
    ${EXTRA_GEN_ARGS} \
    > "${OUT}/logs/${SPLIT}_cond${COND}/gen_s${shard}${LOG_SUFFIX}.log" 2>&1 &
done
wait

GEN_DIR="${OUT}/${SPLIT}/cond${COND}_latent_prefix"
echo "[${SPLIT}-cond${COND}-gen-done] npz=$(find "${GEN_DIR}" -maxdepth 1 -name '*.npz' | wc -l)"

if [ "${RUN_EVAL}" = "0" ]; then
  echo "[done] generation only"
  exit 0
fi

echo "[${SPLIT}-cond${COND}-eval] $(date)"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file "${ANNO}" \
  --data_dir data/motionhub \
  --pred_dir "${GEN_DIR}" \
  --chunk_size "${CHUNK_SIZE}" \
  --out_json "${OUT}/metrics/${SPLIT}_cond${COND}_motionstreamer_c64.json" \
  --n_repeats "${N_REPEATS}" \
  --seed 42 \
  > "${OUT}/logs/${SPLIT}_cond${COND}/eval${LOG_SUFFIX}.log" 2>&1

echo "[done] ${OUT}/metrics/${SPLIT}_cond${COND}_motionstreamer_c64.json"
