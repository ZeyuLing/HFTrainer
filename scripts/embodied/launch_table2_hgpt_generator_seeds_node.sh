#!/usr/bin/env bash
# Launch independent Table-2 generator+Humanoid-GPT runs on one multi-GPU node.
set -euo pipefail

REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
CONFIG="${CONFIG:-configs/physflow/table2_g1_generator_humanoidgpt.py}"
PREFIX="${PREFIX:-humanoidgpt}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
SEED_OFFSET="${SEED_OFFSET:-0}"
MAX_ITERS="${MAX_ITERS:-3000}"
GENCKPT="${GENCKPT:-work_dirs/hymotion_g1_t2m_38dim/checkpoint-iter_339000}"
PHYSFLOW_HGPT_VENV="${PHYSFLOW_HGPT_VENV:-/dev/shm/hgpt_venv311_table2}"

cd "${REPO}"
mkdir -p work_dirs/table2_logs

for gpu in ${GPU_LIST}; do
  seed=$((SEED_OFFSET + gpu))
  seed_tag="$(printf "%02d" "${seed}")"
  session="table2_${PREFIX}_s${seed_tag}"
  work_dir="work_dirs/table2_g1_generator_${PREFIX}_seed${seed_tag}"
  log_path="work_dirs/table2_logs/${session}.log"

  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "bash -lc 'cd \"${REPO}\" && \
      export CONFIG=\"${CONFIG}\" && \
      export CUDA_VISIBLE_DEVICES=\"${gpu}\" && \
      export WORK_DIR=\"${work_dir}\" && \
      export GENCKPT=\"${GENCKPT}\" && \
      export MAX_ITERS=\"${MAX_ITERS}\" && \
      export PHYSFLOW_HGPT_VENV=\"${PHYSFLOW_HGPT_VENV}\" && \
      export OMP_NUM_THREADS=\"${OMP_NUM_THREADS:-4}\" && \
      export MKL_NUM_THREADS=\"${MKL_NUM_THREADS:-4}\" && \
      bash scripts/embodied/physflow_table2_generator_hgpt_node.sh > \"${log_path}\" 2>&1'"

  echo "[launch-table2-hgpt] ${session} gpu=${gpu} work_dir=${work_dir} log=${log_path}"
done

tmux ls || true
