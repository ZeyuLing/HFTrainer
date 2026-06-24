#!/usr/bin/env bash
# Launch independent Table-2 generator seed runs on multiple GPUs of one node.
set -euo pipefail

REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
CONFIG="${CONFIG:?CONFIG is required}"
PREFIX="${PREFIX:?PREFIX is required}"
GPU_LIST="${GPU_LIST:-1 2 3 4 5 6 7}"
SEED_OFFSET="${SEED_OFFSET:-0}"
MAX_KEEP="${MAX_KEEP:-3}"

cd "${REPO}"
mkdir -p work_dirs/table2_logs

for gpu in ${GPU_LIST}; do
  seed=$((SEED_OFFSET + gpu))
  seed_tag="$(printf "%02d" "${seed}")"
  session="table2_${PREFIX}_s${seed_tag}"
  work_dir="work_dirs/table2_g1_generator_${PREFIX}_seed${seed_tag}"
  log_path="work_dirs/table2_logs/${session}.log"
  pool_dir="${work_dir}/tracker_motion_pool"

  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "bash -lc 'cd \"${REPO}\" && \
      export CONFIG=\"${CONFIG}\" && \
      export CUDA_VISIBLE_DEVICES=\"${gpu}\" && \
      export OMP_NUM_THREADS=\"${OMP_NUM_THREADS:-4}\" && \
      export MKL_NUM_THREADS=\"${MKL_NUM_THREADS:-4}\" && \
      export WORK_DIR_OVERRIDE=\"${work_dir}\" && \
      export TRAIN_CFG_OPTIONS=\"trainer.tracker_pool_dir=${pool_dir} default_hooks.checkpoint.max_keep_ckpts=${MAX_KEEP}\" && \
      bash scripts/embodied/physflow_formal_trreward_node.sh > \"${log_path}\" 2>&1'"

  echo "[launch-table2] ${session} gpu=${gpu} work_dir=${work_dir} log=${log_path}"
done

tmux ls || true
