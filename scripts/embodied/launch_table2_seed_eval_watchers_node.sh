#!/usr/bin/env bash
# Launch per-seed final-eval watchers for Table-2 generator runs on one node.
set -euo pipefail

REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
PREFIX="${PREFIX:?PREFIX is required}"
GPU_LIST="${GPU_LIST:-1 2 3 4 5 6 7}"
SEED_OFFSET="${SEED_OFFSET:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"

cd "${REPO}"
mkdir -p work_dirs/table2_logs

for gpu in ${GPU_LIST}; do
  seed=$((SEED_OFFSET + gpu))
  seed_tag="$(printf "%02d" "${seed}")"
  session="table2_eval_${PREFIX}_s${seed_tag}_wait"
  work_dir="work_dirs/table2_g1_generator_${PREFIX}_seed${seed_tag}"
  out_dir="outputs/evaluation/physflow/table2_generator/heldout_agile/${PREFIX}_seed${seed_tag}_iter3000"
  log_path="work_dirs/table2_logs/${session}.log"

  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "bash -lc 'cd \"${REPO}\" && \
      export CUDA_VISIBLE_DEVICES=\"${gpu}\" && \
      export OMP_NUM_THREADS=\"${OMP_NUM_THREADS:-4}\" && \
      export MKL_NUM_THREADS=\"${MKL_NUM_THREADS:-4}\" && \
      export POLL_SECONDS=\"${POLL_SECONDS}\" && \
      export CHECKPOINT=\"${work_dir}/checkpoint-iter_3000\" && \
      export OUT=\"${out_dir}\" && \
      bash scripts/embodied/run_table2_generator_eval.sh > \"${log_path}\" 2>&1'"

  echo "[launch-eval-watcher] ${session} gpu=${gpu} checkpoint=${work_dir}/checkpoint-iter_3000 log=${log_path}"
done

tmux ls || true
