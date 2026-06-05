#!/bin/bash
# Full ViMoGen T2M evaluation for the paper table.
# Runs H3D and MotionHub with rewritten captions, 8-way sharded inference, and
# MotionCLIP chunk_size=64 metrics.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:${PYTHONPATH:-}

TAG=${TAG:-rw_full0605_dn1coord}
NUM_SHARDS=${NUM_SHARDS:-8}
ROOT=${ROOT:-outputs/evaluation/vimogen_t2m_0605}
LOG_ROOT="$ROOT/driver_${TAG}"
mkdir -p "$LOG_ROOT"

run_dataset() {
  local dataset="$1"
  local caption_json="$2"
  local tag="$TAG"

  echo "[$dataset] sharded inference start $(date)" | tee -a "$LOG_ROOT/run.log"
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    (
      export CUDA_VISIBLE_DEVICES="$i"
      export ENC_GPU="$i"
      export DATASET="$dataset"
      export MAX_SAMPLES=0
      export TAG="$tag"
      export NUM_SHARDS="$NUM_SHARDS"
      export SHARD_IDX="$i"
      export NPROC=1
      export TEST_BS=4
      export STEPS=50
      export CFG=5.0
      export DENOISING_STRENGTH=1.0
      export DTYPE=fp16
      export SKIP_EVAL=1
      export EVAL_CHUNK_SIZE=64
      export CAPTION_OVERRIDE_JSON="$caption_json"
      export VIMOGEN_DST_FPS=20
      export MASTER_PORT=$((31100 + i))
      bash scripts/eval/run_vimogen_t2m_eval_0605.sh
    ) > "$LOG_ROOT/${dataset}_s${i}.log" 2>&1 &
  done
  wait

  echo "[$dataset] finalize start $(date)" | tee -a "$LOG_ROOT/run.log"
  DATASET="$dataset" \
    TAG="$tag" \
    NUM_SHARDS="$NUM_SHARDS" \
    CHUNK_SIZE=64 \
    N_REPEATS=20 \
    GPU=0 \
    bash scripts/eval/finalize_vimogen_sharded_eval_0605.sh \
    > "$LOG_ROOT/${dataset}_finalize.log" 2>&1
  echo "[$dataset] done $(date)" | tee -a "$LOG_ROOT/run.log"
}

run_dataset h3d data/annotation/test_hml3d_rewritten.json
run_dataset mh data/annotation/test_motionhub_t2m_rewritten.json

echo "[all] done $(date)" | tee -a "$LOG_ROOT/run.log"
