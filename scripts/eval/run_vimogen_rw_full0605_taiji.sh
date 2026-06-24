#!/bin/bash
# Full ViMoGen T2M evaluation for the paper table.
# Runs H3D and MotionHub with rewritten captions, 8-way sharded inference, and
# MotionCLIP chunk_size=64 metrics.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [[ ! -d "$REPO_ROOT" ]]; then
  REPO_ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$REPO_ROOT"
export PYTHONPATH=$PWD:${PYTHONPATH:-}

TAG=${TAG:-rw_full0605_dn1coord}
NUM_SHARDS=${NUM_SHARDS:-8}
TOTAL_SHARDS=${TOTAL_SHARDS:-$NUM_SHARDS}
LOCAL_SHARDS=${LOCAL_SHARDS:-$NUM_SHARDS}
SHARD_OFFSET=${SHARD_OFFSET:-0}
OUT_BASE=${OUT_BASE:-outputs/evaluation/vimogen_t2m_0605}
DATASETS=${DATASETS:-"h3d mh"}
RUN_SHARDS=${RUN_SHARDS:-1}
RUN_FINALIZE=${RUN_FINALIZE:-1}
H3D_CAPTION_JSON=${H3D_CAPTION_JSON:-data/annotation/test_hml3d_rewritten.json}
MH_CAPTION_JSON=${MH_CAPTION_JSON:-data/annotation/test_motionhub_t2m_rewritten.json}
LOG_ROOT="$OUT_BASE/driver_${TAG}"
mkdir -p "$LOG_ROOT"

run_dataset() {
  local dataset="$1"
  local caption_json="$2"
  local tag="$TAG"

  if [[ "$RUN_SHARDS" == "1" ]]; then
  echo "[$dataset] sharded inference start $(date) offset=$SHARD_OFFSET local=$LOCAL_SHARDS total=$TOTAL_SHARDS" | tee -a "$LOG_ROOT/run.log"
  for local_i in $(seq 0 $((LOCAL_SHARDS - 1))); do
    i=$((SHARD_OFFSET + local_i))
    if [[ "$i" -ge "$TOTAL_SHARDS" ]]; then
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES="$local_i"
      export ENC_GPU="$local_i"
      export DATASET="$dataset"
      export MAX_SAMPLES=0
      export TAG="$tag"
      export NUM_SHARDS="$TOTAL_SHARDS"
      export SHARD_IDX="$i"
      export OUT_ROOT="$OUT_BASE/${dataset}_${tag}_s${i}of${TOTAL_SHARDS}"
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
      export VIMOGEN_TEXT_OVERWRITE=1
      export MASTER_PORT=$((31100 + i))
      bash scripts/eval/run_vimogen_t2m_eval_0605.sh
    ) > "$LOG_ROOT/${dataset}_s${i}.log" 2>&1 &
  done
  wait
  fi

  if [[ "$RUN_FINALIZE" == "1" ]]; then
  echo "[$dataset] finalize start $(date)" | tee -a "$LOG_ROOT/run.log"
  DATASET="$dataset" \
    TAG="$tag" \
    NUM_SHARDS="$TOTAL_SHARDS" \
    ROOT="$OUT_BASE" \
    CHUNK_SIZE=64 \
    N_REPEATS=20 \
    ANNO_OVERRIDE="${ANNO_OVERRIDE:-}" \
    GPU=0 \
    bash scripts/eval/finalize_vimogen_sharded_eval_0605.sh \
    > "$LOG_ROOT/${dataset}_finalize.log" 2>&1
  echo "[$dataset] done $(date)" | tee -a "$LOG_ROOT/run.log"
  fi
}

for dataset in $DATASETS; do
  if [[ "$dataset" == "h3d" ]]; then
    run_dataset h3d "$H3D_CAPTION_JSON"
  elif [[ "$dataset" == "mh" ]]; then
    run_dataset mh "$MH_CAPTION_JSON"
  else
    echo "Unknown dataset in DATASETS=$dataset" >&2
    exit 2
  fi
done

echo "[all] done $(date)" | tee -a "$LOG_ROOT/run.log"
