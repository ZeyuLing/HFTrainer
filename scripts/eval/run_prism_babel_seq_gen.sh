#!/bin/bash
# PRISM sequential-action generation on BABEL val (Table 3), multi-host sharded.
# Each host runs NUM_GPUS parallel shards of gen_prism_babel_seq.py; outputs
# <id>.npz (SMPLX) to OUT. MS-272 repack + eval_babel_seq_ms272.py run afterwards.
set -uo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "${ROOT}" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "${ROOT}"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=${PY:-python3}

OUT=${OUT:-outputs/evaluation/babel_seq/prism_gen}
NUM_GPUS=${NUM_GPUS:-8}
CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
CHECKPOINT=${CHECKPOINT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_19}
MANIFEST=${MANIFEST:-data/babel/babel_seq_val_manifest.jsonl}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-5.0}

# --- Sharding ---
# Preferred: independent single-host jobs set JOB_RANK / JOB_COUNT explicitly
# (proven robust on Taiji, avoids multi-host MPI). Fallback: Taiji multi-host
# INDEX / NODE_LIST.
if [ -n "${JOB_COUNT:-}" ]; then
  HOST_RANK=${JOB_RANK:-0}
  MACHINE_NUM=${JOB_COUNT}
else
  HOST_RANK=${INDEX:-0}
  if [ -n "${NODE_LIST:-}" ]; then
    MACHINE_NUM=$(python3 -c "import os;print(len(os.environ['NODE_LIST'].split(',')))" 2>/dev/null || echo 1)
  else
    MACHINE_NUM=${MACHINE_NUM:-1}
  fi
fi
TOTAL_SHARDS=$((MACHINE_NUM * NUM_GPUS))

mkdir -p "$OUT/logs"
echo "[start] out=$OUT ckpt=$CHECKPOINT host_rank=$HOST_RANK machines=$MACHINE_NUM gpus/node=$NUM_GPUS total_shards=$TOTAL_SHARDS"

for i in $(seq 0 $((NUM_GPUS - 1))); do
  SHARD=$((HOST_RANK * NUM_GPUS + i))
  CUDA_VISIBLE_DEVICES=$i "$PY" -u scripts/eval/gen_prism_babel_seq.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --output-dir "$OUT" \
    --num-inference-steps "$STEPS" \
    --guidance-scale "$GUIDANCE" \
    --kafs-mode "${KAFS_MODE:-none}" \
    --rewrite-captions \
    --num-shards "$TOTAL_SHARDS" \
    --shard-idx "$SHARD" \
    --skip-existing \
    > "$OUT/logs/gen_h${HOST_RANK}_g$i.log" 2>&1 &
done
wait
n=$(find "$OUT" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
echo "[host=$HOST_RANK gen done] total_npz=$n"
echo "[ALL DONE host=$HOST_RANK]"
