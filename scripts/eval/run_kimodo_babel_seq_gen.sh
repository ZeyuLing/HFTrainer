#!/bin/bash
# KIMODO sequential-action generation on BABEL val (Table 3 baseline).
# Per-segment T2M then concat -> <id>.npy (T,22,3). Independent single-host jobs
# set JOB_RANK / JOB_COUNT; each host runs NUM_GPUS parallel shards.
set -uo pipefail
ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
[ -d "$ROOT" ] || ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/checkpoints/kimodo" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="$ROOT/checkpoints/kimodo/text_encoders"
export CHECKPOINT_DIR="$ROOT/checkpoints/kimodo/local_models" TEXT_ENCODER_MODE=local
PY=${PY:-python3}

OUT=${OUT:-outputs/evaluation/babel_seq/kimodo_gen}
NUM_GPUS=${NUM_GPUS:-8}
NS=${NS:-kimodo_soma_t2m_babel_val_llm2vec}

if [ -n "${JOB_COUNT:-}" ]; then
  HOST_RANK=${JOB_RANK:-0}; MACHINE_NUM=${JOB_COUNT}
else
  HOST_RANK=${INDEX:-0}; MACHINE_NUM=${MACHINE_NUM:-1}
fi
TOTAL_SHARDS=$((MACHINE_NUM * NUM_GPUS))

mkdir -p "$OUT/logs"
echo "[kimodo-start] out=$OUT host_rank=$HOST_RANK total_shards=$TOTAL_SHARDS"
for i in $(seq 0 $((NUM_GPUS - 1))); do
  SHARD=$((HOST_RANK * NUM_GPUS + i))
  CUDA_VISIBLE_DEVICES=$i "$PY" -u scripts/eval/gen_kimodo_babel_seq.py \
    --out-dir "$OUT" --min-total 24 --max-total 360 \
    --num-shards "$TOTAL_SHARDS" --shard-index "$SHARD" \
    --text-feature-namespace "$NS" --max-episodes "${MAX_EP:-0}" \
    --skip-existing --device cuda \
    > "$OUT/logs/gen_h${HOST_RANK}_g$i.log" 2>&1 &
done
wait
n=$(find "$OUT" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
echo "[kimodo host=$HOST_RANK done] total_npy=$n"
