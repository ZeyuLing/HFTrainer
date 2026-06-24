#!/bin/bash
# MotionStreamer sequential-action generation on BABEL val (Table 3 baseline).
# Independent single-host jobs set JOB_RANK / JOB_COUNT; each host runs NUM_GPUS
# parallel shards. Writes <id>.npz (motion_272) -> consumable directly by
# eval_babel_seq_ms272.py (no repack).
set -uo pipefail
ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
[ -d "$ROOT" ] || ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/ref_repo/MotionStreamer/MotionStreamer:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/checkpoints/huggingface_motionstreamer}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=${PY:-python3}

OUT=${OUT:-outputs/evaluation/babel_seq/ms_gen}
NUM_GPUS=${NUM_GPUS:-8}
MANIFEST=${MANIFEST:-data/babel/babel_seq_val_manifest.jsonl}
CFG=${CFG:-4.0}

if [ -n "${JOB_COUNT:-}" ]; then
  HOST_RANK=${JOB_RANK:-0}; MACHINE_NUM=${JOB_COUNT}
else
  HOST_RANK=${INDEX:-0}; MACHINE_NUM=${MACHINE_NUM:-1}
fi
TOTAL_SHARDS=$((MACHINE_NUM * NUM_GPUS))

mkdir -p "$OUT/logs"
echo "[ms-start] out=$OUT host_rank=$HOST_RANK machines=$MACHINE_NUM gpus/node=$NUM_GPUS total_shards=$TOTAL_SHARDS"
for i in $(seq 0 $((NUM_GPUS - 1))); do
  SHARD=$((HOST_RANK * NUM_GPUS + i))
  CUDA_VISIBLE_DEVICES=$i "$PY" -u scripts/eval/gen_motionstreamer_babel_seq.py \
    --manifest "$MANIFEST" --out-dir "$OUT" \
    --cfg "$CFG" --min-total 24 --max-total 360 \
    --num-shards "$TOTAL_SHARDS" --shard-index "$SHARD" \
    --max-episodes "${MAX_EP:-0}" \
    --skip-existing --device cuda \
    > "$OUT/logs/gen_h${HOST_RANK}_g$i.log" 2>&1 &
done
wait
n=$(find "$OUT" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
echo "[ms host=$HOST_RANK done] total_npz=$n"
