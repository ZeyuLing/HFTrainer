#!/usr/bin/env bash
# PRISM epoch-43 BABEL official-val generation with configurable AR prefix length.
# Used to diagnose whether first-frame/rollout failures depend on condition frames.
set -uo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "${ROOT}" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "${ROOT}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

PY=${PY:-python3}
NUM_GPUS=${NUM_GPUS:-8}
AR_COND_FRAMES=${AR_COND_FRAMES:-5}
BASE=${BASE:-outputs/evaluation/babel/official_val/msstyle_30fps_gt}
KAFS_MODE=${KAFS_MODE:-depth_driven}
LENGTH_POLICY=${LENGTH_POLICY:-pad360_crop}
PAD_TO_FRAMES=${PAD_TO_FRAMES:-360}
TRANSLATION_DECODE_MODE=${TRANSLATION_DECODE_MODE:-xz_rollout_y_absolute}
OUT=${OUT:-${BASE}/prism_epoch43_pad360crop_arcond${AR_COND_FRAMES}_${KAFS_MODE}}
CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
CHECKPOINT=${CHECKPOINT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_43}
MANIFEST=${MANIFEST:-${BASE}/manifest.jsonl}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-5.0}
ORIENTATION_RETRIES=${ORIENTATION_RETRIES:-4}

LOCAL_HOST_RANK=${INDEX:-0}
if [ -n "${NODE_LIST:-}" ]; then
  MACHINE_NUM=$("${PY}" - <<'PY'
import os
print(len(os.environ["NODE_LIST"].split(",")))
PY
)
else
  MACHINE_NUM=${MACHINE_NUM:-1}
fi
JOB_RANK=${JOB_RANK:-0}
JOB_COUNT=${JOB_COUNT:-1}
TOTAL_SHARDS=$((JOB_COUNT * MACHINE_NUM * NUM_GPUS))
GLOBAL_HOST_RANK=$((JOB_RANK * MACHINE_NUM + LOCAL_HOST_RANK))
SHARD_BASE=$((GLOBAL_HOST_RANK * NUM_GPUS))

mkdir -p "$OUT/logs"
cat > "$OUT/command_job${JOB_RANK}_host${LOCAL_HOST_RANK}.txt" <<EOF
ROOT=$ROOT
CONFIG=$CONFIG
CHECKPOINT=$CHECKPOINT
MANIFEST=$MANIFEST
OUT=$OUT
AR_COND_FRAMES=$AR_COND_FRAMES
STEPS=$STEPS
GUIDANCE=$GUIDANCE
KAFS_MODE=$KAFS_MODE
LENGTH_POLICY=$LENGTH_POLICY
PAD_TO_FRAMES=$PAD_TO_FRAMES
TRANSLATION_DECODE_MODE=$TRANSLATION_DECODE_MODE
ORIENTATION_RETRIES=$ORIENTATION_RETRIES
JOB_RANK=$JOB_RANK
JOB_COUNT=$JOB_COUNT
LOCAL_HOST_RANK=$LOCAL_HOST_RANK
GLOBAL_HOST_RANK=$GLOBAL_HOST_RANK
MACHINE_NUM=$MACHINE_NUM
NUM_GPUS=$NUM_GPUS
TOTAL_SHARDS=$TOTAL_SHARDS
SHARD_BASE=$SHARD_BASE
EOF

echo "[start] $(date) out=$OUT ckpt=$CHECKPOINT ar_cond=$AR_COND_FRAMES job=$JOB_RANK/$JOB_COUNT host=$LOCAL_HOST_RANK machines=$MACHINE_NUM total_shards=$TOTAL_SHARDS shard_base=$SHARD_BASE length_policy=$LENGTH_POLICY translation=$TRANSLATION_DECODE_MODE"

for i in $(seq 0 $((NUM_GPUS - 1))); do
  SHARD=$((SHARD_BASE + i))
  CUDA_VISIBLE_DEVICES=$i "$PY" -u scripts/eval/gen_prism_babel_official_seq.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --output-dir "$OUT" \
    --num-inference-steps "$STEPS" \
    --guidance-scale "$GUIDANCE" \
    --kafs-mode "$KAFS_MODE" \
    --length-policy "$LENGTH_POLICY" \
    --pad-to-frames "$PAD_TO_FRAMES" \
    --translation-decode-mode "$TRANSLATION_DECODE_MODE" \
    --ar-cond-frames "$AR_COND_FRAMES" \
    --num-shards "$TOTAL_SHARDS" \
    --shard-idx "$SHARD" \
    --orientation-retries "$ORIENTATION_RETRIES" \
    --skip-existing \
    > "$OUT/logs/gen_job${JOB_RANK}_h${LOCAL_HOST_RANK}_g${i}.log" 2>&1 &
done
wait

n=$(find "$OUT" -maxdepth 1 -name 'val_*.npz' 2>/dev/null | wc -l)
bad=$(find "$OUT/_meta" -maxdepth 1 -name 'val_*.json' 2>/dev/null | xargs -r grep -l '"bad": true' | wc -l)
echo "[done] $(date) out=$OUT npz=$n meta_with_bad_attempts=$bad"
