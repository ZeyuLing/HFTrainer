#!/usr/bin/env bash
# PRISM epoch-43 TP2M generation for Table 2 on HumanML3D official test.
# Multi-host Taiji worker: each host runs NUM_GPUS shards; all hosts share CephFS.
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
OUT=${OUT:-outputs/evaluation/tp2m/humanml3d_official_test/motion135/prism_epoch43_pad360crop_selected_20260628}
CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
CHECKPOINT=${CHECKPOINT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_43}
ANNO=${ANNO:-outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/test_hml3d_official272_gtlen_motionclip_selected_caption.json}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-5.0}
MAXF=${MAXF:-360}
CONDS=${CONDS:-"1 5 9"}
LENGTH_POLICY=${LENGTH_POLICY:-pad360_crop}
PAD_TO_FRAMES=${PAD_TO_FRAMES:-360}
TRANSLATION_DECODE_MODE=${TRANSLATION_DECODE_MODE:-xz_rollout_y_absolute}

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

mkdir -p "$OUT/logs" "$OUT/metrics"
cat > "$OUT/command_job${JOB_RANK}_host${LOCAL_HOST_RANK}.txt" <<EOF
ROOT=$ROOT
CONFIG=$CONFIG
CHECKPOINT=$CHECKPOINT
ANNO=$ANNO
OUT=$OUT
CONDS=$CONDS
STEPS=$STEPS
GUIDANCE=$GUIDANCE
MAXF=$MAXF
LENGTH_POLICY=$LENGTH_POLICY
PAD_TO_FRAMES=$PAD_TO_FRAMES
TRANSLATION_DECODE_MODE=$TRANSLATION_DECODE_MODE
JOB_RANK=$JOB_RANK
JOB_COUNT=$JOB_COUNT
LOCAL_HOST_RANK=$LOCAL_HOST_RANK
GLOBAL_HOST_RANK=$GLOBAL_HOST_RANK
MACHINE_NUM=$MACHINE_NUM
NUM_GPUS=$NUM_GPUS
TOTAL_SHARDS=$TOTAL_SHARDS
SHARD_BASE=$SHARD_BASE
EOF

echo "[start] $(date) out=$OUT ckpt=$CHECKPOINT job=$JOB_RANK/$JOB_COUNT host=$LOCAL_HOST_RANK machines=$MACHINE_NUM total_shards=$TOTAL_SHARDS shard_base=$SHARD_BASE conds=$CONDS length_policy=$LENGTH_POLICY translation=$TRANSLATION_DECODE_MODE"

for cond in $CONDS; do
  echo "[cond=$cond start] $(date)"
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD=$((SHARD_BASE + i))
    CUDA_VISIBLE_DEVICES=$i "$PY" -u scripts/eval/eval_prism_tp2m_prefix.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --anno-file "$ANNO" \
      --data-dir data/motionhub \
      --output-dir "$OUT" \
      --condition-num-frames "$cond" \
      --kafs-mode depth_driven \
      --num-inference-steps "$STEPS" \
      --guidance-scale "$GUIDANCE" \
      --length-policy "$LENGTH_POLICY" \
      --pad-to-frames "$PAD_TO_FRAMES" \
      --translation-decode-mode "$TRANSLATION_DECODE_MODE" \
      --min-frames "$((cond + 1))" \
      --max-frames "$MAXF" \
      --num-shards "$TOTAL_SHARDS" \
      --shard-idx "$SHARD" \
      --skip-existing \
      > "$OUT/logs/cond${cond}_gen_job${JOB_RANK}_h${LOCAL_HOST_RANK}_g${i}.log" 2>&1 &
  done
  wait
  n=$(find "$OUT/cond${cond}_depth_driven" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  echo "[cond=$cond done] $(date) total_npz=$n"
done

echo "[done] $(date)"
