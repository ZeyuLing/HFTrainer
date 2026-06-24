#!/bin/bash
# Regenerate PRISM TP2M (prefix-pose) predictions with the condition-normalization
# fix (load_condition_pose now normalizes the raw GT prefix before VAE encode).
# Generation only (cond 1/5/9 on HumanML3D test), 8-GPU sharded. MS-272 eval and
# the m2m viewer NPZ are built separately afterwards.
set -uo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "${ROOT}" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "${ROOT}"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=${PY:-python3}

OUT=${OUT:-outputs/evaluation/prism_tp2m_epoch15_fix/h3d}
NUM_GPUS=${NUM_GPUS:-8}
CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
CHECKPOINT=${CHECKPOINT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_15}
ANNO=${ANNO:-data/annotation/test_hml3d.json}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-5.0}
MAXF=${MAXF:-360}
CONDS=${CONDS:-"1 5 9"}

# --- Multi-host sharding (Taiji sets INDEX=node rank, NODE_LIST=comma node list) ---
HOST_RANK=${INDEX:-0}
if [ -n "${NODE_LIST:-}" ]; then
  MACHINE_NUM=$(python3 -c "import os;print(len(os.environ['NODE_LIST'].split(',')))" 2>/dev/null || echo 1)
else
  MACHINE_NUM=${MACHINE_NUM:-1}
fi
TOTAL_SHARDS=$((MACHINE_NUM * NUM_GPUS))

mkdir -p "$OUT/logs"
echo "[start] out=$OUT ckpt=$CHECKPOINT host_rank=$HOST_RANK machines=$MACHINE_NUM gpus/node=$NUM_GPUS total_shards=$TOTAL_SHARDS conds=$CONDS"

for cond in $CONDS; do
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD=$((HOST_RANK * NUM_GPUS + i))
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
      --min-frames "$((cond + 1))" \
      --max-frames "$MAXF" \
      --num-shards "$TOTAL_SHARDS" \
      --shard-idx "$SHARD" \
      --skip-existing \
      > "$OUT/logs/cond${cond}_gen_h${HOST_RANK}_g$i.log" 2>&1 &
  done
  wait
  n=$(find "$OUT/cond${cond}_depth_driven" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  echo "[cond=$cond host=$HOST_RANK gen done] total_npz=$n"
done

echo "[ALL DONE host=$HOST_RANK]"
