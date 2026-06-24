#!/usr/bin/env bash
# Launch 8 parallel HY-Motion-1.0 T2M -> MS-272 generation shards on an 8-GPU box.
# Each shard loads its own HYTextModel (Qwen3-8B + CLIP-L) + HunyuanMotionMMDiT,
# generates per MS-272 test pair, converts motion_135 -> 272, and writes
# <idx:06d>.npy. Detaches via setsid so it survives the launching exec session.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer || exit 1
export PYTHONPATH="$PWD"
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false

OUT=outputs/evaluation/hymotion_h3d272/hy_272
LOGDIR=outputs/evaluation/hymotion_h3d272/logs
N=8
GUIDANCE=${GUIDANCE:-5.0}
BATCH=${BATCH:-8}
DTYPE=${DTYPE:-bf16}
LIMIT=${LIMIT:-0}
mkdir -p "$OUT" "$LOGDIR"

for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i setsid nohup python3 -u scripts/eval/hymotion_t2m_h3d272.py \
    --out_dir "$OUT" --device cuda --guidance "$GUIDANCE" --batch_size "$BATCH" \
    --dtype "$DTYPE" --num_shards "$N" --shard_index "$i" --limit "$LIMIT" --skip_existing \
    > "$LOGDIR/shard_$i.log" 2>&1 < /dev/null &
done
echo "launched $N shards -> $OUT (logs: $LOGDIR)"
sleep 3
echo "running python procs: $(pgrep -fc hymotion_t2m_h3d272.py)"
