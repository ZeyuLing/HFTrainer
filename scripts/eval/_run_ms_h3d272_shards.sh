#!/usr/bin/env bash
# Launch 8 parallel MotionStreamer-272 generation shards (hftrainer-native,
# ref_repo-independent) on an 8-GPU box. Detaches via setsid so it survives the
# launching exec session.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer || exit 1
export PYTHONPATH="$PWD"
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false

OUT=outputs/evaluation/ms_h3d272/ms_272
LOGDIR=outputs/evaluation/ms_h3d272/logs
N=8
mkdir -p "$OUT" "$LOGDIR"

for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i setsid nohup python3 -u scripts/eval/ms_t2m_h3d272.py \
    --out_dir "$OUT" --device cuda --guidance_param 4.0 \
    --num_shards "$N" --shard_index "$i" --skip_existing \
    > "$LOGDIR/shard_$i.log" 2>&1 < /dev/null &
done
echo "launched $N shards -> $OUT (logs: $LOGDIR)"
sleep 3
echo "running python procs: $(pgrep -fc ms_t2m_h3d272.py)"
