#!/usr/bin/env bash
# Re-generate Go-to-Zero / MotionMillion full HumanML3D test set with the corrected
# 150-token (KV-cached) sampler. 8 GPUs, staggered launch to avoid the 8x-18GB
# simultaneous-load OOM. Writes <idx>.npy to mm_272_len150.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD" HFTRAINER_SKIP_AUTOREGISTER=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

OUT=outputs/evaluation/motionmillion_h3d272/mm_272_len150
LOG=outputs/evaluation/motionmillion_h3d272/logs
mkdir -p "$OUT" "$LOG"

for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g nohup python3 -u scripts/eval/motionmillion_h3d272.py \
    --out_dir "$OUT" --device cuda --dtype bf16 \
    --ar_path checkpoints/motionmillion/pretrained_models/t2m_7B_all.zip \
    --text_model_name checkpoints/flan-t5-xl \
    --max_sample_steps 150 --num_shards 8 --shard_index "$g" --skip_existing \
    > "$LOG/len150_shard_$g.log" 2>&1 &
  sleep 45
done
# Keep this (foreground) launcher alive briefly so the nohup children stabilize
# before the exec session closes; then return and let them run in background.
sleep 60
echo "LAUNCHED procs=$(ps aux | grep motionmillion_h3d272 | grep -v grep | wc -l) files=$(ls $OUT/*.npy 2>/dev/null | wc -l)"
