#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
SHARDS=${SHARDS:-8}
BATCH_SIZE=${BATCH_SIZE:-2}

cd "$ROOT"
mkdir -p outputs/debug_caption_qwen3_full data/eval/m2m_v2/caption_embeddings/shards

for i in $(seq 0 $((SHARDS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$i
    export PYTHONPATH=.
    # Avoid the vermo-image NVML NVLink-topology probe that crashes on
    # large-model .to(cuda) ("undefined symbol: nvmlDeviceGetNvLinkRemoteDeviceType").
    # Native allocator (no expandable_segments) + no NVML device check + no P2P
    # probe. Verified torch.zeros(1).cuda() works under these on V100.
    export PYTORCH_CUDA_ALLOC_CONF=
    export PYTORCH_NVML_BASED_CUDA_CHECK=0
    export NCCL_P2P_DISABLE=1
    python3 scripts/caption/extract_eval_caption_embeddings.py \
      --force \
      --llm-type qwen3 \
      --batch-size "$BATCH_SIZE" \
      --num-shards "$SHARDS" \
      --shard-index "$i" \
      --out-file "data/eval/m2m_v2/caption_embeddings/shards/cache_shard_${i}.pt"
  ) > "outputs/debug_caption_qwen3_full/shard_${i}.log" 2>&1 &
done

wait

export PYTHONPATH=.
python3 scripts/debug/merge_caption_cache_shards.py \
  --base data/eval/m2m_v2/caption_embeddings/cache.pt \
  --out data/eval/m2m_v2/caption_embeddings/cache.pt \
  --input data/eval/m2m_v2/caption_embeddings/shards/cache_shard_*.pt \
  > outputs/debug_caption_qwen3_full/merge.log 2>&1
