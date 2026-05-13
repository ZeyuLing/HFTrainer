#!/bin/bash
# Distributed embedding extraction for PerMo dataset
# Usage: ./run_permo_embedding_extraction.sh <num_gpus> [num_nodes] [node_rank]

set -e

NUM_GPUS=${1:-1}
NUM_NODES=${2:-1}
NODE_RANK=${3:-0}

PERMO_ROOT="data/hymotion_data/PerMo/PerMo/20260513"
BATCH_SIZE=4
MAX_LENGTH_LLM=512
TORCH_DTYPE="bfloat16"

# Total shards = NUM_NODES * NUM_GPUS
TOTAL_SHARDS=$((NUM_NODES * NUM_GPUS))

echo "=========================================="
echo "PerMo Embedding Extraction Configuration"
echo "=========================================="
echo "GPUs per node: $NUM_GPUS"
echo "Total nodes: $NUM_NODES"
echo "This node rank: $NODE_RANK"
echo "Total shards: $TOTAL_SHARDS"
echo "Batch size: $BATCH_SIZE"
echo "Max LLM length: $MAX_LENGTH_LLM"
echo "Data root: $PERMO_ROOT"
echo ""

# Process each GPU on this node
for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD_ID=$((NODE_RANK * NUM_GPUS + gpu_idx))
    DEVICE="cuda:$gpu_idx"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting shard $SHARD_ID/$TOTAL_SHARDS on $DEVICE..."
    
    python3 scripts/data/prepare_permo_embeddings.py \
        --permo-root "$PERMO_ROOT" \
        --splits train test \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --max-length-llm "$MAX_LENGTH_LLM" \
        --torch-dtype "$TORCH_DTYPE" \
        --num-shards "$TOTAL_SHARDS" \
        --shard-id "$SHARD_ID" \
        2>&1 | sed "s/^/[GPU$gpu_idx] /" &
done

# Wait for all shards to complete
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All shards completed"

# Verify output
echo ""
echo "Output files generated:"
for split in train test; do
    count=$(find "$PERMO_ROOT/qwen3embedding_augmented/$split" -name "*.pt" 2>/dev/null | wc -l)
    echo "  $split: $count .pt files"
done
