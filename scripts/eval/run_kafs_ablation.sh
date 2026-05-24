#!/bin/bash
# KAFS ablation launcher — runs all 4 modes in parallel on separate GPUs
# Usage: bash scripts/eval/run_kafs_ablation.sh [max_samples]
#   max_samples: optional, default 200

set -e

WORKDIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WORKDIR"

MAX_SAMPLES=${1:-200}
CKPT="work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"
CONFIG="configs/prism/prism_1b_tp2m_multiframe.py"
ANNO="data/annotation/test_motionhub_t2m.json"
DATA_DIR="data/motionhub"
OUT_DIR="work_dirs/prism_kafs_ablation"
STEPS=50
SEED=42

mkdir -p "$OUT_DIR"

echo "=== KAFS Ablation Launcher ==="
echo "Config:      $CONFIG"
echo "Checkpoint:  $CKPT"
echo "Annotation:  $ANNO"
echo "Max samples: $MAX_SAMPLES"
echo "Output:      $OUT_DIR"
echo ""

MODES=("none" "depth_driven" "uniform" "random")
GPUS=(0 1 2 3)

for i in "${!MODES[@]}"; do
    MODE="${MODES[$i]}"
    GPU="${GPUS[$i]}"
    LOG="$OUT_DIR/log_${MODE}.txt"
    echo "[$(date)] Starting mode=$MODE on GPU $GPU -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval/eval_prism_kafs_ablation.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --kafs-mode "$MODE" \
        --anno-file "$ANNO" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUT_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --num-inference-steps "$STEPS" \
        --seed "$SEED" \
        > "$LOG" 2>&1 &
    echo "  PID=$!"
done

echo ""
echo "All 4 modes launched in background. Monitor with:"
echo "  tail -f $OUT_DIR/log_*.txt"
echo "  # or check completion:"
echo "  ls $OUT_DIR/*/manifest.json"
echo ""

wait
echo "[$(date)] All modes finished!"

# Print summaries
for MODE in "${MODES[@]}"; do
    LOG="$OUT_DIR/log_${MODE}.txt"
    echo ""
    echo "=== $MODE ==="
    tail -12 "$LOG" 2>/dev/null || echo "(log not found)"
done
