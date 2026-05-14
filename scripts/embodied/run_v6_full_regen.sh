#!/bin/bash
# Full V6 regeneration pipeline for V4 output
# Run on Taiji debug machine
#
# Step 1: Parallel retarget all 115 NPZ files (12 workers, ~10 hours)
# Step 2: Rebuild caches + JSONs from .motion files (fast, ~5 min)
# Step 3: Verify correctness (root Z check)

set -e

HF_ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
V4_DIR="$HF_ROOT/output/embodied_t2m_v4"
NPZ_DIR="$V4_DIR/data/npz"
RETARGET_DIR="$V4_DIR/data/retarget"

cd "$HF_ROOT"

echo "============================================================"
echo "  V6 Full Regeneration Pipeline"
echo "  V4 dir:      $V4_DIR"
echo "  NPZ dir:     $NPZ_DIR"
echo "  Retarget dir: $RETARGET_DIR"
echo "============================================================"

# Backup old caches
if [ -d "$V4_DIR/data/caches" ]; then
    BACKUP="$V4_DIR/data/caches_old_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up old caches to $BACKUP"
    mv "$V4_DIR/data/caches" "$BACKUP"
    mkdir -p "$V4_DIR/data/caches"
fi

# Backup old motions (JSONs)
if [ -d "$V4_DIR/data/motions" ]; then
    BACKUP="$V4_DIR/data/motions_old_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up old motions JSONs to $BACKUP"
    mv "$V4_DIR/data/motions" "$BACKUP"
    mkdir -p "$V4_DIR/data/motions"
fi

# Step 1: Parallel retarget
echo ""
echo "============================================================"
echo "  Step 1: Parallel PyRoki Retarget (12 workers)"
echo "============================================================"
python3 scripts/embodied/batch_retarget_parallel.py \
    --npz-dir "$NPZ_DIR" \
    --output-dir "$RETARGET_DIR" \
    --workers 12 \
    --skip-existing \
    --keep-intermediates \
    2>&1 | tee /tmp/v6_retarget_full.log

# Step 2: Rebuild caches + JSONs
echo ""
echo "============================================================"
echo "  Step 2: Rebuild caches + JSONs from .motion files"
echo "============================================================"
python3 scripts/embodied/rebuild_v4_from_motion.py \
    --retarget-dir "$RETARGET_DIR" \
    --v4-dir "$V4_DIR" \
    2>&1 | tee /tmp/v6_rebuild.log

# Step 3: Verify
echo ""
echo "============================================================"
echo "  Step 3: Verification"
echo "============================================================"
echo "Checking root Z values in new caches..."
python3 -c "
import torch, os, glob, numpy as np
cache_dir = '$V4_DIR/data/caches'
files = sorted(glob.glob(os.path.join(cache_dir, '*.pt')))
underground = 0
total = 0
for f in files:
    d = torch.load(f, map_location='cpu', weights_only=False)
    if 'body_pos' in d:
        root_z = d['body_pos'][:, 0, 2]
    elif 'rigid_body_pos' in d:
        root_z = d['rigid_body_pos'][:, 0, 2].numpy()
    else:
        continue
    total += 1
    mean_z = float(np.mean(root_z))
    min_z = float(np.min(root_z))
    if min_z < 0.3:
        underground += 1
        print(f'  LOW Z: {os.path.basename(f)}: mean={mean_z:.3f}, min={min_z:.3f}')
print(f'\nTotal: {total} caches')
print(f'Underground (min Z < 0.3): {underground}/{total} ({100*underground/max(total,1):.1f}%)')
print(f'Above ground: {total-underground}/{total} ({100*(total-underground)/max(total,1):.1f}%)')
"

echo ""
echo "============================================================"
echo "  DONE!"
echo "============================================================"
