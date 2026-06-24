#!/usr/bin/env bash
# Generate + convert ONE T2M-only epoch checkpoint into motionclip135 preds for
# the per-epoch MotionCLIP trend comparison. Reuses the cached qwen3+clip
# caption features (cap_cache_full.pt) so no text encoder is loaded -> the M2M
# model alone fits the local 15GB V100. Run once per epoch; then aggregate all
# epoch pred dirs into a single manifest and call eval_motionclip_table1_dirs.py.
#
# Usage: bash scripts/eval/eval_epoch_t2m.sh <EPOCH> [MAX_SAMPLES=500]
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"

EPOCH="${1:?usage: eval_epoch_t2m.sh <EPOCH> [MAX_SAMPLES]}"
MAXS="${2:-500}"
CKPT="work_dirs/hymotion_m2m_t2m_only_from_lite/checkpoint-epoch_${EPOCH}"
CONFIG=configs/hymotion_m2m/hymotion_m2m_t2m_only_from_lite_046b.py
CACHE=outputs/tmp/20260622_t2m_local/cap_cache_full.pt
ANNO=data/annotation/test_hml3d_official272_gtlen.json
BASE="outputs/tmp/20260623_t2m_epoch_trend/epoch_${EPOCH}"
M135="$BASE/m135"
PRED="$BASE/motionclip135_pred"
mkdir -p "$BASE"

if [ ! -e "$CKPT" ]; then echo "CKPT_MISSING $CKPT"; exit 3; fi

echo "[gen] epoch_${EPOCH} max_samples=${MAXS} ckpt=$CKPT"
python3 scripts/eval/gen_ours_m2m_272.py \
  --config "$CONFIG" --ckpt "$CKPT" \
  --out "$BASE/pred272" --m135-dir "$M135" \
  --text-cache "$CACHE" \
  --num-steps 50 --cfg-scale 5.0 --max-samples "$MAXS" --gpu 0 \
  --skip-existing || { echo "GEN_FAIL epoch_${EPOCH}"; exit 4; }

echo "[convert] epoch_${EPOCH} m135 -> motionclip135 (yaw-aligned)"
python3 scripts/eval/convert_hylite135_to_motionclip_col.py \
  --src-dir "$M135" --out-dir "$PRED" \
  --anno-file "$ANNO" \
  --align-to-gt-root --align-root-mode yaw --workers 16 \
  || { echo "CONVERT_FAIL epoch_${EPOCH}"; exit 5; }

echo "EPOCH_${EPOCH}_READY pred=$PRED count=$(ls "$PRED" 2>/dev/null | wc -l)"
