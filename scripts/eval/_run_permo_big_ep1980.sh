#!/usr/bin/env bash
# PerMo Table-10 big-set (1386) generation + scoring with the CURRENT official
# checkpoint ep1980 (the existing big run used ep1530 -> inconsistent with the
# MotionFix tables). Runs locally on one GPU. ~3h.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT"
PY=/usr/bin/python3
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="outputs/evaluation/semantic_edit/permo_style_test/motion135/editfix_ep1980_cfg2.5_big"
NPZ="$OUT/smpl_caption_editfix_latest/E16_style_edit/npz"
mkdir -p "$OUT/_logs"

echo "[permo-big] GEN start $(date)"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval/eval_m2m_v2_all_tasks.py \
  --models smpl_caption_editfix_latest --tasks E16 --settings style_edit \
  --data-file-override eval_e16_semantic_style_edit_big.json \
  --max-samples 1000000 --save-npz \
  --num-steps 50 --replacement-guidance skip_last --text-guidance-scale 2.5 \
  --output-dir "$OUT" > "$OUT/_logs/gen.log" 2>&1
echo "[permo-big] GEN done $(date) npz=$(find "$NPZ" -name '*.npz' 2>/dev/null | wc -l)"

echo "[permo-big] SCORE start $(date)"
( unset PYTHONPATH; CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval/eval_permo_table10.py \
    --npz-dir "$NPZ" --out "$OUT/metrics_table10_ep1980_big.json" ) > "$OUT/_logs/score.log" 2>&1
echo "[permo-big] SCORE done $(date)"
echo "===== metrics ====="
cat "$OUT/metrics_table10_ep1980_big.json" 2>/dev/null | head -30
echo "PERMO_BIG_ALL_DONE"
