#!/usr/bin/env bash
# Generate ONE T2M-only epoch's motion_135 preds using all 8 A100 GPUs on a
# single idle Taiji host (host 4-7), 8-way sharded. Shards write {sid}.npy into
# one shared --m135-dir (disjoint sids, no collision). Blocks until all shards
# finish (run it inside a detached tmux from the caller). Reuses the cached
# qwen3+clip caption features so no text encoder is loaded.
#
# Usage: bash scripts/eval/gen_epoch_host_parallel.sh <EPOCH> [MAX_SAMPLES=500]
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
PY="$ROOT/.venv_t2m_a100/bin/python"

EPOCH="${1:?usage: gen_epoch_host_parallel.sh <EPOCH> [MAX_SAMPLES]}"
MAXS="${2:-500}"
CKPT="work_dirs/hymotion_m2m_t2m_only_from_lite/checkpoint-epoch_${EPOCH}"
CONFIG=configs/hymotion_m2m/hymotion_m2m_t2m_only_from_lite_046b.py
CACHE=outputs/tmp/20260622_t2m_local/cap_cache_full.pt
BASE="outputs/tmp/20260623_t2m_epoch_trend/epoch_${EPOCH}"
M135="$BASE/m135"
LOGD="$BASE/genlogs"
mkdir -p "$M135" "$LOGD"

if [ ! -x "$PY" ]; then echo "VENV_PYTHON_MISSING $PY"; exit 2; fi
if [ ! -e "$CKPT" ]; then echo "CKPT_MISSING $CKPT"; exit 3; fi

echo "[gen-mn] epoch_${EPOCH} max_samples=${MAXS} on 8 GPUs -> $M135"
for s in 0 1 2 3 4 5 6 7; do
  PYTHONPATH="$ROOT" "$PY" scripts/eval/gen_ours_m2m_272.py \
    --config "$CONFIG" --ckpt "$CKPT" \
    --out "$BASE/pred272" --m135-dir "$M135" \
    --text-cache "$CACHE" \
    --num-steps 50 --cfg-scale 5.0 --max-samples "$MAXS" \
    --num-shards 8 --shard-index "$s" --gpu "$s" --skip-existing \
    > "$LOGD/shard${s}.log" 2>&1 &
done
wait
echo "ALL_SHARDS_DONE epoch_${EPOCH} m135_count=$(ls "$M135" 2>/dev/null | wc -l)"
