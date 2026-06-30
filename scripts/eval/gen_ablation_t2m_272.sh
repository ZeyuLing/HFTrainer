#!/usr/bin/env bash
# Generate T2M-only MS272 preds for the loss-ablation arms at a given epoch,
# reusing ONE shared caption-feature cache (built once with --cache-only).
# Sharded over a fixed GPU list on a single idle Taiji host. Writes
# {sid}.npy (272) into <BASE>/<arm>/epoch_<E>/pred272 and raw motion_135 into
# .../m135, matching eval_ms272_trend.py's <trend>/<arm>/epoch_<E>/pred272 layout.
#
# Usage: bash scripts/eval/gen_ablation_t2m_272.sh <EPOCH> [arm1 arm2 ...]
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
PY="$ROOT/.venv_t2m_a100/bin/python"

EPOCH="${1:?usage: gen_ablation_t2m_272.sh <EPOCH> [arms...]}"; shift || true
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(a0_full a1_velocity_only a2_no_smoothness a3_no_aux_geom)

# GPUs to use (skip GPU2 which is busy on this host).
GPUS=(0 1 3 4 5 6 7)
NSHARD=${#GPUS[@]}
MAXS=500
CFGDIR=configs/experiments/m2m_t2m_loss_ablation
ROOTOUT=outputs/tmp/20260628_ablation_t2m_eval
CACHE="$ROOTOUT/cap_cache.pt"
mkdir -p "$ROOTOUT"

if [ ! -x "$PY" ]; then echo "VENV_PYTHON_MISSING $PY"; exit 2; fi

# ---- Stage-1: build shared caption cache once (model-agnostic). ----
if [ ! -e "$CACHE" ]; then
  echo "[cache] building $CACHE (qwen3+clip, $MAXS captions) on GPU ${GPUS[0]} ..."
  PYTHONPATH="$ROOT" "$PY" scripts/eval/gen_ours_m2m_272.py \
    --cache-only --text-cache "$CACHE" \
    --config "$CFGDIR/a1_velocity_only.py" \
    --ckpt "work_dirs/m2m_t2m_loss_ablation/a1_velocity_only/checkpoint-epoch_${EPOCH}" \
    --out "$ROOTOUT/_cache_dummy" --max-samples "$MAXS" --gpu "${GPUS[0]}" \
    > "$ROOTOUT/cache_build.log" 2>&1
  if [ ! -e "$CACHE" ]; then echo "CACHE_BUILD_FAILED (see $ROOTOUT/cache_build.log)"; exit 3; fi
  echo "[cache] done."
fi

# ---- Stage-2: per-arm sharded generation reusing the cache. ----
for arm in "${ARMS[@]}"; do
  CKPT="work_dirs/m2m_t2m_loss_ablation/$arm/checkpoint-epoch_${EPOCH}"
  CONFIG="$CFGDIR/$arm.py"
  BASE="$ROOTOUT/$arm/epoch_${EPOCH}"
  M135="$BASE/m135"; PRED="$BASE/pred272"; LOGD="$BASE/genlogs"
  mkdir -p "$M135" "$PRED" "$LOGD"
  if [ ! -e "$CKPT" ]; then echo "CKPT_MISSING $CKPT -> skip $arm"; continue; fi
  echo "[gen] $arm epoch_${EPOCH} on GPUs ${GPUS[*]} -> $PRED"
  for i in "${!GPUS[@]}"; do
    g=${GPUS[$i]}
    PYTHONPATH="$ROOT" "$PY" scripts/eval/gen_ours_m2m_272.py \
      --config "$CONFIG" --ckpt "$CKPT" \
      --out "$PRED" --m135-dir "$M135" --text-cache "$CACHE" \
      --num-steps 50 --cfg-scale 5.0 --max-samples "$MAXS" \
      --num-shards "$NSHARD" --shard-index "$i" --gpu "$g" --skip-existing \
      > "$LOGD/shard${i}_gpu${g}.log" 2>&1 &
  done
  wait
  echo "[gen] $arm DONE pred272_count=$(ls "$PRED" 2>/dev/null | wc -l)"
done
echo "ALL_ARMS_GEN_DONE epoch_${EPOCH}"
