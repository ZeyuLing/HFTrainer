#!/usr/bin/env bash
# Multi-GPU MotionStreamer generation of FlowMDM-precomputed (rots+transl) for the
# 64 babel_val_set.json compositions. Run on Taiji (T5-XXL + TAE + AR).
set -uo pipefail
ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
[ -d "$ROOT" ] || ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# NOTE: expandable_segments + old driver NVML triggers
#   "nvmlDeviceGetNvLinkRemoteDeviceType undefined symbol" during model .to(cuda).
# MS loads T5-XXL which reliably hits it, so keep the default CUDA allocator here.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"

PC="${PC:-$ROOT/ref_repo/FlowMDM/results/babel/MotionStreamer/evaluation_precomputed/Motion_MotionStreamer_001300000_gscale1.5_debug_s10/00}"
NUM_GPUS=${NUM_GPUS:-8}
UPFIX=${UPFIX:-y2z}
EXTRA_GEN_ARGS=${EXTRA_GEN_ARGS:-}

NTOTAL=${NODE_NUM:-$NUM_GPUS}
INDEX=${INDEX:-0}
TOTAL_SHARDS=$NTOTAL
[ "$TOTAL_SHARDS" -lt "$NUM_GPUS" ] && TOTAL_SHARDS=$NUM_GPUS

mkdir -p "$ROOT/work_dirs/ms_babel_precomp_logs"
# Use the persistent local T5-XXL on CephFS to avoid HF download/rate-limit/race.
T5_LOCAL="$ROOT/checkpoints/sentencet5-xxl"
echo "[start] PC=$PC gpus=$NUM_GPUS node=$INDEX NODE_NUM=$NTOTAL total_shards=$TOTAL_SHARDS upfix=$UPFIX t5=$T5_LOCAL"
for g in $(seq 0 $((NUM_GPUS-1))); do
  GSHARD=$((INDEX * NUM_GPUS + g))
  [ "$GSHARD" -ge "$TOTAL_SHARDS" ] && continue
  CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/gen_motionstreamer_babel_precomp.py \
    --precomp-dir "$PC" --up-fix "$UPFIX" --skip-existing --t5-model "$T5_LOCAL" $EXTRA_GEN_ARGS \
    --num-shards "$TOTAL_SHARDS" --shard-idx "$GSHARD" \
    > "$ROOT/work_dirs/ms_babel_precomp_logs/n${INDEX}_g$g.log" 2>&1 &
done
wait
echo "MS_BABEL_PRECOMP_NODE${INDEX}_LOCAL_DONE  npy=$(ls "$PC"/*.npy 2>/dev/null | wc -l)/64"

if [ "${INDEX:-0}" = "0" ]; then
  for t in $(seq 1 720); do
    n=$(ls "$PC"/*.npy 2>/dev/null | wc -l)
    echo "[chief-barrier] npy=$n/64 (t=$t)"
    [ "$n" -ge 64 ] && break
    # early abort: if no output after ~5 min, workers likely all failed
    if [ "$t" -ge 60 ] && [ "$n" -eq 0 ]; then
      echo "[chief-barrier] ABORT: no npy after $((t*5))s, workers likely failed"; break
    fi
    sleep 5
  done
fi
echo "MS_BABEL_PRECOMP_NODE${INDEX}_DONE  npy=$(ls "$PC"/*.npy 2>/dev/null | wc -l)/64"
