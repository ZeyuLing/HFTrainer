#!/usr/bin/env bash
# Multi-GPU PRISM generation of FlowMDM-precomputed (rots+transl) for the 64
# babel_val_set.json compositions (~2040 frames each). Run on Taiji (big GPU).
# Output: {i:02d}.npy under the PRISM precomputed folder (kwargs.json pre-staged).
set -uo pipefail
ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
[ -d "$ROOT" ] || ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PC="${PC:-$ROOT/ref_repo/FlowMDM/results/babel/PRISM_e19/evaluation_precomputed/Motion_PRISM_e19_001300000_gscale1.5_debug_s10/00}"
NUM_GPUS=${NUM_GPUS:-8}
UPFIX=${UPFIX:-y2z}
EXTRA_GEN_ARGS=${EXTRA_GEN_ARGS:-}

# Multi-host aware global sharding.
# NOTE: on Taiji, NODE_NUM = TOTAL number of GPUs across all hosts (e.g. 8x8=64),
#       and INDEX = host rank (0..num_hosts-1). So total_shards = NODE_NUM and
#       a worker's global shard = INDEX*NUM_GPUS + local_gpu.
NTOTAL=${NODE_NUM:-$NUM_GPUS}
INDEX=${INDEX:-0}
TOTAL_SHARDS=$NTOTAL
[ "$TOTAL_SHARDS" -lt "$NUM_GPUS" ] && TOTAL_SHARDS=$NUM_GPUS

mkdir -p "$ROOT/work_dirs/prism_babel_precomp_logs"
echo "[start] PC=$PC gpus=$NUM_GPUS node=$INDEX NODE_NUM=$NTOTAL total_shards=$TOTAL_SHARDS upfix=$UPFIX"
for g in $(seq 0 $((NUM_GPUS-1))); do
  GSHARD=$((INDEX * NUM_GPUS + g))
  [ "$GSHARD" -ge "$TOTAL_SHARDS" ] && continue
  CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/gen_prism_babel_precomp.py \
    --precomp-dir "$PC" --up-fix "$UPFIX" --skip-existing $EXTRA_GEN_ARGS \
    --num-shards "$TOTAL_SHARDS" --shard-idx "$GSHARD" \
    > "$ROOT/work_dirs/prism_babel_precomp_logs/n${INDEX}_g$g.log" 2>&1 &
done
wait
echo "PRISM_BABEL_PRECOMP_NODE${INDEX}_LOCAL_DONE  npy=$(ls "$PC"/*.npy 2>/dev/null | wc -l)/64"

# Chief barrier: keep the job alive until ALL 64 npy exist, so Taiji does not
# kill still-running worker hosts when the chief's start_cmd exits early.
if [ "${INDEX:-0}" = "0" ]; then
  for t in $(seq 1 720); do
    n=$(ls "$PC"/*.npy 2>/dev/null | wc -l)
    echo "[chief-barrier] npy=$n/64 (t=$t)"
    [ "$n" -ge 64 ] && break
    sleep 5
  done
fi
echo "PRISM_BABEL_PRECOMP_NODE${INDEX}_DONE  npy=$(ls "$PC"/*.npy 2>/dev/null | wc -l)/64"
