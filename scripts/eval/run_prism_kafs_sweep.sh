#!/usr/bin/env bash
# KAFS alpha sweep (single config per Taiji host) on a FIXED id subset.
# 8 local GPUs each generate samples[g::8][:NUM_SAMPLES] of the rewritten test
# set -> union = the first 8*NUM_SAMPLES ids, identical across configs (sample
# order is deterministic), so configs are compared on the SAME ids.
# Repack + MS-272 eval are done locally afterwards.
#
#   CKPT=...epoch_16 OUT_SUBDIR=invstrong KAFS_ALPHA="1.0,1.0,..." \
#   NUM_SAMPLES=64 bash scripts/eval/run_prism_kafs_sweep.sh
#   # or a built-in mode (no alpha): KAFS_MODE=depth_driven OUT_SUBDIR=dd ...
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

NGPU=${NGPU:-8}
NUM_SAMPLES=${NUM_SAMPLES:-64}   # per shard -> 8*NUM_SAMPLES total ids
CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_16}
OUT=${OUT:-outputs/evaluation/prism_kafs_sweep_e16/h3d}
OUT_SUBDIR=${OUT_SUBDIR:?set OUT_SUBDIR}
KAFS_MODE=${KAFS_MODE:-none}
KAFS_ALPHA=${KAFS_ALPHA:-}
ANNO=data/annotation/test_hml3d.json
REWRITTEN=data/annotation/test_hml3d_rewritten.json
STEPS=50
GUIDANCE=5.0

alpha_flag=""
# --kafs-mode only accepts none/depth_driven/uniform/random; when a custom alpha
# vector is given we pass mode=none (valid) and let --kafs-alpha override it.
if [ -n "$KAFS_ALPHA" ]; then alpha_flag="--kafs-alpha $KAFS_ALPHA"; KAFS_MODE=none; fi
mkdir -p "$OUT/$OUT_SUBDIR" "$OUT/_logs"
echo "[sweep] $(date) CKPT=$CKPT subdir=$OUT_SUBDIR mode=$KAFS_MODE alpha=[$KAFS_ALPHA] nsamp=$NUM_SAMPLES"

pids=()
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/eval_prism_kafs_ablation.py \
    --config "$CONFIG" --checkpoint "$CKPT" --kafs-mode "$KAFS_MODE" $alpha_flag \
    --out-subdir "$OUT_SUBDIR" \
    --anno-file "$ANNO" --rewritten-caption-file "$REWRITTEN" \
    --data-dir data/motionhub --output-dir "$OUT" \
    --num-inference-steps $STEPS --guidance-scale $GUIDANCE \
    --num-shards $NGPU --shard-idx $g --num-samples $NUM_SAMPLES --skip-existing \
    > "$OUT/_logs/${OUT_SUBDIR}_g${g}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
n=$(python3 -c "import os;d='$OUT/$OUT_SUBDIR';print(sum(1 for e in os.scandir(d) if e.name.endswith('.npz')))")
echo "[sweep done] $(date) $OUT_SUBDIR total now=$n"
