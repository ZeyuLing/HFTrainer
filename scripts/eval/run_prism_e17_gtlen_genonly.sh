#!/usr/bin/env bash
# Generation-ONLY shard worker: PRISM (latest epoch_17, depth_driven KAFS) on the
# HumanML3D test set, but with num_frames overridden to the GT humanml3d_272 length
# (data/annotation/test_hml3d_gtlen.json) so the model NATIVELY generates at the GT
# time-base (real re-inference, NOT post-hoc resample). Fan out across Taiji hosts:
# each invocation runs NGPU local GPUs as global shards [SHARD_BASE..SHARD_BASE+NGPU-1]
# of TOTAL_SHARDS, all writing the SAME depth_driven/ dir with --skip-existing.
# Repack + MS-272 eval are done by the local watcher once depth_driven/ is full.
#
#   TOTAL_SHARDS=48 SHARD_BASE=0 bash scripts/eval/run_prism_e17_gtlen_genonly.sh
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

NGPU=${NGPU:-8}
TOTAL_SHARDS=${TOTAL_SHARDS:-48}
SHARD_BASE=${SHARD_BASE:-0}
CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py
CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_17}
ANNO=data/annotation/test_hml3d_gtlen.json
REWRITTEN=data/annotation/test_hml3d_rewritten.json
STEPS=50
GUIDANCE=5.0

out="outputs/evaluation/prism_kt_spectral_epoch17_gtlen/h3d"
mkdir -p "$out/depth_driven" "$out/_logs"
echo "[genonly] $(date) TOTAL_SHARDS=$TOTAL_SHARDS SHARD_BASE=$SHARD_BASE -> $out/depth_driven"

pids=()
for g in $(seq 0 $((NGPU-1))); do
  gidx=$((SHARD_BASE + g))
  [ "$gidx" -ge "$TOTAL_SHARDS" ] && continue
  CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/eval_prism_kafs_ablation.py \
    --config "$CONFIG" --checkpoint "$CKPT" --kafs-mode depth_driven \
    --anno-file "$ANNO" --rewritten-caption-file "$REWRITTEN" \
    --data-dir data/motionhub --output-dir "$out" \
    --num-inference-steps $STEPS --guidance-scale $GUIDANCE \
    --num-shards $TOTAL_SHARDS --shard-idx $gidx --skip-existing \
    > "$out/_logs/dd_g${gidx}of${TOTAL_SHARDS}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
n=$(python3 -c "import os;d='$out/depth_driven';print(sum(1 for e in os.scandir(d) if e.name.endswith('.npz')))")
echo "[genonly done] $(date) base=$SHARD_BASE depth_driven total now=$n"
