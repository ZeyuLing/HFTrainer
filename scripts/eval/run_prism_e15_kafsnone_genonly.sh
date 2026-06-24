#!/usr/bin/env bash
# Generation-ONLY shard worker for the e15 No-KAFS HumanML3D run, used to fan the
# job out across multiple Taiji hosts for higher parallelism. Each invocation runs
# 8 local GPUs as global shards [SHARD_BASE .. SHARD_BASE+7] out of TOTAL_SHARDS,
# all writing the SAME none/ dir with --skip-existing (so overlapping/已完成的 clip
# 不会重复生成). Repack + MS-272 eval are intentionally NOT done here — the local
# watcher (/tmp/nokafs_watch.sh) repacks into the viz prep path and evaluates once
# none/ is full.
#
#   TOTAL_SHARDS=48 SHARD_BASE=0  bash scripts/eval/run_prism_e15_kafsnone_genonly.sh
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

NGPU=${NGPU:-8}
TOTAL_SHARDS=${TOTAL_SHARDS:-48}
SHARD_BASE=${SHARD_BASE:-0}
CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py
CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_15}
ANNO=data/annotation/test_hml3d.json
REWRITTEN=data/annotation/test_hml3d_rewritten.json
STEPS=50
GUIDANCE=5.0

out="outputs/evaluation/prism_kt_spectral_epoch15_rw/h3d"
mkdir -p "$out/none" "$out/_logs"
echo "[genonly] $(date) TOTAL_SHARDS=$TOTAL_SHARDS SHARD_BASE=$SHARD_BASE -> $out/none"

pids=()
for g in $(seq 0 $((NGPU-1))); do
  gidx=$((SHARD_BASE + g))
  [ "$gidx" -ge "$TOTAL_SHARDS" ] && continue
  CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/eval_prism_kafs_ablation.py \
    --config "$CONFIG" --checkpoint "$CKPT" --kafs-mode none \
    --anno-file "$ANNO" --rewritten-caption-file "$REWRITTEN" \
    --data-dir data/motionhub --output-dir "$out" \
    --num-inference-steps $STEPS --guidance-scale $GUIDANCE \
    --num-shards $TOTAL_SHARDS --shard-idx $gidx --skip-existing \
    > "$out/_logs/none_g${gidx}of${TOTAL_SHARDS}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
n=$(python3 -c "import os;d='$out/none';print(sum(1 for e in os.scandir(d) if e.name.endswith('.npz')))")
echo "[genonly done] $(date) base=$SHARD_BASE none total now=$n"
