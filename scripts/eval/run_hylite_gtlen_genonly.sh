#!/usr/bin/env bash
# Generation-ONLY shard worker: HY-Motion-Lite T2M on the HumanML3D test set, but
# with per-sample target length overridden to the GT humanml3d_272 length via
# --length-map-file (HY reads frames from the source motion file, so the gtlen anno
# alone is not enough). Real re-inference at the GT time-base, NOT post-hoc resample.
# Fan out across Taiji hosts: each invocation runs NGPU local GPUs as global shards
# [SHARD_BASE..SHARD_BASE+NGPU-1] of TOTAL_SHARDS, all writing the SAME motionclip135/
# dir (raw row-major 135 .npy) with --skip-existing.
#
#   TOTAL_SHARDS=32 SHARD_BASE=0 bash scripts/eval/run_hylite_gtlen_genonly.sh
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"
pick_py() {
  for c in "${PY:-}" python3 python3.10 python3.9 python3.8 python3.11 \
      /usr/local/bin/python3 /usr/bin/python3.10 /usr/bin/python3.9 /usr/bin/python3.8 \
      /opt/conda/bin/python /root/miniconda3/bin/python /opt/miniconda3/bin/python \
      /usr/local/miniconda3/bin/python "$HOME/miniconda3/bin/python"; do
    [ -n "$c" ] || continue
    command -v "$c" >/dev/null 2>&1 || [ -x "$c" ] || continue
    "$c" -c 'import sys; import numpy; import torch; sys.exit(0 if sys.version_info[:2]>=(3,8) else 1)' 2>/dev/null && { echo "$c"; return 0; }
  done
  return 1
}
PY=$(pick_py) || { echo "[error] Python >=3.8 with numpy+torch not found"; exit 2; }
echo "[python] $(command -v "$PY") $("$PY" --version 2>&1)"

NGPU=${NGPU:-8}
TOTAL_SHARDS=${TOTAL_SHARDS:-32}
SHARD_BASE=${SHARD_BASE:-0}
ANNO=${ANNO:-data/annotation/test_hml3d.json}
LENMAP=${LENMAP:-data/annotation/test_hml3d_gtlen_lenmap.json}
REWRITTEN=${REWRITTEN-data/annotation/test_hml3d_rewritten.json}
DATA_DIR=${DATA_DIR:-data/motionhub}
OUT=${OUT:-outputs/evaluation/hylite_gtlen/h3d}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_STEPS=${NUM_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-5.0}
MIN_FRAMES=${MIN_FRAMES:-24}
MAX_FRAMES=${MAX_FRAMES:-360}
caption_flag=()
[ -n "$REWRITTEN" ] && caption_flag=(--caption-file "$REWRITTEN")

out="$OUT"
pred="$out/motionclip135"
mkdir -p "$pred" "$out/_logs"
echo "[genonly] $(date) TOTAL_SHARDS=$TOTAL_SHARDS SHARD_BASE=$SHARD_BASE min/max=$MIN_FRAMES/$MAX_FRAMES rewritten=[$REWRITTEN] -> $pred"

pids=()
for g in $(seq 0 $((NGPU-1))); do
  gidx=$((SHARD_BASE + g))
  [ "$gidx" -ge "$TOTAL_SHARDS" ] && continue
  "$PY" scripts/eval/hylite_t2m_anno_infer.py \
    --anno-file "$ANNO" --length-map-file "$LENMAP" \
    "${caption_flag[@]}" \
    --data-dir "$DATA_DIR" --out-dir "$pred" \
    --min-frames "$MIN_FRAMES" --max-frames "$MAX_FRAMES" \
    --num-shards $TOTAL_SHARDS --shard-index $gidx --gpu $g \
    --batch-size "$BATCH_SIZE" --num-steps "$NUM_STEPS" --cfg-scale "$CFG_SCALE" --skip-existing \
    > "$out/_logs/hy_g${gidx}of${TOTAL_SHARDS}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
n=$("$PY" -c "import os;d='$pred';print(sum(1 for e in os.scandir(d) if e.name.endswith('.npy')))")
echo "[genonly done] $(date) base=$SHARD_BASE motionclip135 total now=$n"
