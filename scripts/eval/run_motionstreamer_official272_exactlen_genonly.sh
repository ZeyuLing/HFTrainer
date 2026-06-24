#!/usr/bin/env bash
# Generation-only MotionStreamer T2M over the official HumanML3D-272 test split.
# Outputs canonical-id NPZ files with motion_272/motion_135 and exact GT length.
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
OUT=${OUT:-outputs/evaluation/motionstreamer_official272_exactlen_0617/prep}
GENERATION_MODE=${GENERATION_MODE:-paper_eval}

mkdir -p "$OUT" "$(dirname "$OUT")/_logs"
LOGDIR="$(dirname "$OUT")/_logs"
echo "[ms-exactlen] $(date) TOTAL_SHARDS=$TOTAL_SHARDS SHARD_BASE=$SHARD_BASE -> $OUT"

pids=()
for g in $(seq 0 $((NGPU-1))); do
  gidx=$((SHARD_BASE + g))
  [ "$gidx" -ge "$TOTAL_SHARDS" ] && continue
  CUDA_VISIBLE_DEVICES=$g "$PY" scripts/eval/gen_motionstreamer_smpl_npz.py \
    --dataset humanml3d \
    --out-dir "$OUT" \
    --num-shards "$TOTAL_SHARDS" \
    --shard-index "$gidx" \
    --humanml3d-protocol all \
    --caption-protocol original \
    --generation-mode "$GENERATION_MODE" \
    --skip-existing \
    > "$LOGDIR/ms_g${gidx}of${TOTAL_SHARDS}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
n=$("$PY" -c "import os;d='$OUT';print(sum(1 for e in os.scandir(d) if e.name.endswith('.npz')))")
echo "[ms-exactlen done] $(date) base=$SHARD_BASE total now=$n"
