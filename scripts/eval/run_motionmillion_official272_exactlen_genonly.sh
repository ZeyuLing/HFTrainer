#!/usr/bin/env bash
# Generation-only MotionMillion / Go-To-Zero T2M over the official HumanML3D-272
# test split. Outputs canonical-id raw 272 .npy files with exact GT length.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HFTRAINER_SKIP_AUTOREGISTER=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
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
OUT=${OUT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/gotozero_7b_train}
DTYPE=${DTYPE:-bf16}
STEPS=${STEPS:-150}
LIMIT=${LIMIT:-0}
ARTIFACT=${ARTIFACT:-checkpoints/gotozero/hftrainer_7b_train_humanml272}
FSQ=${FSQ:-}
AR=${AR:-checkpoints/motionmillion/pretrained_models/motionmillion_7B.pth}
TEXT=${TEXT:-}
ANNO=${ANNO:-outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json}

ckpt_args=()
if [ -n "$ARTIFACT" ]; then
  ckpt_args+=(--artifact "$ARTIFACT")
else
  [ -n "$FSQ" ] && ckpt_args+=(--fsq_path "$FSQ")
  ckpt_args+=(--ar_path "$AR")
fi
text_args=()
[ -n "$TEXT" ] && text_args+=(--text_model_name "$TEXT")

LOGDIR="${LOGDIR:-$(dirname "$OUT")/_logs}"
mkdir -p "$OUT" "$LOGDIR"
echo "[mm-exactlen] $(date) TOTAL_SHARDS=$TOTAL_SHARDS SHARD_BASE=$SHARD_BASE STEPS=$STEPS LIMIT=$LIMIT ANNO=$ANNO ARTIFACT=${ARTIFACT:-<raw>} -> $OUT"

pids=()
for g in $(seq 0 $((NGPU-1))); do
  gidx=$((SHARD_BASE + g))
  [ "$gidx" -ge "$TOTAL_SHARDS" ] && continue
  CUDA_VISIBLE_DEVICES=$g "$PY" -u scripts/eval/motionmillion_h3d272.py \
    --out_dir "$OUT" --device cuda --dtype "$DTYPE" "${ckpt_args[@]}" \
    "${text_args[@]}" \
    --max_sample_steps "$STEPS" \
    --pair_source annotation --anno_file "$ANNO" --anno_data_dir . --canonical_output \
    --num_shards "$TOTAL_SHARDS" --shard_index "$gidx" \
    --limit "$LIMIT" \
    --skip_existing \
    > "$LOGDIR/mm_g${gidx}of${TOTAL_SHARDS}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
n=$("$PY" -c "import os;d='$OUT';print(sum(1 for e in os.scandir(d) if e.name.endswith('.npy')))")
echo "[mm-exactlen done] $(date) base=$SHARD_BASE total now=$n fail=$fail"
exit "$fail"
