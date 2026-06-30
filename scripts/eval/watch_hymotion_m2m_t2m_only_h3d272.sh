#!/usr/bin/env bash
# Watch HYMotion-M2M T2M-only checkpoints and evaluate every N epochs on the
# HumanML3D official-test MotionStreamer-272 leaderboard protocol.
#
# Intended use on a spare 8-GPU H20 host while 64 GPUs train:
#   bash scripts/eval/watch_hymotion_m2m_t2m_only_h3d272.sh

set -u

cd /apdcephfs_zwfy7/share_305994131/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CONFIG=${CONFIG:-configs/hymotion_m2m/hymotion_m2m_smpl_caption_t2m_only_046b.py}
WORK_DIR=${WORK_DIR:-work_dirs/hymotion_m2m_v2_smpl_caption_t2m_only_h20x64_20260630}
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/m2m_t2m_only_h20x64_20260630}
DATA_ROOT=${DATA_ROOT:-ref_repo/MotionStreamer/MotionStreamer/humanml3d_272}
NGPU=${NGPU:-8}
EVAL_EVERY=${EVAL_EVERY:-10}
START_EPOCH=${START_EPOCH:-10}
MAX_EPOCH=${MAX_EPOCH:-10000}
SLEEP_SECONDS=${SLEEP_SECONDS:-300}
CFG=${CFG:-5.0}
STEPS=${STEPS:-50}
MIN_LEN=${MIN_LEN:-1}
MAX_LEN=${MAX_LEN:-361}
TEXT_CACHE=${TEXT_CACHE:-$OUT_ROOT/text_cache/hml3d_test_qwen3_clip.pt}
LOG_ROOT="$OUT_ROOT/logs"
SUMMARY="$OUT_ROOT/summary.tsv"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$(dirname "$TEXT_CACHE")"

echo -e "epoch\tckpt\tpred_count\tresult_json\tstatus\tdate" > "$SUMMARY.tmp"
if [ -f "$SUMMARY" ]; then
  cat "$SUMMARY" >> "$SUMMARY.tmp"
fi
awk '!seen[$0]++' "$SUMMARY.tmp" > "$SUMMARY"
rm -f "$SUMMARY.tmp"

echo "[watch] repo=$PWD"
echo "[watch] config=$CONFIG"
echo "[watch] work_dir=$WORK_DIR"
echo "[watch] out_root=$OUT_ROOT"
echo "[watch] target HYMotion-Lite MS272: R1=0.6528 R2=0.7932 R3=0.8500 FID=10.5127 MM=16.6585 Div=27.6548"

if [ ! -f "$TEXT_CACHE" ]; then
  echo "[watch] building text cache -> $TEXT_CACHE"
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/gen_ours_m2m_272.py \
    --config "$CONFIG" \
    --ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --data-root "$DATA_ROOT" \
    --out "$OUT_ROOT/_cache_probe" \
    --m135-dir "$OUT_ROOT/_cache_probe_m135" \
    --gpu 0 \
    --cache-only \
    --text-cache "$TEXT_CACHE" \
    --min-len "$MIN_LEN" --max-len "$MAX_LEN" \
    > "$LOG_ROOT/text_cache.log" 2>&1
fi

if [ -f scripts/eval/_cache_272_data.sh ]; then
  bash scripts/eval/_cache_272_data.sh > "$LOG_ROOT/cache_272_data.log" 2>&1 || true
fi

run_epoch() {
  local epoch="$1"
  local ckpt="$WORK_DIR/checkpoint-epoch_${epoch}"
  local root="$OUT_ROOT/epoch_${epoch}"
  local pred="$root/pred272"
  local m135="$root/m135"
  local logs="$root/logs"
  local result="$root/results/ms272.json"
  mkdir -p "$pred" "$m135" "$logs" "$root/results"

  if [ -s "$result" ]; then
    local n_done
    n_done=$(find "$pred" -maxdepth 1 -name '*.npy' | wc -l)
    echo -e "${epoch}\t${ckpt}\t${n_done}\t${result}\tskipped_existing\t$(date -Is)" >> "$SUMMARY"
    return 0
  fi

  echo "[watch] evaluating epoch=$epoch ckpt=$ckpt"
  local pids=()
  local g
  for g in $(seq 0 $((NGPU - 1))); do
    python3 -u scripts/eval/gen_ours_m2m_272.py \
      --config "$CONFIG" \
      --ckpt "$ckpt" \
      --data-root "$DATA_ROOT" \
      --out "$pred" \
      --m135-dir "$m135" \
      --num-steps "$STEPS" \
      --cfg-scale "$CFG" \
      --rotation-space local \
      --gpu "$g" \
      --num-shards "$NGPU" \
      --shard-index "$g" \
      --skip-existing \
      --text-cache "$TEXT_CACHE" \
      --min-len "$MIN_LEN" --max-len "$MAX_LEN" \
      > "$logs/gen_shard_${g}.log" 2>&1 &
    pids+=($!)
  done

  local fail=0
  for p in "${pids[@]}"; do
    wait "$p" || fail=1
  done

  local n
  n=$(find "$pred" -maxdepth 1 -name '*.npy' | wc -l)
  echo "[watch] epoch=$epoch generation done n=$n fail=$fail"
  if [ "$fail" -ne 0 ]; then
    echo -e "${epoch}\t${ckpt}\t${n}\t${result}\tgeneration_failed\t$(date -Is)" >> "$SUMMARY"
    return 1
  fi

  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$pred" \
    --tag "m2m_t2m_only_ep${epoch}" \
    --also-refk \
    --min-motion-len "$MIN_LEN" \
    --max-motion-length "$MAX_LEN" \
    --out-json "$result" \
    > "$logs/eval_motionstreamer_272.log" 2>&1

  local status="ok"
  if [ ! -s "$result" ]; then
    status="eval_failed"
  fi
  echo -e "${epoch}\t${ckpt}\t${n}\t${result}\t${status}\t$(date -Is)" >> "$SUMMARY"
}

while true; do
  epoch="$START_EPOCH"
  while [ "$epoch" -le "$MAX_EPOCH" ]; do
    ckpt="$WORK_DIR/checkpoint-epoch_${epoch}"
    if [ -e "$ckpt" ]; then
      run_epoch "$epoch" || true
    fi
    epoch=$((epoch + EVAL_EVERY))
  done
  sleep "$SLEEP_SECONDS"
done
