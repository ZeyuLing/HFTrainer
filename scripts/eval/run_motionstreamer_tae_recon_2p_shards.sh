#!/usr/bin/env bash
set -euo pipefail

ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
if [[ ! -d "$ROOT" ]]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$ROOT"

export PYTHONPATH="$PWD:$PWD/ref_repo/MotionStreamer/MotionStreamer:${PYTHONPATH:-}"

LOG_DIR=output/evaluation/table3_recon_baselines_0606/logs
OUT_BASE=output/evaluation/table3_recon_baselines_0606/motionstreamer_tae_recon_2p_vermoimg
mkdir -p "$LOG_DIR" "$OUT_BASE"

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$shard"
    python3 scripts/eval/reconstruct_motionstreamer_tae272.py \
      --anno-file data/annotation/vermo_recon_motionhub_2p_test_20260606.json \
      --data-dir data/motionhub \
      --out-dir "$OUT_BASE/shard_${shard}" \
      --num-person 2 \
      --max-duration 12 \
      --num-shards 8 \
      --shard-index "$shard" \
      --device cuda
  ) > "$LOG_DIR/motionstreamer_tae_vermoimg_dha_script_2p_s${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
