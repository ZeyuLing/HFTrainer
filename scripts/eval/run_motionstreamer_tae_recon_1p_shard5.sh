#!/usr/bin/env bash
set -euo pipefail

ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
if [[ ! -d "$ROOT" ]]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$ROOT"

export PYTHONPATH="$PWD:$PWD/ref_repo/MotionStreamer/MotionStreamer:${PYTHONPATH:-}"

LOG_DIR=output/evaluation/table3_recon_baselines_0606/logs
OUT_DIR=output/evaluation/table3_recon_baselines_0606/motionstreamer_tae_recon_1p_min40_vermoimg/shard_5
mkdir -p "$LOG_DIR" "$OUT_DIR"

python3 scripts/eval/reconstruct_motionstreamer_tae272.py \
  --anno-file data/annotation/vermo_recon_motionhub_1p_test_20260606.json \
  --data-dir data/motionhub \
  --out-dir "$OUT_DIR" \
  --id-list output/evaluation/table3_recon_baselines_0606/hml263_gt_1p_max12_min40/test.txt \
  --num-person 1 \
  --max-duration 12 \
  --num-shards 8 \
  --shard-index 5 \
  --device cuda \
  > "$LOG_DIR/motionstreamer_tae_vermoimg_dha_script_1p_s5.log" 2>&1
