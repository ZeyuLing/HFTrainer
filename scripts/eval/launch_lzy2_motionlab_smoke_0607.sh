#!/usr/bin/env bash
# Launch a short MotionLab TP2M smoke on lzy_debug_machine_2.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "${ROOT}"
mkdir -p outputs/evaluation/tp2m_lzy2_0607/logs

setsid env \
  OUT=outputs/evaluation/motionlab_tp2m_debug_0607/h3d_c5_demo201_smoke_lzy2 \
  GT263_ROOT=outputs/evaluation/humanml3d/gt_smpl135_to_hml263 \
  SPLITS=h3d \
  CONDS=5 \
  STAGES="infer retarget summary" \
  GPU_LIST=4,5,6,7 \
  NUM_SHARDS=4 \
  BATCH_SIZE=16 \
  MAX_SAMPLES=32 \
  STAGE=demo \
  EXTRA_INFER_ARGS="--no-cfg-from-checkpoint --cfg configs/config_rfmotion.yaml --num-steps 201" \
  bash scripts/eval/run_motionlab_tp2m_table2_0606.sh \
  > outputs/evaluation/tp2m_lzy2_0607/logs/motionlab_h3d_c5_demo201_smoke.log 2>&1 < /dev/null &
echo "motionlab_demo201_smoke_pid=$!"
