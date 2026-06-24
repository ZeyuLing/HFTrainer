#!/usr/bin/env bash
# Launch full MotionLab TP2M Table 2 rerun with the released demo/201-step sampler.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "${ROOT}"
mkdir -p outputs/evaluation/tp2m_lzy2_0607/logs

setsid env \
  OUT=outputs/evaluation/motionlab_tp2m_table2_0607_demo201_lzy2 \
  GT263_ROOT=outputs/evaluation/humanml3d/gt_smpl135_to_hml263 \
  SPLITS="h3d mh" \
  CONDS="1 5 9" \
  STAGES="infer retarget eval summary" \
  GPU_LIST=4,5,6,7 \
  NUM_SHARDS=4 \
  BATCH_SIZE=32 \
  STAGE=demo \
  EVAL_GPU_H3D=4 \
  EVAL_GPU_MH=5 \
  N_REPEATS=20 \
  EXTRA_INFER_ARGS="--no-cfg-from-checkpoint --cfg configs/config_rfmotion.yaml --num-steps 201" \
  bash scripts/eval/run_motionlab_tp2m_table2_0606.sh \
  > outputs/evaluation/tp2m_lzy2_0607/logs/motionlab_full_demo201_0607.log 2>&1 < /dev/null &
echo "motionlab_full_demo201_pid=$!"
