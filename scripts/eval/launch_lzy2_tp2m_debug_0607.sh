#!/usr/bin/env bash
# Launch critical TP2M debug runs on lzy_debug_machine_2.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "${ROOT}"
mkdir -p outputs/evaluation/tp2m_lzy2_0607/logs

setsid env \
  OUT=outputs/evaluation/motionstreamer_tp2m_debug_0607/mh_c5_rewrite_sample_lzy2 \
  SPLIT=motionhub \
  COND=5 \
  NUM_GPUS=4 \
  CAPTION_PROTOCOL=rewritten \
  REWRITTEN_FILE=data/annotation/test_motionhub_t2m_rewritten.json \
  PREFIX_LATENT_SOURCE=sample \
  SAMPLING_METHOD=new_demo \
  MS_TEMPERATURE=1.0 \
  N_REPEATS=20 \
  GT272_MH=outputs/evaluation/motionstreamer_tp2m_table2_0606_direct272_full/gt272_motionhub \
  bash scripts/eval/run_motionstreamer_tp2m_single_0606.sh \
  > outputs/evaluation/tp2m_lzy2_0607/logs/motionstreamer_mh_c5_rewrite_sample.log 2>&1 < /dev/null &
echo "motionstreamer_pid=$!"

setsid env \
  OUT=outputs/evaluation/flowmdm_tp2m_debug_0607/h3d_c5_g75_lzy2 \
  GT263_ROOT=outputs/evaluation/humanml3d/gt_smpl135_to_hml263 \
  SPLITS=h3d \
  CONDS=5 \
  STAGES="infer retarget summary" \
  GPU_LIST=4,5,6,7 \
  NUM_SHARDS=4 \
  GUIDANCE=7.5 \
  BPE_STEP=60 \
  N_REPEATS=20 \
  bash scripts/eval/run_flowmdm_tp2m_table2_0606.sh \
  > outputs/evaluation/tp2m_lzy2_0607/logs/flowmdm_h3d_c5_g75.log 2>&1 < /dev/null &
echo "flowmdm_pid=$!"
