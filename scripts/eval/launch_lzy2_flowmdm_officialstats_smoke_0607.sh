#!/usr/bin/env bash
# Launch a short FlowMDM TP2M smoke with official HumanML3D stats on lzy_debug_machine_2.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "${ROOT}"
mkdir -p outputs/evaluation/tp2m_lzy2_0607/logs

setsid env \
  OUT=outputs/evaluation/flowmdm_tp2m_debug_0607/h3d_c5_g75_officialstats_smoke_lzy2 \
  GT263_ROOT=outputs/evaluation/humanml3d/gt_smpl135_to_hml263 \
  SPLITS=h3d \
  CONDS=5 \
  STAGES="infer retarget summary" \
  GPU_LIST=4,5,6,7 \
  NUM_SHARDS=4 \
  TOTAL_SHARDS=4 \
  GUIDANCE=7.5 \
  BPE_STEP=60 \
  MAX_SAMPLES=16 \
  N_REPEATS=3 \
  bash scripts/eval/run_flowmdm_tp2m_table2_0606.sh \
  > outputs/evaluation/tp2m_lzy2_0607/logs/flowmdm_h3d_c5_g75_officialstats_smoke.log 2>&1 < /dev/null &
echo "flowmdm_officialstats_smoke_pid=$!"
