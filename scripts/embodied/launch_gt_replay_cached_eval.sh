#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
RUN_TAG="${RUN_TAG:-amass_sanity100k_v3_20260615_225029}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/gt_replay_tracker_train_eval/${RUN_TAG}/eval_cached_shard0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
NUM_ENVS="${NUM_ENVS:-256}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-600}"
CHECKPOINT_SPECS="${CHECKPOINT_SPECS:-pretrained_g1_bones=${PROJECT_ROOT}/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt,gt_replay_after=${PROJECT_ROOT}/ref_repo/ProtoMotions/results/physflow_g1_gt_replay_amass_sanity100k_v3/last.ckpt}"

export PROJECT_ROOT OUT_ROOT NUM_SHARDS NUM_ENVS MAX_EVAL_STEPS CHECKPOINT_SPECS
cd "${PROJECT_ROOT}"
exec bash scripts/embodied/run_cached_amass_g1_tracker_eval.sh
