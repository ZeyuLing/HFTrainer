#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/amass_g1_proto_baseline_eval/physflow_0605h_rollout_metrics_v100g470_20260605}"
CKPT_PATH="${CKPT_PATH:-${PROJECT_ROOT}/ref_repo/ProtoMotions/results/physflow_g1_guarded_adv_anchor_w099_adv00075_jump00025_task3_disc01_fromofficial_700k_0605h_retry1/last.ckpt}"

cd "${PROJECT_ROOT}"
export PROJECT_ROOT
export OUT_ROOT
export CHECKPOINT_SPECS="physflow0605h_fast=${CKPT_PATH}"
export NUM_SHARDS="${NUM_SHARDS:-4}"
export NUM_ENVS="${NUM_ENVS:-256}"
export MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-600}"
export SAVE_PREDICTED_MOTION_LIB="${SAVE_PREDICTED_MOTION_LIB:-1}"
export PREDICTED_MOTION_LIB_EVERY="${PREDICTED_MOTION_LIB_EVERY:-1}"
export FORCE_EVAL="${FORCE_EVAL:-1}"
export RUN_NODE_SETUP="${RUN_NODE_SETUP:-0}"
export GPU_OFFSET="${GPU_OFFSET:-4}"

exec bash scripts/embodied/run_amass_g1_proto_baseline_eval.sh
