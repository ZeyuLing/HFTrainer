#!/usr/bin/env bash
# Run MBench Pose_Quality and Body_Penetration for one Table-3 method.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
METHOD=${METHOD:?Set METHOD, e.g. hymotion_lite or motionstreamer}
GPU_POSE=${GPU_POSE:-1}
GPU_BODY=${GPU_BODY:-7}
SUFFIX=${SUFFIX:-debug2}
INFO=${INFO:-$ROOT/ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json}

METHOD_ROOT="$ROOT/output/evaluation/table3_mbench/$METHOD"
SRC_DIR=${SRC_DIR:-$METHOD_ROOT/mbench_eval_input}
EVAL_DIR="$METHOD_ROOT/mbench_eval_input_${SUFFIX}"
POSE_OUT="$METHOD_ROOT/mbench_results_pose_${SUFFIX}"
BODY_OUT="$METHOD_ROOT/mbench_results_body_${SUFFIX}"
LOG="$METHOD_ROOT/mbench_posebody_${SUFFIX}_runner.log"

export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$ROOT/tools/table3_pycompat:$ROOT/third_party/table3_pydeps:$ROOT:$ROOT/ref_repo/ViMoGen:${PYTHONPATH:-}"

if [ ! -d "$SRC_DIR" ]; then
  echo "Missing SRC_DIR=$SRC_DIR" >&2
  exit 2
fi

{
  echo "[start] method=$METHOD src=$SRC_DIR gpu_pose=$GPU_POSE gpu_body=$GPU_BODY $(date -Is)"
  mkdir -p "$EVAL_DIR"
  find "$EVAL_DIR" -maxdepth 1 -type l -name '*.npy' -delete
  find "$EVAL_DIR" -maxdepth 1 -type f -name '*.npy' -delete
  for npy in "$SRC_DIR"/*.npy; do
    target="$npy"
    if [ -L "$npy" ]; then
      target="$(readlink "$npy")"
      target="${target/#\/apdcephfs\/AILab_DHA\/apdcephfs_cq11/\/apdcephfs_cq11}"
    fi
    ln -sf "$target" "$EVAL_DIR/$(basename "$npy")"
  done
  echo "[prepared] eval_input=$(find "$EVAL_DIR" -maxdepth 1 -name '*.npy' | wc -l)"

  (
    cd "$ROOT/ref_repo/ViMoGen"
    CUDA_VISIBLE_DEVICES="$GPU_POSE" python3 evaluate_mbench.py \
      --evaluation_path "$EVAL_DIR" \
      --output_path "$POSE_OUT" \
      --full_info_json "$INFO" \
      --device cuda \
      --dimension Pose_Quality
  ) > "$METHOD_ROOT/mbench_pose_${SUFFIX}.log" 2>&1 &
  pose_pid=$!

  (
    cd "$ROOT/ref_repo/ViMoGen"
    CUDA_VISIBLE_DEVICES="$GPU_BODY" python3 evaluate_mbench.py \
      --evaluation_path "$EVAL_DIR" \
      --output_path "$BODY_OUT" \
      --full_info_json "$INFO" \
      --device cuda \
      --dimension Body_Penetration
  ) > "$METHOD_ROOT/mbench_body_${SUFFIX}.log" 2>&1 &
  body_pid=$!

  rc=0
  wait "$pose_pid" || rc=1
  wait "$body_pid" || rc=1
  echo "[done] method=$METHOD rc=$rc $(date -Is)"
  exit "$rc"
} > "$LOG" 2>&1
