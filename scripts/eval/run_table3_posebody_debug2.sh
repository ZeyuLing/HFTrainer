#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
INFO=$ROOT/ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json
METHOD_SPECS=(
  "mdm:0"
  "motionlcm:2"
  "t2mgpt:3"
  "momask:4"
  "motiongpt3:5"
  "vimogen_official:6"
)

export PYTHONPATH="$ROOT/tools/table3_pycompat:$ROOT/third_party/table3_pydeps:$ROOT:$ROOT/ref_repo/ViMoGen:${PYTHONPATH:-}"
cd "$ROOT"

run_one() {
  local method="$1"
  local gpu="$2"
  local method_root="$ROOT/output/evaluation/table3_mbench/$method"
  local src_dir="$method_root/mbench_eval_input"
  local eval_dir="$method_root/mbench_eval_input_debug2"
  local log="$method_root/mbench_posebody_debug2_runner.log"

  {
    echo "[start] method=$method gpu=$gpu $(date -Is)"
    mkdir -p "$eval_dir"
    find "$eval_dir" -maxdepth 1 -type l -name '*.npy' -delete
    for npy in "$src_dir"/*.npy; do
      target="$npy"
      if [ -L "$npy" ]; then
        target="$(readlink "$npy")"
        target="${target/#\/apdcephfs\/AILab_DHA\/apdcephfs_cq11/\/apdcephfs_cq11}"
      fi
      ln -sf "$target" "$eval_dir/$(basename "$npy")"
    done

    cd "$ROOT/ref_repo/ViMoGen"
    CUDA_VISIBLE_DEVICES="$gpu" python3 evaluate_mbench.py \
      --evaluation_path "$eval_dir" \
      --output_path "$method_root/mbench_results_pose_debug2" \
      --full_info_json "$INFO" \
      --device cuda \
      --dimension Pose_Quality \
      > "$method_root/mbench_pose_debug2.log" 2>&1

    CUDA_VISIBLE_DEVICES="$gpu" python3 evaluate_mbench.py \
      --evaluation_path "$eval_dir" \
      --output_path "$method_root/mbench_results_body_debug2" \
      --full_info_json "$INFO" \
      --device cuda \
      --dimension Body_Penetration \
      > "$method_root/mbench_body_debug2.log" 2>&1
    echo "[done] method=$method $(date -Is)"
  } > "$log" 2>&1
}

pids=()
for spec in "${METHOD_SPECS[@]}"; do
  method=${spec%%:*}
  gpu=${spec##*:}
  run_one "$method" "$gpu" &
  pids+=("$!")
  echo "[launch] $method pid=${pids[-1]}"
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
exit "$rc"
