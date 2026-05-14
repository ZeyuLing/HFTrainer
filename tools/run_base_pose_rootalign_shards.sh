#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <machine_id> <start_case> <end_case>" >&2
  exit 2
fi

machine_id="$1"
start_case="$2"
end_case="$3"
root="output/eval_keyframe_pose_v3/local_rot/kimodo_base_pose_keypose_rootalign_nosplit_s1_20260513/shards"
mkdir -p "$root/logs"

for gpu in 0 1 2 3 4 5 6 7; do
  s=$((start_case + gpu * 10))
  e=$((s + 10))
  if [[ "$s" -ge "$end_case" ]]; then
    continue
  fi
  if [[ "$e" -gt "$end_case" ]]; then
    e="$end_case"
  fi
  out="$root/${machine_id}_gpu${gpu}_${s}_${e}"
  log="$root/logs/${machine_id}_gpu${gpu}_${s}_${e}.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python3 scripts/kimodo/run_kimodo_base_pose_edit.py \
    --start-idx "$s" \
    --end-idx "$e" \
    --context-stride 1 \
    --context-mode fullbody_pos_only \
    --num-steps 100 \
    --output-dir "$out" \
    > "$log" 2>&1 &
  echo "launched ${machine_id} gpu=${gpu} cases=${s}-${e} log=${log}"
done

wait
