#!/usr/bin/env bash
# Sharded MBench Pose_Quality + Body_Penetration for one Table-3 method.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
METHOD=${METHOD:?Set METHOD, e.g. motionstreamer_fixed}
NUM_SHARDS=${NUM_SHARDS:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
SUFFIX=${SUFFIX:-sharded0605}
INFO=${INFO:-$ROOT/ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json}

METHOD_ROOT="$ROOT/output/evaluation/table3_mbench/$METHOD"
SRC_DIR=${SRC_DIR:-$METHOD_ROOT/mbench_eval_input}
LOG="$METHOD_ROOT/mbench_posebody_${SUFFIX}_runner.log"

export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$ROOT/tools/table3_pycompat:$ROOT/third_party/table3_pydeps:$ROOT:$ROOT/ref_repo/ViMoGen:${PYTHONPATH:-}"

if [ ! -d "$SRC_DIR" ]; then
  echo "Missing SRC_DIR=$SRC_DIR" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -lt "$NUM_SHARDS" ]; then
  echo "GPUS=$GPUS has fewer entries than NUM_SHARDS=$NUM_SHARDS" >&2
  exit 2
fi

mkdir -p "$METHOD_ROOT"
{
  echo "[start] method=$METHOD src=$SRC_DIR shards=$NUM_SHARDS gpus=$GPUS suffix=$SUFFIX $(date -Is)"
  find "$METHOD_ROOT" -maxdepth 1 -type d -name "mbench_eval_input_${SUFFIX}_s*" -exec rm -rf {} +
  find "$METHOD_ROOT" -maxdepth 1 -type d -name "mbench_results_posebody_${SUFFIX}_s*" -exec rm -rf {} +
  rm -rf "$METHOD_ROOT/mbench_results_posebody_${SUFFIX}"

  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPU_LIST[$shard]}"
    eval_dir="$METHOD_ROOT/mbench_eval_input_${SUFFIX}_s${shard}"
    subset_info="$METHOD_ROOT/mbench_eval_info_${SUFFIX}_s${shard}.json"
    shard_out="$METHOD_ROOT/mbench_results_posebody_${SUFFIX}_s${shard}"
    shard_log="$METHOD_ROOT/mbench_posebody_${SUFFIX}_s${shard}.log"

    python3 - "$INFO" "$SRC_DIR" "$eval_dir" "$subset_info" "$shard" "$NUM_SHARDS" <<'PY'
import json
import os
import sys
from pathlib import Path

info_path, src_dir, eval_dir, subset_info, shard, num_shards = sys.argv[1:]
src_dir = Path(src_dir)
eval_dir = Path(eval_dir)
shard = int(shard)
num_shards = int(num_shards)
dims = {"Pose_Quality", "Body_Penetration"}

rows = json.load(open(info_path, "r", encoding="utf-8"))
ids = sorted({int(row["id"]) for row in rows if row.get("dimension") in dims})
selected = ids[shard::num_shards]
selected_set = set(selected)
subset = [row for row in rows if row.get("dimension") in dims and int(row["id"]) in selected_set]

eval_dir.mkdir(parents=True, exist_ok=True)
for old in eval_dir.glob("*"):
    old.unlink()
for motion_id in selected:
    src = src_dir / f"{motion_id}.npy"
    if not src.exists() and not src.is_symlink():
        raise FileNotFoundError(src)
    target = os.path.realpath(src)
    target = target.replace("/apdcephfs/AILab_DHA/apdcephfs_cq11", "/apdcephfs_cq11")
    os.symlink(target, eval_dir / f"{motion_id}.npy")

Path(subset_info).write_text(json.dumps(subset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"shard": shard, "num_ids": len(selected), "min_id": selected[0] if selected else None, "max_id": selected[-1] if selected else None}))
PY

    (
      cd "$ROOT/ref_repo/ViMoGen"
      CUDA_VISIBLE_DEVICES="$gpu" python3 evaluate_mbench.py \
        --evaluation_path "$eval_dir" \
        --output_path "$shard_out" \
        --full_info_json "$subset_info" \
        --device cuda \
        --dimension Pose_Quality Body_Penetration
    ) > "$shard_log" 2>&1 &
    pids+=("$!")
    echo "[launch] shard=$shard gpu=$gpu pid=${pids[-1]} log=$shard_log"
  done

  rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  if [ "$rc" -ne 0 ]; then
    echo "[fail] one or more shards failed $(date -Is)"
    exit "$rc"
  fi

  python3 "$ROOT/tools/aggregate_mbench_eval_results.py" \
    --inputs "$METHOD_ROOT"/mbench_results_posebody_${SUFFIX}_s*/mbench_results_*_eval_results.json \
    --output-dir "$METHOD_ROOT/mbench_results_posebody_${SUFFIX}" \
    --name mbench_results
  echo "[done] method=$METHOD suffix=$SUFFIX $(date -Is)"
} > "$LOG" 2>&1
