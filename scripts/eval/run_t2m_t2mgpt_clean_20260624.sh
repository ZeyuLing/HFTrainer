#!/usr/bin/env bash
# Clean full-dataset T2M-GPT rerun for HumanML3D official test.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HFTRAINER_SKIP_AUTOREGISTER=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

BASE="outputs/evaluation/t2m/humanml3d_official_test"
ANNO="${ANNO:-$BASE/captions/gt_motionclip_selected_20260622/test_hml3d_official272_gtlen_motionclip_selected_caption.json}"
SPLIT="${SPLIT:-ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt}"
RUN_TAG="${RUN_TAG:-t2mgpt_clean_20260624}"
RUN_ROOT="$BASE/_runs/$RUN_TAG"
LOG_DIR="$RUN_ROOT/logs"

HML_DIR="$BASE/hml263/t2mgpt"
M135_DIR="$BASE/motion135/t2mgpt"
MS272_DIR="$BASE/ms272/t2mgpt"

TOTAL_SHARDS="${TOTAL_SHARDS:-8}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
LOCAL_SHARDS="${LOCAL_SHARDS:-${TJ_GPU_NUM:-8}}"
NUM_GPUS="${NUM_GPUS:-${TJ_GPU_NUM:-8}}"
WORKERS="${WORKERS:-32}"
REFINE_ITERS="${REFINE_ITERS:-80}"
REFINE_LR="${REFINE_LR:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CLEAN="${CLEAN:-1}"
PY_BIN="${PY:-python3}"

if [[ "$NUM_GPUS" -lt 1 ]]; then
  NUM_GPUS=1
fi
if [[ "$LOCAL_SHARDS" -lt 1 ]]; then
  LOCAL_SHARDS=1
fi

mkdir -p "$LOG_DIR"
if [[ "$CLEAN" == "1" ]]; then
  rm -rf "$HML_DIR" "$M135_DIR" "$MS272_DIR"
fi
mkdir -p "$HML_DIR" "$M135_DIR" "$MS272_DIR"

write_meta() {
  local rep_dir="$1" rep="$2"
  "$PY_BIN" - <<'PY' "$rep_dir" "$rep" "$ANNO" "$RUN_ROOT" "$HML_DIR" "$M135_DIR" "$MS272_DIR" "$TOTAL_SHARDS" "$REFINE_ITERS" "$REFINE_LR"
import json
import os
import sys
from pathlib import Path

rep_dir, rep, anno, run_root, hml_dir, m135_dir, ms272_dir, total_shards, refine_iters, refine_lr = sys.argv[1:]
cfg = {
    "task": "t2m",
    "dataset": "humanml3d_official_test",
    "method": "t2mgpt",
    "representation": rep,
    "caption_protocol": "motionclip_selected_official_humanml3d_caption",
    "annotation": anno,
    "hml263_dir": hml_dir,
    "motion135_dir": m135_dir,
    "ms272_dir": ms272_dir,
    "source_fps": 20,
    "target_fps": 30,
    "target_length_policy": "annotation_length_at_model_fps_then_resample_to_annotation",
    "hml263_to_smpl": {
        "rotation_init": "position_ik",
        "floor_align": True,
        "refine_iters": int(refine_iters),
        "refine_lr": float(refine_lr),
        "skip_existing": False
    },
    "total_shards": int(total_shards),
    "runner": run_root,
    "created_by": "scripts/eval/run_t2m_t2mgpt_clean_20260624.sh"
}
path = Path(rep_dir)
path.mkdir(parents=True, exist_ok=True)
(path / "run_config.json").write_text(json.dumps(cfg, indent=2))
(path / "command.txt").write_text(
    " ".join([
        f"ROOT={os.environ.get('ROOT', '')}",
        f"RUN_TAG={os.environ.get('RUN_TAG', '')}",
        f"TOTAL_SHARDS={os.environ.get('TOTAL_SHARDS', '')}",
        f"LOCAL_SHARDS={os.environ.get('LOCAL_SHARDS', '')}",
        f"NUM_GPUS={os.environ.get('NUM_GPUS', '')}",
        "bash scripts/eval/run_t2m_t2mgpt_clean_20260624.sh"
    ]).strip() + "\n"
)
PY
}

run_shards() {
  local phase="$1"
  shift
  echo "[phase-start] $phase total=$TOTAL_SHARDS offset=$SHARD_OFFSET local=$LOCAL_SHARDS $(date -Is)" | tee -a "$LOG_DIR/t2mgpt.log"
  local pids=()
  local local_idx shard gpu log
  for ((local_idx=0; local_idx<LOCAL_SHARDS; local_idx++)); do
    shard=$((SHARD_OFFSET + local_idx))
    if (( shard >= TOTAL_SHARDS )); then
      continue
    fi
    gpu=$((local_idx % NUM_GPUS))
    log="$LOG_DIR/${phase}_s$(printf '%02d' "$shard")_of_$(printf '%02d' "$TOTAL_SHARDS").log"
    (
      set +e
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$@" "$TOTAL_SHARDS" "$shard" >"$log" 2>&1
      code=$?
      echo "exit_code=$code finished_at=$(date -Is)" >"${log}.status"
      exit "$code"
    ) &
    pids+=("$!")
    echo "[launch] phase=$phase shard=$shard/$TOTAL_SHARDS gpu=$gpu pid=${pids[-1]} log=$log" | tee -a "$LOG_DIR/t2mgpt.log"
  done

  local fail=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[phase-fail] $phase $(date -Is)" | tee -a "$LOG_DIR/t2mgpt.log"
    return 1
  fi
  echo "[phase-done] $phase $(date -Is)" | tee -a "$LOG_DIR/t2mgpt.log"
}

run_t2mgpt() {
  local shards="$1" shard="$2"
  "$PY_BIN" scripts/eval/t2mgpt_t2m_h3d263.py \
    --anno_file "$ANNO" \
    --anno_data_dir "." \
    --out_dir "$HML_DIR" \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --batch_size "$BATCH_SIZE" \
    --truncate_to_gt
}

run_ik() {
  local shards="$1" shard="$2"
  "$PY_BIN" scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$HML_DIR" \
    --out-dir "$M135_DIR" \
    --ids "$SPLIT" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --source-fps 20 \
    --target-fps 30 \
    --target-length-anno "$ANNO" \
    --device cuda \
    --batch-size 1 \
    --floor-align \
    --refine-iters "$REFINE_ITERS" \
    --refine-lr "$REFINE_LR"
}

coverage() {
  local rep="$1" directory="$2" suffix="$3" key="$4" out="$5"
  "$PY_BIN" - <<'PY' "$rep" "$directory" "$suffix" "$key" "$ANNO" "$out"
import json
import sys
from pathlib import Path

import numpy as np

rep, directory, suffix, key, anno, out = sys.argv[1:]
directory = Path(directory)
data = json.loads(Path(anno).read_text())["data_list"]
files = {p.stem: p for p in directory.glob(f"*{suffix}") if not p.name.startswith("_")}
missing = sorted(set(data) - set(files))
extra = sorted(set(files) - set(data))
mismatch = []
if key:
    for sid, path in files.items():
        if sid not in data:
            continue
        if suffix == ".npz":
            with np.load(path) as z:
                length = int(z[key].shape[0])
        else:
            length = int(np.load(path, mmap_mode="r").shape[0])
        expected = int(data[sid]["num_frames"])
        if length != expected:
            mismatch.append({"sid": sid, "frames": length, "expected": expected})
summary = {
    "representation": rep,
    "count": len(files),
    "expected_count": len(data),
    "missing_count": len(missing),
    "extra_count": len(extra),
    "length_mismatch_count": len(mismatch),
    "missing_first50": missing[:50],
    "extra_first50": extra[:50],
    "length_mismatch_first50": mismatch[:50],
}
Path(out).write_text(json.dumps(summary, indent=2))
print(f"[coverage-{rep}] " + json.dumps(summary, ensure_ascii=False))
if missing or extra or mismatch:
    raise SystemExit(1)
PY
}

echo "[start] clean T2M-GPT rerun $(date -Is)" | tee "$LOG_DIR/t2mgpt.log"
echo "[paths] hml=$HML_DIR motion135=$M135_DIR ms272=$MS272_DIR anno=$ANNO" | tee -a "$LOG_DIR/t2mgpt.log"
write_meta "$HML_DIR" hml263
write_meta "$M135_DIR" motion135
write_meta "$MS272_DIR" ms272

run_shards infer_t2mgpt run_t2mgpt
coverage hml263 "$HML_DIR" .npy "" "$RUN_ROOT/hml263_coverage.json"

run_shards ik_t2mgpt run_ik
coverage motion135 "$M135_DIR" .npz motion_135 "$RUN_ROOT/motion135_coverage.json"

"$PY_BIN" scripts/data/convert_motion135_to_h3d272.py \
  --in-dir "$M135_DIR" \
  --out-dir "$MS272_DIR" \
  --rotation-space local \
  --workers "$WORKERS" \
  >"$LOG_DIR/motion135_to_ms272.log" 2>&1
coverage ms272 "$MS272_DIR" .npy "" "$RUN_ROOT/ms272_coverage.json"

"$PY_BIN" scripts/eval/audit_table1_lengths.py \
  --out-dir "$RUN_ROOT/length_audit" \
  --method "T2MGPT=$M135_DIR" \
  >"$LOG_DIR/length_audit.log" 2>&1

echo "[done] clean T2M-GPT rerun $(date -Is)" | tee -a "$LOG_DIR/t2mgpt.log"
