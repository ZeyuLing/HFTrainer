#!/usr/bin/env bash
# Convert framework-native MotionGPT HumanML3D-263 outputs to canonical SMPL
# motion135 and MotionStreamer-272 representations.
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

BASE="${BASE:-outputs/evaluation/t2m/humanml3d_official_test}"
ANNO="${ANNO:-$BASE/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json}"
RUN_TAG="${RUN_TAG:-motiongpt_framework_native_20260626}"
RUN_ROOT="${RUN_ROOT:-$BASE/_runs/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
METRIC_DIR="${METRIC_DIR:-$RUN_ROOT/metrics}"
SPLIT="${SPLIT:-$RUN_ROOT/test_ids.txt}"

HML_DIR="${HML_DIR:-$BASE/hml263/motiongpt}"
M135_DIR="${M135_DIR:-$BASE/motion135/motiongpt}"
MS272_DIR="${MS272_DIR:-$BASE/ms272/motiongpt}"

TOTAL_SHARDS="${TOTAL_SHARDS:-64}"
LOCAL_SHARDS="${LOCAL_SHARDS:-${TJ_GPU_NUM:-$(python3 - <<'PY'
import torch
print(max(1, torch.cuda.device_count()))
PY
)}}"
NUM_GPUS="${NUM_GPUS:-${TJ_GPU_NUM:-$(python3 - <<'PY'
import torch
print(max(1, torch.cuda.device_count()))
PY
)}}"
DEVICE_IDS="${DEVICE_IDS:-}"
WORKERS="${WORKERS:-32}"
REFINE_ITERS="${REFINE_ITERS:-80}"
REFINE_LR="${REFINE_LR:-0.02}"
CLEAN_MOTION="${CLEAN_MOTION:-0}"

mkdir -p "$LOG_DIR" "$METRIC_DIR" "$RUN_ROOT"
if [[ "$CLEAN_MOTION" == "1" ]]; then
  rm -rf "$M135_DIR" "$MS272_DIR"
fi
mkdir -p "$M135_DIR" "$MS272_DIR"

pick_python() {
  local candidates=()
  if [[ -n "${PY:-}" ]]; then
    candidates+=("$PY")
  fi
  candidates+=(python3 python /opt/conda/bin/python /root/miniconda3/bin/python /opt/miniconda3/bin/python /usr/local/miniconda3/bin/python)
  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    if ! command -v "$candidate" >/dev/null 2>&1 && [[ ! -x "$candidate" ]]; then
      continue
    fi
    if "$candidate" - <<'PY' >/dev/null 2>&1
import scipy  # noqa: F401
import torch  # noqa: F401
PY
    then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

PY_BIN="$(pick_python)" || {
  echo "[error] could not find a Python with torch/scipy" >&2
  exit 2
}
echo "[python] $PY_BIN $("$PY_BIN" --version 2>&1)"

if [[ -z "$DEVICE_IDS" ]]; then
  DEVICE_IDS="$("$PY_BIN" - <<'PY'
import torch
n = torch.cuda.device_count()
print(",".join(str(i) for i in range(n)))
PY
)"
fi
IFS=',' read -r -a GPUS <<< "$DEVICE_IDS"
if [[ "${#GPUS[@]}" -eq 0 || -z "${GPUS[0]}" ]]; then
  GPUS=("0")
fi
if [[ "$NUM_GPUS" -lt 1 ]]; then
  NUM_GPUS=1
fi
if [[ "$LOCAL_SHARDS" -lt 1 ]]; then
  LOCAL_SHARDS=1
fi

ensure_python_deps() {
  local stamp="${DEPS_STAMP:-/tmp/hftrainer_motiongpt_convert_deps_v1.stamp}"
  if [[ -f "$stamp" ]]; then
    return 0
  fi
  local missing
  missing="$("$PY_BIN" - <<'PY'
import importlib.util
checks = [
    ("chumpy", "chumpy>=0.70"),
    ("smplx", "smplx>=0.1.28"),
    ("torchgeometry", "torchgeometry>=0.1.2"),
]
for module, package in checks:
    if importlib.util.find_spec(module) is None:
        print(package)
PY
)"
  if [[ -n "$missing" ]]; then
    echo "[deps] installing: $(tr '\n' ' ' <<<"$missing")"
    "$PY_BIN" -m pip install -q \
      -i https://mirrors.tencent.com/pypi/simple \
      --trusted-host mirrors.tencent.com \
      --no-build-isolation \
      $missing
  else
    echo "[deps] all required optional packages importable"
  fi
  touch "$stamp"
}

prepare_split() {
  "$PY_BIN" - <<'PY' "$ANNO" "$SPLIT"
import json
import sys
from pathlib import Path

anno, split = map(Path, sys.argv[1:])
data = json.loads(anno.read_text())["data_list"]
split.parent.mkdir(parents=True, exist_ok=True)
split.write_text("".join(f"{sid}\n" for sid in sorted(data)))
print(f"[split] wrote {len(data)} ids -> {split}")
PY
}

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
    "method": "motiongpt",
    "representation": rep,
    "model_bundle": "hftrainer.models.motion.motiongpt.MotionGPTBundle",
    "pipeline": "hftrainer.pipelines.motiongpt.MotionGPTPipeline",
    "caption_protocol": "humanml3d_official_corrected_caption",
    "annotation": anno,
    "hml263_dir": hml_dir,
    "motion135_dir": m135_dir,
    "ms272_dir": ms272_dir,
    "source_fps": 20,
    "target_fps": 30,
    "target_length_policy": "official_annotation_num_frames",
    "hml263_to_smpl": {
        "rotation_init": "position_ik",
        "floor_align": True,
        "refine_iters": int(refine_iters),
        "refine_lr": float(refine_lr),
        "skip_existing": False,
    },
    "total_shards": int(total_shards),
    "runner": run_root,
    "created_by": "scripts/eval/run_t2m_motiongpt_convert_20260626.sh",
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
        "bash scripts/eval/run_t2m_motiongpt_convert_20260626.sh",
    ]).strip() + "\n"
)
PY
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

run_all_shards() {
  local phase="$1"
  shift
  echo "[phase-start] $phase total=$TOTAL_SHARDS local=$LOCAL_SHARDS gpus=${DEVICE_IDS:-${GPUS[*]}} $(date -Is)" | tee -a "$LOG_DIR/motiongpt_convert.log"
  local start shard local_idx gpu log pids fail pid
  for ((start=0; start<TOTAL_SHARDS; start+=LOCAL_SHARDS)); do
    pids=()
    echo "[wave] $phase start=$start $(date -Is)" | tee -a "$LOG_DIR/motiongpt_convert.log"
    for ((local_idx=0; local_idx<LOCAL_SHARDS; local_idx++)); do
      shard=$((start + local_idx))
      if (( shard >= TOTAL_SHARDS )); then
        continue
      fi
      gpu="${GPUS[$((local_idx % ${#GPUS[@]}))]}"
      log="$LOG_DIR/${phase}_s$(printf '%02d' "$shard")_of_$(printf '%02d' "$TOTAL_SHARDS").log"
      rm -f "${log}.status"
      (
        set +e
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$@" "$TOTAL_SHARDS" "$shard" >"$log" 2>&1
        code=$?
        echo "exit_code=$code finished_at=$(date -Is)" >"${log}.status"
        exit "$code"
      ) &
      pids+=("$!")
      echo "[launch] phase=$phase shard=$shard/$TOTAL_SHARDS gpu=$gpu pid=${pids[-1]} log=$log" | tee -a "$LOG_DIR/motiongpt_convert.log"
    done
    fail=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        fail=1
      fi
    done
    if [[ "$fail" -ne 0 ]]; then
      echo "[phase-fail] $phase wave_start=$start $(date -Is)" | tee -a "$LOG_DIR/motiongpt_convert.log"
      return 1
    fi
  done
  echo "[phase-done] $phase $(date -Is)" | tee -a "$LOG_DIR/motiongpt_convert.log"
}

coverage_hml() {
  "$PY_BIN" - <<'PY' "$HML_DIR" "$ANNO" "$METRIC_DIR/hml263_coverage_after_convert.json"
import json
import sys
from pathlib import Path

directory, anno, out = map(Path, sys.argv[1:])
data = json.loads(anno.read_text())["data_list"]
files = {p.stem for p in directory.glob("*.npy") if not p.name.startswith("_")}
missing = sorted(set(data) - files)
extra = sorted(files - set(data))
summary = {
    "representation": "hml263",
    "count": len(files),
    "expected_count": len(data),
    "missing_count": len(missing),
    "extra_count": len(extra),
    "missing_first50": missing[:50],
    "extra_first50": extra[:50],
}
out.write_text(json.dumps(summary, indent=2))
print("[coverage-hml263] " + json.dumps(summary, ensure_ascii=False))
if missing or extra:
    raise SystemExit(1)
PY
}

coverage_motion135() {
  "$PY_BIN" - <<'PY' "$M135_DIR" "$ANNO" "$METRIC_DIR/motion135_coverage.json"
import json
import sys
from pathlib import Path

import numpy as np

directory, anno, out = map(Path, sys.argv[1:])
data = json.loads(anno.read_text())["data_list"]
files = {p.stem: p for p in directory.glob("*.npz") if not p.name.startswith("_")}
missing = sorted(set(data) - set(files))
extra = sorted(set(files) - set(data))
mismatch = []
for sid, path in files.items():
    if sid not in data:
        continue
    with np.load(path) as z:
        length = int(z["motion_135"].shape[0])
    expected = int(data[sid]["num_frames"])
    if length != expected:
        mismatch.append({"sid": sid, "frames": length, "expected": expected})
summary = {
    "representation": "motion135",
    "count": len(files),
    "expected_count": len(data),
    "missing_count": len(missing),
    "extra_count": len(extra),
    "length_mismatch_count": len(mismatch),
    "missing_first50": missing[:50],
    "extra_first50": extra[:50],
    "length_mismatch_first50": mismatch[:50],
}
out.write_text(json.dumps(summary, indent=2))
print("[coverage-motion135] " + json.dumps(summary, ensure_ascii=False))
if missing or extra or mismatch:
    raise SystemExit(1)
PY
}

coverage_ms272() {
  "$PY_BIN" - <<'PY' "$MS272_DIR" "$ANNO" "$METRIC_DIR/ms272_coverage.json"
import json
import sys
from pathlib import Path

import numpy as np

directory, anno, out = map(Path, sys.argv[1:])
data = json.loads(anno.read_text())["data_list"]
files = {p.stem: p for p in directory.glob("*.npy") if not p.name.startswith("_")}
missing = sorted(set(data) - set(files))
extra = sorted(set(files) - set(data))
mismatch = []
for sid, path in files.items():
    if sid not in data:
        continue
    length = int(np.load(path, mmap_mode="r").shape[0])
    expected = int(data[sid]["num_frames"])
    if length != expected:
        mismatch.append({"sid": sid, "frames": length, "expected": expected})
summary = {
    "representation": "ms272",
    "count": len(files),
    "expected_count": len(data),
    "missing_count": len(missing),
    "extra_count": len(extra),
    "length_mismatch_count": len(mismatch),
    "missing_first50": missing[:50],
    "extra_first50": extra[:50],
    "length_mismatch_first50": mismatch[:50],
}
out.write_text(json.dumps(summary, indent=2))
print("[coverage-ms272] " + json.dumps(summary, ensure_ascii=False))
if missing or extra or mismatch:
    raise SystemExit(1)
PY
}

echo "[start] MotionGPT HML263 -> SMPL/MS272 convert $(date -Is)" | tee "$LOG_DIR/motiongpt_convert.log"
echo "[paths] hml=$HML_DIR motion135=$M135_DIR ms272=$MS272_DIR anno=$ANNO" | tee -a "$LOG_DIR/motiongpt_convert.log"
ensure_python_deps
prepare_split
write_meta "$M135_DIR" motion135
write_meta "$MS272_DIR" ms272
coverage_hml

run_all_shards ik_motiongpt run_ik
coverage_motion135

"$PY_BIN" scripts/data/convert_motion135_to_h3d272.py \
  --in-dir "$M135_DIR" \
  --out-dir "$MS272_DIR" \
  --rotation-space local \
  --workers "$WORKERS" \
  >"$LOG_DIR/motiongpt_motion135_to_ms272.log" 2>&1
coverage_ms272

"$PY_BIN" scripts/eval/audit_table1_lengths.py \
  --out-dir "$METRIC_DIR/length_audit" \
  --method "MotionGPT=$M135_DIR" \
  >"$LOG_DIR/motiongpt_length_audit.log" 2>&1

echo "[done] MotionGPT HML263 -> SMPL/MS272 convert $(date -Is)" | tee -a "$LOG_DIR/motiongpt_convert.log"
