#!/usr/bin/env bash
# Clean full-dataset MotionGPT3 rerun for HumanML3D official test.
#
# Canonical outputs:
#   outputs/evaluation/t2m/humanml3d_official_test/hml263/motiongpt3
#   outputs/evaluation/t2m/humanml3d_official_test/motion135/motiongpt3
#   outputs/evaluation/t2m/humanml3d_official_test/ms272/motiongpt3
#
# This script intentionally starts from empty canonical method directories and
# runs the HML263 generation stage through the hftrainer MotionGPT3
# ModelBundle/Pipeline implementation.
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
RUN_TAG="${RUN_TAG:-motiongpt3_clean_20260624}"
RUN_ROOT="$BASE/_runs/$RUN_TAG"
LOG_DIR="$RUN_ROOT/logs"
RUNTIME_DIR="$RUN_ROOT/runtime"
SPLIT="${SPLIT:-$RUN_ROOT/test_ids.txt}"

HML_DIR="${HML_DIR:-$BASE/hml263/motiongpt3}"
M135_DIR="${M135_DIR:-$BASE/motion135/motiongpt3}"
MS272_DIR="${MS272_DIR:-$BASE/ms272/motiongpt3}"

TOTAL_SHARDS="${TOTAL_SHARDS:-64}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
LOCAL_SHARDS="${LOCAL_SHARDS:-${TJ_GPU_NUM:-8}}"
NUM_GPUS="${NUM_GPUS:-${TJ_GPU_NUM:-8}}"
WORKERS="${WORKERS:-32}"
REFINE_ITERS="${REFINE_ITERS:-80}"
REFINE_LR="${REFINE_LR:-0.02}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CLEAN="${CLEAN:-1}"
MOTIONGPT3_ARTIFACT_DIR="${MOTIONGPT3_ARTIFACT_DIR:-checkpoints/baselines/motiongpt3}"
HML263_MEAN_PATH="${HML263_MEAN_PATH:-$MOTIONGPT3_ARTIFACT_DIR/assets/meta/mean.npy}"
HML263_STD_PATH="${HML263_STD_PATH:-$MOTIONGPT3_ARTIFACT_DIR/assets/meta/std.npy}"
export ROOT RUN_TAG TOTAL_SHARDS LOCAL_SHARDS NUM_GPUS GUIDANCE_SCALE REFINE_ITERS REFINE_LR
export HML263_MEAN_PATH HML263_STD_PATH HML_DIR M135_DIR MS272_DIR
export MOTIONGPT3_ARTIFACT_DIR

if [[ "$NUM_GPUS" -lt 1 ]]; then
  NUM_GPUS=1
fi
if [[ "$LOCAL_SHARDS" -lt 1 ]]; then
  LOCAL_SHARDS=1
fi

mkdir -p "$LOG_DIR" "$RUNTIME_DIR"
if [[ "$CLEAN" == "1" ]]; then
  rm -rf "$HML_DIR" "$M135_DIR" "$MS272_DIR"
fi
mkdir -p "$HML_DIR" "$M135_DIR" "$MS272_DIR"

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
import torch  # noqa: F401
import scipy  # noqa: F401
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

ensure_python_deps() {
  local stamp="${DEPS_STAMP:-/tmp/hftrainer_motiongpt3_clean_deps_v1.stamp}"
  if [[ -f "$stamp" ]]; then
    return 0
  fi
  local missing
  missing="$("$PY_BIN" - <<'PY'
import importlib.util
checks = [
    ("einops", "einops>=0.7"),
    ("hydra", "hydra-core>=1.3"),
    ("omegaconf", "omegaconf>=2.3"),
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
      $missing
  else
    echo "[deps] all required optional packages importable"
  fi
  touch "$stamp"
}

write_meta() {
  local rep_dir="$1" rep="$2"
  "$PY_BIN" - <<'PY' "$rep_dir" "$rep" "$ANNO" "$RUN_ROOT" "$HML_DIR" "$M135_DIR" "$MS272_DIR" "$TOTAL_SHARDS" "$GUIDANCE_SCALE" "$REFINE_ITERS" "$REFINE_LR" "$HML263_MEAN_PATH" "$HML263_STD_PATH"
import json
import os
import sys
from pathlib import Path

rep_dir, rep, anno, run_root, hml_dir, m135_dir, ms272_dir, total_shards, guidance, refine_iters, refine_lr, mean_path, std_path = sys.argv[1:]
cfg = {
    "task": "t2m",
    "dataset": "humanml3d_official_test",
    "method": "motiongpt3",
    "representation": rep,
    "model_bundle": "hftrainer.models.motion.motiongpt3.MotionGPT3Bundle",
    "pipeline": "hftrainer.pipelines.motiongpt3.MotionGPT3Pipeline",
    "artifact_dir": os.environ.get("MOTIONGPT3_ARTIFACT_DIR", ""),
    "caption_protocol": "motionclip_selected_official_humanml3d_caption",
    "annotation": anno,
    "hml263_dir": hml_dir,
    "motion135_dir": m135_dir,
    "ms272_dir": ms272_dir,
    "hml263_mean_path": mean_path,
    "hml263_std_path": std_path,
    "guidance_scale": float(guidance),
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
    "created_by": "scripts/eval/run_t2m_motiongpt3_clean_20260624.sh + scripts/eval/framework_t2m_hml263_infer.py",
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
        f"HML263_MEAN_PATH={os.environ.get('HML263_MEAN_PATH', '')}",
        f"HML263_STD_PATH={os.environ.get('HML263_STD_PATH', '')}",
        "bash scripts/eval/run_t2m_motiongpt3_clean_20260624.sh",
    ]).strip() + "\n"
)
PY
}

run_shards() {
  local phase="$1"
  shift
  echo "[phase-start] $phase total=$TOTAL_SHARDS offset=$SHARD_OFFSET local=$LOCAL_SHARDS $(date -Is)" | tee -a "$LOG_DIR/motiongpt3.log"
  local pids=()
  local local_idx shard gpu log
  for ((local_idx=0; local_idx<LOCAL_SHARDS; local_idx++)); do
    shard=$((SHARD_OFFSET + local_idx))
    if (( shard >= TOTAL_SHARDS )); then
      continue
    fi
    gpu=$((local_idx % NUM_GPUS))
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
    echo "[launch] phase=$phase shard=$shard/$TOTAL_SHARDS gpu=$gpu pid=${pids[-1]} log=$log" | tee -a "$LOG_DIR/motiongpt3.log"
  done

  local fail=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[phase-fail] $phase $(date -Is)" | tee -a "$LOG_DIR/motiongpt3.log"
    return 1
  fi
  echo "[phase-done] $phase $(date -Is)" | tee -a "$LOG_DIR/motiongpt3.log"
}

run_motiongpt3() {
  local shards="$1" shard="$2"
  "$PY_BIN" scripts/eval/framework_t2m_hml263_infer.py \
    --method motiongpt3 \
    --artifact-dir "$MOTIONGPT3_ARTIFACT_DIR" \
    --anno-file "$ANNO" \
    --caption-file "$ANNO" \
    --anno-data-dir "." \
    --out-dir "$HML_DIR" \
    --motiongpt3-runtime-dir "$RUNTIME_DIR/shard_${shard}" \
    --motiongpt3-guidance-scale "$GUIDANCE_SCALE" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --batch-size "$BATCH_SIZE" \
    --device cuda
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

coverage_hml() {
  "$PY_BIN" - <<'PY' "$HML_DIR" "$ANNO" "$RUN_ROOT/hml263_coverage.json"
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
  "$PY_BIN" - <<'PY' "$M135_DIR" "$ANNO" "$RUN_ROOT/motion135_coverage.json"
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
  "$PY_BIN" - <<'PY' "$MS272_DIR" "$ANNO" "$RUN_ROOT/ms272_coverage.json"
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

echo "[start] clean MotionGPT3 rerun $(date -Is)" | tee "$LOG_DIR/motiongpt3.log"
echo "[paths] hml=$HML_DIR motion135=$M135_DIR ms272=$MS272_DIR anno=$ANNO" | tee -a "$LOG_DIR/motiongpt3.log"
ensure_python_deps
prepare_split
write_meta "$HML_DIR" hml263
write_meta "$M135_DIR" motion135
write_meta "$MS272_DIR" ms272

run_shards infer_motiongpt3 run_motiongpt3
coverage_hml

run_shards ik_motiongpt3 run_ik
coverage_motion135

"$PY_BIN" scripts/data/convert_motion135_to_h3d272.py \
  --in-dir "$M135_DIR" \
  --out-dir "$MS272_DIR" \
  --rotation-space local \
  --workers "$WORKERS" \
  >"$LOG_DIR/motion135_to_ms272.log" 2>&1
coverage_ms272

"$PY_BIN" scripts/eval/audit_table1_lengths.py \
  --out-dir "$RUN_ROOT/length_audit" \
  --method "MotionGPT3=$M135_DIR" \
  >"$LOG_DIR/length_audit.log" 2>&1

echo "[done] clean MotionGPT3 rerun $(date -Is)" | tee -a "$LOG_DIR/motiongpt3.log"
