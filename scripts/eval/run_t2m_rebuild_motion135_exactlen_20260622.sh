#!/usr/bin/env bash
# Rebuild clean HumanML3D official-test SMPL/motion135 predictions from the
# canonical HML263 outputs. This is intentionally full-dataset, exact-length,
# and writes to a new method_setting directory before any canonical replacement.
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

METHOD="${METHOD:?set METHOD=flowmdm|motionlab|mdm|motiongpt3}"
TOTAL_SHARDS="${TOTAL_SHARDS:-8}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
LOCAL_SHARDS="${LOCAL_SHARDS:-${TJ_GPU_NUM:-1}}"
NUM_GPUS="${NUM_GPUS:-${TJ_GPU_NUM:-1}}"
REFINE_ITERS="${REFINE_ITERS:-80}"
REFINE_LR="${REFINE_LR:-0.02}"
RUN_TAG="${RUN_TAG:-rebuild_motion135_exactlen_20260622}"

ANNO="data/annotation/test_hml3d_official272_gtlen.json"
BASE="outputs/evaluation/t2m/humanml3d_official_test"
RUN_ROOT="$BASE/_runs/$RUN_TAG"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

case "$METHOD" in
  flowmdm|motionlab|mdm|motiongpt3) ;;
  *)
    echo "[error] unsupported METHOD=$METHOD" >&2
    exit 2
    ;;
esac

IN_DIR="${IN_DIR:-$BASE/hml263/${METHOD}_official/predictions/hml263}"
OUT_SETTING="${OUT_SETTING:-${METHOD}_official_smpl_ik_exactlen_20260622}"
OUT_ROOT="$BASE/motion135/$OUT_SETTING"
OUT_DIR="$OUT_ROOT/predictions/motion135"
mkdir -p "$OUT_DIR"

if [[ ! -d "$IN_DIR" ]]; then
  echo "[error] missing input dir: $IN_DIR" >&2
  exit 2
fi

pick_python() {
  local candidates=()
  if [[ -n "${PY:-}" ]]; then
    candidates+=("$PY")
  fi
  candidates+=(python3 python /opt/conda/bin/python /root/miniconda3/bin/python /opt/miniconda3/bin/python)
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
  local stamp="${DEPS_STAMP:-/tmp/hftrainer_t2m_rebuild_motion135_exactlen_deps_v1.stamp}"
  if [[ -f "$stamp" ]]; then
    return 0
  fi
  local missing
  missing="$("$PY_BIN" - <<'PY'
import importlib.util
checks = [
    ("chumpy", "chumpy>=0.70"),
    ("smplx", "smplx>=0.1.28"),
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

ensure_python_deps

"$PY_BIN" - <<'PY' "$OUT_ROOT" "$METHOD" "$IN_DIR" "$OUT_DIR" "$ANNO" "$RUN_ROOT" "$TOTAL_SHARDS" "$REFINE_ITERS" "$REFINE_LR"
import json
import sys
from pathlib import Path

out_root, method, in_dir, out_dir, anno, run_root, total_shards, refine_iters, refine_lr = sys.argv[1:]
cfg = {
    "method": method,
    "dataset": "humanml3d_official_test",
    "task": "t2m",
    "native_representation": "hml263",
    "target_representation": "motion135",
    "source_dir": in_dir,
    "pred_dir": out_dir,
    "annotation": anno,
    "target_length_policy": "official_annotation_num_frames",
    "source_fps": 20,
    "target_fps": 30,
    "floor_align": True,
    "refine_iters": int(refine_iters),
    "refine_lr": float(refine_lr),
    "total_shards": int(total_shards),
    "runner": run_root,
    "created_by": "scripts/eval/run_t2m_rebuild_motion135_exactlen_20260622.sh",
}
Path(out_root).mkdir(parents=True, exist_ok=True)
(Path(out_root) / "run_config.json").write_text(json.dumps(cfg, indent=2))
PY

echo "[setup] method=$METHOD in=$IN_DIR out=$OUT_DIR shards=$TOTAL_SHARDS offset=$SHARD_OFFSET local=$LOCAL_SHARDS gpus=$NUM_GPUS refine=$REFINE_ITERS"

pids=()
for ((local_idx=0; local_idx<LOCAL_SHARDS; local_idx++)); do
  shard=$((SHARD_OFFSET + local_idx))
  if (( shard >= TOTAL_SHARDS )); then
    continue
  fi
  gpu=$((local_idx % NUM_GPUS))
  log="$LOG_DIR/${METHOD}_s$(printf '%02d' "$shard")_of_$(printf '%02d' "$TOTAL_SHARDS").log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[shard-start] method=$METHOD shard=$shard/$TOTAL_SHARDS gpu=$gpu $(date -Is)"
    "$PY_BIN" scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "$IN_DIR" \
      --out-dir "$OUT_DIR" \
      --num-shards "$TOTAL_SHARDS" \
      --shard-index "$shard" \
      --source-fps 20 \
      --target-fps 30 \
      --target-length-anno "$ANNO" \
      --device cuda \
      --batch-size 1 \
      --floor-align \
      --refine-iters "$REFINE_ITERS" \
      --refine-lr "$REFINE_LR" \
      --skip-existing
    echo "[shard-end] method=$METHOD shard=$shard/$TOTAL_SHARDS $(date -Is)"
  ) >"$log" 2>&1 &
  pids+=("$!")
  echo "[launch] shard=$shard/$TOTAL_SHARDS gpu=$gpu pid=${pids[-1]} log=$log"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done
if (( failed > 0 )); then
  echo "[error] failed_shards=$failed" >&2
  exit 1
fi

"$PY_BIN" - <<'PY' "$OUT_DIR" "$ANNO" "$OUT_ROOT"
import glob
import json
import numpy as np
import sys
from pathlib import Path

out_dir, anno, out_root = map(Path, sys.argv[1:])
data = json.loads(anno.read_text())["data_list"]
files = sorted(out_dir.glob("*.npz"))
bad = []
for path in files:
    sid = path.stem
    expected = data.get(sid, {}).get("num_frames")
    if expected is None:
        continue
    try:
        with np.load(path) as z:
            n = int(z["motion_135"].shape[0])
    except Exception as exc:
        bad.append({"sid": sid, "error": repr(exc)})
        continue
    if n != int(expected):
        bad.append({"sid": sid, "frames": n, "expected": int(expected)})
summary = {
    "count": len(files),
    "expected_count": len(data),
    "missing_count": max(0, len(data) - len(files)),
    "length_mismatch_count": len(bad),
    "length_mismatch_examples": bad[:20],
}
(out_root / "coverage_summary.json").write_text(json.dumps(summary, indent=2))
print("[coverage] " + json.dumps(summary, ensure_ascii=False))
if len(files) != len(data) or bad:
    raise SystemExit(1)
PY

echo "[done] method=$METHOD out=$OUT_DIR"
