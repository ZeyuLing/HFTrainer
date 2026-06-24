#!/usr/bin/env bash
# Score the fixed 24-clip AGILE hard-overfit set under frozen and each exported
# co-evolved tracker ONNX. This is the clean metric for "did the tracker improve":
# same references, different policies.
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"
echo "[hardovf-score] $(date) host=$(hostname)"

nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[hardovf-score] host CUDA driver version: ${CUDA_DRV:-unknown}"
if [ -n "$CUDA_DRV" ] && awk "BEGIN{exit !($CUDA_DRV < 11.4)}"; then
  echo "[hardovf-score] FATAL_BAD_NODE: CUDA driver $CUDA_DRV < 11.4. Aborting fast for reschedule."
  exit 42
fi
echo "[hardovf-score] driver gate OK (>=11.4)"

ln -sfn "$ENVDIR/isaacgym" /root/isaacgym
ln -sfn "$ENVDIR/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38 python38-devel gcc gcc-c++ make 2>&1 | tail -1 || true
PY38RT="$ENVDIR/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  cp -a "$PY38RT/bin/python3.8" /usr/bin/python3.8
  cp -a "$PY38RT/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "$PY38RT/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi

export PIP_DEFAULT_TIMEOUT=30
PY310="${PY310:-/usr/local/bin/python3}"
if [ ! -x "$PY310" ]; then
  PY310="$(command -v python3)"
fi
timeout 300 "$PY310" -m pip install --quiet \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -1 || true

export PATH=/usr/local/bin:$PATH
export PYTHONPATH="$REPO:$REPO/ref_repo/ProtoMotions:${PYTHONPATH:-}"
export MUJOCO_GL=disable
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PHYSFLOW_CONVERT_PYTHON=/root/physflow_isaacgym_py38_cu118/bin/python

ANNO=data/annotation/_coevo_hardovf_agile_eval.json
BASE_OUT="${SCORE_OUT:-output/hardovf_frontier_fixed_score}"
ROOT="${SCORE_ROOT:-work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_cons/hardovf_frontier_gtreplay_cons}"

score_one () {  # label, optional onnx
  local label="$1"; shift
  local out="$BASE_OUT/$label"
  echo "[hardovf-score] scoring $label -> $out"
  "$PY310" scripts/embodied/score_heldout_frozen.py \
    --heldout "$ANNO" --out "$out" "$@"
}

SCORED_LABELS=()
score_one frozen
SCORED_LABELS+=("frozen")
for r in 0 1 2 3 4; do
  onnx="$ROOT/judge_onnx/r${r}/unified_pipeline.onnx"
  if [ -f "$onnx" ]; then
    score_one "r${r}" --onnx "$onnx"
    SCORED_LABELS+=("r${r}")
  else
    echo "[hardovf-score] skip r${r}: missing $onnx"
  fi
done

"$PY310" - "$BASE_OUT" "${SCORED_LABELS[@]}" <<'PY'
import json, os, glob
import sys
base = sys.argv[1]
print("\n[hardovf-score] SUMMARY")
for label in sys.argv[2:]:
    p = os.path.join(base, label, "heldout_score.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    rows = d["rows"]
    mean = sum(r["completion"] for r in rows) / max(len(rows), 1)
    head = d["n_headroom"]
    falls = sum(1 for r in rows if r["fall"])
    print(f"  {label:>6}: mean_completion={mean:.3f} headroom={head}/{len(rows)} falls={falls}")
PY
