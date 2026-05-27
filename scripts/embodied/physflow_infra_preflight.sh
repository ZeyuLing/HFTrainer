#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
LOG_DIR="${LOG_DIR:-$ROOT/output/physflow/infra_preflight/$(date +%Y%m%d_%H%M%S)}"
SAMPLE_NPZ="${SAMPLE_NPZ:-$ROOT/output/physflow/eval_demo/data/npz/original_000_a_person_stands_still.npz}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "[preflight] root=$ROOT"
echo "[preflight] log_dir=$LOG_DIR"
echo "[preflight] sample_npz=$SAMPLE_NPZ"

{
  echo "===== host ====="
  hostname
  date
  pwd
  echo
  echo "===== gpu ====="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi not found"
  fi
  echo
  echo "===== python ====="
  which python3
  python3 --version
  python3 - <<'PY'
import importlib
mods = ["torch", "numpy", "mujoco", "onnxruntime", "scipy", "yaml"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"{m}: OK {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"{m}: FAIL {type(e).__name__}: {e}")
PY
} 2>&1 | tee "$LOG_DIR/00_env.log"

echo "[preflight] verifying MuJoCo collision/contact config"
python3 scripts/embodied/verify_mujoco_collision_config.py \
  2>&1 | tee "$LOG_DIR/01_mujoco_collision_config.log"

if [[ -f "$SAMPLE_NPZ" ]]; then
  echo "[preflight] running one-sample SMPL RL tracker smoke test"
  mkdir -p "$LOG_DIR/tracker_json" "$LOG_DIR/tracker_stats"
  python3 scripts/embodied/run_smpl_rl_tracker.py \
    --npz-file "$SAMPLE_NPZ" \
    --output-dir "$LOG_DIR/tracker_json" \
    --stats-dir "$LOG_DIR/tracker_stats" \
    --max-motions 1 \
    2>&1 | tee "$LOG_DIR/02_tracker_smoke.log"
else
  echo "[preflight] sample NPZ not found, skipping tracker smoke test" | tee "$LOG_DIR/02_tracker_smoke.log"
fi

echo "[preflight] done: $LOG_DIR"
