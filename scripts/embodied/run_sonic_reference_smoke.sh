#!/usr/bin/env bash
set -euo pipefail

# Smoke-run the official SONIC deploy stack on one G1 reference directory.
# It is intentionally conservative: it only clears large placeholder processes
# on the selected GPU and writes all artifacts under a caller-provided output dir.

GPU_ID="${GPU_ID:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
SONIC_REPO="${SONIC_REPO:-$PROJECT_ROOT/ref_repo/GR00T-WholeBodyControl}"
REFERENCE_DIR="${REFERENCE_DIR:-$SONIC_REPO/gear_sonic_deploy/reference/sonic_sample_smoke}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/evaluation/physflow/sonic_smoke_gpu${GPU_ID}}"
INTERFACE="${INTERFACE:-bond1}"
RUN_SECONDS="${RUN_SECONDS:-75}"
KILL_LARGE_OCCUPANTS="${KILL_LARGE_OCCUPANTS:-1}"
SONIC_DEPS_ROOT="${SONIC_DEPS_ROOT:-}"

mkdir -p "$OUT_DIR"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
SONIC_REPO="$(cd "$SONIC_REPO" && pwd)"
REFERENCE_DIR="$(cd "$REFERENCE_DIR" && pwd)"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

if [[ -z "$SONIC_DEPS_ROOT" ]]; then
  if [[ -d "$SONIC_REPO/tools/sonic_deps/install" ]]; then
    SONIC_DEPS_ROOT="$SONIC_REPO/tools/sonic_deps"
  elif [[ -d "$PROJECT_ROOT/tools/sonic_deps/install" ]]; then
    SONIC_DEPS_ROOT="$PROJECT_ROOT/tools/sonic_deps"
  else
    echo "[sonic-smoke] cannot locate sonic_deps/install under SONIC_REPO or PROJECT_ROOT" >&2
    exit 2
  fi
fi
SONIC_DEPS_ROOT="$(cd "$SONIC_DEPS_ROOT" && pwd)"

if [[ "$KILL_LARGE_OCCUPANTS" == "1" ]]; then
  mapfile -t large_pids < <(
    nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,used_memory \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -F, '$2 + 0 >= 50000 {gsub(/ /, "", $1); print $1}'
  )
  if (( ${#large_pids[@]} > 0 )); then
    echo "[sonic-smoke] killing large occupants on GPU ${GPU_ID}: ${large_pids[*]}"
    kill "${large_pids[@]}" || true
    sleep 3
  fi
fi

python "$PROJECT_ROOT/scripts/embodied/patch_sonic_qpos_logger.py" --sonic-repo "$SONIC_REPO"

cd "$SONIC_REPO"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SONIC_QPOS_LOGFILE="$OUT_DIR/sim_qpos.csv"
export PYTHONPATH="$SONIC_REPO:$SONIC_REPO/external_dependencies/unitree_sdk2_python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SONIC_DEPS_ROOT/install/zeromq/lib:$SONIC_DEPS_ROOT/install/conda-runtime/lib:$SONIC_DEPS_ROOT/install/tensorrt-10.0.1-cu11/lib:$SONIC_DEPS_ROOT/install/onnxruntime/lib:$SONIC_DEPS_ROOT/install/yaml-cpp/lib64:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"

rm -f "$OUT_DIR/deploy.stdin"
rm -rf "$OUT_DIR/deploy_logs"
mkdir -p "$OUT_DIR/deploy_logs"
mkfifo "$OUT_DIR/deploy.stdin"

cleanup() {
  kill "${DEPLOY_PID:-}" "${SIM_PID:-}" "${KEY_PID:-}" 2>/dev/null || true
  rm -f "$OUT_DIR/deploy.stdin"
}
trap cleanup EXIT

python gear_sonic/scripts/run_sim_loop.py \
  --interface "$INTERFACE" \
  --no-enable-onscreen \
  --no-enable-offscreen \
  --no-verbose \
  --no-data-collection \
  --no-enable-real-device \
  > "$OUT_DIR/sim.log" 2>&1 &
SIM_PID=$!

sleep 8

(
  exec 3>"$OUT_DIR/deploy.stdin"
  sleep 20
  for _ in $(seq 1 30); do printf ']' >&3; sleep 0.25; done
  sleep 1
  for _ in $(seq 1 30); do printf 't' >&3; sleep 0.25; done
  sleep "$RUN_SECONDS"
  exec 3>&-
) > "$OUT_DIR/keyinject.log" 2>&1 &
KEY_PID=$!

gear_sonic_deploy/target/release/g1_deploy_onnx_ref \
  "$INTERFACE" \
  gear_sonic_deploy/policy/release/model_decoder.onnx \
  "$REFERENCE_DIR" \
  --obs-config gear_sonic_deploy/policy/release/observation_config.yaml \
  --encoder-file gear_sonic_deploy/policy/release/model_encoder.onnx \
  --planner-file gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx \
  --input-type keyboard \
  --disable-crc-check \
  --enable-csv-logs \
  --enable-motion-recording \
  --logs-dir "$OUT_DIR/deploy_logs" \
  --target-motion-logfile "$OUT_DIR/target_motion.csv" \
  < "$OUT_DIR/deploy.stdin" > "$OUT_DIR/deploy.log" 2>&1 &
DEPLOY_PID=$!

sleep "$RUN_SECONDS"

echo "== control cues =="
grep -nE "Playing motion|operator_state.start|CONTROL state|Safety check failed|StateLogger|ERROR|WARN|Token" "$OUT_DIR/deploy.log" | tail -100 || true
echo "== artifact sizes =="
for file in "$OUT_DIR/sim_qpos.csv" "$OUT_DIR/target_motion.csv" "$OUT_DIR/deploy_logs/q.csv" "$OUT_DIR/deploy_logs/action.csv"; do
  [[ -e "$file" ]] && wc -l -c "$file"
done
echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
