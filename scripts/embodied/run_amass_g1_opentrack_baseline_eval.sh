#!/usr/bin/env bash
# Run OpenTrack ONNX+MuJoCo baseline on AMASS_Retarged_for_G1.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
AMASS_ROOT="${AMASS_ROOT:-${PROJECT_ROOT}/data/AMASS_Retarged_for_G1/g1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/opentrack_amass_g1/$(date +%Y%m%d_%H%M%S)}"
NUM_SHARDS="${NUM_SHARDS:-4}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
MAX_STEPS="${MAX_STEPS:-600}"
MAX_MOTIONS="${MAX_MOTIONS:-}"
QUAT_ORDER="${QUAT_ORDER:-wxyz}"
WAIT_FOR_PATTERN="${WAIT_FOR_PATTERN:-}"
PYTHON_BIN="${PYTHON_BIN:-/root/physflow_isaacgym_py38_cu118/bin/python}"

OPENTRACK_ROOT="${OPENTRACK_ROOT:-${PROJECT_ROOT}/ref_repo/OpenTrack}"
XML_PATH="${XML_PATH:-${OPENTRACK_ROOT}/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml}"
CONFIG_PATH="${CONFIG_PATH:-${OPENTRACK_ROOT}/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/config.json}"
ONNX_PATH="${ONNX_PATH:-${OPENTRACK_ROOT}/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/model.onnx}"

mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

cd "${PROJECT_ROOT}"
echo "[opentrack-eval] start $(date)"
echo "[opentrack-eval] out=${OUT_ROOT}"
echo "[opentrack-eval] shards=${NUM_SHARDS} output_fps=${OUTPUT_FPS} max_steps=${MAX_STEPS} quat_order=${QUAT_ORDER}"
echo "[opentrack-eval] python=${PYTHON_BIN}"

if [[ -n "${WAIT_FOR_PATTERN}" ]]; then
    echo "[opentrack-eval] waiting for processes matching: ${WAIT_FOR_PATTERN}"
    while pgrep -af "${WAIT_FOR_PATTERN}" >/dev/null; do
        pgrep -af "${WAIT_FOR_PATTERN}" | head -8 || true
        sleep 300
    done
fi

MOTION_DIR="${OUT_ROOT}/UnitreeG1"
mkdir -p "${MOTION_DIR}"

echo "[opentrack-eval] converting AMASS-G1 to OpenTrack npz shards"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    (
        "${PYTHON_BIN}" scripts/embodied/convert_amass_g1_to_opentrack_npz.py \
            --input-dir "${AMASS_ROOT}" \
            --output-dir "${MOTION_DIR}" \
            --xml "${XML_PATH}" \
            --output-fps "${OUTPUT_FPS}" \
            --quat-order "${QUAT_ORDER}" \
            --num-rank "${NUM_SHARDS}" \
            --slurm-rank "${shard}" \
            --manifest "${OUT_ROOT}/manifest_shard_${shard}.json" \
            --force \
            > "${OUT_ROOT}/convert_shard_${shard}.log" 2>&1
    ) &
done
wait

echo "[opentrack-eval] evaluating OpenTrack shards"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    extra=()
    if [[ -n "${MAX_MOTIONS}" ]]; then
        extra+=(--max-motions "${MAX_MOTIONS}")
    fi
    (
        "${PYTHON_BIN}" scripts/embodied/eval_opentrack_onnx_mujoco.py \
            --motion-dir "${MOTION_DIR}" \
            --manifest "${OUT_ROOT}/manifest_shard_${shard}.json" \
            --xml "${XML_PATH}" \
            --config "${CONFIG_PATH}" \
            --onnx "${ONNX_PATH}" \
            --output-json "${OUT_ROOT}/eval_shard_${shard}.json" \
            --output-csv "${OUT_ROOT}/eval_shard_${shard}.csv" \
            --max-steps "${MAX_STEPS}" \
            "${extra[@]}" \
            > "${OUT_ROOT}/eval_shard_${shard}.log" 2>&1
    ) &
done
wait

python3 scripts/embodied/aggregate_opentrack_eval.py --eval-root "${OUT_ROOT}"
echo "[opentrack-eval] done $(date)"
echo "[opentrack-eval] summary=${OUT_ROOT}/summary.md"
