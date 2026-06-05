#!/usr/bin/env bash
# Run ProtoMotions-format tracker baselines on the public LAFAN1-G1 set.
#
# Intended for an 8-GPU node/container with IsaacGym available. The script:
#   1. converts LAFAN1-G1 qpos/qvel NPZ files to ProtoMotions .motion shards,
#   2. packs each shard into a MotionLib .pt,
#   3. evaluates each checkpoint with one GPU per shard,
#   4. aggregates shard logs into JSON/Markdown.
set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
LAFAN_ROOT="${LAFAN_ROOT:-${PROJECT_ROOT}/ref_repo/OpenTrack/storage/data/mocap/lafan1/UnitreeG1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/lafan1_g1_proto_baseline_eval/$(date +%Y%m%d_%H%M%S)}"
NUM_SHARDS="${NUM_SHARDS:-8}"
NUM_ENVS="${NUM_ENVS:-64}"
# Match the OpenTrack/ONNX LAFAN1-G1 local benchmark horizon unless explicitly
# overridden. The released LAFAN1 sequences are 100-270s long, so full-clip
# IsaacGym evaluation is much slower and should be scheduled separately.
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-600}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
FORCE_PACK="${FORCE_PACK:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
MAX_FILES="${MAX_FILES:-}"
SAVE_PREDICTED_MOTION_LIB="${SAVE_PREDICTED_MOTION_LIB:-0}"
PREDICTED_MOTION_LIB_EVERY="${PREDICTED_MOTION_LIB_EVERY:-1}"

DEFAULT_CHECKPOINTS="protomotions_g1_bones=data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt,physflow_rehearsal_v2=results/physflow_g1_released_rehearsal_v2_taskheavy/last.ckpt"
CHECKPOINT_SPECS="${CHECKPOINT_SPECS:-${DEFAULT_CHECKPOINTS}}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

echo "[lafan1-g1-proto] start $(date)"
echo "[lafan1-g1-proto] host=$(hostname)"
echo "[lafan1-g1-proto] project=${PROJECT_ROOT}"
echo "[lafan1-g1-proto] lafan=${LAFAN_ROOT}"
echo "[lafan1-g1-proto] out=${OUT_ROOT}"
echo "[lafan1-g1-proto] shards=${NUM_SHARDS} envs_per_gpu=${NUM_ENVS} max_eval_steps=${MAX_EVAL_STEPS}"
echo "[lafan1-g1-proto] output_fps=${OUTPUT_FPS} max_files=${MAX_FILES:-all}"
echo "[lafan1-g1-proto] checkpoints=${CHECKPOINT_SPECS}"
echo "[lafan1-g1-proto] save_predicted_motion_lib=${SAVE_PREDICTED_MOTION_LIB} every=${PREDICTED_MOTION_LIB_EVERY}"

if [[ ! -d "${LAFAN_ROOT}" ]]; then
    echo "[lafan1-g1-proto] ERROR: LAFAN root missing: ${LAFAN_ROOT}" >&2
    exit 2
fi

wait_for_background() {
    local stage="$1"
    shift || true
    local failed=0
    local pid
    for pid in "$@"; do
        if ! wait "${pid}"; then
            echo "[lafan1-g1-proto] ERROR: ${stage} subprocess ${pid} failed" >&2
            failed=1
        fi
    done
    if [[ "${failed}" != "0" ]]; then
        exit 6
    fi
}

if [[ "${RUN_NODE_SETUP:-1}" == "1" ]]; then
    PHYSFLOW_NODE_SETUP_ONLY=1 bash scripts/embodied/cursor_physflow_taiji_node_setup.sh
fi

if [[ -n "${PHYSFLOW_TRACKER_PYTHON_CMD:-}" ]]; then
    read -r -a TRACKER_PY <<< "${PHYSFLOW_TRACKER_PYTHON_CMD}"
elif [[ -x /root/physflow_isaacgym_py38_cu118/bin/python ]]; then
    TRACKER_PY=(/root/physflow_isaacgym_py38_cu118/bin/python)
else
    TRACKER_PY=(python3)
fi

cd "${PROJECT_ROOT}/ref_repo/ProtoMotions"
export PYTHONPATH="${PWD}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/.cache/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-8}"

for version in 14 13 12 11 10 9; do
    gcc_root="/opt/rh/gcc-toolset-${version}/root/usr"
    if [[ -d "${gcc_root}/bin" ]]; then
        export PATH="${gcc_root}/bin:${PATH}"
        export CC="${gcc_root}/bin/gcc"
        export CXX="${gcc_root}/bin/g++"
        export LD_LIBRARY_PATH="${gcc_root}/lib64:${LD_LIBRARY_PATH:-}"
        echo "[lafan1-g1-proto] using gcc-toolset-${version}: CC=${CC}"
        break
    fi
done

echo "[lafan1-g1-proto] tracker_python=${TRACKER_PY[*]}"
"${TRACKER_PY[@]}" - <<'PY'
import importlib.util, sys
print("python", sys.version)
for name in ("torch", "isaacgym", "mujoco", "lightning", "tensordict"):
    print(f"import_check {name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
PY

if ! "${TRACKER_PY[@]}" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("isaacgym") else 1)
PY
then
    echo "[lafan1-g1-proto] ERROR: IsaacGym is unavailable in tracker_python; run inside the prepared tracker container." >&2
    exit 5
fi

echo "[lafan1-g1-proto] warm IsaacGym gymtorch"
if [[ "${REBUILD_GYMTORCH:-0}" == "1" ]]; then
    rm -rf "${TORCH_EXTENSIONS_DIR}/gymtorch"
fi
"${TRACKER_PY[@]}" - <<'PY'
import isaacgym  # noqa: F401
from isaacgym import gymtorch  # noqa: F401
print("gymtorch warmup OK")
PY

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/[lafan1-g1-proto] gpu /' || true

MOTION_BASE="${OUT_ROOT}/motion_shards"
mkdir -p "${MOTION_BASE}"

echo "[lafan1-g1-proto] converting LAFAN1-G1 to ProtoMotions .motion shards"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_dir="${MOTION_BASE}/shard_${shard}"
    mkdir -p "${shard_dir}"
    if [[ "${FORCE_CONVERT}" == "1" ]]; then
        rm -rf "${shard_dir}"
        mkdir -p "${shard_dir}"
    fi
    if find "${shard_dir}" -type f -name '*.motion' | grep -q .; then
        echo "[lafan1-g1-proto] shard ${shard}: conversion exists"
        continue
    fi
    (
        cmd=(
            "${TRACKER_PY[@]}" data/scripts/convert_g1_qpos_npz_to_proto.py
            --input-dir "${LAFAN_ROOT}"
            --output-dir "${shard_dir}"
            --output-fps "${OUTPUT_FPS}"
            --num-rank "${NUM_SHARDS}"
            --slurm-rank "${shard}"
        )
        if [[ -n "${MAX_FILES}" ]]; then
            cmd+=(--max-files "${MAX_FILES}")
        fi
        "${cmd[@]}" > "${OUT_ROOT}/convert_shard_${shard}.log" 2>&1
    ) &
    pids+=("$!")
done
wait_for_background "conversion" "${pids[@]}"

echo "[lafan1-g1-proto] packing motion shards"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_dir="${MOTION_BASE}/shard_${shard}"
    shard_pt="${MOTION_BASE}/lafan1_g1_shard_${shard}.pt"
    motion_count=$(find "${shard_dir}" -type f -name '*.motion' | wc -l)
    echo "[lafan1-g1-proto] shard ${shard}: motion_count=${motion_count}"
    if [[ "${motion_count}" -lt 1 ]]; then
        echo "[lafan1-g1-proto] ERROR: shard ${shard} has no motions" >&2
        exit 3
    fi
    if [[ "${FORCE_PACK}" == "1" || ! -s "${shard_pt}" ]]; then
        "${TRACKER_PY[@]}" protomotions/components/motion_lib.py \
            --motion-path "${shard_dir}" \
            --output-file "${shard_pt}" \
            --device cpu \
            > "${OUT_ROOT}/pack_shard_${shard}.log" 2>&1
    fi
done

IFS=',' read -r -a CKPT_ARRAY <<< "${CHECKPOINT_SPECS}"
for spec in "${CKPT_ARRAY[@]}"; do
    name="${spec%%=*}"
    ckpt="${spec#*=}"
    ckpt="${ckpt#./}"
    if [[ "${ckpt}" != /* ]]; then
        ckpt="${PROJECT_ROOT}/ref_repo/ProtoMotions/${ckpt}"
    fi
    if [[ ! -f "${ckpt}" ]]; then
        echo "[lafan1-g1-proto] ERROR: checkpoint missing for ${name}: ${ckpt}" >&2
        exit 4
    fi

    eval_dir="${OUT_ROOT}/eval_${name}"
    mkdir -p "${eval_dir}"
    echo "[lafan1-g1-proto] evaluating ${name}: ${ckpt}"

    pids=()
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
        shard_pt="${MOTION_BASE}/lafan1_g1_shard_${shard}.pt"
        log="${eval_dir}/shard_${shard}.log"
        pred_root="${eval_dir}/predicted_shard_${shard}"
        if [[ "${SAVE_PREDICTED_MOTION_LIB}" == "1" ]]; then
            mkdir -p "${pred_root}"
        fi
        if [[ "${FORCE_EVAL}" != "1" && -s "${log}" ]] && grep -q "EVALUATION RESULTS" "${log}"; then
            echo "[lafan1-g1-proto] ${name} shard ${shard}: already done"
            continue
        fi
        (
            export CUDA_VISIBLE_DEVICES="${shard}"
            save_override="agent.evaluator.save_predicted_motion_lib_every=None"
            root_args=()
            if [[ "${SAVE_PREDICTED_MOTION_LIB}" == "1" ]]; then
                save_override="agent.evaluator.save_predicted_motion_lib_every=${PREDICTED_MOTION_LIB_EVERY}"
                root_args=(--root-dir "${pred_root}")
            fi
            "${TRACKER_PY[@]}" protomotions/inference_agent.py \
                --checkpoint "${ckpt}" \
                --motion-file "${shard_pt}" \
                --simulator isaacgym \
                --num-envs "${NUM_ENVS}" \
                --headless \
                --full-eval \
                "${root_args[@]}" \
                --overrides \
                    "agent.evaluator.max_eval_steps=${MAX_EVAL_STEPS}" \
                    "${save_override}" \
                > "${log}" 2>&1
        ) &
        pids+=("$!")
    done
    wait_for_background "${name} eval" "${pids[@]}"
done

cd "${PROJECT_ROOT}"
python3 scripts/embodied/aggregate_proto_eval_logs.py \
    --eval-root "${OUT_ROOT}" \
    --motion-base "${MOTION_BASE}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-file-template 'lafan1_g1_shard_{shard}.pt'

if [[ "${SAVE_PREDICTED_MOTION_LIB}" == "1" ]]; then
    python3 scripts/embodied/aggregate_proto_predicted_motion_metrics.py \
        --eval-root "${OUT_ROOT}" \
        --motion-base "${MOTION_BASE}" \
        --num-shards "${NUM_SHARDS}" \
        --shard-file-template 'lafan1_g1_shard_{shard}.pt'
fi

echo "[lafan1-g1-proto] done $(date)"
echo "[lafan1-g1-proto] summary=${OUT_ROOT}/summary.md"
