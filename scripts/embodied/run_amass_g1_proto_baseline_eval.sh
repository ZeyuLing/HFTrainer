#!/usr/bin/env bash
# Run ProtoMotions-format tracker baselines on AMASS_Retarged_for_G1.
#
# Intended for an 8-GPU Taiji node with the staged IsaacGym env restored by
# cursor_physflow_taiji_node_setup.sh. The script:
#   1. converts recursive AMASS-G1 *_jpos.npz to ProtoMotions .motion shards,
#   2. packs each shard into a MotionLib .pt,
#   3. evaluates each checkpoint on each shard with one GPU per shard,
#   4. aggregates logs into JSON/Markdown.
set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
AMASS_ROOT="${AMASS_ROOT:-${PROJECT_ROOT}/data/AMASS_Retarged_for_G1/g1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/amass_g1_proto_baseline_eval/$(date +%Y%m%d_%H%M%S)}"
NUM_SHARDS="${NUM_SHARDS:-8}"
NUM_ENVS="${NUM_ENVS:-256}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-600}"
OUTPUT_FPS="${OUTPUT_FPS:-30}"
QUAT_ORDER="${QUAT_ORDER:-wxyz}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
FORCE_PACK="${FORCE_PACK:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
SAVE_PREDICTED_MOTION_LIB="${SAVE_PREDICTED_MOTION_LIB:-0}"
PREDICTED_MOTION_LIB_EVERY="${PREDICTED_MOTION_LIB_EVERY:-1}"

DEFAULT_CHECKPOINTS="protomotions_g1_bones=data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt,physflow_rehearsal_v2=results/physflow_g1_released_rehearsal_v2_taskheavy/last.ckpt"
CHECKPOINT_SPECS="${CHECKPOINT_SPECS:-${DEFAULT_CHECKPOINTS}}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

echo "[amass-g1-eval] start $(date)"
echo "[amass-g1-eval] host=$(hostname)"
echo "[amass-g1-eval] project=${PROJECT_ROOT}"
echo "[amass-g1-eval] amass=${AMASS_ROOT}"
echo "[amass-g1-eval] out=${OUT_ROOT}"
echo "[amass-g1-eval] shards=${NUM_SHARDS} envs_per_gpu=${NUM_ENVS} max_eval_steps=${MAX_EVAL_STEPS}"
echo "[amass-g1-eval] output_fps=${OUTPUT_FPS} quat_order=${QUAT_ORDER}"
echo "[amass-g1-eval] checkpoints=${CHECKPOINT_SPECS}"
echo "[amass-g1-eval] save_predicted_motion_lib=${SAVE_PREDICTED_MOTION_LIB} every=${PREDICTED_MOTION_LIB_EVERY}"

if [[ ! -d "${AMASS_ROOT}" ]]; then
    echo "[amass-g1-eval] ERROR: AMASS root missing: ${AMASS_ROOT}" >&2
    exit 2
fi

if [[ "${RUN_NODE_SETUP:-1}" == "1" ]]; then
    PHYSFLOW_NODE_SETUP_ONLY=1 bash scripts/embodied/cursor_physflow_taiji_node_setup.sh
fi

wait_for_background() {
    local stage="$1"
    shift || true
    local failed=0
    local pid
    for pid in "$@"; do
        if ! wait "${pid}"; then
            echo "[amass-g1-eval] ERROR: ${stage} subprocess ${pid} failed" >&2
            failed=1
        fi
    done
    if [[ "${failed}" != "0" ]]; then
        exit 6
    fi
}

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
        echo "[amass-g1-eval] using gcc-toolset-${version}: CC=${CC}"
        break
    fi
done

echo "[amass-g1-eval] tracker_python=${TRACKER_PY[*]}"
"${TRACKER_PY[@]}" - <<'PY'
import importlib.util, sys
print("python", sys.version)
for name in ("torch", "isaacgym", "mujoco", "lightning", "tensordict"):
    print(f"import_check {name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
PY

echo "[amass-g1-eval] warm IsaacGym gymtorch"
if [[ "${REBUILD_GYMTORCH:-0}" == "1" ]]; then
    rm -rf "${TORCH_EXTENSIONS_DIR}/gymtorch"
fi
"${TRACKER_PY[@]}" - <<'PY'
import isaacgym  # noqa: F401
from isaacgym import gymtorch  # noqa: F401
print("gymtorch warmup OK")
PY

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/[amass-g1-eval] gpu /' || true

MOTION_BASE="${OUT_ROOT}/motion_shards"
mkdir -p "${MOTION_BASE}"

PACKED_SHARDS_REUSABLE=0
packed_count=$(find -L "${MOTION_BASE}" -maxdepth 1 -type f -name 'amass_g1_full_shard_*.pt' | wc -l)
packed_complete=1
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    if [[ ! -s "${MOTION_BASE}/amass_g1_full_shard_${shard}.pt" ]]; then
        packed_complete=0
        break
    fi
done
if [[ "${FORCE_CONVERT}" != "1" && "${FORCE_PACK}" != "1" && "${packed_count}" -gt 0 ]]; then
    if [[ "${packed_count}" -ne "${NUM_SHARDS}" || "${packed_complete}" != "1" ]]; then
        echo "[amass-g1-eval] ERROR: packed shard cache has ${packed_count} file(s), expected ${NUM_SHARDS}; use matching NUM_SHARDS or a clean motion_shards directory" >&2
        exit 7
    fi
    PACKED_SHARDS_REUSABLE=1
fi

echo "[amass-g1-eval] converting AMASS-G1 to ProtoMotions .motion shards"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_dir="${MOTION_BASE}/shard_${shard}"
    shard_pt="${MOTION_BASE}/amass_g1_full_shard_${shard}.pt"
    mkdir -p "${shard_dir}"
    if [[ "${FORCE_CONVERT}" == "1" ]]; then
        rm -rf "${shard_dir}"
        mkdir -p "${shard_dir}"
    fi
    if [[ "${PACKED_SHARDS_REUSABLE}" == "1" && -s "${shard_pt}" ]]; then
        echo "[amass-g1-eval] shard ${shard}: packed shard exists; skipping conversion"
        continue
    fi
    if find "${shard_dir}" -type f -name '*.motion' | grep -q .; then
        echo "[amass-g1-eval] shard ${shard}: conversion exists"
        continue
    fi
    (
        "${TRACKER_PY[@]}" data/scripts/convert_amass_g1_npz_to_proto.py \
            --input-dir "${AMASS_ROOT}" \
            --output-dir "${shard_dir}" \
            --output-fps "${OUTPUT_FPS}" \
            --quat-order "${QUAT_ORDER}" \
            --num-rank "${NUM_SHARDS}" \
            --slurm-rank "${shard}" \
            > "${OUT_ROOT}/convert_shard_${shard}.log" 2>&1
    ) &
    pids+=("$!")
done
wait_for_background "conversion" "${pids[@]}"

echo "[amass-g1-eval] packing motion shards"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_dir="${MOTION_BASE}/shard_${shard}"
    shard_pt="${MOTION_BASE}/amass_g1_full_shard_${shard}.pt"
    if [[ "${PACKED_SHARDS_REUSABLE}" == "1" && -s "${shard_pt}" ]]; then
        echo "[amass-g1-eval] shard ${shard}: packed shard exists"
        continue
    fi
    motion_count=$(find "${shard_dir}" -type f -name '*.motion' | wc -l)
    echo "[amass-g1-eval] shard ${shard}: motion_count=${motion_count}"
    if [[ "${motion_count}" -lt 1 ]]; then
        echo "[amass-g1-eval] ERROR: shard ${shard} has no motions" >&2
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
        echo "[amass-g1-eval] ERROR: checkpoint missing for ${name}: ${ckpt}" >&2
        exit 4
    fi

    eval_dir="${OUT_ROOT}/eval_${name}"
    mkdir -p "${eval_dir}"
    echo "[amass-g1-eval] evaluating ${name}: ${ckpt}"

    pids=()
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
        shard_pt="${MOTION_BASE}/amass_g1_full_shard_${shard}.pt"
        log="${eval_dir}/shard_${shard}.log"
        pred_root="${eval_dir}/predicted_shard_${shard}"
        if [[ "${SAVE_PREDICTED_MOTION_LIB}" == "1" ]]; then
            mkdir -p "${pred_root}"
        fi
        if [[ "${FORCE_EVAL}" != "1" && -s "${log}" ]] && grep -q "EVALUATION RESULTS" "${log}"; then
            echo "[amass-g1-eval] ${name} shard ${shard}: already done"
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
    --num-shards "${NUM_SHARDS}"

if [[ "${SAVE_PREDICTED_MOTION_LIB}" == "1" ]]; then
    python3 scripts/embodied/aggregate_proto_predicted_motion_metrics.py \
        --eval-root "${OUT_ROOT}" \
        --motion-base "${MOTION_BASE}" \
        --num-shards "${NUM_SHARDS}"
fi

echo "[amass-g1-eval] done $(date)"
echo "[amass-g1-eval] summary=${OUT_ROOT}/summary.md"
