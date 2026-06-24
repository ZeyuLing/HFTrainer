#!/usr/bin/env bash
set -euo pipefail

# PhysFlow tracker upgrade path on the Any2Track/OpenTrack training stack.
# Stage hard motions -> train an adversarial specialist -> export ONNX -> distill
# with the released LAFAN specialists through DAgger.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPENTRACK_ROOT="${OPENTRACK_ROOT:-${PROJECT_ROOT}/ref_repo/OpenTrack}"
TAG="${TAG:-physflow_adv_$(date +%m%d%H%M)}"
ADV_SOURCE_DIR="${ADV_SOURCE_DIR:-${PROJECT_ROOT}/output/opentrack_amass_g1/debug2_20260604_1915_wait_proto_wxyz/UnitreeG1}"
ADV_KEYWORDS="${ADV_KEYWORDS-jump,fall,getup,run,sprint}"
ADV_MAX_FILES="${ADV_MAX_FILES:-96}"
ADV_SELECTION_STRATEGY="${ADV_SELECTION_STRATEGY:-first}"
ADV_SELECTION_SEED="${ADV_SELECTION_SEED:-0}"
ADV_PROB="${ADV_PROB:-0.25}"
STAGE_MODE="${STAGE_MODE:-symlink}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-2000000000}"
BASE_TEACHER_CKPT_DIR="${BASE_TEACHER_CKPT_DIR:-storage/logs/dagger/general_tracker_lafan1_v2}"
BASE_TEACHER_ONNX_PATH="${BASE_TEACHER_ONNX_PATH:-storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/model.onnx}"

MANIFEST_DIR="${PROJECT_ROOT}/output/opentrack_physflow_adversarial/${TAG}"
MANIFEST_JSON="${MANIFEST_DIR}/adversarial_motions.json"
MANIFEST_TXT="${MANIFEST_DIR}/adversarial_motions.txt"
SPECIALIST_EXP="${SPECIALIST_EXP:-${TAG}_specialist}"
DAGGER_EXP="${DAGGER_EXP:-${TAG}_dagger}"
DAGGER_CONFIG="${MANIFEST_DIR}/dagger_physflow_adversarial.json"
WARMSTART_PTH="${MANIFEST_DIR}/general_tracker_lafan1_v2.pth"

cd "${PROJECT_ROOT}"
python3 scripts/embodied/stage_opentrack_adversarial_motions.py \
  --input-dir "${ADV_SOURCE_DIR}" \
  --manifest-json "${MANIFEST_JSON}" \
  --manifest-txt "${MANIFEST_TXT}" \
  --keywords "${ADV_KEYWORDS}" \
  --max-files "${ADV_MAX_FILES}" \
  --selection-strategy "${ADV_SELECTION_STRATEGY}" \
  --selection-seed "${ADV_SELECTION_SEED}" \
  --mode "${STAGE_MODE}" \
  --force

cd "${OPENTRACK_ROOT}"
export GLI_PATH="${GLI_PATH:-${OPENTRACK_ROOT}}"
python -m track_mj.learning.train.train_ppo_track \
  --task G1TrackingGeneralDR \
  --exp-name "${SPECIALIST_EXP}" \
  --trajectory-manifest "${MANIFEST_TXT}" \
  --trajectory-dataset-name lafan1 \
  --num-timesteps "${NUM_TIMESTEPS}"

SPECIALIST_LOGDIR="$(find storage/logs/track -maxdepth 1 -type d -name "*_${SPECIALIST_EXP}" | sort | tail -1)"
if [[ -z "${SPECIALIST_LOGDIR}" ]]; then
  echo "Could not locate specialist logdir for ${SPECIALIST_EXP}" >&2
  exit 1
fi

python -m track_mj.eval.tracking.export_onnx \
  --task G1TrackingGeneralDR \
  --exp_name "$(basename "${SPECIALIST_LOGDIR}")"

SPECIALIST_ONNX="$(find "${SPECIALIST_LOGDIR}/checkpoints" -name policy.onnx | sort | tail -1)"
if [[ -z "${SPECIALIST_LOGDIR}" || -z "${SPECIALIST_ONNX}" ]]; then
  echo "Could not locate specialist checkpoint/ONNX for ${SPECIALIST_EXP}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
python3 scripts/embodied/opentrack_onnx_to_dagger_pth.py \
  --onnx "${OPENTRACK_ROOT}/${BASE_TEACHER_ONNX_PATH}" \
  --out-pth "${WARMSTART_PTH}" \
  --force

python3 scripts/embodied/build_opentrack_physflow_adversarial_dagger.py \
  --motion-manifest "${MANIFEST_JSON}" \
  --teacher-ckpt-dir "${SPECIALIST_LOGDIR}" \
  --teacher-onnx-path "${SPECIALIST_ONNX}" \
  --base-teacher-ckpt-dir "${BASE_TEACHER_CKPT_DIR}" \
  --base-teacher-onnx-path "${BASE_TEACHER_ONNX_PATH}" \
  --out-config "${DAGGER_CONFIG}" \
  --adversarial-prob "${ADV_PROB}"

cd "${OPENTRACK_ROOT}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NUM_GPUS}" -m track_mj.learning.train.train_dagger \
    --task G1TrackingGeneralDR \
    --exp-name "${DAGGER_EXP}" \
    --dagger-config-path "${DAGGER_CONFIG}" \
    --load-pretrained-path "${WARMSTART_PTH}" \
    --use-ddp
else
  python -m track_mj.learning.train.train_dagger \
    --task G1TrackingGeneralDR \
    --exp-name "${DAGGER_EXP}" \
    --dagger-config-path "${DAGGER_CONFIG}" \
    --load-pretrained-path "${WARMSTART_PTH}"
fi

DAGGER_LOGDIR="$(find storage/logs/dagger -maxdepth 1 -type d -name "*_${DAGGER_EXP}" | sort | tail -1)"
if [[ -z "${DAGGER_LOGDIR}" ]]; then
  echo "Could not locate DAgger logdir for ${DAGGER_EXP}" >&2
  exit 1
fi

python -m track_mj.eval.dagger.torch2onnx \
  --ckpt-dir "${DAGGER_LOGDIR}"
