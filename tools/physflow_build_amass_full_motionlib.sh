#!/usr/bin/env bash
# Build AMASS full MotionLib for tracker benchmark evaluation.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
AMASS_ROOT="${PHYSFLOW_AMASS_ROOT:-${1:-}}"
OUTPUT_DIR="${PHYSFLOW_AMASS_OUTPUT_DIR:-data/motion_for_trackers}"
YAML_PATH="${PHYSFLOW_AMASS_YAML:-data/yaml_files/amass_smpl_full.yaml}"
DEVICE="${PHYSFLOW_AMASS_DEVICE:-cpu}"
OUTPUT_FILE="${PROJECT_ROOT}/ref_repo/ProtoMotions/${OUTPUT_DIR}/amass_smpl_full.pt"

if [[ -z "${AMASS_ROOT}" ]]; then
  echo "usage: PHYSFLOW_AMASS_ROOT=/path/to/amass bash tools/physflow_build_amass_full_motionlib.sh" >&2
  echo "   or: bash tools/physflow_build_amass_full_motionlib.sh /path/to/amass" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
python3 scripts/embodied/build_amass_full_yaml.py

cd "${PROJECT_ROOT}/ref_repo/ProtoMotions"
python3 data/scripts/convert_amass_to_motionlib.py "${AMASS_ROOT}" "${OUTPUT_DIR}" \
  --motion-config "${YAML_PATH}" \
  --humanoid-type smpl \
  --device "${DEVICE}"

echo "[amass-full] output=${OUTPUT_FILE}"
