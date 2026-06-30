#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-vimogen_1_3b}"
EXPECTED="${2:-4042}"
ROOT="outputs/evaluation/t2m/humanml3d_official_test"
NATIVE="${ROOT}/vimogen276/${METHOD}"
M135="${ROOT}/motion135/${METHOD}"
MS272="${ROOT}/ms272/${METHOD}"

while pgrep -f "scripts/eval/vimogen_t2m_humanml3d.py" >/dev/null; do
  date
  find "${NATIVE}" -maxdepth 1 -name "*.npy" | wc -l
  sleep 300
done

COUNT="$(find "${NATIVE}" -maxdepth 1 -name "*.npy" | wc -l)"
echo "final_count=${COUNT}"
if [[ "${COUNT}" != "${EXPECTED}" ]]; then
  echo "skip_postprocess_count_${COUNT}_expected_${EXPECTED}"
  exit 2
fi

PYTHONPATH=. python3 scripts/eval/convert_vimogen276_to_motionclip135.py \
  --input-root "${NATIVE}" \
  --out-dir "${M135}" \
  --pattern "*.npy" \
  --out-format npz \
  --src-fps 20 \
  --dst-fps 30 \
  --max-frames 300 \
  --overwrite

PYTHONPATH=. python3 scripts/data/convert_motion135_to_h3d272.py \
  --in-dir "${M135}" \
  --out-dir "${MS272}" \
  --pattern "*.npz" \
  --workers 16 \
  --skip-existing
