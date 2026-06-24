#!/usr/bin/env bash
# Run the official HumanML3D/MoMask evaluator for HML3D-263 prediction dirs.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/native_hml263_diag_0606}
RECON_ROOT=${RECON_ROOT:-work_dirs/h3d263_eval/h3d263_test_recon_fk}
SRC_H3D272=${SRC_H3D272:-ref_repo/MotionStreamer/MotionStreamer/humanml3d_272}
MOMASK_ROOT=${MOMASK_ROOT:-ref_repo/Momask/momask-codes}
NUM_REPEATS=${NUM_REPEATS:-20}
CAPTION_SELECTION=${CAPTION_SELECTION:-first}
GPU=${GPU:-0}
MAX_SAMPLES=${MAX_SAMPLES:-}

mkdir -p "${OUT_ROOT}/logs"

run_one() {
  local name="$1"
  local pred_dir="$2"
  local out_json="${OUT_ROOT}/${name}.json"
  local extra=()
  if [ -n "${MAX_SAMPLES}" ]; then
    extra+=(--max_samples "${MAX_SAMPLES}")
  fi
  echo "[eval] ${name} pred=${pred_dir} $(date)" | tee -a "${OUT_ROOT}/logs/run.log"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/eval_momask_native_h3d263.py \
    --recon_root "${RECON_ROOT}" \
    --src_h3d272 "${SRC_H3D272}" \
    --momask_root "${MOMASK_ROOT}" \
    --mode pred \
    --pred_dir "${pred_dir}" \
    --num_repeats "${NUM_REPEATS}" \
    --drop_mirrored \
    --caption_selection "${CAPTION_SELECTION}" \
    "${extra[@]}" \
    --output "${out_json}" \
    > "${OUT_ROOT}/logs/${name}.log" 2>&1
}

if [ "$#" -lt 2 ] || [ $(( $# % 2 )) -ne 0 ]; then
  echo "Usage: $0 <name> <pred_dir> [<name> <pred_dir> ...]" >&2
  exit 2
fi

while [ "$#" -gt 0 ]; do
  run_one "$1" "$2"
  shift 2
done

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
for p in sorted(root.glob("*.json")):
    d = json.load(open(p))
    print(
        p.stem,
        "samples", d.get("n_samples"),
        "R1", f"{d['r_precision']['mean'][0]:.4f}",
        "R3", f"{d['r_precision']['mean'][2]:.4f}",
        "FID", f"{d['fid']['mean']:.4f}",
        "MM", f"{d['matching_score']['mean']:.4f}",
        "Div", f"{d['diversity']['mean']:.4f}",
    )
PY

touch "${OUT_ROOT}/_DONE"
