#!/usr/bin/env bash
# Build and evaluate the GT -> HML3D-263 -> SMPL control row.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

METHOD=${METHOD:-gt_hml263_control}
BASE_263=${BASE_263:-outputs/evaluation/humanml3d/gt_smpl135_to_hml263}
H3D_SRC=${H3D_SRC:-${BASE_263}/humanml3d}
MH_SRC=${MH_SRC:-${BASE_263}/motionhub}
EVAL_ROOT=${EVAL_ROOT:-outputs/evaluation/${METHOD}_rw_c64_0605}
SMPL_ROOT=${SMPL_ROOT:-outputs/evaluation/${METHOD}_smpl135_0605}
MC135_ROOT=${MC135_ROOT:-outputs/evaluation/${METHOD}_smpl135_0605_motionclip135}
NUM_SHARDS=${NUM_SHARDS:-8}
CONVERT_WORKERS=${CONVERT_WORKERS:-16}
LOGDIR="${EVAL_ROOT}/logs"

mkdir -p "${LOGDIR}" "${H3D_SRC}" "${MH_SRC}"

echo "[start] method=${METHOD} $(date)" | tee "${LOGDIR}/run.log"

echo "[convert-h3d] $(date)" | tee -a "${LOGDIR}/run.log"
python3 scripts/eval/build_gt_smpl135_to_hml263.py \
  --anno-file data/annotation/test_hml3d.json \
  --data-dir data/motionhub \
  --out-dir "${H3D_SRC}" \
  --workers "${CONVERT_WORKERS}" \
  --skip-existing \
  > "${LOGDIR}/convert_h3d.log" 2>&1

echo "[convert-mh] $(date)" | tee -a "${LOGDIR}/run.log"
python3 scripts/eval/build_gt_smpl135_to_hml263.py \
  --anno-file data/annotation/test_motionhub_t2m.json \
  --data-dir data/motionhub \
  --out-dir "${MH_SRC}" \
  --workers "${CONVERT_WORKERS}" \
  --skip-existing \
  > "${LOGDIR}/convert_mh.log" 2>&1

echo "[retarget-eval] $(date)" | tee -a "${LOGDIR}/run.log"
METHOD="${METHOD}" \
  H3D_SRC="${H3D_SRC}" \
  MH_SRC="${MH_SRC}" \
  EVAL_ROOT="${EVAL_ROOT}" \
  SMPL_ROOT="${SMPL_ROOT}" \
  MC135_ROOT="${MC135_ROOT}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  bash scripts/eval/run_hml263_method_rw_c64_eval_0605.sh \
  > "${LOGDIR}/postprocess_eval.log" 2>&1

echo "[done] $(date)" | tee -a "${LOGDIR}/run.log"
