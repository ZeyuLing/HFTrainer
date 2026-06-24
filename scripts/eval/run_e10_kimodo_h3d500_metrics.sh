#!/usr/bin/env bash
# Table 6 Exp A: KIMODO E10 H3D-aligned raw -> rotation retarget -> eval npz
# -> Ctrl.Err / Foot / Jitter / multi-seed FID.
#
# IMPORTANT:
# - Use H3D-aligned raw source only:
#     output/evaluation/paper_baseline_kimodo_e10_h3d500/E10/E10_<setting>/npz
# - Use eval_h3d_editing.json, matching \ours E10 ordering/captions.
# - Do NOT use old paper_baseline_kimodo/E10 or e10n400 small-datalist outputs.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

RAW_ROOT=output/evaluation/paper_baseline_kimodo_e10_h3d500/E10
OUT_ROOT=output/evaluation/bodypart_table6_rot/kimodo_h3d500
DATA_FILE=data/eval/m2m_v2/eval_h3d_editing.json
FILT='Warning|deprecat|ds_accelerator|def (forward|backward)|from torch|body_model|mesh_dist|tbs_axes|warnings.warn|torch.load'

SETTINGS=(
  "A_upper:upper"
  "B_lower:lower"
  "C_spine_only:spine_only"
  "D_arms_only:arms_only"
  "E_legs_only:legs_only"
  "F_left_arm:left_arm"
  "G_right_arm:right_arm"
  "H_left_leg:left_leg"
  "I_right_leg:right_leg"
  "J_feet_only:feet_only"
  "K_no_feet:no_feet"
)

for pair in "${SETTINGS[@]}"; do
  setting="${pair%%:*}"
  key="${pair##*:}"
  raw_dir="${RAW_ROOT}/E10_${setting}/npz"
  smplx_dir="${OUT_ROOT}/${setting}/smplx"
  eval_dir="${OUT_ROOT}/${setting}/npz"
  metrics_dir="${OUT_ROOT}/${setting}/metrics"

  echo "==================== ${setting} key=${key} ===================="
  raw_count=$(ls "${raw_dir}"/*.npz 2>/dev/null | wc -l)
  echo "[raw] ${raw_dir} count=${raw_count}"
  if [ "${raw_count}" -lt 64 ]; then
    echo "[skip] too few raw npz for stable eval: ${raw_count}"
    continue
  fi

  mkdir -p "${smplx_dir}" "${eval_dir}" "${metrics_dir}"

  python3 scripts/eval/kimodo_soma_to_smpl_byid.py \
    --data-file "${DATA_FILE}" --max-samples 500 \
    --raw-npz-dir "${raw_dir}" \
    --out-dir "${smplx_dir}" \
    --mode rotation --device cpu --num-shards 1 --shard-index 0 \
    2>&1 | grep -vE "${FILT}" | tail -5

  python3 scripts/eval/build_e10_kimodo_eval_npz.py \
    --smplx-dir "${smplx_dir}" \
    --setting-key "${key}" \
    --out-dir "${eval_dir}" \
    --data-file "${DATA_FILE}" \
    --max-samples 500 \
    2>&1 | grep -vE "${FILT}" | tail -3

  python3 scripts/eval/e10_ctrl_err.py \
    --npz-dir "${eval_dir}" \
    --tag "kimodo_h3d500_${setting}" \
    --out-json "${metrics_dir}/ctrlerr.json" \
    2>&1 | grep -vE "${FILT}" | tail -3

  python3 scripts/eval/paper_npz_ric_mpjpe.py \
    --npz-dir "${eval_dir}" \
    --tag "kimodo_h3d500_${setting}" \
    --out-json "${metrics_dir}/ric.json" \
    2>&1 | grep -vE "${FILT}" | tail -5
done

FID_ARGS=()
for pair in "${SETTINGS[@]}"; do
  setting="${pair%%:*}"
  eval_dir="${OUT_ROOT}/${setting}/npz"
  if [ -d "${eval_dir}" ]; then
    FID_ARGS+=("${setting}=${eval_dir}")
  fi
done

if [ "${#FID_ARGS[@]}" -gt 0 ]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python3 scripts/eval/e10_fid_multiseed.py \
    --seeds 0 1 2 3 4 5 6 \
    --out-json "${OUT_ROOT}/all_fid_multiseed.json" \
    --dirs "${FID_ARGS[@]}" \
    2>&1 | grep -vE "${FILT}" | tee "${OUT_ROOT}/fid_multiseed.log"
fi

echo "H3D500_METRICS_DONE"
