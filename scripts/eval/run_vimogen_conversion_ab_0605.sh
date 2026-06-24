#!/usr/bin/env bash
# Conversion-only A/B for an already generated ViMoGen run.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [ ! -d "${ROOT}" ]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY=${PY:-python3}

BASE=${BASE:-outputs/evaluation/vimogen_t2m_0605}
RUN=${RUN:-h3d_seq_dn07_dur0}
DATASET=${DATASET:-h3d}
GPU=${GPU:-4}
CHUNK_SIZE=${CHUNK_SIZE:-32}

if [[ "${DATASET}" == "h3d" ]]; then
  ANNO=${ANNO:-data/annotation/test_hml3d.json}
elif [[ "${DATASET}" == "mh" ]]; then
  ANNO=${ANNO:-data/annotation/test_motionhub_t2m.json}
else
  echo "Unsupported DATASET=${DATASET}" >&2
  exit 2
fi

IN_ROOT="${BASE}/${RUN}/vimogen_exp/test_visualization/${RUN}"
CAPMAP=${CAPMAP:-ref_repo/ViMoGen/data/eval/${RUN}/vimogen_${DATASET}_captions.json}
SUMMARY=${SUMMARY:-${BASE}/driver_conv_ab_0605_summary.txt}
mkdir -p "$(dirname "${SUMMARY}")"
: > "${SUMMARY}"

run_one() {
  local conv="$1"
  local rfv="$2"
  local tag="conv_${conv}_rfv${rfv}"
  local out="${BASE}/${RUN}_${tag}"
  mkdir -p "${out}/logs"

  local args=(
    --vimogen-root ref_repo/ViMoGen
    --input-root "${IN_ROOT}"
    --out-dir "${out}/motionclip135"
    --src-fps 20
    --dst-fps 20
    --coord-conversion "${conv}"
    --overwrite
  )
  if [[ "${rfv}" == "0" ]]; then
    args+=(--no-recover-from-velocity)
    args+=(--no-equal-length)
  fi

  echo "[${tag}] convert $(date)"
  CUDA_VISIBLE_DEVICES="${GPU}" "$PY" scripts/eval/convert_vimogen276_to_motionclip135.py "${args[@]}" \
    > "${out}/logs/convert.log" 2>&1

  echo "[${tag}] eval $(date)"
  CUDA_VISIBLE_DEVICES="${GPU}" "$PY" scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${ANNO}" \
    --data_dir data/motionhub \
    --pred_dir "${out}/motionclip135" \
    --rewritten_caption_file "${CAPMAP}" \
    --out_json "${out}/metrics_motionclip.json" \
    --forward_batch_size 64 \
    --chunk_size "${CHUNK_SIZE}" \
    --n_repeats 20 \
    > "${out}/logs/eval_motionclip.log" 2>&1

  TAG="${tag}" METRICS="${out}/metrics_motionclip.json" "$PY" - <<'PY' >> "${SUMMARY}"
import json
import os

d = json.load(open(os.environ["METRICS"]))
print(
    os.environ["TAG"],
    "samples", d.get("samples"),
    "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
    "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
    "FID", f"{d.get('fid_mean', float('nan')):.4f}",
    "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
    "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
)
PY
}

for conv in mbench none; do
  for rfv in 1 0; do
    run_one "${conv}" "${rfv}"
  done
done

echo "[done] $(date)"
