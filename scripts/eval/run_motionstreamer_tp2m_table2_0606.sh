#!/usr/bin/env bash
# MotionStreamer latent-prefix TP2M rerun for Table 2.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [ ! -d "${ROOT}" ]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PY=${PY:-python3}

OUT=${OUT:-outputs/evaluation/motionstreamer_tp2m_table2_0606}
NUM_GPUS=${NUM_GPUS:-8}
WORKERS=${WORKERS:-16}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
PREFIX_LATENT_SOURCE=${PREFIX_LATENT_SOURCE:-sample}
SAMPLING_METHOD=${SAMPLING_METHOD:-new_demo}
MS_CFG=${MS_CFG:-4.5}
MS_TEMPERATURE=${MS_TEMPERATURE:-1.0}
CAPTION_PROTOCOL=${CAPTION_PROTOCOL:-original}
REWRITTEN_FILE=${REWRITTEN_FILE:-}
T5_MODEL=${T5_MODEL:-}
SPLITS=${SPLITS:-"h3d motionhub"}
CONDS=${CONDS:-"1 5 9"}
GT272_H3D=${GT272_H3D:-${OUT}/gt272_humanml3d}
GT272_MH=${GT272_MH:-${OUT}/gt272_motionhub}

mkdir -p "${OUT}/logs" "${OUT}/metrics" "${GT272_H3D}" "${GT272_MH}"
echo "[start] MotionStreamer TP2M out=${OUT}"

(
  flock 9
  "$PY" - <<'PY' > "${OUT}/logs/install_check.log" 2>&1
missing = []
for mod in ["sentence_transformers", "mmengine", "smplx"]:
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
print("missing", missing)
PY
  if grep -q "missing \\[\\]" "${OUT}/logs/install_check.log"; then
    :
  else
    "$PY" -m pip install -q --upgrade 'pip<25' setuptools wheel >> "${OUT}/logs/install_check.log" 2>&1
    "$PY" -m pip install -q --ignore-installed PyYAML 'numpy<2' sentence-transformers smplx addict yapf rich termcolor >> "${OUT}/logs/install_check.log" 2>&1
    "$PY" -m pip install -q --no-deps mmengine >> "${OUT}/logs/install_check.log" 2>&1
  fi
) 9>/tmp/motionstreamer_tp2m_pip_install.lock

echo "[h3d-build-gt272-direct-smpl] $(date)"
"$PY" scripts/data/convert_motion135_to_h3d272.py \
  --anno-file data/annotation/test_hml3d.json \
  --data-dir data/motionhub \
  --out-dir "${GT272_H3D}" \
  --workers "${WORKERS}" \
  --skip-existing \
  > "${OUT}/logs/build_gt_h3d_272_direct.log" 2>&1

echo "[mh-build-gt272-direct-smpl] $(date)"
"$PY" scripts/data/convert_motion135_to_h3d272.py \
  --anno-file data/annotation/test_motionhub_t2m.json \
  --data-dir data/motionhub \
  --out-dir "${GT272_MH}" \
  --workers "${WORKERS}" \
  --skip-existing \
  > "${OUT}/logs/build_gt_mh_272_direct.log" 2>&1

run_gen() {
  local split="$1"
  local anno="$2"
  local gt272="$3"
  local cond="$4"
  local out_root="${OUT}/${split}"
  mkdir -p "${out_root}" "${OUT}/logs/${split}_cond${cond}"
  echo "[${split}-cond${cond}-gen] $(date)"
  local gen_args=(
    --dataset "${split/h3d/humanml3d}"
    --out-dir "${out_root}"
    --gt-272-dir "${gt272}"
    --condition-num-frames "${cond}"
    --anno-file "${anno}"
    --data-dir data/motionhub
    --caption-protocol "${CAPTION_PROTOCOL}"
    --prefix-latent-source "${PREFIX_LATENT_SOURCE}"
    --sampling-method "${SAMPLING_METHOD}"
    --cfg "${MS_CFG}"
    --temperature "${MS_TEMPERATURE}"
    --align-to-gt-root
    --align-root-mode yaw
    --skip-existing
  )
  if [ -n "${T5_MODEL}" ]; then
    gen_args+=(--t5-model "${T5_MODEL}")
  fi
  if [ -n "${REWRITTEN_FILE}" ]; then
    gen_args+=(--rewritten-file "${REWRITTEN_FILE}")
  fi
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES="${i}" "$PY" scripts/eval/gen_motionstreamer_tp2m_smpl_npz.py \
      "${gen_args[@]}" \
      --num-shards "${NUM_GPUS}" \
      --shard-index "${i}" \
      > "${OUT}/logs/${split}_cond${cond}/gen_s${i}.log" 2>&1 &
  done
  wait
  echo "[${split}-cond${cond}-gen-done] npz=$(find "${out_root}/cond${cond}_latent_prefix" -maxdepth 1 -name '*.npz' | wc -l)"
}

run_eval() {
  local split="$1"
  local anno="$2"
  local cond="$3"
  local gpu="$4"
  echo "[${split}-cond${cond}-eval] $(date)"
  CUDA_VISIBLE_DEVICES="${gpu}" "$PY" scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${OUT}/${split}/cond${cond}_latent_prefix" \
    --chunk_size "${CHUNK_SIZE}" \
    --out_json "${OUT}/metrics/${split}_cond${cond}_motionstreamer_c64.json" \
    --n_repeats "${N_REPEATS}" \
    --seed 42 \
    > "${OUT}/logs/${split}_cond${cond}/eval.log" 2>&1
}

want_item() {
  local needle="$1"
  local haystack="$2"
  for item in ${haystack}; do
    if [ "${item}" = "${needle}" ]; then
      return 0
    fi
  done
  return 1
}

for cond in ${CONDS}; do
  if want_item h3d "${SPLITS}"; then
    run_gen h3d data/annotation/test_hml3d.json "${GT272_H3D}" "${cond}"
  fi
  if want_item motionhub "${SPLITS}"; then
    run_gen motionhub data/annotation/test_motionhub_t2m.json "${GT272_MH}" "${cond}"
  fi
  if want_item h3d "${SPLITS}"; then
    run_eval h3d data/annotation/test_hml3d.json "${cond}" 0 &
  fi
  if want_item motionhub "${SPLITS}"; then
    run_eval motionhub data/annotation/test_motionhub_t2m.json "${cond}" 1 &
  fi
  wait
done

"$PY" - <<'PY'
import json
from pathlib import Path

import os
root = Path(os.environ.get("OUT", "outputs/evaluation/motionstreamer_tp2m_table2_0606")) / "metrics"
for path in sorted(root.glob("*.json")):
    d = json.loads(path.read_text())
    print(path.name, {
        "samples": d.get("samples"),
        "r3": d.get("r_precision_pred_top3_mean"),
        "fid": d.get("fid_mean"),
        "mm": d.get("mm_dist_pred_mean"),
        "div": d.get("diversity_pred_mean"),
    })
PY

echo "[done] $(date)"
