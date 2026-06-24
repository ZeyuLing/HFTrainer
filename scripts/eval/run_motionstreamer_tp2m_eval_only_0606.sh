#!/usr/bin/env bash
# Eval existing MotionStreamer TP2M outputs without regenerating motions.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [ ! -d "${ROOT}" ]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PY=${PY:-python3}

NUM_GPUS=${NUM_GPUS:-1}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
OUT_SUFFIX=${OUT_SUFFIX:-evalrerun}

# Semicolon-separated specs: out_dir:split:condition:annotation_file
SPECS=${SPECS:-}
if [ -z "${SPECS}" ]; then
  echo "Missing SPECS, e.g. OUT:motionhub:1:data/annotation/test_motionhub_t2m.json" >&2
  exit 2
fi

(
  flock 9
  "$PY" - <<'PY' >/tmp/motionstreamer_tp2m_eval_deps.log 2>&1
missing = []
for mod in ["mmengine", "smplx", "torchgeometry", "einops", "sentence_transformers"]:
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
print("missing", missing)
PY
  if grep -q "missing \\[\\]" /tmp/motionstreamer_tp2m_eval_deps.log; then
    :
  else
    "$PY" -m pip install -q 'numpy<2' mmengine smplx torchgeometry einops sentence-transformers \
      >>/tmp/motionstreamer_tp2m_eval_deps.log 2>&1
  fi
) 9>/tmp/motionstreamer_tp2m_eval_pip_install.lock

IFS=';' read -r -a specs <<< "${SPECS}"
pids=()
job_idx=0
for spec in "${specs[@]}"; do
  [ -z "${spec}" ] && continue
  IFS=':' read -r out split cond anno <<< "${spec}"
  if [ -z "${out:-}" ] || [ -z "${split:-}" ] || [ -z "${cond:-}" ] || [ -z "${anno:-}" ]; then
    echo "Bad spec: ${spec}" >&2
    exit 2
  fi
  gen_dir="${out}/${split}/cond${cond}_latent_prefix"
  out_json="${out}/metrics/${split}_cond${cond}_motionstreamer_c64_${OUT_SUFFIX}.json"
  log_dir="${out}/logs/${split}_cond${cond}"
  mkdir -p "${out}/metrics" "${log_dir}"
  if [ ! -d "${gen_dir}" ]; then
    echo "Missing pred dir: ${gen_dir}" >&2
    exit 2
  fi
  gpu=$((job_idx % NUM_GPUS))
  echo "[eval-launch] gpu=${gpu} split=${split} cond=${cond} pred=${gen_dir}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "$PY" scripts/eval/eval_with_motionclip_evaluator.py \
      --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
      --anno_file "${anno}" \
      --data_dir data/motionhub \
      --pred_dir "${gen_dir}" \
      --chunk_size "${CHUNK_SIZE}" \
      --out_json "${out_json}" \
      --n_repeats "${N_REPEATS}" \
      --seed 42 \
      > "${log_dir}/eval_${OUT_SUFFIX}.log" 2>&1
    echo "[eval-done] ${out_json}"
  ) &
  pids+=("$!")
  job_idx=$((job_idx + 1))
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

"$PY" - <<'PY'
import json
import os
from pathlib import Path

suffix = os.environ.get("OUT_SUFFIX", "evalrerun")
for spec in os.environ["SPECS"].split(";"):
    if not spec:
        continue
    out, split, cond, _anno = spec.split(":", 3)
    path = Path(out) / "metrics" / f"{split}_cond{cond}_motionstreamer_c64_{suffix}.json"
    d = json.loads(path.read_text())
    print(path, {
        "samples": d.get("samples"),
        "r3": d.get("r_precision_pred_top3_mean"),
        "fid": d.get("fid_mean"),
        "mm": d.get("mm_dist_pred_mean"),
        "div": d.get("diversity_pred_mean"),
    })
PY
