#!/usr/bin/env bash
# ViMoGen small A/B over denoising strength and duration prompt.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BASE=${BASE:-outputs/evaluation/vimogen_t2m_0605}
MAX_SAMPLES=${MAX_SAMPLES:-64}
GPU_OFFSET=${GPU_OFFSET:-0}
mkdir -p "${BASE}/driver_ab_smoke/logs"

run_one() {
  local tag="$1"
  local gpu="$2"
  local denoise="$3"
  local append_duration="$4"
  local physical_gpu=$((gpu + GPU_OFFSET))
  {
    echo "[${tag}] start $(date)"
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    ENC_GPU=0 \
    DATASET=h3d \
    MAX_SAMPLES="${MAX_SAMPLES}" \
    TAG="${tag}" \
    NUM_SHARDS=1 \
    SHARD_IDX=0 \
    NPROC=1 \
    TEST_BS=4 \
    STEPS=50 \
    CFG=5.0 \
    DENOISING_STRENGTH="${denoise}" \
    DTYPE=fp16 \
    SKIP_EVAL=0 \
    EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-32}" \
    APPEND_DURATION_TO_PROMPT="${append_duration}" \
    CAPTION_OVERRIDE_JSON=data/annotation/test_hml3d_rewritten.json \
    VIMOGEN_DST_FPS=20 \
    MASTER_PORT="$((32100 + physical_gpu))" \
    bash scripts/eval/run_vimogen_t2m_eval_0605.sh
    echo "[${tag}] done $(date)"
  } > "${BASE}/driver_ab_smoke/logs/${tag}.log" 2>&1
}

run_one ab_dn07_dur0 0 0.7 0 &
run_one ab_dn07_dur1 1 0.7 1 &
run_one ab_dn10_dur0 2 1.0 0 &
run_one ab_dn10_dur1 3 1.0 1 &
wait

python3 - <<'PY' | tee "${BASE}/driver_ab_smoke/summary.txt"
import json
from pathlib import Path

base = Path("outputs/evaluation/vimogen_t2m_0605")
for tag in ("ab_dn07_dur0", "ab_dn07_dur1", "ab_dn10_dur0", "ab_dn10_dur1"):
    p = base / f"h3d_{tag}" / "metrics_motionclip.json"
    if not p.exists():
        print(tag, "missing")
        continue
    d = json.load(open(p))
    print(
        tag,
        "samples", d.get("samples"),
        "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
        "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
        "FID", f"{d.get('fid_mean', float('nan')):.4f}",
        "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
        "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
    )
PY
