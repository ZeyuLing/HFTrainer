#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/humanml3d/motiongpt_official_cond_smoke_0605}
SRC_ROOT=${SRC_ROOT:-outputs/evaluation/humanml3d/motiongpt_official_h3d263_repro_0605}
RECON_ROOT=${RECON_ROOT:-work_dirs/h3d263_eval/h3d263_test_recon_fk}
SRC_H3D272=${SRC_H3D272:-ref_repo/MotionStreamer/MotionStreamer/humanml3d_272}
MAX_SAMPLES=${MAX_SAMPLES:-256}
NUM_REPEATS=${NUM_REPEATS:-3}
GPU=${GPU:-4}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/pred"

CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/motiongpt_infer_hml3d263.py \
  --anno-file "${SRC_ROOT}/official_h3d263_test_anno.json" \
  --caption-file "${SRC_ROOT}/official_h3d263_first_caption.json" \
  --out-dir "${OUT_ROOT}/pred" \
  --num-shards 1 \
  --shard-index 0 \
  --batch-size "${BATCH_SIZE:-16}" \
  --gt-fps 20 \
  --model-fps 20 \
  --prompt-mode "${PROMPT_MODE:-official_nolen}" \
  --instruction-key "${INSTRUCTION_KEY:-caption_framelen}" \
  --max-samples "${MAX_SAMPLES}" \
  --seed "${SEED:-42}" \
  --debug-generations \
  ${RESTORE_T5_LOGIT_SCALE:+--restore-t5-logit-scale} \
  ${TIE_WORD_EMBEDDINGS:+--tie-word-embeddings} \
  > "${OUT_ROOT}/logs/infer.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/eval_momask_native_h3d263.py \
  --recon_root "${RECON_ROOT}" \
  --src_h3d272 "${SRC_H3D272}" \
  --momask_root ref_repo/Momask/momask-codes \
  --mode pred \
  --pred_dir "${OUT_ROOT}/pred" \
  --num_repeats "${NUM_REPEATS}" \
  --drop_mirrored \
  --caption_selection first \
  --max_samples "${MAX_SAMPLES}" \
  --output "${OUT_ROOT}/eval_momask_native_first_rep${NUM_REPEATS}.json" \
  > "${OUT_ROOT}/logs/eval.log" 2>&1

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
p = Path("${OUT_ROOT}/eval_momask_native_first_rep${NUM_REPEATS}.json")
d = json.load(open(p))
print(
    "samples", d.get("n_samples"),
    "R1", f"{d['r_precision']['mean'][0]:.4f}",
    "R3", f"{d['r_precision']['mean'][2]:.4f}",
    "FID", f"{d['fid']['mean']:.4f}",
    "MM", f"{d['matching_score']['mean']:.4f}",
    "Div", f"{d['diversity']['mean']:.4f}",
)
PY
