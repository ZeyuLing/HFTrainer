#!/usr/bin/env bash
# Run the fixed Table-2 generator evaluation for one G1 generator checkpoint.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
OUT="${OUT:?OUT is required}"
CONFIG="${CONFIG:-configs/physflow/hymotion_g1_t2m_38dim_long.py}"
ANNO="${ANNO:-data/annotation/_heldout_agile_ground_only.json}"
# Style-bank diagnostics are optional; do not enable the custom robot-style FID
# by default because it is not a main-paper distribution metric.
STYLE_BANK="${STYLE_BANK:-}"
NUM_SAMPLES="${NUM_SAMPLES:-60}"
MAX_ITEMS="${MAX_ITEMS:-60}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GUIDANCE="${GUIDANCE:-1.0}"
SEED="${SEED:-20260623}"
POLL_SECONDS="${POLL_SECONDS:-300}"

while [[ ! -f "${CHECKPOINT}/model.pt" ]]; do
  echo "[table2-eval] waiting for ${CHECKPOINT}/model.pt at $(date)"
  sleep "${POLL_SECONDS}"
done

bash scripts/embodied/physflow_g1_eval_node.sh \
  scripts/embodied/eval_hymotion_g1_checkpoint_frozen.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --out "${OUT}" \
  --anno-override "${ANNO}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-items "${MAX_ITEMS}" \
  --sample-steps "${SAMPLE_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --guidance "${GUIDANCE}" \
  --seed "${SEED}" \
  --score-gt \
  --style-bank "${STYLE_BANK}"
