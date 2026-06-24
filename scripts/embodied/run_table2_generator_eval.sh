#!/usr/bin/env bash
# Run the fixed Table-2 generator evaluation for one G1 generator checkpoint.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
OUT="${OUT:?OUT is required}"
CONFIG="${CONFIG:-configs/physflow/hymotion_g1_t2m_38dim_long.py}"
ANNO="${ANNO:-data/annotation/_heldout_agile.json}"
STYLE_BANK="${STYLE_BANK:-data/g1_style_bank/heldout_agile_80.npz}"
NUM_SAMPLES="${NUM_SAMPLES:-80}"
MAX_ITEMS="${MAX_ITEMS:-80}"
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
