#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
cd "$ROOT"

SUITE=${SUITE:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_epoch43_translation_decode_t2m_20260629}
mkdir -p "$SUITE/logs"

echo "[stop] before"
pgrep -af "eval_prism_kafs_ablation.py.*prism_epoch43_translation_decode_t2m_20260629" || true
pgrep -af "run_prism_t2m_translation_decode_ablation_20260629.sh" || true

pkill -f "eval_prism_kafs_ablation.py.*prism_epoch43_translation_decode_t2m_20260629" || true
pkill -f "run_prism_t2m_translation_decode_ablation_20260629.sh" || true
sleep 3

echo "[stop] after"
pgrep -af "eval_prism_kafs_ablation.py.*prism_epoch43_translation_decode_t2m_20260629" || true
pgrep -af "run_prism_t2m_translation_decode_ablation_20260629.sh" || true
if [[ "${RESET_OUTPUTS:-0}" == "1" ]]; then
  echo "[stop] reset partial outputs"
  rm -rf "$SUITE/raw" "$SUITE/prep" "$SUITE/results" "$SUITE/analysis"
  rm -f "$SUITE/_GEN_DONE" "$SUITE/_EVAL_DONE" "$SUITE/driver.pid" "$SUITE/summary_translation_decode.json"
  mkdir -p "$SUITE"/{raw,prep,results,analysis,logs}
fi
echo "[stop] done"
