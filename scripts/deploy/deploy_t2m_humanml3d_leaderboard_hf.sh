#!/usr/bin/env bash
# Upload the built T2M HumanML3D static leaderboard to a HuggingFace Space.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

SPACE_DIR="${SPACE_DIR:-docs/leaderboards/hf_space_t2m_humanml3d}"
HF_SPACE_ID="${HF_SPACE_ID:?set HF_SPACE_ID, e.g. username/t2m-humanml3d-leaderboard}"

if [[ ! -f "$SPACE_DIR/README.md" || ! -f "$SPACE_DIR/index.html" ]]; then
  echo "[error] expected built Space files under $SPACE_DIR" >&2
  exit 1
fi

hf upload \
  --repo-type space \
  --commit-message "Update T2M HumanML3D leaderboard" \
  "$HF_SPACE_ID" \
  "$SPACE_DIR" \
  .
