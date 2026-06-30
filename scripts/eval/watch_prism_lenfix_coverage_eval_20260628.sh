#!/usr/bin/env bash
# Wait for the PRISM epoch-43 official-selected length-fix generation suite to
# reach full raw coverage, then run validation, repack, and MS-272 evaluation.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SUITE="${SUITE:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_epoch43_official_selected_lenfix_20260628}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TARGET="${TARGET:-4042}"
mkdir -p "$SUITE/_remote_logs"

while true; do
  date -Is
  ok=1
  for policy in direct_len pad360_crop; do
    for mode in depth_driven none uniform random; do
      dir="$SUITE/raw/$policy/$mode"
      count=$(find "$dir" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
      echo "raw/$policy/$mode=$count/$TARGET"
      if [[ "$count" -lt "$TARGET" ]]; then
        ok=0
      fi
    done
  done
  if [[ "$ok" -eq 1 ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

python3 scripts/eval/validate_repack_eval_prism_lenfix_20260628.py --skip-eval-existing
