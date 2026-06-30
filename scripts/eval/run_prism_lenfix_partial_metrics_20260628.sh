#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_epoch43_official_selected_lenfix_20260628}"
VIEW="${VIEW:-${ROOT}/viewer_partial_before_after}"
OUT="${OUT:-${ROOT}/partial_metrics_82}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${OUT}"
export PYTHONPATH="${PYTHONPATH:-.}:."

run_one() {
  local tag="$1"
  local dir_name="$2"
  echo "[run] ${tag} $(date)"
  "${PYTHON_BIN}" scripts/eval/eval_motionstreamer_272.py \
    --split "${VIEW}/ids.txt" \
    --pred-dir "${VIEW}/${dir_name}" \
    --tag "prism_depth_${tag}" \
    --also-refk \
    --min-motion-len 1 \
    --out-json "${OUT}/${tag}.json" \
    --device "${DEVICE}" 2>&1 | tee "${OUT}/${tag}.log"
}

run_one old prism_old_table6_depth
run_one direct_len prism_direct_len_depth
run_one pad360_crop prism_pad360_depth

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

root = Path(os.environ.get(
    "OUT",
    "outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_epoch43_official_selected_lenfix_20260628/partial_metrics_82",
))
rows = []
for tag in ["old", "direct_len", "pad360_crop"]:
    data = json.loads((root / f"{tag}.json").read_text())
    pred = data.get("pred", {})
    r = pred.get("r_precision", [None, None, None])
    rows.append({
        "tag": tag,
        "ids": data.get("ids_with_required_files"),
        "nb": pred.get("nb"),
        "fid_native": pred.get("fid_vs_gt_native"),
        "fid_refk": pred.get("fid_vs_gt_refk"),
        "r1": r[0],
        "r2": r[1],
        "r3": r[2],
        "mm_dist": pred.get("matching_score"),
        "diversity": pred.get("diversity"),
    })
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY

echo "[done] $(date) OUT=${OUT}"
