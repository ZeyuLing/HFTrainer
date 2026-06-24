#!/usr/bin/env bash
# Export Humanoid-GPT real rollout frames for the Table 2 case comparison page.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/case_compare_viz}"
MANIFEST="${MANIFEST:-${OUT_ROOT}/manifest.json}"
LAFAN_ROOT="${LAFAN_ROOT:-${PROJECT_ROOT}/data/LAFAN1_Retargeted_for_G1/UnitreeG1}"
WILD_ROOT="${WILD_ROOT:-${PROJECT_ROOT}/output/heldout_frozen_score}"
HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-/dev/shm/hgpt_venv311/bin/python}"
TIMEOUT_S="${TIMEOUT_S:-3600}"

cd "${PROJECT_ROOT}"

if [[ ! -x "${HGPT_PYTHON}" ]]; then
  echo "[hgpt-case-dump] building HGPT worker env"
  bash scripts/embodied/physflow_hgpt_node_setup.sh
fi
HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-/dev/shm/hgpt_venv311/bin/python}"

mkdir -p "${OUT_ROOT}/manifests"
"${HGPT_PYTHON}" - "${MANIFEST}" "${OUT_ROOT}/manifests/hgpt_lafan1_selected.json" "${OUT_ROOT}/manifests/hgpt_wild_selected.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
lafan, wild = [], []
for row in manifest.get("rows", []):
    dataset = str(row.get("dataset", ""))
    stem = str(row.get("stem") or row.get("match_key"))
    if dataset.startswith("LAFAN"):
        lafan.append(stem)
    elif dataset.startswith("Wild"):
        wild.append(f"{stem.split('_', 1)[0]}_gen")
Path(sys.argv[2]).write_text(json.dumps(lafan, indent=2) + "\n")
Path(sys.argv[3]).write_text(json.dumps(wild, indent=2) + "\n")
print(f"[hgpt-case-dump] selected lafan={len(lafan)} wild={len(wild)}")
PY

echo "[hgpt-case-dump] run LAFAN1 cases"
"${HGPT_PYTHON}" scripts/embodied/run_table2_hgpt_eval.py \
  --motion-dir "${LAFAN_ROOT}" \
  --manifest "${OUT_ROOT}/manifests/hgpt_lafan1_selected.json" \
  --out-dir "${OUT_ROOT}/humanoid_gpt_eval/lafan1" \
  --frames-out-dir "${OUT_ROOT}/humanoid_gpt/lafan1" \
  --hgpt-python "${HGPT_PYTHON}" \
  --timeout-s "${TIMEOUT_S}"

echo "[hgpt-case-dump] run Wild-G1 cases"
"${HGPT_PYTHON}" scripts/embodied/run_table2_hgpt_eval.py \
  --motion-dir "${WILD_ROOT}" \
  --manifest "${OUT_ROOT}/manifests/hgpt_wild_selected.json" \
  --out-dir "${OUT_ROOT}/humanoid_gpt_eval/wild" \
  --frames-out-dir "${OUT_ROOT}/humanoid_gpt/wild" \
  --hgpt-python "${HGPT_PYTHON}" \
  --timeout-s "${TIMEOUT_S}"

python3 scripts/embodied/add_hgpt_to_table2_case_compare_viz.py --manifest "${MANIFEST}"
echo "[hgpt-case-dump] manifest=${MANIFEST}"
