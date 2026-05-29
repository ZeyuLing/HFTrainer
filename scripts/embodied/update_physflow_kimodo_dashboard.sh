#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

RAW_RUN="${1:-output/physflow_kimodo_g1/smoke_lzy2_caption_rp_3prompts}"
OPT_RUN="${2:-}"
MONITOR_DIR="${3:-output/physflow_kimodo_g1/online_triplet_monitor}"

if [[ -n "${OPT_RUN}" && -f "${OPT_RUN}/summary.json" ]]; then
  python3 scripts/embodied/physflow_triplet_manifest.py \
    --raw-run-dir "${RAW_RUN}" \
    --optimized-run-dir "${OPT_RUN}" \
    --out-dir "${MONITOR_DIR}" \
    --iteration 0
else
  python3 scripts/embodied/physflow_triplet_manifest.py \
    --raw-run-dir "${RAW_RUN}" \
    --out-dir "${MONITOR_DIR}" \
    --iteration 0
fi

python3 - <<'PY'
import json
from pathlib import Path
p = Path("output/physflow_kimodo_g1/online_triplet_monitor/manifest.json")
d = json.loads(p.read_text())
print(f"dashboard_manifest={p} rows={len(d.get('rows', []))}")
for row in d.get("rows", []):
    cols = {k: v.get("status") for k, v in row.get("columns", {}).items()}
    print(row.get("prompt_id"), cols)
PY
