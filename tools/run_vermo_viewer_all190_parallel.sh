#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-output/evaluation/vermo_overfit_viewer_all190}"
CONFIG="${2:-configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage4_singlepretrain_lr1e5_from_stage3ckpt250.py}"
CHECKPOINT="${3:-work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage4_singlepretrain_lr1e5_from_stage3ckpt250/checkpoint-iter_300}"

rm -rf "$ROOT"
mkdir -p "$ROOT/logs"

TASK_GROUPS=(
  "pretrain,t2m,m2t"
  "n2tm,pred,inbetween"
  "m2d,d2m,t2md"
  "g2md,n2md"
  "m2d_ar,d2m_ar"
  "s2g,g2s"
  "t2sg,n2sg"
  "ss2sg,s2g_ar"
)

pids=()
for i in "${!TASK_GROUPS[@]}"; do
  CUDA_VISIBLE_DEVICES="$i" python3 tools/export_vermo_overfit_viewer.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$ROOT/g$i" \
    --samples-per-task 10 \
    --tasks "${TASK_GROUPS[$i]}" \
    --device cuda \
    > "$ROOT/logs/g$i.log" 2>&1 &
  pids+=("$!")
  echo "GROUP=$i PID=${pids[-1]} TASKS=${TASK_GROUPS[$i]}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "EXPORT_FAILED"
  exit "$status"
fi

ROOT="$ROOT" python3 - <<'PY'
import glob
import json
import os
from datetime import datetime

from tools.export_vermo_overfit_viewer import summarize_cases

root = os.environ["ROOT"]
manifests = sorted(glob.glob(os.path.join(root, "g*", "manifest.json")))
if len(manifests) != 8:
    raise SystemExit(f"expected 8 manifests, got {len(manifests)}")

cases = []
config = ""
checkpoint = ""
for manifest_path in manifests:
    group = os.path.basename(os.path.dirname(manifest_path))
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    config = config or manifest.get("config", "")
    checkpoint = checkpoint or manifest.get("checkpoint", "")
    for case in manifest.get("cases", []):
        case = json.loads(json.dumps(case, ensure_ascii=False))
        case["case_id"] = f"{group}__{case['case_id']}"
        for bucket in ("inputs", "targets", "predictions"):
            for item in case.get(bucket, []):
                if item.get("path"):
                    item["path"] = f"{group}/{item['path']}"
        cases.append(case)

summary = summarize_cases(cases, expected_cases=len(cases), complete=True)

merged = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "config": config,
    "checkpoint": checkpoint,
    "output_dir": os.path.abspath(root),
    "summary": summary,
    "cases": cases,
}
tmp_path = os.path.join(root, "manifest.json.tmp")
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
os.replace(tmp_path, os.path.join(root, "manifest.json"))
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
