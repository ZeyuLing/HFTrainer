#!/usr/bin/env bash
# Recompute TP2M HumanML3D MS272 evaluator metrics from canonical paths.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [ ! -d "${ROOT}" ]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PY="${PY:-python3}"
CONDS="${CONDS:-1 5 9}"
METHODS="${METHODS:-prism motionstreamer flowmdm motionlab kimodo}"
TASK_SPECS="${TASK_SPECS:-}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
N_REPEATS="${N_REPEATS:-20}"
RUN_ROOT="${RUN_ROOT:-outputs/evaluation/tp2m/_runs/ms272_metrics_20260629}"
TEXT_DIR="${TEXT_DIR:-outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/texts}"
SKIP_CACHE="${SKIP_CACHE:-0}"
LOG_ROOT="${RUN_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -lt 1 ]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

tasks=()
if [ -n "${TASK_SPECS}" ]; then
  for pair in ${TASK_SPECS}; do
    IFS=':' read -r cond method <<< "${pair}"
    pred_dir="outputs/evaluation/tp2m/humanml3d_official_test_c${cond}/ms272/${method}"
    metric_dir="${pred_dir}/metrics"
    if [ ! -d "${pred_dir}" ]; then
      echo "[skip] missing ${pred_dir}" >&2
      continue
    fi
    tasks+=("${cond}:${method}:${pred_dir}:${metric_dir}")
  done
else
  for cond in ${CONDS}; do
    for method in ${METHODS}; do
      pred_dir="outputs/evaluation/tp2m/humanml3d_official_test_c${cond}/ms272/${method}"
      metric_dir="${pred_dir}/metrics"
      if [ ! -d "${pred_dir}" ]; then
        echo "[skip] missing ${pred_dir}" >&2
        continue
      fi
      tasks+=("${cond}:${method}:${pred_dir}:${metric_dir}")
    done
  done
fi

echo "[start] tasks=${#tasks[@]} conds=${CONDS} methods=${METHODS} task_specs=${TASK_SPECS} n_repeats=${N_REPEATS} text_dir=${TEXT_DIR} skip_cache=${SKIP_CACHE}"
if [ "${SKIP_CACHE}" != "1" ]; then
  bash scripts/eval/_cache_272_data.sh > "${LOG_ROOT}/cache_272_data.log" 2>&1 || true
else
  echo "[cache] skipped by SKIP_CACHE=1" > "${LOG_ROOT}/cache_272_data.log"
fi
fail=0
running=0
idx=0
pids=()

run_one() {
  local spec="$1"
  local gpu="$2"
  local cond method pred_dir metric_dir log_file
  IFS=':' read -r cond method pred_dir metric_dir <<< "${spec}"
  mkdir -p "${metric_dir}"
  log_file="${LOG_ROOT}/c${cond}_${method}.log"
  {
    echo "[metric-start] $(date -Is) cond=${cond} method=${method} gpu=${gpu}"
    echo "python3 scripts/eval/eval_motionstreamer_272.py --pred-dir ${pred_dir} --tag ${method}_c${cond} --also-refk --text-dir ${TEXT_DIR} --out-json ${metric_dir}/motionstreamer.json"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/eval/eval_motionstreamer_272.py \
      --pred-dir "${pred_dir}" \
      --tag "${method}_c${cond}" \
      --also-refk \
      --text-dir "${TEXT_DIR}" \
      --out-json "${metric_dir}/motionstreamer.json"
    echo "[metric-done] $(date -Is) cond=${cond} method=${method}"
  } > "${log_file}" 2>&1
}

for spec in "${tasks[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_one "${spec}" "${gpu}" &
  pids+=("$!")
  running=$((running + 1))
  idx=$((idx + 1))
  if [ "${running}" -ge "${#GPUS[@]}" ]; then
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        fail=1
      fi
    done
    pids=()
    running=0
  fi
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

"${PY}" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/evaluation/tp2m")
rows = []
for cond in (1, 5, 9):
    for method in ("prism", "motionstreamer", "flowmdm", "motionlab", "kimodo"):
        path = root / f"humanml3d_official_test_c{cond}" / "ms272" / method / "metrics" / "motionstreamer.json"
        row = {"condition": cond, "method": method, "path": str(path), "exists": path.exists()}
        if path.exists():
            try:
                data = json.loads(path.read_text())
                pred = data.get("pred", {})
                row.update({
                    "fid_vs_gt_native": pred.get("fid_vs_gt_native"),
                    "fid_vs_gt_refk": pred.get("fid_vs_gt_refk"),
                    "matching_score": pred.get("matching_score"),
                    "r_precision": pred.get("r_precision"),
                    "diversity": pred.get("diversity"),
                    "nb": pred.get("nb"),
                    "ids_with_required_files": data.get("ids_with_required_files"),
                    "text_dir": data.get("text_dir"),
                    "min_motion_len": data.get("min_motion_len"),
                })
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
out = Path("outputs/evaluation/tp2m/_runs/ms272_metrics_20260629/summary.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
PY

echo "[done] fail=${fail} logs=${LOG_ROOT}"
exit "${fail}"
