#!/usr/bin/env bash
# Evaluate the latest two HYMotion PhysFlow checkpoints on the established
# HML3D 40-case visualization benchmark and build four-column dashboard
# manifests:
#   KIMODO before | KIMODO after | Tracker before | Tracker after
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "${PROJECT_ROOT}"

PY="${PHYSFLOW_PYTHON_CMD:-/usr/local/bin/python3}"
export PATH="/usr/local/bin:${PATH}"
export HF_HOME="${PROJECT_ROOT}/checkpoints/kimodo"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="${PROJECT_ROOT}/checkpoints/kimodo/text_encoders"
export PHYSFLOW_CONVERT_PYTHON="${PY}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

CONFIG="${PHYSFLOW_CONFIG:-configs/physflow/physflow_online_adv_mn_hymotion_real.py}"
CKPT_ROOT="${PHYSFLOW_CKPT_ROOT:-work_dirs/physflow_online_adv_mn_hymotion_real}"
ITERS="${PHYSFLOW_EVAL_ITERS:-1400 1500}"
TAG="${PHYSFLOW_EVAL_TAG:-hymotion_real_latest2_20260604}"
VIZ_ROOT="${PHYSFLOW_VIZ_DIR:-${CKPT_ROOT}/viz}"
CORPUS="${PHYSFLOW_CORPUS:-configs/experiments/physflow_kimodo_g1/physflow_bench_hml3d_test_g1_noscene.jsonl}"
FEATURE_DIR="${PHYSFLOW_FEATURE_DIR:-data/kimodo_text_feature/kimodo_g1_llm2vec_hml3dtest}"
BASE_RUN="${PHYSFLOW_BASE_RUN:-work_dirs/physflow_online_adv_v3/viz/hml3dtest_base_run}"
TRACKER_AFTER_ONNX="${PHYSFLOW_TRACKER_AFTER_ONNX:-ref_repo/ProtoMotions/results/physflow_g1_released_rehearsal_v2_taskheavy/compiled_best/unified_pipeline.onnx}"
NUM_PROMPTS="${PHYSFLOW_NUM_PROMPTS:-40}"
GEN_BATCH="${PHYSFLOW_GEN_BATCH:-4}"
DIFFUSION_STEPS="${PHYSFLOW_DIFFUSION_STEPS:-20}"
GPU_ID="${GPU_ID:-0}"
SEED="${PHYSFLOW_SEED:-}"

mkdir -p "${VIZ_ROOT}" "output/physflow_reports/20260604_current"

"${PY}" -c "import mujoco, onnxruntime, dm_control, typer" 2>/dev/null || {
  echo "[eval-hymotion] installing mujoco onnxruntime dm_control typer ..."
  "${PY}" -m pip install --quiet mujoco onnxruntime dm_control typer
}
"${PY}" -c "import sys, mujoco, onnxruntime; print('[eval-hymotion] python', sys.version.split()[0], 'mujoco', mujoco.__version__, 'onnxruntime', onnxruntime.__version__)"

if [[ -n "${SEED}" ]]; then
  SEED_ARGS=(--seed "${SEED}")
else
  SEED_ARGS=()
fi

metric_files=()
for iter in ${ITERS}; do
  ckpt="${CKPT_ROOT}/checkpoint-iter_${iter}"
  run_dir="${VIZ_ROOT}/hml3dtest_${TAG}_iter${iter}_run"
  manifest_dir="${VIZ_ROOT}/hml3dtest_${TAG}_iter${iter}_manifest"
  tracker_after_dir="${VIZ_ROOT}/hml3dtest_${TAG}_iter${iter}_tracker_rehearsal_v2_run"
  fourway_dir="${VIZ_ROOT}/hml3dtest_${TAG}_iter${iter}_fourway_manifest"
  log_path="output/physflow_reports/20260604_current/eval_${TAG}_iter${iter}.log"

  echo "[eval-hymotion] iter=${iter} ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" scripts/embodied/physflow_coevolve_viz.py \
    --config "${CONFIG}" \
    --ckpt "${ckpt}" \
    --eval-corpus "${CORPUS}" \
    --feature-dir "${FEATURE_DIR}" \
    --split test \
    --num-prompts "${NUM_PROMPTS}" \
    --diffusion-steps "${DIFFUSION_STEPS}" \
    --gen-batch "${GEN_BATCH}" \
    --out-dir "${run_dir}" \
    --manifest-dir "${manifest_dir}" \
    --iteration "${iter}" \
    "${SEED_ARGS[@]}" 2>&1 | tee "${log_path}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" scripts/embodied/score_tracker_on_physflow_run.py \
    --source-run "${run_dir}" \
    --onnx "${TRACKER_AFTER_ONNX}" \
    --out-dir "${tracker_after_dir}" \
    --label "rehearsal_v2_taskheavy_after" 2>&1 | tee -a "${log_path}"

  "${PY}" scripts/embodied/physflow_fourway_manifest.py \
    --kimodo-before-dir "${BASE_RUN}" \
    --kimodo-after-dir "${run_dir}" \
    --tracker-before-dir "${run_dir}" \
    --tracker-after-dir "${tracker_after_dir}" \
    --out-dir "${fourway_dir}" | tee -a "${log_path}"

  metric_path="${VIZ_ROOT}/hml3dtest_${TAG}_iter${iter}_metrics.json"
  "${PY}" - "${iter}" "${BASE_RUN}" "${run_dir}" "${tracker_after_dir}" "${fourway_dir}/manifest.json" "${metric_path}" <<'PY'
import json
import math
import sys
from pathlib import Path

iter_id, base_run, run_dir, tracker_after_dir, manifest_path, out_path = sys.argv[1:7]

def records(path):
    data = json.loads((Path(path) / "summary.json").read_text())
    return [r for r in data.get("records", []) if r.get("status") == "scored"]

def mean(rows, key, nested=None):
    vals = []
    for r in rows:
        obj = r.get(nested, {}) if nested else r
        v = obj.get(key)
        if isinstance(v, bool):
            v = float(v)
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None

def block(rows):
    return {
        "n_scored": len(rows),
        "completion_ratio": mean(rows, "completion_ratio"),
        "fall_rate": mean(rows, "fall_detected"),
        "max_joint_error_rad": mean(rows, "max_joint_error_rad"),
        "adversarial_score": mean(rows, "adversarial_score"),
        "root_trajectory_error_mean_m": mean(rows, "root_trajectory_error_mean_m"),
        "foot_skate_speed": mean(rows, "foot_skate_speed", nested="kinematic"),
        "joint_vel_max": mean(rows, "joint_vel_max", nested="kinematic"),
        "jerk": mean(rows, "jerk", nested="kinematic"),
    }

base = records(base_run)
after = records(run_dir)
tracker_after = records(tracker_after_dir)
out = {
    "iter": int(iter_id),
    "base_run": str(base_run),
    "kimodo_after_run": str(run_dir),
    "tracker_after_run": str(tracker_after_dir),
    "fourway_manifest": str(manifest_path),
    "kimodo_before": block(base),
    "kimodo_after_and_tracker_before": block(after),
    "tracker_after": block(tracker_after),
}
Path(out_path).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY
  metric_files+=("${metric_path}")
done

"${PY}" - "${VIZ_ROOT}/hml3dtest_${TAG}_latest_two_metrics.json" "${metric_files[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
metrics = [json.loads(Path(p).read_text()) for p in sys.argv[2:]]
out.write_text(json.dumps({"metrics": metrics}, indent=2))
print(f"[eval-hymotion] wrote {out}")
for m in metrics:
    print(f"[eval-hymotion] iter {m['iter']} manifest={m['fourway_manifest']}")
PY
