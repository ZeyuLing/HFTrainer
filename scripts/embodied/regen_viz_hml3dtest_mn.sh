#!/usr/bin/env bash
# Paired HumanML3D-test visualization/evaluation for the PhysFlow multi-node run.
# Defaults to the finished mn@1500 checkpoint and reuses the established v3 base
# arm so the dashboard comparison keeps the same baseline as earlier mn@800 runs.
set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "${PROJECT_ROOT}"

export HF_HOME="${PROJECT_ROOT}/checkpoints/kimodo"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="${PROJECT_ROOT}/checkpoints/kimodo/text_encoders"

PY="${PHYSFLOW_PYTHON_CMD:-/usr/local/bin/python3}"
GPU="${GPU_ID:-0}"
TAG="${PHYSFLOW_TAG:-mn1500}"
ITER="${PHYSFLOW_ITER:-1500}"
CONFIG="${PHYSFLOW_CONFIG:-configs/physflow/physflow_online_adv_mn.py}"
CKPT="${PHYSFLOW_CKPT:-work_dirs/physflow_online_adv_mn/checkpoint-iter_1500}"
VIZ="${PHYSFLOW_VIZ_DIR:-work_dirs/physflow_online_adv_mn/viz}"
CORPUS="${PHYSFLOW_CORPUS:-configs/experiments/physflow_kimodo_g1/physflow_bench_hml3d_test.jsonl}"
NS="${PHYSFLOW_TEXT_NS:-kimodo_g1_llm2vec_hml3dtest}"
FEATDIR="${PHYSFLOW_FEATURE_DIR:-data/kimodo_text_feature/${NS}}"
SEED="${PHYSFLOW_SEED:-}"
BASE_SUMMARY="${PHYSFLOW_BASE_SUMMARY:-work_dirs/physflow_online_adv_v3/viz/hml3dtest_base_run/summary.json}"
BASE_MANIFEST="${PHYSFLOW_BASE_MANIFEST:-work_dirs/physflow_online_adv_v3/viz/hml3dtest_base_manifest/manifest.json}"

RUN_DIR="${VIZ}/hml3dtest_${TAG}_run"
MANIFEST_DIR="${VIZ}/hml3dtest_${TAG}_manifest"
COMPARE_JSON="${VIZ}/hml3dtest_compare_${TAG}.json"
CANON_COMPARE_JSON="${VIZ}/hml3dtest_compare.json"
COMPARE_MANIFEST="${VIZ}/hml3dtest_compare_${TAG}_manifest/manifest.json"
CANON_COMPARE_MANIFEST="${VIZ}/hml3dtest_compare_manifest/manifest.json"

mkdir -p "${VIZ}"

if [[ -f "${CANON_COMPARE_JSON}" && ! -f "${VIZ}/hml3dtest_compare_mn800.json" ]]; then
    cp "${CANON_COMPARE_JSON}" "${VIZ}/hml3dtest_compare_mn800.json"
fi
if [[ -d "${VIZ}/hml3dtest_compare_manifest" && ! -d "${VIZ}/hml3dtest_compare_mn800_manifest" ]]; then
    cp -a "${VIZ}/hml3dtest_compare_manifest" "${VIZ}/hml3dtest_compare_mn800_manifest"
fi

echo "[regen-mn] start $(date)"
echo "[regen-mn] host=$(hostname) gpu=${GPU} tag=${TAG} iter=${ITER}"
echo "[regen-mn] config=${CONFIG}"
echo "[regen-mn] ckpt=${CKPT}"
if [[ -n "${SEED}" ]]; then
    echo "[regen-mn] seed=${SEED}"
fi
SEED_ARGS=()
if [[ -n "${SEED}" ]]; then
    SEED_ARGS=(--seed "${SEED}")
fi

"${PY}" -c "import mujoco, onnxruntime, dm_control, typer" 2>/dev/null || {
    echo "[regen-mn] installing mujoco onnxruntime dm_control typer ..."
    "${PY}" -m pip install --quiet mujoco onnxruntime dm_control typer
}
"${PY}" -c "import mujoco, onnxruntime, dm_control; print('[regen-mn] deps OK', mujoco.__version__, onnxruntime.__version__)"

export PHYSFLOW_CONVERT_PYTHON="${PY}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/embodied/cursor_extract_kimodo_text_feature.py \
    --corpus "${CORPUS}" --namespace "${NS}" --text-encoder llm2vec --device cuda --batch-size 16

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/embodied/physflow_coevolve_viz.py \
    --config "${CONFIG}" --ckpt "${CKPT}" \
    --eval-corpus "${CORPUS}" --feature-dir "${FEATDIR}" --split test \
    --num-prompts 40 --gen-batch 8 \
    --out-dir "${RUN_DIR}" \
    --manifest-dir "${MANIFEST_DIR}" --iteration "${ITER}" \
    "${SEED_ARGS[@]}"

"${PY}" - "${BASE_SUMMARY}" "${RUN_DIR}/summary.json" "${COMPARE_JSON}" "${CANON_COMPARE_JSON}" "${TAG}" <<'PYEOF'
import json
import sys

import numpy as np

base_p, opt_p, out_p, canon_p, tag = sys.argv[1:6]
kin_keys = [
    "foot_skate_ratio",
    "foot_skate_speed",
    "penetration_ratio",
    "penetration_depth",
    "float_ratio",
    "jump_rate",
    "jerk",
    "joint_std",
]
tracker_keys = [
    "completion_ratio",
    "max_joint_error_rad",
    "fall_detected",
    "adversarial_score",
    "root_trajectory_error_mean_m",
]


def aggregate(path):
    records = json.load(open(path))["records"]
    ok = [r for r in records if r.get("status") == "scored"]
    out = {}
    for key in tracker_keys:
        vals = [float(r[key]) for r in ok if r.get(key) is not None]
        if vals:
            out[key] = round(float(np.mean(vals)), 4)
    for key in kin_keys:
        vals = [
            float(r["kinematic"][key])
            for r in ok
            if r.get("kinematic", {}).get(key) is not None
        ]
        if vals:
            out[key] = round(float(np.mean(vals)), 4)
    out["n_scored"] = len(ok)
    out["n_total"] = len(records)
    return out


base = aggregate(base_p)
opt = aggregate(opt_p)
compare = {
    "base": base,
    f"{tag}_optimized": opt,
    f"delta({tag}-base)": {
        key: round(opt[key] - base[key], 4)
        for key in opt
        if key in base and isinstance(opt[key], float)
    },
}
for path in (out_p, canon_p):
    json.dump(compare, open(path, "w"), indent=2)

print(f"=== base vs {tag} (HumanML3D test) ===")
for key in tracker_keys + kin_keys:
    if key in base and key in opt:
        print(f"  {key:28s} base={base[key]:<10} {tag}={opt[key]:<10} d={opt[key]-base[key]:+.4f}")
print("[compare] wrote", out_p, "and", canon_p)
PYEOF

"${PY}" scripts/embodied/build_compare_manifest.py \
    --base "${BASE_MANIFEST}" \
    --opt "${MANIFEST_DIR}/manifest.json" \
    --out "${COMPARE_MANIFEST}" \
    --base-label "KIMODO-G1 (base)" \
    --opt-label "PhysFlow ${TAG}"

"${PY}" scripts/embodied/build_compare_manifest.py \
    --base "${BASE_MANIFEST}" \
    --opt "${MANIFEST_DIR}/manifest.json" \
    --out "${CANON_COMPARE_MANIFEST}" \
    --base-label "KIMODO-G1 (base)" \
    --opt-label "PhysFlow ${TAG}"

echo "[regen-mn] done $(date)"
echo "[regen-mn] compare=${COMPARE_JSON}"
echo "[regen-mn] manifest=${CANON_COMPARE_MANIFEST}"
