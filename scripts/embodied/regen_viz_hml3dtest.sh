#!/usr/bin/env bash
# PAIRED base-vs-optimized physical-realism / robot-compatibility evaluation of
# the PhysFlow generator on the OFFICIAL HumanML3D test split (SOTA-aligned prompts).
#
# The research question is NOT "is this a good generic T2M model" but "does our
# optimization make the GENERATED MOTION more physically realistic and more
# robot-executable". So we run TWO arms on the SAME prompts and compare:
#   - base : un-optimized KIMODO-G1
#   - v3   : PhysFlow-optimized (latest checkpoint)
# Per motion we report BOTH metric families:
#   - simulation-free kinematic artifacts (foot slip / penetration / float / jump / jerk)
#   - tracker-in-the-loop executability (MuJoCo G1 judge: completion / fall / joint err / score)
#
# Runs on a KIMODO py3.10 machine (kimodo + mujoco + onnxruntime). Outputs land on
# shared cephfs; the dashboard on the T4 box serves them. Cache-backed + idempotent.
set -eo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# The LLM2Vec 8B text encoder weights are cached on shared cephfs under
# checkpoints/kimodo/hub. taiji containers run HF in offline mode, so point every
# HF cache var at the local cache (absolute paths) BEFORE python starts; otherwise
# transformers tries to resolve the revision over the network and aborts.
export HF_HOME=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/checkpoints/kimodo
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# transformers==5.0.0rc3 makes an unconditional huggingface.co model_info call in
# the tokenizer's mistral-regex patch whenever the model id is NOT a local path,
# which aborts under offline mode. KIMODO's LLM2Vec wrapper joins TEXT_ENCODERS_DIR
# with the model names, so pointing it at a flat dir of LOCAL snapshot symlinks
# makes _is_local=True and skips that network call.
export TEXT_ENCODERS_DIR=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/checkpoints/kimodo/text_encoders

PY="${PHYSFLOW_PYTHON_CMD:-python3}"
GPU="${GPU_ID:-0}"
CORPUS=configs/experiments/physflow_kimodo_g1/physflow_bench_hml3d_test.jsonl
NS=kimodo_g1_llm2vec_hml3dtest
FEATDIR=data/kimodo_text_feature/${NS}
CONFIG=configs/physflow/physflow_online_adv_v3.py
CKPT="${PHYSFLOW_CKPT:-$(ls -dt work_dirs/physflow_online_adv_v3/checkpoint-iter_* | head -1)}"
VIZ=work_dirs/physflow_online_adv_v3/viz

set +e  # diagnostics must never abort the run
echo "[regen] === env diagnostics ==="
echo "[regen] host=$(hostname) pwd=$(pwd)"
echo "[regen] which python3=$(command -v "${PY}")"
"${PY}" --version 2>&1 | sed 's/^/[regen] /'
"${PY}" -c "import sys; sys.path.insert(0,'ref_repo/KIMODO/kimodo'); import kimodo; print('kimodo import OK', kimodo.__file__)" 2>&1 | tail -2 | sed 's/^/[regen] /'
"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | sed 's/^/[regen] /'
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>&1 | sed 's/^/[regen] gpu /' | head -2
set -e  # re-arm: real pipeline steps must fail loudly

# The vermo image ships KIMODO + LLM2Vec + torch but NOT the MuJoCo tracker-scoring
# deps. Tencent clusters can reach the internal PyPI mirror even when huggingface.co
# is blocked, so install them here (idempotent: skips if already importable).
# dm_control + typer are needed by the csv->.motion converter (protomotions.pose_lib);
# mujoco + onnxruntime by the tracker rollout. NONE of these need IsaacGym.
"${PY}" -c "import mujoco, onnxruntime, dm_control, typer" 2>/dev/null || {
  echo "[regen] installing mujoco onnxruntime dm_control typer from internal mirror ..."
  "${PY}" -m pip install --quiet mujoco onnxruntime dm_control typer 2>&1 | tail -3 | sed 's/^/[regen] pip /'
}
"${PY}" -c "import mujoco, onnxruntime, dm_control; print('[regen] mujoco', mujoco.__version__, 'onnxruntime', onnxruntime.__version__, 'dm_control OK')"

# The converter defaults to an IsaacGym py3.8 venv that only exists on lzy_debug_machine.
# In this container the converter has no IsaacGym dependency, so point it at our py3.10.
export PHYSFLOW_CONVERT_PYTHON="${PY}"
echo "[regen] PHYSFLOW_CONVERT_PYTHON=${PHYSFLOW_CONVERT_PYTHON}"

echo "[regen] v3 ckpt=${CKPT}  gpu=${GPU}"

# 1) text features for the HumanML3D-test prompts (own LLM2Vec namespace, cache-backed)
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/embodied/cursor_extract_kimodo_text_feature.py \
  --corpus "${CORPUS}" --namespace "${NS}" --text-encoder llm2vec --device cuda --batch-size 16

run_arm () {
  local arm="$1" ckpt="$2"
  echo "[regen] === arm=${arm} ckpt=${ckpt} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/embodied/physflow_coevolve_viz.py \
    --config "${CONFIG}" --ckpt "${ckpt}" \
    --eval-corpus "${CORPUS}" --feature-dir "${FEATDIR}" --split test \
    --num-prompts 40 --gen-batch 8 \
    --out-dir "${VIZ}/hml3dtest_${arm}_run" \
    --manifest-dir "${VIZ}/hml3dtest_${arm}_manifest" --iteration 900
}

# 2) paired arms on identical prompts
run_arm base "base"
run_arm v3   "${CKPT}"

# 3) base-vs-optimized comparison table
"${PY}" - "${VIZ}/hml3dtest_base_run/summary.json" "${VIZ}/hml3dtest_v3_run/summary.json" \
        "${VIZ}/hml3dtest_compare.json" <<'PYEOF'
import json, sys
import numpy as np
base_p, v3_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
KIN = ["foot_skate_ratio","foot_skate_speed","penetration_ratio","penetration_depth",
       "float_ratio","jump_rate","jerk","joint_std"]
TRK = ["completion_ratio","max_joint_error_rad","fall_detected","adversarial_score",
       "root_trajectory_error_mean_m"]
def agg(path):
    recs = json.load(open(path))["records"]
    ok = [r for r in recs if r.get("status") == "scored"]
    out = {}
    for k in TRK:
        vals = [float(r[k]) for r in ok if r.get(k) is not None]
        if vals: out[k] = round(float(np.mean(vals)), 4)
    for k in KIN:
        vals = [float(r["kinematic"][k]) for r in ok if r.get("kinematic", {}).get(k) is not None]
        if vals: out[k] = round(float(np.mean(vals)), 4)
    out["n_scored"] = len(ok); out["n_total"] = len(recs)
    return out
b, v = agg(base_p), agg(v3_p)
cmp = {"base": b, "v3_optimized": v,
       "delta(v3-base)": {k: round(v[k]-b[k], 4) for k in v if k in b and isinstance(v[k], float)}}
json.dump(cmp, open(out_p, "w"), indent=2)
print("=== base vs v3 (HumanML3D test, n=%d/%d) ===" % (v.get("n_scored",0), v.get("n_total",0)))
for k in TRK + KIN:
    if k in b and k in v:
        print(f"  {k:28s} base={b[k]:<10} v3={v[k]:<10} d={v[k]-b[k]:+.4f}")
print("[compare] wrote", out_p)
PYEOF

echo "[regen] DONE -> ${VIZ}/hml3dtest_{base,v3}_manifest/manifest.json + hml3dtest_compare.json"
