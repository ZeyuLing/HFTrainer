#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
TAG="${TAG:-preflight_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/output/opentrack_physflow_adversarial/${TAG}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/preflight.log") 2>&1

echo "[preflight] root=${ROOT}"
echo "[preflight] tag=${TAG}"
date

cd "${ROOT}/ref_repo/OpenTrack"
if command -v uv >/dev/null 2>&1; then
  UV_CMD=(uv)
else
  python3 -m pip install --user -q uv
  UV_CMD=(python3 -m uv)
fi

echo "[preflight] syncing OpenTrack environment"
"${UV_CMD[@]}" sync -i https://pypi.org/simple
source .venv/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled

python - <<'PY'
import colorlog, jax, mujoco, onnxruntime, torch
import track_mj
print("runtime_ok", "jax", jax.__version__, "torch", torch.__version__, "mujoco", mujoco.__version__)
PY

cd "${ROOT}"
python3 scripts/embodied/stage_opentrack_adversarial_motions.py \
  --input-dir "${ROOT}/output/opentrack_amass_g1/debug2_20260604_1915_wait_proto_wxyz/UnitreeG1" \
  --manifest-json "${LOG_DIR}/adversarial_motions.json" \
  --manifest-txt "${LOG_DIR}/adversarial_motions.txt" \
  --keywords jump \
  --max-files 8 \
  --mode symlink \
  --force

python3 scripts/embodied/opentrack_onnx_to_dagger_pth.py \
  --onnx "${ROOT}/ref_repo/OpenTrack/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/model.onnx" \
  --out-pth "${LOG_DIR}/general_tracker_lafan1_v2.pth" \
  --force

cd "${ROOT}/ref_repo/OpenTrack"
python -m track_mj.learning.train.train_ppo_track \
  --task G1TrackingGeneralDR \
  --exp-name debug_physflow_adv_preflight \
  --trajectory-manifest "${LOG_DIR}/adversarial_motions.txt" \
  --trajectory-dataset-name lafan1 \
  --num-timesteps 100000 \
  --obs-noise-level 0.0

echo "[preflight] completed"
date
