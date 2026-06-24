#!/usr/bin/env bash
# Build or repair an OpenTrack environment that does not depend on jax[cuda12].
set -euo pipefail

OPEN_VENV_PATH="${1:?usage: prepare_opentrack_cuda11_env.sh <venv_path>}"
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
OPENTRACK_ROOT="${OPENTRACK_ROOT:-${PROJECT_ROOT}/ref_repo/OpenTrack}"
PYTHON_BIN="${OPENTRACK_CUDA11_PYTHON:-}"
export GLI_PATH="${GLI_PATH:-${OPENTRACK_ROOT}}"
mkdir -p "${GLI_PATH}/storage/logs" "${GLI_PATH}/storage/assets"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.10)"
  elif command -v /usr/local/bin/python3 >/dev/null 2>&1; then
    PYTHON_BIN="/usr/local/bin/python3"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

mkdir -p "$(dirname "${OPEN_VENV_PATH}")"
CUDA_LIB_DIRS=()
for d in /usr/local/cuda*/extras/CUPTI/lib64 /usr/local/cuda*/lib64; do
  [[ -d "${d}" ]] && CUDA_LIB_DIRS+=("${d}")
done
if [[ "${#CUDA_LIB_DIRS[@]}" -gt 0 ]]; then
  CUDA_LD_PATH="$(IFS=:; echo "${CUDA_LIB_DIRS[*]}")"
  export LD_LIBRARY_PATH="${CUDA_LD_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "[opentrack-cuda11-env] LD_LIBRARY_PATH prepended with ${CUDA_LD_PATH}"
fi
if [[ ! -x "${OPEN_VENV_PATH}/bin/python" ]]; then
  echo "[opentrack-cuda11-env] creating venv=${OPEN_VENV_PATH} with ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m venv --system-site-packages "${OPEN_VENV_PATH}"
fi

source "${OPEN_VENV_PATH}/bin/activate"
python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"OpenTrack CUDA11 env requires Python >=3.10, got {sys.version}")
print("[opentrack-cuda11-env] python", sys.version.replace("\n", " "))
PY

python -m pip install --upgrade "pip<25" setuptools wheel

# The upstream OpenTrack pyproject pins Python 3.12 and jax[cuda12].  For V100
# CUDA11 jobs we install the local package without dependencies, then install a
# CUDA11-compatible runtime explicitly. The V100 pool is capped at driver
# CUDA 11.4, so use a cudnn82 wheel instead of newer cuda11 wheels built
# against CUDA 11.8. Override these pins from the submit command if the
# resource pool image has a more suitable CUDA11 stack.
JAX_CUDA11_PACKAGES="${JAX_CUDA11_PACKAGES:-jax==0.4.7 jaxlib==0.4.7+cuda11.cudnn82}"
JAX_CUDA11_FIND_LINKS="${JAX_CUDA11_FIND_LINKS:-https://storage.googleapis.com/jax-releases/jax_cuda_releases.html}"
JAX_CUDA11_CONSTRAINTS="${OPEN_VENV_PATH}/opentrack_cuda11_constraints.txt"
cat > "${JAX_CUDA11_CONSTRAINTS}" <<'EOF'
jax==0.4.7
jaxlib==0.4.7+cuda11.cudnn82
numpy<2
scipy<1.13
chex==0.1.7
optax==0.1.7
flax==0.6.11
orbax-checkpoint==0.1.6
mujoco-mjx==3.3.1
jaxlie==1.5.0
EOF

python -m pip install --ignore-requires-python --no-deps -e "${OPENTRACK_ROOT}"
read -r -a JAX_CUDA11_PACKAGE_ARGS <<< "${JAX_CUDA11_PACKAGES}"
python -m pip install --upgrade --force-reinstall -f "${JAX_CUDA11_FIND_LINKS}" -c "${JAX_CUDA11_CONSTRAINTS}" "${JAX_CUDA11_PACKAGE_ARGS[@]}"

SITE_PACKAGES="$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else site.getusersitepackages())
PY
)"
cat > "${SITE_PACKAGES}/sitecustomize.py" <<'PY'
"""Compatibility shims for the CUDA11 OpenTrack runtime."""
try:
    import jax
    from jax._src import config as _jax_config_mod

    if not hasattr(jax.config, "define_bool_state") and hasattr(_jax_config_mod, "define_bool_state"):
        try:
            setattr(jax.config, "define_bool_state", _jax_config_mod.define_bool_state)
        except Exception:
            setattr(type(jax.config), "define_bool_state", staticmethod(_jax_config_mod.define_bool_state))

    if not hasattr(jax.random, "KeyArray"):
        jax.random.KeyArray = type(jax.random.PRNGKey(0))
except Exception:
    pass
PY
echo "[opentrack-cuda11-env] wrote sitecustomize compatibility shim to ${SITE_PACKAGES}"

JAX_PACKAGE_DIR="$(python - <<'PY'
from pathlib import Path
import jax
print(Path(jax.__file__).resolve().parent)
PY
)"
mkdir -p "${JAX_PACKAGE_DIR}/scipy/spatial"
cp "${PROJECT_ROOT}/scripts/embodied/jax047_spatial_transform_shim.py" "${JAX_PACKAGE_DIR}/scipy/spatial/transform.py"
cat > "${JAX_PACKAGE_DIR}/scipy/spatial/__init__.py" <<'PY'
from jax.scipy.spatial.transform import Rotation, Slerp

__all__ = ["Rotation", "Slerp"]
PY
echo "[opentrack-cuda11-env] installed jax.scipy.spatial.transform compatibility shim"

python -m pip install \
  -f "${JAX_CUDA11_FIND_LINKS}" -c "${JAX_CUDA11_CONSTRAINTS}" \
  "numpy<2" "scipy<1.13" tqdm pytz absl-py ml-collections tyro wandb colorlog rich toml \
  jaxlie==1.5.0 jaxopt==0.8.5 flax==0.6.11 brax==0.12.3 mujoco==3.3.1 mujoco-mjx==3.3.1 playground==0.0.4 \
  onnx onnxscript onnxruntime tf2onnx einops hydra-core scikit-learn pillow imageio imageio-ffmpeg trimesh \
  "osqp>=1.0.5"

python - <<'PY'
import importlib
mods = [
    "jax",
    "jaxlib",
    "mujoco",
    "mujoco_playground._src.mjx_env",
    "brax.training.agents.ppo.networks",
    "track_mj",
    "torch",
]
for mod in mods:
    importlib.import_module(mod)

import jax
import os
devices = jax.devices()
print("[opentrack-cuda11-env] jax devices:", devices)
if not any(getattr(d, "platform", "") in {"gpu", "cuda"} for d in devices):
    if os.environ.get("ALLOW_JAX_CPU", "0") != "1":
        raise SystemExit("JAX did not expose a GPU device; set ALLOW_JAX_CPU=1 only for smoke tests")
PY

echo "[opentrack-cuda11-env] ready: ${OPEN_VENV_PATH}"
