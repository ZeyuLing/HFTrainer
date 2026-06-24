#!/usr/bin/env bash
# Build the Humanoid-GPT tracking-judge venv (py3.11) on a Taiji node's FAST
# local disk (/dev/shm) or into the bundled Humanoid-GPT directory. The vermo trainer image is py3.10 and cannot import
# HGPT's jax / mujoco-mjx stack, so the PhysFlow HGPT judge runs in this separate
# venv via a long-lived worker bundled at
# hftrainer/models/motion/physflow/trackers/humanoid_gpt/physflow_hgpt_judge_server.py.
#
# CephFS venvs are pathologically slow (tiny-file I/O), so we build on /dev/shm,
# which is node-local and fast but ephemeral -- fine for a job's lifetime; just
# re-run this at node start. Idempotent: skips install if the stack imports.
#
# Usage (on the node, fast ephemeral venv):
#   bash scripts/embodied/physflow_hgpt_node_setup.sh
#
# Persistent in-repo venv for local validation:
#   PHYSFLOW_HGPT_VENV=hftrainer/models/motion/physflow/trackers/humanoid_gpt/.venv311 \
#     bash scripts/embodied/physflow_hgpt_node_setup.sh
# Prints the venv python path on the last line.
set -eo pipefail

VENV="${PHYSFLOW_HGPT_VENV:-/dev/shm/hgpt_venv311}"
MIRROR="${PIP_MIRROR:-https://mirrors.tencent.com/pypi/simple}"
PY311="${PY311:-python3.11}"
ORT_PACKAGE="${PHYSFLOW_HGPT_ORT_PACKAGE:-onnxruntime<1.24}"

if "$VENV/bin/python" -c "import jax,mujoco,onnxruntime,flax,scipy,tyro,tree" 2>/dev/null; then
    echo "[hgpt-setup] venv already complete: $VENV"
    echo "$VENV/bin/python"
    exit 0
fi

echo "[hgpt-setup] building $VENV with $PY311 from $MIRROR ..."
rm -rf "$VENV"
"$PY311" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q -U pip wheel -i "$MIRROR" 2>&1 | tail -1
"$VENV/bin/python" -m pip install -i "$MIRROR" \
    numpy scipy "$ORT_PACKAGE" "jax==0.8.0" jaxlie "flax==0.12.0" ml_collections \
    "mujoco==3.3.7" "mujoco-mjx==3.3.7" dm-tree tyro tqdm absl-py colorlog \
    loop_rate_limiters matplotlib opencv-python imageio imageio-ffmpeg pillow 2>&1 | tail -4
"$VENV/bin/python" -c "import jax,mujoco,onnxruntime,flax,scipy,tyro,tree; print('[hgpt-setup] import OK')"
echo "$VENV/bin/python"
