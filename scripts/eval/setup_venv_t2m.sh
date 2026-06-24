#!/usr/bin/env bash
# Build an ISOLATED venv (--system-site-packages) inside the Taiji keyframe
# instance so hftrainer deps (mmengine) don't pollute the container's base
# python. Reuses the image's torch/accelerate/transformers; only mmengine +
# a numpy<2 pin live in the venv.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

echo "--- stop stale training (was using polluted system python) ---"
pkill -f 'accelerate launch' 2>/dev/null || true
pkill -f 'tools/train.py' 2>/dev/null || true
sleep 2

echo "--- restore base python: remove mmengine I installed there ---"
pip uninstall -y mmengine addict yapf >/dev/null 2>&1 || true
echo "done_uninstall_system"

echo "--- create venv with system site packages ---"
rm -rf .venv_t2m_a100
python3 -m venv --system-site-packages .venv_t2m_a100 || { echo "VENV_CREATE_FAIL"; exit 3; }

echo "--- install hftrainer deps into venv only ---"
# overrides + mmcv-lite + transformers 4.57.3 are needed by the motionhub
# MultiTaskMultiAgent dataset import chain (dataset -> vermo.llama needs
# TransformersKwargs from transformers>=4.55; BaseTransform from mmcv). The
# container base image only ships transformers 4.53.3 and lacks overrides/mmcv.
.venv_t2m_a100/bin/pip install -q mmengine==0.10.7 overrides mmcv-lite 2>&1 | tail -3
.venv_t2m_a100/bin/pip install -q transformers==4.57.3 2>&1 | tail -3
# numpy<2 must be pinned LAST (mmcv-lite/transformers may pull numpy 2.x).
.venv_t2m_a100/bin/pip install -q 'numpy==1.26.4' 2>&1 | tail -3

echo "--- verify import inside venv ---"
PYTHONPATH="$PWD" .venv_t2m_a100/bin/python -c \
  "import numpy,mmengine,hftrainer;print('VENV_OK np',numpy.__version__,'mm',mmengine.__version__)" 2>&1 | tail -6
