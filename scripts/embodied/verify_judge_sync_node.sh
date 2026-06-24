#!/usr/bin/env bash
# Panel 2 (JUDGE SYNC proof) on a Taiji node: roll out the same generated
# motions under frozen / trainee-r0 / trainee-r1 judges. No CSV->.motion convert
# (reuses existing .motion), so NO py3.8 needed -- just py3.10 MuJoCo+ONNX.
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$REPO"
echo "[judge-sync] $(date) host=$(hostname)"
export PIP_DEFAULT_TIMEOUT=30
timeout 300 /usr/local/bin/python3 -m pip install --quiet \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -1 || true
export PATH=/usr/local/bin:$PATH
export PYTHONPATH="$REPO:$REPO/ref_repo/ProtoMotions:${PYTHONPATH:-}"
export MUJOCO_GL=disable
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
/usr/local/bin/python3 scripts/embodied/verify_judge_sync_viz.py
echo "[judge-sync] exit=$?"
