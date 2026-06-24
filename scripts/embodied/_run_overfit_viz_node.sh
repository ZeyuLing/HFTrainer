#!/usr/bin/env bash
# Produce the PhysFlow overfit demo data (base vs optimized) on a Taiji vermo
# node: for each arm, generate the 100 train-prompt G1 motions and roll them out
# under the frozen ProtoMotions judge in MuJoCo, emitting robot_frames JSON +
# per-arm viewer manifest that motion_annot_web/embodied_viz can render.
#
# Wraps tools/physflow_eval_trainset.sh but first adds the two env bits that
# script assumes are pre-set in its original "already-running container":
#   * PYTHONPATH -> in-repo kimodo (not pip-installed in the vermo image)
#   * the MuJoCo / convert deps (idempotent install, like physflow_mn_start.sh)
set -eo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PATH=/usr/local/bin:$PATH
export PYTHONPATH="$PWD/ref_repo/KIMODO/kimodo:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

GPU="${1:-1}"
OPT_CKPT="${2:-work_dirs/physflow_overfit100_hgpt/checkpoint-iter_2000}"
OPT_TAG="${3:-opt_iter2000}"

# frozen-judge rollout deps (mujoco + onnxruntime) and CSV->.motion converter
# deps (dm_control + typer); skip if already importable.
python3 -c "import mujoco, onnxruntime, dm_control, typer" 2>/dev/null || {
    echo "[viz] installing mujoco onnxruntime dm_control typer ..."
    python3 -m pip install --quiet mujoco onnxruntime dm_control typer 2>&1 | tail -3
}

echo "=== BASE arm (un-optimized KIMODO-G1) ==="
bash tools/physflow_eval_trainset.sh base base "$GPU"
echo "=== OPT arm ($OPT_CKPT) ==="
bash tools/physflow_eval_trainset.sh "$OPT_CKPT" "$OPT_TAG" "$GPU"
echo "=== OVERFIT VIZ DONE (arms: base, $OPT_TAG) ==="
