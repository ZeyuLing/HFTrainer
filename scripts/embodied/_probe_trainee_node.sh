#!/usr/bin/env bash
# Standalone probe of the ProtoMotions trainee training + ONNX export on a Taiji
# node, reusing an existing accepted-motion snapshot (so we don't pay for another
# generator round). Validates that the recovered train_agent.py + isaacgym
# simulator train to a checkpoint and that the BeyondMimic tracker exports to the
# ONNX the judge consumes -- the two trainee-side links of the bidirectional loop.
set -eo pipefail
HFT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$HFT/ref_repo/ProtoMotions"
PY=/root/physflow_isaacgym_py38_cu118/bin/python
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ACCEPT_EULA=Y
export MUJOCO_GL=disable
export CUDA_VISIBLE_DEVICES="${1:-0}"
for v in 14 13 12 11 10 9; do
    r="/opt/rh/gcc-toolset-$v/root/usr"
    [ -d "$r/bin" ] && { export PATH="$r/bin:$PATH" CC="$r/bin/gcc" CXX="$r/bin/g++" \
        LD_LIBRARY_PATH="$r/lib64:${LD_LIBRARY_PATH:-}"; break; }
done

SNAP="$HFT/work_dirs/physflow_coevolve_smoke/smoke_co2/trainee/r0_snap"
CK="$HFT/ref_repo/ProtoMotions/results/physflow_online_g1_trainee_gpu2/last.ckpt"
EXP=smoke_trainee_probe
OUT="$HFT/work_dirs/physflow_coevolve_smoke/probe_export"
mkdir -p "$OUT"

echo "=== TRAINEE TRAIN (+2 epochs from warm ckpt) ==="
$PY protomotions/train_agent.py --robot-name g1 --simulator isaacgym \
    --experiment-path examples/experiments/mimic/physflow_g1_xy_offset.py \
    --experiment-name "$EXP" \
    --motion-file "$SNAP" --checkpoint "$CK" \
    --num-envs 512 --batch-size 4096 \
    --training-max-steps 50020352 \
    --headless True \
    --overrides agent.save_last_checkpoint_every=50

TRCK="$HFT/ref_repo/ProtoMotions/results/$EXP/last.ckpt"
echo "=== trainee ckpt: $TRCK ($(ls -la "$TRCK" 2>&1)) ==="
echo "=== ONNX EXPORT ==="
$PY deployment/export_bm_tracker_onnx.py --checkpoint "$TRCK" --output "$OUT"
echo "=== export onnx: $(ls -la "$OUT/unified_pipeline.onnx" 2>&1) ==="
echo "=== TRAINEE PROBE DONE ==="
