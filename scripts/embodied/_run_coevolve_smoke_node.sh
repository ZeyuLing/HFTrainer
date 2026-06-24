#!/usr/bin/env bash
# Bidirectional co-evolution SMOKE on a Taiji node: exercise the full closed loop
# GENERATOR (py3.10 KIMODO-G1 RAFT vs ProtoMotions judge) -> TRAINEE (py3.8
# IsaacGym PPO+AMP+BM on the accepted-motion pool) -> JUDGE SYNC (export trainee
# ONNX, feed back as next round's judge). Tiny iters/epochs/rounds just to prove
# the plumbing end-to-end before a full run.
set -eo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PATH=/usr/local/bin:$PATH
# kimodo is not pip-installed in the vermo image; orchestrator prepends HFT to
# PYTHONPATH, we add the in-repo kimodo here so the generator round can import it.
export PYTHONPATH="$PWD/ref_repo/KIMODO/kimodo:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TEXT_ENCODERS_DIR="$PWD/checkpoints/kimodo/text_encoders"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

ARM="${1:-smoke_co1}"
GPU="${2:-0}"

python3.10 scripts/embodied/physflow_coevolve_orchestrator.py \
    --arm-name "$ARM" \
    --judge-mode trainee \
    --num-rounds 2 \
    --gen-iters 30 \
    --trainee-epochs 6 \
    --num-envs 512 \
    --batch-size 4096 \
    --gpu "$GPU" \
    --gen-config configs/physflow/physflow_overfit100.py \
    --gen-init-ckpt work_dirs/physflow_overfit100_hgpt/checkpoint-iter_2000 \
    --root work_dirs/physflow_coevolve_smoke
echo "=== COEVOLVE SMOKE WRAPPER EXIT $? ==="
