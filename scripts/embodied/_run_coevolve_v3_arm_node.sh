#!/usr/bin/env bash
# One arm of the PhysFlow bidirectional co-evolution ablation (v3 anti-freeze
# generator + real 11k HumanML3D prompt corpus), on a Taiji node. Each arm runs
# the full GEN<->TRAINEE<->JUDGE loop with a different judge mode:
#   frozen  : control (judge never changes)
#   trainee : pure adversarial (judge == latest trainee tracker)
#   anchor  : 0.5 frozen + 0.5 trainee (anti reward-hack)
set -eo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PATH=/usr/local/bin:$PATH
export PYTHONPATH="$PWD/ref_repo/KIMODO/kimodo:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TEXT_ENCODERS_DIR="$PWD/checkpoints/kimodo/text_encoders"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

ARM="${1:?arm name}"
MODE="${2:?judge mode: frozen|trainee|anchor}"
GPU="${3:?gpu id}"
ALPHA="${4:-0.5}"

python3.10 scripts/embodied/physflow_coevolve_orchestrator.py \
    --arm-name "$ARM" \
    --judge-mode "$MODE" \
    --anchor-alpha "$ALPHA" \
    --num-rounds 8 \
    --gen-iters 100 \
    --trainee-epochs 50 \
    --num-envs 1024 \
    --batch-size 8192 \
    --gpu "$GPU" \
    --gen-config configs/physflow/physflow_online_adv_v3.py \
    --gen-init-ckpt work_dirs/physflow_online_adv_v3/checkpoint-iter_3000 \
    --root work_dirs/physflow_coevolve_v3
echo "=== COEVOLVE ARM $ARM ($MODE) EXIT $? ==="
