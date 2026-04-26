#!/usr/bin/env bash
set -euo pipefail

# =========================
# launch_ablation_experiments.sh — Launch all M2M ablation experiments
#
# Usage:
#   bash scripts/launch_ablation_experiments.sh [round]
#
# Rounds:
#   1 (default): Launch all Round 1 training experiments (8 configs)
#   2: Launch evaluation on all completed checkpoints
# =========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

ROUND="${1:-1}"
CONFIG_DIR="configs/hymotion_m2m/ablation"

echo "============================================="
echo "HyMotion M2M Ablation Experiments — Round ${ROUND}"
echo "============================================="

if [[ "${ROUND}" == "1" ]]; then
    # Round 1: Training experiments
    # Each experiment: 20 epochs, 16 GPUs
    CONFIGS=(
        "ablation_m2_baseline.py"
        "ablation_m1_t2m_only.py"
        "ablation_m3_t2m_heavy.py"
        "ablation_l1_fk_loss.py"
        "ablation_l3a_trans_w1.py"
        "ablation_l3b_trans_w10.py"
        "ablation_l4_velocity_loss.py"
        "ablation_t1_ema.py"
        "ablation_t2_curriculum_p1.py"
    )

    echo ""
    echo "Will launch ${#CONFIGS[@]} training experiments:"
    for cfg in "${CONFIGS[@]}"; do
        echo "  - ${cfg}"
    done
    echo ""

    for cfg in "${CONFIGS[@]}"; do
        config_path="${CONFIG_DIR}/${cfg}"
        exp_name="${cfg%.py}"
        echo ""
        echo "--- Launching: ${exp_name} ---"
        echo "Config: ${config_path}"

        # Submit via taiji_dist_train.sh
        # Each experiment uses 16 GPUs (2 nodes × 8 GPUs)
        bash tools/taiji_dist_train.sh "${config_path}" &
        echo "Launched ${exp_name} (PID: $!)"

        # Small delay to avoid port conflicts
        sleep 5
    done

    echo ""
    echo "All Round 1 experiments launched."
    echo "Monitor with: ls work_dirs/ablation_*/*/train.log"

elif [[ "${ROUND}" == "2" ]]; then
    # Round 2: Evaluation
    EXPERIMENTS=(
        "ablation_m2_baseline"
        "ablation_m1_t2m_only"
        "ablation_m3_t2m_heavy"
        "ablation_l1_fk_loss"
        "ablation_l3a_trans_w1"
        "ablation_l3b_trans_w10"
        "ablation_l4_velocity_loss"
        "ablation_t1_ema"
        "ablation_t2_curriculum_p2"
    )

    for exp in "${EXPERIMENTS[@]}"; do
        work_dir="work_dirs/${exp}"
        # Find latest checkpoint
        latest_ckpt=$(ls -d "${work_dir}"/checkpoint-epoch_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [[ -z "${latest_ckpt}" ]]; then
            echo "SKIP: ${exp} — no checkpoint found"
            continue
        fi

        config_path="${CONFIG_DIR}/${exp}.py"
        echo ""
        echo "--- Evaluating: ${exp} ---"
        echo "Checkpoint: ${latest_ckpt}"

        python scripts/eval_m2m_ablation.py \
            --config "${config_path}" \
            --checkpoint "${latest_ckpt}" \
            --num-samples 200 \
            --num-steps 50 \
            --output "${work_dir}/eval_results.json"
    done

    echo ""
    echo "All evaluations complete."
    echo "Results in: work_dirs/ablation_*/eval_results.json"

else
    echo "Unknown round: ${ROUND}. Use 1 or 2."
    exit 1
fi
