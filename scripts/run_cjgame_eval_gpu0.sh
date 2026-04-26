#!/bin/bash
# CJGame repair eval - Debug Machine 1
# Runs: MoGenDIT (masks + denoise + ada_denoise) + Uncond M2M (_man variants)
#
# Usage: taiji_client exec lzy_debug_machine_1 <instance_id> bash
#   cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
#   nohup bash scripts/run_cjgame_eval_gpu0.sh > output/cjgame_repair_eval/run_machine1.log 2>&1 &

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
mkdir -p output/cjgame_repair_eval

echo "======================================"
echo "CJGame Eval - Debug Machine 1"
echo "Started: $(date)"
echo "======================================"

# Phase 1: MoGenDIT adaptive masks + repair (sequential, single GPU)
echo ""
echo "[Phase 1] MoGenDIT: masks + denoise + ada_denoise (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_cjgame_repair.py \
    --max-samples 0 \
    --num-steps 50 \
    --mogendit-steps 10 \
    --device cuda:0 \
    --output-dir output/cjgame_repair_eval \
    --skip-m2m \
    --skip-checker \
    --seed 42 \
    2>&1 | tee output/cjgame_repair_eval/log_phase1_mogendit.txt

echo ""
echo "[Phase 1] MoGenDIT done. Masks: $(ls output/cjgame_repair_eval/adaptive_masks/*.npz 2>/dev/null | wc -l)"

# Phase 2: M2M uncond configs (sequential per model to avoid OOM, one GPU each)
# Each M2M model is ~2GB; with T4/V100 we can run one at a time safely.
M2M_CONFIGS=(
    "uncond_fm_man"
    "uncond_jit_man"
    "uncond_fm_man_globalrot"
    "uncond_jit_man_globalrot"
)

for cfg in "${M2M_CONFIGS[@]}"; do
    echo ""
    echo "[Phase 2] M2M $cfg (GPU 0)"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_cjgame_repair.py \
        --max-samples 0 \
        --num-steps 50 \
        --device cuda:0 \
        --output-dir output/cjgame_repair_eval \
        --skip-mogendit \
        --m2m-configs $cfg \
        --skip-checker \
        --seed 42 \
        2>&1 | tee output/cjgame_repair_eval/log_m2m_${cfg}.txt
    echo "  Done: $(ls output/cjgame_repair_eval/m2m_${cfg}_completion/repaired/*.npz 2>/dev/null | wc -l) completion + $(ls output/cjgame_repair_eval/m2m_${cfg}_edit/repaired/*.npz 2>/dev/null | wc -l) edit"
done

echo ""
echo "======================================"
echo "Debug Machine 1 complete: $(date)"
echo "======================================"
