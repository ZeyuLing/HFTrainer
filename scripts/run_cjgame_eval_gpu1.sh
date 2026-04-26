#!/bin/bash
# CJGame repair eval - Debug Machine 2
# Runs: Caption M2M (_man variants)
# Pre-requisite: Debug Machine 1 must compute adaptive masks first.
#
# Usage: taiji_client exec lzy_debug_machine_2 <instance_id> bash
#   cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
#   nohup bash scripts/run_cjgame_eval_gpu1.sh > output/cjgame_repair_eval/run_machine2.log 2>&1 &

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
mkdir -p output/cjgame_repair_eval

echo "======================================"
echo "CJGame Eval - Debug Machine 2"
echo "Started: $(date)"
echo "======================================"

# Wait for adaptive masks from Machine 1
echo "Waiting for adaptive masks (from debug machine 1)..."
while [ ! -d output/cjgame_repair_eval/adaptive_masks ] || [ $(ls output/cjgame_repair_eval/adaptive_masks/*.npz 2>/dev/null | wc -l) -lt 500 ]; do
    count=$(ls output/cjgame_repair_eval/adaptive_masks/*.npz 2>/dev/null | wc -l)
    echo "  $(date +%H:%M:%S) - $count masks found, waiting..."
    sleep 60
done
echo "Masks ready: $(ls output/cjgame_repair_eval/adaptive_masks/*.npz 2>/dev/null | wc -l)"

M2M_CONFIGS=(
    "caption_fm_man"
    "caption_jit_man"
    "caption_fm_man_globalrot"
    "caption_jit_man_globalrot"
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
echo "Debug Machine 2 complete: $(date)"
echo "======================================"
