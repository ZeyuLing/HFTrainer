#!/bin/bash
# Force-kill any PRISM generation driver/workers on this node.
for p in $(pgrep -f 'run_epoch4_rw_node|run_gen_node|eval_prism_kafs_ablation'); do
    kill -9 "$p" 2>/dev/null
done
sleep 8
echo "AFTER_KILL remaining=$(pgrep -f 'run_epoch4_rw_node|run_gen_node|eval_prism_kafs_ablation' | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
