#!/bin/bash
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/KIMODO/kimodo:/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:$PYTHONPATH

echo "=== KIMODO E8-D re-run with T_PAD_MAX=240 ==="
python3 scripts/kimodo/run_kimodo_all_tasks.py \
    --tasks E8 \
    --settings D \
    --max-samples 200 \
    --use-caption no \
    --output-dir work_dirs/e8d_kimodo_fixed_20260511 \
    2>&1 | tail -20
echo "KIMODO_E8D_DONE"
