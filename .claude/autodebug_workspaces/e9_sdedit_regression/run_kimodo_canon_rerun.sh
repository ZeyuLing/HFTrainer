#!/bin/bash
# Re-run KIMODO across all constraint tasks with the new canonical fix
# (2026-04-21). Tasks E2/E3/E4/E5/E6/E7/E8/E10/E14 — both caption + uncond.
# Each job on its own GPU. KIMODO T2M (E1) is unchanged; we don't rerun it.
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUT_ROOT=work_dirs/kimodo_canon_rerun_20260421
mkdir -p "$OUT_ROOT"

# Caption variants (use sample captions from datalist) — GPUs 0-3
CUDA_VISIBLE_DEVICES=0 python3 tools/run_kimodo_all_tasks.py \
    --tasks E2 E3 E4 --max-samples 50 \
    --use-caption yes \
    --output-dir "$OUT_ROOT/kcap_234" > "$OUT_ROOT/kcap_234.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 tools/run_kimodo_all_tasks.py \
    --tasks E5 E6 E7 --max-samples 50 \
    --use-caption yes \
    --output-dir "$OUT_ROOT/kcap_567" > "$OUT_ROOT/kcap_567.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 tools/run_kimodo_all_tasks.py \
    --tasks E8 E10 E14 --max-samples 50 \
    --use-caption yes \
    --output-dir "$OUT_ROOT/kcap_8_10_14" > "$OUT_ROOT/kcap_8_10_14.log" 2>&1 &

# Uncond variants — GPUs 4-7
CUDA_VISIBLE_DEVICES=4 python3 tools/run_kimodo_all_tasks.py \
    --tasks E2 E3 E4 --max-samples 50 \
    --use-caption no \
    --output-dir "$OUT_ROOT/kunc_234" > "$OUT_ROOT/kunc_234.log" 2>&1 &

CUDA_VISIBLE_DEVICES=5 python3 tools/run_kimodo_all_tasks.py \
    --tasks E5 E6 E7 --max-samples 50 \
    --use-caption no \
    --output-dir "$OUT_ROOT/kunc_567" > "$OUT_ROOT/kunc_567.log" 2>&1 &

CUDA_VISIBLE_DEVICES=6 python3 tools/run_kimodo_all_tasks.py \
    --tasks E8 E10 E14 --max-samples 50 \
    --use-caption no \
    --output-dir "$OUT_ROOT/kunc_8_10_14" > "$OUT_ROOT/kunc_8_10_14.log" 2>&1 &

echo "Launched 6 KIMODO canon-rerun jobs on GPUs 0-2 + 4-6"
wait
echo "ALL DONE"
