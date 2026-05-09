#!/bin/bash
# Re-run E3/E8-D/E14/E15 with latest HyMotion M2M v2 checkpoints.
# 8 parallel jobs across 8 GPUs (one process per GPU). Heaviest E3 work
# is split across 2 GPUs per uncond model so E14/E15/E8 fit alongside.
# Designed for lzy_debug_machine_2 (8x V100-32GB).
#
# Usage:
#   bash scripts/eval/run_e3_e8d_e14_e15_latest_v2.sh [OUT_DIR]
#
# Output:
#   work_dirs/m2m_v2_e3_e8d_e14_e15_latest_<TS>/
#     <model>_<tag>/eval_v2_<TS>.json
#     <model>_<tag>/<model>/<task>_<setting>/npz/*.npz
#     logs/<model>_<tag>.log

set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

TS=${OUT_TS:-$(date +%Y%m%d_%H%M)}
OUT=${1:-work_dirs/m2m_v2_e3_e8d_e14_e15_latest_${TS}}
mkdir -p "$OUT/logs"
echo "OUT=$OUT" | tee "$OUT/logs/run_meta.txt"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "$OUT/logs/run_meta.txt"

EVAL=tools/eval_m2m_v2_all_tasks.py
COMMON="--max-samples 250 --num-steps 50 --save-npz"
# --use-rewritten only matters for caption models; harmless for uncond
COMMON_CAP="$COMMON --use-rewritten"

declare -a JOBS=(
    "0|uncond_global|--tasks E3 --settings adaptive every_5f every_10f"
    "1|uncond_global|--tasks E3 E8 E14 E15 --settings every_15f every_30f every_60f D L M default"
    "2|uncond_local|--tasks E3 --settings adaptive every_5f every_10f"
    "3|uncond_local|--tasks E3 E8 E14 E15 --settings every_15f every_30f every_60f D L M default"
    "4|caption_global_phase2|--tasks E3 --settings adaptive every_5f every_10f"
    "5|caption_global_phase2|--tasks E3 --settings every_15f every_30f every_60f"
    "6|caption_local_phase2|--tasks E3 --settings adaptive every_5f every_10f"
    "7|caption_local_phase2|--tasks E3 --settings every_15f every_30f every_60f"
)

declare -a PIDS=()
for spec in "${JOBS[@]}"; do
    IFS='|' read -r gpu model rest <<< "$spec"
    tag=$(echo "$rest" | tr -s ' ' '_' | tr -dc 'a-zA-Z0-9_+-' | cut -c1-80)
    job_dir="$OUT/${model}__${tag}"
    log="$OUT/logs/${model}__${tag}.log"
    mkdir -p "$job_dir"
    case "$model" in
        caption_*) cmd="python3 $EVAL --models $model $rest $COMMON_CAP --output-dir $job_dir" ;;
        *)         cmd="python3 $EVAL --models $model $rest $COMMON --output-dir $job_dir" ;;
    esac
    echo "[GPU $gpu] $cmd" | tee -a "$OUT/logs/run_meta.txt"
    CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "$log" 2>&1 &
    PIDS+=("$!")
    sleep 1
done

echo "Launched ${#PIDS[@]} jobs (PIDs: ${PIDS[*]})" | tee -a "$OUT/logs/run_meta.txt"
echo "Waiting for completion ..." | tee -a "$OUT/logs/run_meta.txt"

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  pid=$pid FAILED" | tee -a "$OUT/logs/run_meta.txt"
        FAIL=$((FAIL+1))
    fi
done
echo "Done. $FAIL job(s) failed." | tee -a "$OUT/logs/run_meta.txt"
date | tee -a "$OUT/logs/run_meta.txt"

echo "OUT=$OUT" > "$OUT/logs/last_out_dir.txt"
exit $FAIL
