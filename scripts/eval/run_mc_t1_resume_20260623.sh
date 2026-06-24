#!/usr/bin/env bash
# Resume the interrupted 2026-06-19 HumanML3D MotionCLIP Table-1 sweep.
# Each method runs on its own GPU (one host, 8 GPUs). Per-method results land in
# the original eval/results/<method>.json so they merge with the already-done
# real/motiongpt3/mld/momask. The shared summary.* is ignored (aggregated later).
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

B=outputs/evaluation/t2m/humanml3d_official_test/motionclip_table1_20260619
MC=$B/motionclip135
EVAL=$B/eval
LOG=$B/logs
MF=outputs/tmp/20260623_mc_t1_resume
mkdir -p "$EVAL/results" "$LOG" "$MF"

CKPT=checkpoints/motion_clip/motionclip_base_1p_aug_hq
ANNO=data/annotation/test_hml3d_official272_gtlen.json

METHODS=(mogents mdm t2mgpt flowmdm motionlab kimodo gotozero motionstreamer)

i=0
for m in "${METHODS[@]}"; do
  d="$MC/$m"
  if [[ ! -d "$d" ]]; then echo "[skip] $m missing $d"; continue; fi
  n=$(find "$d" -maxdepth 1 -name '*.npy' | wc -l)
  if [[ "$n" -eq 0 ]]; then echo "[skip] $m empty"; continue; fi
  mf="$MF/$m.tsv"
  printf "%s\t%s\n" "$m" "$d" > "$mf"
  echo "[launch] $m gpu=$i n=$n"
  CUDA_VISIBLE_DEVICES=$i nohup python3 scripts/eval/eval_motionclip_table1_dirs.py \
    --evaluator-ckpt "$CKPT" \
    --anno-file "$ANNO" \
    --data-dir . \
    --real-dir "$MC/real" \
    --pred-manifest "$mf" \
    --out-dir "$EVAL" \
    --min-frames 60 --max-frames 300 \
    --chunk-size 32 --forward-batch-size 32 --n-repeats 20 --seed 0 \
    > "$LOG/resume_${m}.log" 2>&1 &
  i=$((i + 1))
done
echo "[launched] $i jobs; pids: $(jobs -p | tr '\n' ' ')"
wait
echo "[all-finished] $(date -Is)"
