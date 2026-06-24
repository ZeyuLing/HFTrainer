#!/usr/bin/env bash
# VerMo Table-3 (T2M) MotionHub MotionCLIP eval for all reusable methods.
# Waits for the in-flight HumanML3D eval to release the single GPU, then scores
# each method's MotionHub prediction dir with the MotionCLIP evaluator using the
# established MH protocol (chunk 64, n_repeats 20, seed 42, fwd 64, column).
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CKPT=checkpoints/motion_clip/motionclip_base_1p_aug_hq
ANNO=data/annotation/test_motionhub_t2m.json
DATA=data/motionhub
OUT=outputs/evaluation/t2m/motionhub_test/motionclip_vermo_t3_20260622
MET="$OUT/metrics"; LOG="$OUT/logs"
mkdir -p "$MET" "$LOG"
WLOG="$LOG/mh_driver.log"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WLOG"; }

# 1) wait for the HumanML3D eval to finish (frees the GPU)
log "waiting for HumanML3D eval to finish..."
while pgrep -f "eval_motionclip_table1_dirs.py" >/dev/null 2>&1; do
  sleep 30
done
log "HumanML3D eval done; starting MotionHub eval"

declare -A DIRS=(
  [mdm]="outputs/evaluation/motionhub_smpl135_fpsfix_0605_motionclip135/mdm_fixed"
  [mld]="outputs/evaluation/motionhub_smpl135_fpsfix_0605_motionclip135/mld_adapter"
  [vimogen]="outputs/evaluation/vimogen_t2m_0606/mh_rw_full0606_ow2_dn1_merged/motionclip135"
  [motionstreamer]="$OUT/pred_mc135/motionstreamer"
  [hymotion]="outputs/evaluation/hylite_t2m_rerun0607_rootalign/mh_rw_row2col_yaw"
  [tm2t]="outputs/evaluation/tm2t_0611/mc135/mh"
  [t2mgpt]="outputs/evaluation/aligned_hml263_baselines_0605_seq/motionclip135/mh/t2mgpt"
  [momask]="outputs/evaluation/motionhub_smpl135_fpsfix_0605_motionclip135/momask"
  [motiongpt]="outputs/evaluation/aligned_hml263_baselines_0605_seq/motionclip135/mh/motiongpt"
  [motiongpt3]="outputs/evaluation/motionhub_smpl135_fpsfix_0605_motionclip135/motiongpt3_fixed"
  [lom]="outputs/evaluation/lom_0611/mc135/mh"
  [gotozero]="outputs/evaluation/motionmillion_gtz/motionclip135_mh"
  [vermo]="outputs/evaluation/vermo_mh_motionclip_col_0610/pred_col"
)

ORDER=(mdm mld vimogen motionstreamer hymotion tm2t t2mgpt momask motiongpt motiongpt3 lom gotozero vermo)

# Real row (GT only): real R-P / MM / Div, FID=0 by definition
if [[ ! -s "$MET/real.json" ]]; then
  log "eval real (gt_only)"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt "$CKPT" --anno_file "$ANNO" --data_dir "$DATA" \
    --gt_only --rot6d_convention column \
    --chunk_size 64 --n_repeats 20 --seed 42 --forward_batch_size 64 \
    --out_json "$MET/real.json" > "$LOG/eval_real.log" 2>&1 \
    && log "done real" || log "FAIL real (see $LOG/eval_real.log)"
fi

for name in "${ORDER[@]}"; do
  dir="${DIRS[$name]}"
  oj="$MET/$name.json"
  if [[ -s "$oj" ]]; then log "skip $name (exists)"; continue; fi
  if [[ ! -d "$dir" ]]; then log "MISS $name dir=$dir"; continue; fi
  n=$(find "$dir" -maxdepth 1 \( -name '*.npy' -o -name '*.npz' \) | wc -l)
  log "eval $name n=$n dir=$dir"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt "$CKPT" --anno_file "$ANNO" --data_dir "$DATA" \
    --pred_dir "$dir" --rot6d_convention column \
    --chunk_size 64 --n_repeats 20 --seed 42 --forward_batch_size 64 \
    --out_json "$oj" > "$LOG/eval_${name}.log" 2>&1 \
    && log "done $name" || log "FAIL $name (see $LOG/eval_${name}.log)"
done

# combined summary
python3 - "$MET" > "$MET/summary.tsv" <<'PY'
import json, sys, glob, os
md = sys.argv[1]
def g(d,*ks):
    for k in ks:
        if k in d: return d[k]
    return float('nan')
rows=["method\tN\tR1\tR3\tFID\tMM\tDiv"]
for p in sorted(glob.glob(os.path.join(md,"*.json"))):
    name=os.path.splitext(os.path.basename(p))[0]
    d=json.load(open(p))
    r1=g(d,"r_precision_pred_top1_mean"); r3=g(d,"r_precision_pred_top3_mean")
    fid=g(d,"fid_mean"); mm=g(d,"mm_dist_pred_mean"); div=g(d,"diversity_pred_mean")
    rows.append(f"{name}\t{d.get('samples')}\t{r1:.4f}\t{r3:.4f}\t{fid:.4f}\t{mm:.4f}\t{div:.4f}")
print("\n".join(rows))
PY
cat "$MET/summary.tsv" | tee -a "$WLOG"
touch "$MET/_DONE"
log "ALL DONE"
