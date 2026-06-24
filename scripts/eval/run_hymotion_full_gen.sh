#!/usr/bin/env bash
# HY-Motion-1.0 FULL (1.04B) T2M generation for Table 4 (H3D + MotionHub).
#   - H3D : original HumanML3D captions (no rewrite)
#   - MH  : REWRITTEN captions (HY-Motion's official input pipeline)
# Outputs raw 135-dim npy named by annotation id; convert+eval done separately.
# Designed for an 8-GPU single node (Taiji).
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYMOTION_T2M_CONFIG="configs/hymotion_t2m/hymotion_t2m_201dim_full.py"

OUT_ROOT="${OUT_ROOT:-outputs/evaluation/hymotion_full_0611}"
H3D_OUT="$OUT_ROOT/h3d/raw135"
MH_OUT="$OUT_ROOT/mh_rw/raw135"
LOG="$OUT_ROOT/_logs"
mkdir -p "$H3D_OUT" "$MH_OUT" "$LOG"

NGPU="${NGPU:-8}"
BATCH="${BATCH:-8}"
STEPS="${STEPS:-50}"
CFG="${CFG:-5.0}"

gen() {
  local tag="$1" anno="$2" outdir="$3" capfile="$4"
  local capargs=()
  [ -n "$capfile" ] && capargs=(--caption-file "$capfile")
  echo "[gen] $tag anno=$anno cap=${capfile:-<orig>} -> $outdir $(date)" | tee -a "$LOG/run.log"
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES="$i" python3 scripts/eval/hylite_t2m_anno_infer.py \
      --anno-file "$anno" \
      "${capargs[@]}" \
      --data-dir data/motionhub \
      --out-dir "$outdir" \
      --num-shards "$NGPU" --shard-index "$i" --gpu "$i" \
      --batch-size "$BATCH" --num-steps "$STEPS" --cfg-scale "$CFG" \
      --skip-existing \
      > "$LOG/${tag}_g${i}.log" 2>&1 &
  done
  wait
  echo "[gen] $tag done n=$(ls "$outdir" 2>/dev/null | wc -l) $(date)" | tee -a "$LOG/run.log"
}

# 1) HumanML3D — original captions
gen h3d data/annotation/test_hml3d.json "$H3D_OUT" ""

# 2) MotionHub — REWRITTEN captions (HY-Motion rewrite pipeline)
gen mhrw data/annotation/test_motionhub_t2m.json "$MH_OUT" \
  data/annotation/test_motionhub_t2m_rewritten.json

touch "$OUT_ROOT/_GEN_DONE"
echo "[all] generation complete $(date)" | tee -a "$LOG/run.log"
