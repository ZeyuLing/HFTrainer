#!/usr/bin/env bash
# Table 4 all-ORIGINAL-caption recompute for the methods that were previously
# evaluated with rewritten captions. The MotionCLIP evaluator is aligned to the
# ORIGINAL MotionHub/HumanML3D captions (GT real R1 ~0.94 vs ~0.65 rewritten),
# so we unify the protocol to original captions across all methods.
#   - Real (gt_only)
#   - HY-Motion MH : use the ORIGINAL-generated predictions (mh_orig_row2col_yaw)
#   - ViMoGen H3D/MH : only rewritten-generated preds exist; eval w/ original.
# Already-original results are reused elsewhere (HML263 baselines original_eval,
# MotionGPT embed-repair rel_orig, MotionStreamer h3d/mh, HY-Motion h3d, GTZ h3d).
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CKPT=checkpoints/motion_clip/motionclip_base_1p_aug_hq
DATA=data/motionhub
H3D_ANNO=data/annotation/test_hml3d.json
MH_ANNO=data/annotation/test_motionhub_t2m.json
HYLITE=outputs/evaluation/hylite_t2m_rerun0607_rootalign
VIMOGEN=outputs/evaluation/vimogen_t2m_0606

OUT=outputs/evaluation/table4_allorig_remaining_0610
RES="$OUT/results"; LOG="$OUT/logs"; mkdir -p "$RES" "$LOG"
NGPU=8; CHUNK=64; NREP=20; SEED=42

# tag | anno | pred(GT for gt_only)
ENTRIES=(
  "real_h3d|$H3D_ANNO|GT"
  "real_mh|$MH_ANNO|GT"
  "hymotion_mh|$MH_ANNO|$HYLITE/mh_orig_row2col_yaw"
  "vimogen_h3d|$H3D_ANNO|$VIMOGEN/h3d_rw_full0606_ow2_dn1_merged/motionclip135"
  "vimogen_mh|$MH_ANNO|$VIMOGEN/mh_rw_full0606_ow2_dn1_merged/motionclip135"
)

run_one() {
  local tag=$1 anno=$2 pred=$3 gpu=$4
  local oj="$RES/$tag.json"
  [ -s "$oj" ] && { echo "[skip-done] $tag"; return 0; }
  local args=(--evaluator_ckpt "$CKPT" --anno_file "$anno" --data_dir "$DATA"
    --chunk_size "$CHUNK" --n_repeats "$NREP" --seed "$SEED"
    --forward_batch_size 64 --out_json "$oj")
  if [ "$pred" = "GT" ]; then args+=(--gt_only); else args+=(--pred_dir "$pred"); fi
  echo "[run] $tag gpu=$gpu pred=$pred (ORIGINAL caption) $(date)" | tee -a "$LOG/run.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    "${args[@]}" > "$LOG/$tag.log" 2>&1 || echo "[FAIL] $tag rc=$?" | tee -a "$LOG/run.log"
}

echo "[start] $(date)" | tee "$LOG/run.log"
i=0
for e in "${ENTRIES[@]}"; do
  IFS='|' read -r tag anno pred <<< "$e"
  if [ "$pred" != "GT" ] && [ ! -e "$pred" ]; then
    echo "[MISS] $tag pred=$pred" | tee -a "$LOG/run.log"; continue
  fi
  run_one "$tag" "$anno" "$pred" $((i % NGPU)) &
  i=$((i + 1))
done
wait

echo "[aggregate] $(date)" | tee -a "$LOG/run.log"
RES="$RES" python3 - <<'PY' | tee "$OUT/summary.txt"
import json, os, math
from pathlib import Path
root = Path(os.environ["RES"])
def f(d,k):
    v=d.get(k); return "nan" if v is None or (isinstance(v,float) and math.isnan(v)) else f"{float(v):.4f}"
for p in sorted(root.glob("*.json")):
    d=json.load(open(p)); real="real" in p.stem
    r1="r_precision_real_top1_mean" if real else "r_precision_pred_top1_mean"
    r3="r_precision_real_top3_mean" if real else "r_precision_pred_top3_mean"
    mm="mm_dist_real_mean" if real else "mm_dist_pred_mean"
    dv="diversity_real_mean" if real else "diversity_pred_mean"
    print(p.stem, "N", d.get("samples"), "R1", f(d,r1), "R3", f(d,r3),
          "FID", f(d,"fid_mean"), "MM", f(d,mm), "Div", f(d,dv), "RealR1", f(d,"r_precision_real_top1_mean"))
PY
touch "$OUT/_DONE"
echo "[done] $(date)" | tee -a "$LOG/run.log"
