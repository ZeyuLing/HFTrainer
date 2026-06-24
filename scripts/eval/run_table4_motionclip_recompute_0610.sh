#!/usr/bin/env bash
# Unified Table 4 (tab:eval_t2m) MotionCLIP recompute — eval-only.
# All ready methods x {HumanML3D, MotionHub}, paperproto protocol:
#   evaluator = motionclip_base_1p_aug_hq, chunk=64, n_repeats=20, seed=42.
# Caption protocol per method matches its generation captions
#   (rewritten for HML263 baselines / Real / VerMo / ViMoGen / HY-Motion-MH;
#    original for HY-Motion-H3D / MotionStreamer).
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CKPT=checkpoints/motion_clip/motionclip_base_1p_aug_hq
DATA=data/motionhub
H3D_ANNO=data/annotation/test_hml3d.json
MH_ANNO=data/annotation/test_motionhub_t2m.json
H3D_RW=data/annotation/test_hml3d_rewritten.json
MH_RW=data/annotation/test_motionhub_t2m_rewritten.json

OUT=${OUT:-outputs/evaluation/table4_motionclip_recompute_0610}
RES="$OUT/results"; LOG="$OUT/logs"; mkdir -p "$RES" "$LOG"
NGPU=${NGPU:-8}
CHUNK=${CHUNK:-64}; NREP=${NREP:-20}; SEED=${SEED:-42}

ALIGN=outputs/evaluation/aligned_hml263_baselines_0605_seq/motionclip135
HYLITE=outputs/evaluation/hylite_t2m_rerun0607_rootalign
MS=outputs/evaluation/motionstreamer_align0606
VERMO=outputs/evaluation/prism_kt_spectral_epoch7_rw
VIMOGEN=outputs/evaluation/vimogen_t2m_0606

# tag | anno | pred(GT for gt_only) | caption(rw file | "" for original)
ENTRIES=(
  "real_h3d|$H3D_ANNO|GT|$H3D_RW"
  "real_mh|$MH_ANNO|GT|$MH_RW"
  "mdm_h3d|$H3D_ANNO|$ALIGN/h3d/mdm|$H3D_RW"
  "mdm_mh|$MH_ANNO|$ALIGN/mh/mdm|$MH_RW"
  "mld_h3d|$H3D_ANNO|$ALIGN/h3d/mld|$H3D_RW"
  "mld_mh|$MH_ANNO|$ALIGN/mh/mld|$MH_RW"
  "momask_h3d|$H3D_ANNO|$ALIGN/h3d/momask|$H3D_RW"
  "momask_mh|$MH_ANNO|$ALIGN/mh/momask|$MH_RW"
  "t2mgpt_h3d|$H3D_ANNO|$ALIGN/h3d/t2mgpt|$H3D_RW"
  "t2mgpt_mh|$MH_ANNO|$ALIGN/mh/t2mgpt|$MH_RW"
  "motiongpt3_h3d|$H3D_ANNO|$ALIGN/h3d/motiongpt3|$H3D_RW"
  "motiongpt3_mh|$MH_ANNO|$ALIGN/mh/motiongpt3|$MH_RW"
  "motiongpt_h3d|$H3D_ANNO|$ALIGN/h3d/motiongpt|$H3D_RW"
  "motiongpt_mh|$MH_ANNO|$ALIGN/mh/motiongpt|$MH_RW"
  "hymotion_h3d|$H3D_ANNO|$HYLITE/h3d_row2col_yaw|"
  "hymotion_mh|$MH_ANNO|$HYLITE/mh_rw_row2col_yaw|$MH_RW"
  "motionstreamer_h3d|$H3D_ANNO|$MS/h3d_all_npz|"
  "motionstreamer_mh|$MH_ANNO|$MS/mh_npz|"
  "vimogen_h3d|$H3D_ANNO|$VIMOGEN/h3d_rw_full0606_ow2_dn1_merged/motionclip135|$H3D_RW"
  "vimogen_mh|$MH_ANNO|$VIMOGEN/mh_rw_full0606_ow2_dn1_merged/motionclip135|$MH_RW"
  "vermo_h3d|$H3D_ANNO|$VERMO/h3d/depth_driven|$H3D_RW"
  "vermo_mh|$MH_ANNO|$VERMO/mh/depth_driven|$MH_RW"
)

run_one() {
  local tag=$1 anno=$2 pred=$3 cap=$4 gpu=$5
  local oj="$RES/$tag.json"
  [ -s "$oj" ] && { echo "[skip-done] $tag"; return 0; }
  local args=(--evaluator_ckpt "$CKPT" --anno_file "$anno" --data_dir "$DATA"
    --chunk_size "$CHUNK" --n_repeats "$NREP" --seed "$SEED"
    --forward_batch_size 64 --out_json "$oj")
  if [ "$pred" = "GT" ]; then args+=(--gt_only); else args+=(--pred_dir "$pred"); fi
  if [ -n "$cap" ]; then args+=(--rewritten_caption_file "$cap"); fi
  echo "[run] $tag gpu=$gpu pred=$pred cap=${cap:-<orig>} $(date)" | tee -a "$LOG/run.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    "${args[@]}" > "$LOG/$tag.log" 2>&1 || echo "[FAIL] $tag rc=$?" | tee -a "$LOG/run.log"
}

echo "[start] $(date) out=$OUT ngpu=$NGPU" | tee "$LOG/run.log"
i=0
for e in "${ENTRIES[@]}"; do
  IFS='|' read -r tag anno pred cap <<< "$e"
  if [ "$pred" != "GT" ] && [ ! -e "$pred" ]; then
    echo "[MISS] $tag pred=$pred (not found)" | tee -a "$LOG/run.log"; continue
  fi
  gpu=$((i % NGPU))
  run_one "$tag" "$anno" "$pred" "$cap" "$gpu" &
  i=$((i + 1))
  if (( i % NGPU == 0 )); then wait; fi
done
wait

echo "[aggregate] $(date)" | tee -a "$LOG/run.log"
RES="$RES" python3 - <<'PY' | tee "$OUT/summary.txt"
import json, os, math
from pathlib import Path
root = Path(os.environ["RES"])
def f(d, k):
    v = d.get(k)
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.4f}"
rows = []
for p in sorted(root.glob("*.json")):
    d = json.load(open(p))
    real = "real" in p.stem
    r1 = "r_precision_real_top1_mean" if real else "r_precision_pred_top1_mean"
    r3 = "r_precision_real_top3_mean" if real else "r_precision_pred_top3_mean"
    mm = "mm_dist_real_mean" if real else "mm_dist_pred_mean"
    dv = "diversity_real_mean" if real else "diversity_pred_mean"
    rows.append((p.stem, d.get("samples"), f(d, r1), f(d, r3), f(d, "fid_mean"), f(d, mm), f(d, dv)))
print(f"{'tag':<24}{'N':>6}  {'R1':>7}{'R3':>8}{'FID':>8}{'MM':>8}{'Div':>8}")
for r in rows:
    print(f"{r[0]:<24}{str(r[1]):>6}  {r[2]:>7}{r[3]:>8}{r[4]:>8}{r[5]:>8}{r[6]:>8}")
PY

touch "$OUT/_DONE"
echo "[done] $(date)" | tee -a "$LOG/run.log"
