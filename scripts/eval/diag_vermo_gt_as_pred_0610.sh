#!/usr/bin/env bash
# Diagnostic: take VerMo's per-case GT (target_raw_motion, row-major 135D),
# run the SAME row->col + yaw-align conversion used for predictions, then
# evaluate it as if it were the prediction. If R1 ~= real (~0.94) the recipe
# is correct and the low VerMo pred R1 reflects prediction quality; if R1 stays
# low (~0.08) the conversion/format is still mismatched.
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT=output/evaluation/vermo_paper_t2m_m2t_ckpt22000_20260605_full
OUT=outputs/evaluation/vermo_mh_gtdiag_0610
RAW="$OUT/targets_raw135"; PREDC="$OUT/targets_col"; MET="$OUT/metrics"; LOG="$OUT/_logs"
mkdir -p "$RAW" "$PREDC" "$MET" "$LOG"
CKPT=checkpoints/motion_clip/motionclip_base_1p_aug_hq

echo "[export targets] $(date)" | tee "$LOG/run.log"
python3 - <<'PY' 2>&1 | tee -a "$LOG/run.log"
import json, numpy as np
from pathlib import Path
root = Path("output/evaluation/vermo_paper_t2m_m2t_ckpt22000_20260605_full")
man = json.load(open(root/"manifest.json"))
out = Path("outputs/evaluation/vermo_mh_gtdiag_0610/targets_raw135")
n=0; miss=0
for c in man["cases"]:
    if c.get("task")!="t2m": continue
    sk = c["overview"]["source_key"]
    # target_raw path
    tp=None
    for t in c.get("targets",[]):
        if t.get("source")=="raw": tp=root/t["path"]; break
    if tp is None or not tp.exists(): miss+=1; continue
    d=np.load(str(tp), allow_pickle=True)
    m=np.asarray(d["motion_135"], dtype=np.float32)
    np.save(str(out/f"{sk}.npy"), m); n+=1
print(f"[export done] wrote={n} miss={miss}")
PY

echo "[convert row->col+yaw] $(date)" | tee -a "$LOG/run.log"
python3 scripts/eval/convert_hylite135_to_motionclip_col.py \
  --src-dir "$RAW" --out-dir "$PREDC" \
  --anno-file data/annotation/test_motionhub_t2m.json --data-dir data/motionhub \
  --align-to-gt-root --align-root-mode yaw --overwrite --workers 16 \
  > "$LOG/convert.log" 2>&1
echo "[convert done] n=$(ls "$PREDC"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"

echo "[eval GT-as-pred, original caption] $(date)" | tee -a "$LOG/run.log"
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt "$CKPT" --anno_file data/annotation/test_motionhub_t2m.json \
  --data_dir data/motionhub --pred_dir "$PREDC" \
  --chunk_size 64 --n_repeats 20 --seed 42 --forward_batch_size 64 \
  --out_json "$MET/gtdiag_orig_c64.json" > "$LOG/eval.log" 2>&1

python3 - <<'PY' | tee "$MET/summary.txt"
import json
d=json.load(open("outputs/evaluation/vermo_mh_gtdiag_0610/metrics/gtdiag_orig_c64.json"))
print("GT-as-pred  N", d.get("samples"),
      "predR1", round(d["r_precision_pred_top1_mean"],4),
      "predR3", round(d["r_precision_pred_top3_mean"],4),
      "realR1", round(d["r_precision_real_top1_mean"],4),
      "FID", round(d["fid_mean"],4),
      "predDiv", round(d["diversity_pred_mean"],3))
PY
touch "$MET/_DONE"
echo "[done] $(date)" | tee -a "$LOG/run.log"
