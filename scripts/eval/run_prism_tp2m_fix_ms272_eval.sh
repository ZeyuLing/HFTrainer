#!/usr/bin/env bash
# MS-272 eval of the *fixed* (normalize-bug-corrected) PRISM TP2M generation
# for Table 2. Repack SMPLX npz -> canon272 row135 -> eval_motionstreamer_272.py
# (native + refk FID). No generation here; reads epoch15_fix outputs.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/ms272_table2_prism_fix}
PREP="$OUT_ROOT/prep"; LOG="$OUT_ROOT/logs"; RES="$OUT_ROOT/results"
mkdir -p "$PREP" "$LOG" "$RES"
ANNO=data/annotation/test_hml3d.json
MS_REL="ref_repo/MotionStreamer/MotionStreamer"
GEN=outputs/evaluation/prism_tp2m_epoch15_fix/h3d

echo "[start] $(date) out=$OUT_ROOT" | tee "$LOG/run.log"
bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true
if [ ! -f /dev/shm/eval272_epoch99.ckpt ]; then
  cp "$MS_REL/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt" /dev/shm/eval272_epoch99.ckpt 2>/dev/null || true
fi

ENTRIES=(
  "prism_c1|$GEN/cond1_depth_driven"
  "prism_c5|$GEN/cond5_depth_driven"
  "prism_c9|$GEN/cond9_depth_driven"
)

repack_one() {  # name src
  local name="$1" src="$2" dst="$PREP/$1"
  if [ -f "$dst/_DONE" ]; then echo "$dst"; return 0; fi
  mkdir -p "$dst"
  python3 scripts/eval/repack_pred_to_272ids.py --npz-dir "$src" \
    --anno-file "$ANNO" --out-dir "$dst" --workers 16 \
    > "$LOG/repack_$name.log" 2>&1 && touch "$dst/_DONE"
  echo "$dst"
}
eval_one() {  # name pred
  local name="$1" pred="$2" oj="$RES/$1.json"
  [ -s "$oj" ] && return 0
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$pred" --tag "$name" --also-refk --out-json "$oj" \
    > "$LOG/eval_$name.log" 2>&1
}

echo "[repack] $(date)" | tee -a "$LOG/run.log"
declare -A PRED_OF
for e in "${ENTRIES[@]}"; do
  IFS='|' read -r name src <<< "$e"
  if [ ! -e "$src" ]; then echo "[MISS] $name $src" | tee -a "$LOG/run.log"; continue; fi
  p="$(repack_one "$name" "$src")"; PRED_OF["$name"]="$p"
  echo "[repack] $name -> $p (n=$(ls "$p"/*.npz 2>/dev/null | wc -l))" | tee -a "$LOG/run.log"
done

echo "[eval] $(date)" | tee -a "$LOG/run.log"
for name in prism_c1 prism_c5 prism_c9; do
  [ -n "${PRED_OF[$name]:-}" ] || continue
  echo "[eval] $name $(date)" | tee -a "$LOG/run.log"
  eval_one "$name" "${PRED_OF[$name]}"
done
touch "$OUT_ROOT/_DONE"
echo "[done] $(date)" | tee -a "$LOG/run.log"

python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob("outputs/evaluation/ms272_table2_prism_fix/results/*.json")):
    d=json.load(open(f)); p=d.get("pred",{})
    print(os.path.basename(f), {
        "n": p.get("nb"),
        "r3": round(p["r_precision"][2],4) if p.get("r_precision") else None,
        "fid_native": round(p.get("fid_vs_gt_native",0),3),
        "fid_refk": round(p.get("fid_vs_gt_refk",0),3),
        "mm": round(p.get("matching_score",0),3),
        "div": round(p.get("diversity",0),3),
    })
PY
