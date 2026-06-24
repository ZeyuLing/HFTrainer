#!/usr/bin/env bash
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
OUT=output/evaluation/mib_ms272_ikfix
REF="$OUT/gtctrl/gt272ref"
SPLIT="$OUT/common1817.txt"
D="$OUT/_fairfix"; mkdir -p "$D"
EV=scripts/eval/eval_motionstreamer_272.py

run() { # tag pred extra
  local tag="$1" pred="$2"; shift 2
  CUDA_VISIBLE_DEVICES=0 python3 $EV --pred-dir "$pred" --split "$SPLIT" --seed 0 \
    --tag "$tag" --out-json "$D/${tag}.json" "$@" > "$D/${tag}.log" 2>&1
  echo "[done] $tag"
}

# A: condmdi vs native 272 (current protocol, on common subset)
run condmdi_native "$OUT/condmdi/repack272"
# B: condmdi vs 263->IK chain GT ref (fair protocol)
run condmdi_chain  "$OUT/condmdi/repack272" --gt-272-dir "$REF"
# C: gtctrl self vs 263->IK chain ref (floor)
run gtctrl_chain   "$OUT/gtctrl/repack272"  --gt-272-dir "$REF"
# D: gtctrl vs native 272 (reproduce ~47)
run gtctrl_native  "$OUT/gtctrl/repack272"

echo "===== SUMMARY ====="
python3 - <<'PY'
import json,os
d="output/evaluation/mib_ms272_ikfix/_fairfix"
for t in ["gtctrl_native","gtctrl_chain","condmdi_native","condmdi_chain"]:
    p=f"{d}/{t}.json"
    if not os.path.exists(p): print(t,"MISSING"); continue
    j=json.load(open(p)); pr=j.get("pred",{}); gr=j.get("gt_real",{})
    print(f"{t:16s} n={j.get('ids_with_required_files'):>4} "
          f"selfFID={gr.get('self_fid_split_halves',float('nan')):6.2f} "
          f"FID={pr.get('fid_vs_gt_native',float('nan')):7.2f} "
          f"R@3={pr.get('r_precision',[0,0,0])[2]:.3f}")
PY
touch "$D/_DONE"
