#!/usr/bin/env bash
# Full-set FAIR-FIX MS-272 eval.
# Joints-only baselines are scored against the GT pushed through the IDENTICAL
# 263->IK->SMPL->272 chain (gtctrl_full), so the 263<->272 representation gap
# (~50 FID, shown to be a pure artifact: gtctrl-vs-gtctrl-chain FID==0.00)
# cancels and FID reflects true generation quality. `ours` (native SMPL) stays
# scored vs native-272.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
OUT=output/evaluation/mib_ms272_ikfix
D="$OUT/_fairfix_full"; LOG="$D/logs"; mkdir -p "$LOG"
EV=scripts/eval/eval_motionstreamer_272.py
NGPU=${NGPU:-8}
REF_SHM=/dev/shm/gt272ref_full     # fast-read chain GT reference

bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true

# 1) build the 263->IK->272 chain GT reference (full set) into /dev/shm
if [ ! -f "$REF_SHM/_DONE" ]; then
  echo "[build-ref] $(date)" | tee -a "$LOG/run.log"
  SRC="$OUT/gtctrl_full/repack272" OUT="$REF_SHM" \
    python3 scripts/eval/_build_gtctrl272.py > "$LOG/build_ref.log" 2>&1
  touch "$REF_SHM/_DONE"
fi
echo "[ref] n=$(ls "$REF_SHM"/*.npz 2>/dev/null|wc -l)" | tee -a "$LOG/run.log"

# 2) eval each joints-only baseline vs the chain reference (full set)
eval_chain() {  # tag pred gpu
  local tag="$1" pred="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python3 $EV --pred-dir "$pred" \
    --gt-272-dir "$REF_SHM" --tag "$tag" \
    --out-json "$D/${tag}.json" > "$LOG/eval_${tag}.log" 2>&1
  echo "[eval:$tag] done" | tee -a "$LOG/run.log"
}
# ours vs native-272 (native SMPL prediction; no chain)
eval_native() {  # tag pred gpu
  local tag="$1" pred="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python3 $EV --pred-dir "$pred" \
    --tag "$tag" --out-json "$D/${tag}.json" > "$LOG/eval_${tag}.log" 2>&1
  echo "[eval:$tag] done" | tee -a "$LOG/run.log"
}

eval_chain  condmdi   "$OUT/condmdi/repack272"   0 &
eval_chain  flowmdm   "$OUT/flowmdm/repack272"   1 &
eval_chain  motionlab "$OUT/motionlab/repack272" 2 &
eval_chain  kimodo    "$OUT/kimodo/repack272"    3 &
eval_chain  gtctrl    "$OUT/gtctrl_full/repack272" 4 &   # self -> ~0 (floor sanity)
eval_native ours      "$OUT/ours/row_npz"        5 &
wait

python3 - <<'PY' | tee "$D/summary.txt"
import json, os
d="output/evaluation/mib_ms272_ikfix/_fairfix_full"
print(f"{'method':12s} {'n':>5} {'floor':>6} {'FID':>8} {'R@1':>6} {'R@3':>6} {'MM':>7} {'Div':>7}")
for t in ["gtctrl","ours","condmdi","motionlab","flowmdm","kimodo"]:
    p=f"{d}/{t}.json"
    if not os.path.exists(p): print(f"{t:12s} MISSING"); continue
    j=json.load(open(p)); pr=j.get("pred",{}); gr=j.get("gt_real",{})
    rp=pr.get("r_precision",[float('nan')]*3)
    print(f"{t:12s} {j.get('ids_with_required_files','?'):>5} "
          f"{gr.get('self_fid_split_halves',float('nan')):6.2f} "
          f"{pr.get('fid_vs_gt_native',float('nan')):8.2f} "
          f"{rp[0]:6.3f} {rp[2]:6.3f} {pr.get('matching_score',float('nan')):7.3f} "
          f"{pr.get('diversity',float('nan')):7.3f}")
PY
touch "$D/_DONE"
echo "[fairfix-full-done] $(date)" | tee -a "$LOG/run.log"
