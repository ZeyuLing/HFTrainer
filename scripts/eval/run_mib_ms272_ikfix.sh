#!/usr/bin/env bash
# MIB (motion in-betweening) MS-272 re-evaluation using the VALIDATED retarget path:
#   263 -> hml263_to_smpl_ik.py (hierarchical IK on SMPL rest skeleton, fps20->30,
#          floor-align) -> SMPLX npz -> repack_pred_to_272ids.py (row-major canon272)
#   -> eval_motionstreamer_272.py vs NATIVE GT-272.
# GT control should reproduce FID ~1.4 (PRISM "Real (HML3D->SMPL)" row).
set -uo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

OUT=${OUT:-output/evaluation/mib_ms272_ikfix}
LOG="$OUT/logs"; mkdir -p "$LOG"
SPLIT=output/evaluation/mib_h3d_full/_common_ids_272.txt
NGPU=${NGPU:-8}
MODEL_DIR=ref_repo/MDM/body_models

# name | 263_in_dir | ids_filter(optional, '-' = none)
ENTRIES=(
  "gtctrl|ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs|$SPLIT"
  "condmdi|output/evaluation/mib_h3d_full/_condmdi_mib263|-"
  "flowmdm|output/evaluation/flowmdm_impute/mib_full/hml263|-"
  "motionlab|output/evaluation/motionlab_impute/mib_full/hml263|-"
  "kimodo|output/evaluation/mib_h3d_full/_kimodo_mib263|-"
)

ik_method() {  # name in_dir ids
  local name="$1" in_dir="$2" ids="$3"
  local smplx="$OUT/$name/smplx"; mkdir -p "$smplx"
  echo "[ik:$name] $(date) in=$in_dir" | tee -a "$LOG/run.log"
  local idsarg=""; [ "$ids" != "-" ] && idsarg="--ids $ids"
  local pids=()
  for s in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$s python3 scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "$in_dir" --out-dir "$smplx" --model-dir "$MODEL_DIR" \
      $idsarg --source-fps 20 --target-fps 30 --device cuda --batch-size 256 \
      --floor-align --refine-iters 0 --skip-existing \
      --num-shards "$NGPU" --shard-index "$s" \
      > "$LOG/ik_${name}_s${s}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  echo "[ik:$name] done n=$(ls "$smplx"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"
}

repack_method() {  # name
  local name="$1"
  local smplx="$OUT/$name/smplx" rp="$OUT/$name/repack272"; mkdir -p "$rp"
  python3 scripts/eval/repack_pred_to_272ids.py --npz-dir "$smplx" \
    --id-passthrough --out-dir "$rp" --workers 32 \
    > "$LOG/repack_${name}.log" 2>&1
  echo "[repack:$name] n=$(ls "$rp"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"
}

eval_method() {  # name gpu
  local name="$1" gpu="$2"
  local rp="$OUT/$name/repack272" oj="$OUT/$name/ms272.json"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$rp" --tag "$name" --split "$SPLIT" --also-refk --out-json "$oj" \
    > "$LOG/eval_${name}.log" 2>&1
  echo "[eval:$name] -> $oj" | tee -a "$LOG/run.log"
}

echo "[start] $(date) OUT=$OUT" | tee "$LOG/run.log"
bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true

# Phase 1: IK + repack each method (IK uses all GPUs, run sequentially)
for e in "${ENTRIES[@]}"; do
  IFS='|' read -r name in_dir ids <<< "$e"
  [ -f "$OUT/$name/_ik_done" ] || { ik_method "$name" "$in_dir" "$ids"; touch "$OUT/$name/_ik_done"; }
  repack_method "$name"
done

# Phase 2: ours (already SMPL; test BOTH 6D conventions vs native GT)
OURS_SID=output/evaluation/mib_h3d_full/_ours_sid_npz
if [ -d "$OURS_SID" ]; then
  echo "[ours] preparing col/row variants $(date)" | tee -a "$LOG/run.log"
  python3 - <<'PY' 2>> "$LOG/ours_prep.log"
import glob, os, numpy as np
src="output/evaluation/mib_h3d_full/_ours_sid_npz"
col="output/evaluation/mib_ms272_ikfix/ours/col_npy"; os.makedirs(col, exist_ok=True)
rowdir="output/evaluation/mib_ms272_ikfix/ours/row_npz"; os.makedirs(rowdir, exist_ok=True)
n=0
for p in glob.glob(os.path.join(src,'*.npz')):
    sid=os.path.splitext(os.path.basename(p))[0]
    z=np.load(p, allow_pickle=True)
    m=np.asarray(z['motion_135'], dtype=np.float32)
    np.save(os.path.join(col, sid+'.npy'), m)                 # treat as column-major
    np.savez(os.path.join(rowdir, sid+'.npz'), motion_135=m)  # treat as row-major (as-is)
    n+=1
print('ours prepared', n)
PY
  # col variant: repack converts column->row
  mkdir -p "$OUT/ours/col_repack272"
  python3 scripts/eval/repack_pred_to_272ids.py --col-npy-dir "$OUT/ours/col_npy" \
    --id-passthrough --out-dir "$OUT/ours/col_repack272" --workers 32 \
    > "$LOG/repack_ours_col.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$OUT/ours/col_repack272" --tag ours_col --split "$SPLIT" --also-refk \
    --out-json "$OUT/ours/ms272_col.json" > "$LOG/eval_ours_col.log" 2>&1
  # row variant: feed motion_135 as-is (eval does motion135_to_272 assuming row)
  CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$OUT/ours/row_npz" --tag ours_row --split "$SPLIT" --also-refk \
    --out-json "$OUT/ours/ms272_row.json" > "$LOG/eval_ours_row.log" 2>&1
fi

# Phase 3: eval all baselines+gtctrl in parallel
idx=0
for e in "${ENTRIES[@]}"; do
  IFS='|' read -r name in_dir ids <<< "$e"
  eval_method "$name" $((idx % NGPU)) &
  idx=$((idx+1)); (( idx % NGPU == 0 )) && wait
done
wait

# summary
python3 - <<'PY' | tee "$OUT/summary.txt"
import json, glob, os
root="output/evaluation/mib_ms272_ikfix"
def show(tag, p):
    if not os.path.exists(p): print(f"{tag:14s} MISSING"); return
    d=json.load(open(p)); pr=d.get("pred",{})
    rp=pr.get("r_precision",[float('nan')]*3)
    print(f"{tag:14s} n={d.get('ids_with_required_files','?')} "
          f"R1={rp[0]:.3f} R3={rp[2]:.3f} "
          f"FID={pr.get('fid_vs_gt_native',float('nan')):.3f} "
          f"MM={pr.get('matching_score',float('nan')):.3f} "
          f"Div={pr.get('diversity',float('nan')):.3f}")
for name in ["gtctrl","condmdi","flowmdm","motionlab","kimodo"]:
    show(name, f"{root}/{name}/ms272.json")
show("ours_col", f"{root}/ours/ms272_col.json")
show("ours_row", f"{root}/ours/ms272_row.json")
PY
touch "$OUT/_DONE"
echo "[done] $(date)" | tee -a "$LOG/run.log"
