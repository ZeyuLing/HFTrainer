#!/usr/bin/env bash
# Phase 2: full-set MS-272 eval (no --split) to escape the small-N FID floor.
# self-FID on the 1817 common subset was 15.67 (paper expects ~0.002 on full test);
# evaluating on each method's full output vs full native GT reproduces the PRISM
# "Real (HML3D->SMPL)" ~1.4 control and gives credible absolute FID.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
OUT=output/evaluation/mib_ms272_ikfix
LOG="$OUT/logs"; mkdir -p "$LOG"
NGPU=${NGPU:-8}
MODEL_DIR=ref_repo/MDM/body_models

echo "[full-start] $(date)" | tee -a "$LOG/run_full.log"
bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true

# gtctrl: IK the FULL GT-263 test set (all ids present in condmdi repack = 4012)
GT_SMPLX="$OUT/gtctrl_full/smplx"; GT_RP="$OUT/gtctrl_full/repack272"; mkdir -p "$GT_SMPLX" "$GT_RP"
if [ ! -f "$OUT/gtctrl_full/_ik_done" ]; then
  echo "[ik:gtctrl_full] $(date)" | tee -a "$LOG/run_full.log"
  pids=()
  for s in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$s python3 scripts/eval/hml263_to_smpl_ik.py \
      --in-dir ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs \
      --out-dir "$GT_SMPLX" --model-dir "$MODEL_DIR" \
      --source-fps 20 --target-fps 30 --device cuda --batch-size 256 \
      --floor-align --refine-iters 0 --skip-existing \
      --num-shards "$NGPU" --shard-index "$s" \
      > "$LOG/ik_gtctrl_full_s${s}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  touch "$OUT/gtctrl_full/_ik_done"
fi
python3 scripts/eval/repack_pred_to_272ids.py --npz-dir "$GT_SMPLX" \
  --id-passthrough --out-dir "$GT_RP" --workers 32 > "$LOG/repack_gtctrl_full.log" 2>&1
echo "[repack:gtctrl_full] n=$(ls "$GT_RP"/*.npz 2>/dev/null|wc -l)" | tee -a "$LOG/run_full.log"

# eval helper (no --split => full per-method set vs full native GT)
eval_full() {  # tag pred gpu
  local tag="$1" pred="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$pred" --tag "$tag" --also-refk \
    --out-json "$OUT/_full/${tag}.json" > "$LOG/evalfull_${tag}.log" 2>&1
  echo "[evalfull:$tag] done" | tee -a "$LOG/run_full.log"
}
mkdir -p "$OUT/_full"
eval_full gtctrl   "$GT_RP"                       0 &
eval_full condmdi  "$OUT/condmdi/repack272"       1 &
eval_full flowmdm  "$OUT/flowmdm/repack272"       2 &
eval_full motionlab "$OUT/motionlab/repack272"    3 &
eval_full kimodo   "$OUT/kimodo/repack272"        4 &
eval_full ours     "$OUT/ours/row_npz"            5 &
wait

python3 - <<'PY' | tee "$OUT/_full/summary.txt"
import json, os
root="output/evaluation/mib_ms272_ikfix/_full"
def show(tag):
    p=f"{root}/{tag}.json"
    if not os.path.exists(p): print(f"{tag:12s} MISSING"); return
    d=json.load(open(p)); pr=d.get("pred",{}); gr=d.get("gt_real",{})
    rp=pr.get("r_precision",[float('nan')]*3)
    print(f"{tag:12s} n={d.get('ids_with_required_files','?'):>4} "
          f"selfFID={gr.get('self_fid', d.get('real_self_fid',float('nan'))):.3f} "
          f"R1={rp[0]:.3f} R3={rp[2]:.3f} "
          f"FID={pr.get('fid_vs_gt_native',float('nan')):.3f} "
          f"MM={pr.get('matching_score',float('nan')):.3f} "
          f"Div={pr.get('diversity',float('nan')):.3f}")
for t in ["gtctrl","ours","condmdi","motionlab","flowmdm","kimodo"]:
    show(t)
PY
touch "$OUT/_full/_DONE"
echo "[full-done] $(date)" | tee -a "$LOG/run_full.log"
