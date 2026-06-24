#!/usr/bin/env bash
# Re-evaluate already-generated baselines (MDM / MLD / MotionGPT3) under MS-272.
# Their SMPL->motionclip135 outputs already exist; we only repack to canon-272 ids
# and run the MotionStreamer-272 evaluator (same path as HY in gtlen_watch.sh).
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

ANNO=data/annotation/test_hml3d.json
SRC_ROOT=outputs/evaluation/humanml3d_smpl135_fpsfix_v5_fixed0604_motionclip135_v6
OUT=outputs/evaluation/ms272_t1_fill_0609
LOG="$OUT/logs"; RES="$OUT/results"
mkdir -p "$LOG" "$RES"

# ensure GT-272 + evaluator cached (idempotent; reuses /dev/shm if present)
bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true

# tag|src_subdir
ENTRIES=(
  "mdm|mdm_fixed"
  "mld|mld_v1_rootfix"
  "motiongpt3|motiongpt3_fixed"
)

for e in "${ENTRIES[@]}"; do
  tag="${e%%|*}"; sub="${e##*|}"
  src="$SRC_ROOT/$sub"; prep="$OUT/prep/$tag"; oj="$RES/$tag.json"
  echo "[$(date +%H:%M:%S)] $tag: repack $src -> $prep"
  mkdir -p "$prep"
  python3 scripts/eval/repack_pred_to_272ids.py --col-npy-dir "$src" \
    --anno-file "$ANNO" --out-dir "$prep" --workers 16 \
    > "$LOG/repack_$tag.log" 2>&1
  n=$(python3 -c "import os,sys;d=sys.argv[1];print(sum(1 for e in os.scandir(d) if e.name.endswith('.npz')) if os.path.isdir(d) else 0)" "$prep")
  echo "[$(date +%H:%M:%S)] $tag: prep n=$n -> eval"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$prep" --tag "$tag" --also-refk --out-json "$oj" \
    > "$LOG/eval_$tag.log" 2>&1 \
    || CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_motionstreamer_272.py \
         --pred-dir "$prep" --tag "$tag" --out-json "$oj" \
         >> "$LOG/eval_$tag.log" 2>&1 || true
  if [ -s "$oj" ]; then
    python3 -c "import json;d=json.load(open('$oj'));p=d.get('pred',{});rp=p.get('r_precision',[None,None,None]);print('[RESULT] $tag FID=%.2f R1=%.3f R3=%.3f MM=%.2f Div=%.2f'%(p.get('fid_vs_gt_native',float('nan')),rp[0] or 0,(rp[2] if len(rp)>2 else 0) or 0,p.get('matching_score',float('nan')),p.get('diversity',float('nan'))))" 2>/dev/null | tee -a "$LOG/summary.txt"
  else
    echo "[RESULT] $tag FAILED (see $LOG/eval_$tag.log)" | tee -a "$LOG/summary.txt"
  fi
done
echo "[$(date +%H:%M:%S)] ALL DONE -> $RES"
