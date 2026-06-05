#!/bin/bash
# Monitor the 8 parallel PRISM clean-pipeline generation jobs (2 checkpoints x
# 4 nodes, NSHARDS=32). For each checkpoint, wait until ALL 4 node done-markers
# (_done_none_s{0,8,16,24}of32) are present, then repack SMPLX preds ->
# motion_135 272-ids, print a first-frame body_pose magnitude sanity check
# (clean vs the OLD distorted outputs), and run the MotionStreamer-272 evaluator.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
ANNO=data/annotation/test_hml3d.json
LOG=/tmp/prism_clean_eval_monitor.log
NSHARDS=32
SHARD_STARTS=(0 8 16 24)
echo "[monitor] start $(date)" | tee -a "$LOG"

# name|gen_out_dir|old_distorted_dir
JOBS=(
  "iter15k|outputs/evaluation/prism_paper_iter15000_clean0603/h3d|outputs/evaluation/prism_paper_iter15000_nomask/h3d/none"
  "ktspectral|outputs/evaluation/prism_kt_spectral_latest_clean0603/h3d|outputs/evaluation/prism_kt_spectral_epoch4/h3d/none"
)

firstframe_check () {  # newdir olddir tag
  python3 - "$1" "$2" "$3" <<'PY'
import sys, glob, numpy as np
new,old,tag=sys.argv[1],sys.argv[2],sys.argv[3]
def agg(d):
    fs=sorted(glob.glob(d+'/*.npz'))[:300]
    a=np.zeros(8);c=np.zeros(8);hi=0
    for f in fs:
        try: bp=np.abs(np.load(f,allow_pickle=True)['body_pose'].reshape(-1,63)).max(1)
        except: continue
        n=min(8,len(bp));a[:n]+=bp[:n];c[:n]+=1
        if len(bp)>0 and bp[0]>1.8: hi+=1
    return a/np.maximum(c,1), hi, len(fs)
na,nhi,nn=agg(new)
print(f'[{tag}] NEW first8 bodypose absmax:', np.round(na,2), f' frame0>1.8: {nhi}/{nn}')
try:
    oa,ohi,on=agg(old)
    print(f'[{tag}] OLD first8 bodypose absmax:', np.round(oa,2), f' frame0>1.8: {ohi}/{on}')
except Exception as e:
    print(f'[{tag}] OLD dir n/a: {e}')
PY
}

all_markers_present () {  # gdir -> 0 if all 4 markers exist
  local gdir="$1"
  for ss in "${SHARD_STARTS[@]}"; do
    [ -f "$gdir/_logs/_done_none_s${ss}of${NSHARDS}" ] || return 1
  done
  return 0
}

for spec in "${JOBS[@]}"; do
  IFS='|' read -r name gdir olddir <<< "$spec"
  echo "[monitor] waiting for $name: all ${#SHARD_STARTS[@]} node markers ..." | tee -a "$LOG"
  waited=0
  while ! all_markers_present "$gdir"; do
    have=0
    for ss in "${SHARD_STARTS[@]}"; do
      [ -f "$gdir/_logs/_done_none_s${ss}of${NSHARDS}" ] && have=$((have+1))
    done
    nz=$(ls "$gdir/none/"*.npz 2>/dev/null | wc -l)
    echo "[monitor] $(date +%H:%M) $name markers=$have/${#SHARD_STARTS[@]} npz=$nz waited=${waited}m" | tee -a "$LOG"
    sleep 180; waited=$((waited+3))
  done
  echo "[monitor] $name GENERATION DONE $(date) npz=$(ls $gdir/none/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG"

  ids=outputs/evaluation/_272ids_${name}_clean0603
  echo "[monitor] repack $name -> $ids" | tee -a "$LOG"
  python3 scripts/eval/repack_pred_to_272ids.py --npz-dir "$gdir/none" --anno-file "$ANNO" \
      --out-dir "$ids" --workers 16 >> "$LOG" 2>&1

  echo "[monitor] ==== first-frame sanity ($name) ====" | tee -a "$LOG"
  firstframe_check "$gdir/none" "$olddir" "$name" 2>&1 | tee -a "$LOG"

  echo "[monitor] ==== MotionStreamer-272 eval ($name) ====" | tee -a "$LOG"
  python3 scripts/eval/eval_motionstreamer_272.py --pred-dir "$ids" --tag ${name}_clean0603 \
      2>&1 | tee -a "$LOG" | tee "outputs/evaluation/_eval_${name}_clean0603.txt"
done

echo "[monitor] ALL DONE $(date)" | tee -a "$LOG"
echo "MONITOR_COMPLETE" | tee -a "$LOG"
