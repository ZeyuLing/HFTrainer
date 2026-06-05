#!/bin/bash
# Monitor the two PRISM fixed-generation Taiji jobs; when each finishes (done
# marker written by run_gen_node.sh), repack SMPLX preds -> motion_135 272-ids
# and run the MotionStreamer-272 evaluator. Also prints a first-frame body_pose
# magnitude sanity check (new vs old broken-fix outputs) to confirm the
# start-of-sequence distortion is gone.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
ANNO=data/annotation/test_hml3d.json
LOG=/tmp/prism_fix_eval_monitor.log
echo "[monitor] start $(date)" | tee -a "$LOG"

# name|gen_out_dir|taiji_flag|old_broken_dir
JOBS=(
  "iter15k|outputs/evaluation/prism_paper_iter15000_fix0603/h3d|prism_iter15k_fix0603|outputs/evaluation/prism_paper_iter15000_nomask/h3d/none"
  "ktspectral|outputs/evaluation/prism_kt_spectral_latest_fix0603/h3d|prism_ktspectral_fix0603|outputs/evaluation/prism_kt_spectral_epoch4/h3d/none"
)

firstframe_check () {  # newdir olddir tag
  python3 - "$1" "$2" "$3" <<'PY'
import sys, glob, numpy as np
new,old,tag=sys.argv[1],sys.argv[2],sys.argv[3]
def agg(d):
    fs=sorted(glob.glob(d+'/*.npz'))[:200]
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

for spec in "${JOBS[@]}"; do
  IFS='|' read -r name gdir flag olddir <<< "$spec"
  echo "[monitor] waiting for $name ($flag) done-marker ..." | tee -a "$LOG"
  waited=0
  while [ ! -f "$gdir/_logs/_done_none_s0of8" ]; do
    st=$(taiji_client trl 2>/dev/null | grep -E "\b$flag\b" | awk -F'|' '{print $4}' | tr -d ' ')
    nz=$(ls "$gdir/none/"*.npz 2>/dev/null | wc -l)
    echo "[monitor] $(date +%H:%M) $name status=${st:-?} npz=$nz waited=${waited}m" | tee -a "$LOG"
    # bail if the job vanished/failed and produced nothing for a long time
    if [ "${st:-}" = "" ] && [ "$nz" -eq 0 ] && [ "$waited" -gt 30 ]; then
      echo "[monitor] $name appears not running and no output after ${waited}m; continuing to wait but flagging" | tee -a "$LOG"
    fi
    sleep 180; waited=$((waited+3))
  done
  echo "[monitor] $name GENERATION DONE $(date) npz=$(ls $gdir/none/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG"

  ids=outputs/evaluation/_272ids_${name}_fix0603
  echo "[monitor] repack $name -> $ids" | tee -a "$LOG"
  python3 scripts/eval/repack_pred_to_272ids.py --npz-dir "$gdir/none" --anno-file "$ANNO" \
      --out-dir "$ids" --workers 16 >> "$LOG" 2>&1

  echo "[monitor] ==== first-frame sanity ($name) ====" | tee -a "$LOG"
  firstframe_check "$gdir/none" "$olddir" "$name" 2>&1 | tee -a "$LOG"

  echo "[monitor] ==== MotionStreamer-272 eval ($name) ====" | tee -a "$LOG"
  python3 scripts/eval/eval_motionstreamer_272.py --pred-dir "$ids" --tag ${name}_fix0603 \
      2>&1 | tee -a "$LOG" | tee "outputs/evaluation/_eval_${name}_fix0603.txt"
done

echo "[monitor] ALL DONE $(date)" | tee -a "$LOG"
echo "MONITOR_COMPLETE" | tee -a "$LOG"
