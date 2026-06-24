#!/usr/bin/env bash
# Local watcher for the GT-length re-inference jobs (PRISM e17 depth_driven + HY).
# Polls the two Taiji gen dirs; when each is stable & near-full, repacks to a
# canonical-id canon272 prep dir, runs MS-272 eval, and (after both) restarts the
# t2m_compare viz so it shows ours/HY at the native GT length.
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

ANNO=data/annotation/test_hml3d.json
GTANNO=data/annotation/test_hml3d_gtlen.json
WDIR=outputs/evaluation/gtlen_eval
LOG="$WDIR/logs"; RES="$WDIR/results"
mkdir -p "$LOG" "$RES"
WLOG="$LOG/watch.log"

PRISM_GEN=outputs/evaluation/prism_kt_spectral_epoch17_gtlen/h3d/depth_driven
PRISM_PREP=outputs/evaluation/prism_kt_spectral_epoch17_gtlen/prep/ours_e17_gtlen
HY_GEN=outputs/evaluation/hylite_gtlen/h3d/motionclip135
HY_COL=outputs/evaluation/hylite_gtlen/h3d_row2col_yaw
HY_PREP=outputs/evaluation/hylite_gtlen/prep/hymotion_gtlen

MIN_OK=3500          # min files to accept as "full enough"
STALL_POLLS=2        # consecutive unchanged polls => stalled/complete
INTERVAL=180

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WLOG"; }
cnt(){ python3 -c "import os,sys;d=sys.argv[1];ext=sys.argv[2];print(sum(1 for e in os.scandir(d) if e.name.endswith(ext)) if os.path.isdir(d) else 0)" "$1" "$2"; }

# cache GT-272 data + evaluator ckpt to /dev/shm (idempotent)
bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true

eval_prep(){  # name  pred_dir
  local name="$1" pred="$2" oj="$RES/$name.json"
  [ -s "$oj" ] && { log "eval $name already done"; return 0; }
  log "eval $name (pred=$pred) ..."
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$pred" --tag "$name" --also-refk --out-json "$oj" \
    > "$LOG/eval_$name.log" 2>&1 \
    || CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_motionstreamer_272.py \
         --pred-dir "$pred" --tag "$name" --out-json "$oj" \
         >> "$LOG/eval_$name.log" 2>&1 || true
  [ -s "$oj" ] && log "eval $name DONE: $(python3 -c "import json;d=json.load(open('$oj'));print({k:round(v,4) for k,v in d.items() if isinstance(v,(int,float))})" 2>/dev/null)"
}

do_prism(){
  [ -f "$PRISM_PREP/_DONE" ] && return 0
  log "PRISM repack $PRISM_GEN -> $PRISM_PREP"
  mkdir -p "$PRISM_PREP"
  python3 scripts/eval/repack_pred_to_272ids.py --npz-dir "$PRISM_GEN" \
    --anno-file "$GTANNO" --out-dir "$PRISM_PREP" --workers 16 \
    > "$LOG/repack_prism.log" 2>&1 && touch "$PRISM_PREP/_DONE"
  log "PRISM prep n=$(cnt "$PRISM_PREP" .npz)"
  eval_prep ours_e17_gtlen "$PRISM_PREP"
}

do_hy(){
  [ -f "$HY_PREP/_DONE" ] && return 0
  log "HY convert (row2col+yaw) $HY_GEN -> $HY_COL"
  mkdir -p "$HY_COL"
  python3 scripts/eval/convert_hylite135_to_motionclip_col.py \
    --src-dir "$HY_GEN" --out-dir "$HY_COL" --anno-file "$ANNO" \
    --data-dir data/motionhub --align-to-gt-root --align-root-mode yaw \
    --overwrite --workers 16 > "$LOG/convert_hy.log" 2>&1
  log "HY repack $HY_COL -> $HY_PREP"
  mkdir -p "$HY_PREP"
  python3 scripts/eval/repack_pred_to_272ids.py --col-npy-dir "$HY_COL" \
    --anno-file "$ANNO" --out-dir "$HY_PREP" --workers 16 \
    > "$LOG/repack_hy.log" 2>&1 && touch "$HY_PREP/_DONE"
  log "HY prep n=$(cnt "$HY_PREP" .npz)"
  eval_prep hymotion_gtlen "$HY_PREP"
}

restart_viz(){
  log "restarting t2m_compare viz on :8086"
  pkill -f "t2m_compare/app.py" 2>/dev/null || true
  sleep 3
  ( cd motion_annot_web/t2m_compare && PYTHONPATH="$PWD/../..:${PYTHONPATH:-}" \
      nohup python3 app.py --port 8086 > "$OLDPWD/$LOG/viz.log" 2>&1 & )
  sleep 5
  log "viz restarted (see $LOG/viz.log)"
}

log "watcher start. PRISM_GEN=$PRISM_GEN HY_GEN=$HY_GEN"
pp=-1; pp_stall=0; hp=-1; hp_stall=0; prism_done=0; hy_done=0
while :; do
  if [ "$prism_done" -eq 0 ]; then
    n=$(cnt "$PRISM_GEN" .npz)
    [ "$n" -eq "$pp" ] && pp_stall=$((pp_stall+1)) || pp_stall=0
    pp=$n
    log "PRISM gen=$n (stall=$pp_stall)"
    if [ "$n" -ge "$MIN_OK" ] && [ "$pp_stall" -ge "$STALL_POLLS" ]; then
      do_prism; prism_done=1
    fi
  fi
  if [ "$hy_done" -eq 0 ]; then
    n=$(cnt "$HY_GEN" .npy)
    [ "$n" -eq "$hp" ] && hp_stall=$((hp_stall+1)) || hp_stall=0
    hp=$n
    log "HY gen=$n (stall=$hp_stall)"
    if [ "$n" -ge "$MIN_OK" ] && [ "$hp_stall" -ge "$STALL_POLLS" ]; then
      do_hy; hy_done=1
    fi
  fi
  if [ "$prism_done" -eq 1 ] && [ "$hy_done" -eq 1 ]; then
    restart_viz
    log "ALL DONE. PRISM + HY at GT length repacked, evaluated, viz restarted."
    break
  fi
  sleep "$INTERVAL"
done
