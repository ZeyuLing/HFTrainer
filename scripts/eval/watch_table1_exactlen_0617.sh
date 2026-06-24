#!/usr/bin/env bash
# Watch exact-length Table-1 reruns. Once generation dirs are complete/stable,
# repack to MotionStreamer-272 prep format, audit lengths, and run MS evaluator.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

ANNO=${ANNO:-data/annotation/test_hml3d_official272_gtlen.json}
DATA_DIR=${DATA_DIR:-.}
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/table1_exactlen_0617}
LOG="$OUT_ROOT/logs"
RES="$OUT_ROOT/results"
AUD="$OUT_ROOT/length_audit"
mkdir -p "$LOG" "$RES" "$AUD"
WLOG="$LOG/watch.log"

PRISM_GEN=${PRISM_GEN:-outputs/evaluation/prism_epoch31_smooth_official272_exactlen_0617/h3d/depth_driven}
PRISM_PREP=${PRISM_PREP:-outputs/evaluation/prism_epoch31_smooth_official272_exactlen_0617/prep/ours_e31_smooth}
HY_GEN=${HY_GEN:-outputs/evaluation/hylite_official272_exactlen_0617/h3d/motionclip135}
HY_COL=${HY_COL:-outputs/evaluation/hylite_official272_exactlen_0617/h3d_row2col_yaw}
HY_PREP=${HY_PREP:-outputs/evaluation/hylite_official272_exactlen_0617/prep/hymotion}
MS_PREP=${MS_PREP:-outputs/evaluation/motionstreamer_official272_exactlen_0617/prep}
MM_RAW=${MM_RAW:-outputs/evaluation/motionmillion_official272_exactlen_0617/raw272}
MM_PREP=${MM_PREP:-outputs/evaluation/motionmillion_official272_exactlen_0617/prep}

TARGET=${TARGET:-4042}
INTERVAL=${INTERVAL:-180}
STALL_POLLS=${STALL_POLLS:-2}

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WLOG"; }
cnt(){ python3 -c "import os,sys;d=sys.argv[1];ext=sys.argv[2];print(sum(1 for e in os.scandir(d) if e.name.endswith(ext)) if os.path.isdir(d) else 0)" "$1" "$2"; }

eval_prep(){
  local name="$1" pred="$2" oj="$RES/$name.json"
  [ -s "$oj" ] && { log "eval $name already done"; return 0; }
  log "eval $name pred=$pred"
  CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$pred" --tag "$name" --also-refk --out-json "$oj" \
    > "$LOG/eval_$name.log" 2>&1 \
    || CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} python3 scripts/eval/eval_motionstreamer_272.py \
         --pred-dir "$pred" --tag "$name" --out-json "$oj" \
         >> "$LOG/eval_$name.log" 2>&1 || true
  [ -s "$oj" ] && log "eval $name DONE"
}

audit_all(){
  log "length audit"
  python3 scripts/eval/audit_table1_lengths.py \
    --out-dir "$AUD" \
    --method "PRISM e31 smooth exact=$PRISM_PREP" \
    --method "HY-Motion exact=$HY_PREP" \
    --method "MotionStreamer exact=$MS_PREP" \
    --method "Go-To-Zero exact=$MM_PREP" \
    > "$LOG/length_audit.log" 2>&1 || true
  [ -s "$AUD/summary.tsv" ] && tee -a "$WLOG" < "$AUD/summary.tsv" >/dev/null
}

ready_stable(){
  local dir="$1" ext="$2" prev="$3" stall="$4"
  local n
  n=$(cnt "$dir" "$ext")
  if [ "$n" = "$prev" ]; then
    stall=$((stall+1))
  else
    stall=0
  fi
  echo "$n $stall"
}

do_prism(){
  [ -f "$PRISM_PREP/_DONE" ] && return 0
  mkdir -p "$PRISM_PREP"
  log "PRISM repack $PRISM_GEN -> $PRISM_PREP"
  python3 scripts/eval/repack_pred_to_272ids.py --npz-dir "$PRISM_GEN" \
    --anno-file "$ANNO" --id-passthrough --out-dir "$PRISM_PREP" --workers 16 \
    > "$LOG/repack_prism.log" 2>&1 && touch "$PRISM_PREP/_DONE"
  log "PRISM prep n=$(cnt "$PRISM_PREP" .npz)"
  eval_prep prism_e31_smooth_exact "$PRISM_PREP"
}

do_hy(){
  [ -f "$HY_PREP/_DONE" ] && return 0
  mkdir -p "$HY_COL" "$HY_PREP"
  log "HY convert $HY_GEN -> $HY_COL"
  python3 scripts/eval/convert_hylite135_to_motionclip_col.py \
    --src-dir "$HY_GEN" --out-dir "$HY_COL" --anno-file "$ANNO" \
    --data-dir "$DATA_DIR" --overwrite --workers 16 \
    > "$LOG/convert_hy.log" 2>&1
  log "HY repack $HY_COL -> $HY_PREP"
  python3 scripts/eval/repack_pred_to_272ids.py --col-npy-dir "$HY_COL" \
    --anno-file "$ANNO" --id-passthrough --out-dir "$HY_PREP" --workers 16 \
    > "$LOG/repack_hy.log" 2>&1 && touch "$HY_PREP/_DONE"
  log "HY prep n=$(cnt "$HY_PREP" .npz)"
  eval_prep hymotion_exact "$HY_PREP"
}

do_ms(){
  [ -f "$MS_PREP/_DONE_EVAL" ] && return 0
  log "MS prep n=$(cnt "$MS_PREP" .npz)"
  eval_prep motionstreamer_exact "$MS_PREP"
  touch "$MS_PREP/_DONE_EVAL"
}

do_mm(){
  [ -f "$MM_PREP/_DONE" ] && return 0
  mkdir -p "$MM_PREP"
  log "Go-To-Zero repack $MM_RAW -> $MM_PREP"
  python3 scripts/eval/repack_pred_to_272ids.py --gt272-dir "$MM_RAW" \
    --id-passthrough --out-dir "$MM_PREP" --workers 16 \
    > "$LOG/repack_mm.log" 2>&1 && touch "$MM_PREP/_DONE"
  log "Go-To-Zero prep n=$(cnt "$MM_PREP" .npz)"
  eval_prep gotozero_exact "$MM_PREP"
}

log "watch start target=$TARGET interval=$INTERVAL"
bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true

pp=-1; ps=0; hp=-1; hs=0; mp=-1; ms=0; gp=-1; gs=0
pd=0; hd=0; md=0; gd=0
while :; do
  if [ "$pd" -eq 0 ]; then
    read -r pp ps < <(ready_stable "$PRISM_GEN" .npz "$pp" "$ps")
    log "PRISM gen=$pp stall=$ps"
    if [ "$pp" -ge "$TARGET" ] && [ "$ps" -ge "$STALL_POLLS" ]; then do_prism; pd=1; fi
  fi
  if [ "$hd" -eq 0 ]; then
    read -r hp hs < <(ready_stable "$HY_GEN" .npy "$hp" "$hs")
    log "HY gen=$hp stall=$hs"
    if [ "$hp" -ge "$TARGET" ] && [ "$hs" -ge "$STALL_POLLS" ]; then do_hy; hd=1; fi
  fi
  if [ "$md" -eq 0 ]; then
    read -r mp ms < <(ready_stable "$MS_PREP" .npz "$mp" "$ms")
    log "MotionStreamer gen=$mp stall=$ms"
    if [ "$mp" -ge "$TARGET" ] && [ "$ms" -ge "$STALL_POLLS" ]; then do_ms; md=1; fi
  fi
  if [ "$gd" -eq 0 ]; then
    read -r gp gs < <(ready_stable "$MM_RAW" .npy "$gp" "$gs")
    log "Go-To-Zero gen=$gp stall=$gs"
    if [ "$gp" -ge "$TARGET" ] && [ "$gs" -ge "$STALL_POLLS" ]; then do_mm; gd=1; fi
  fi
  audit_all
  if [ "$pd" -eq 1 ] && [ "$hd" -eq 1 ] && [ "$md" -eq 1 ] && [ "$gd" -eq 1 ]; then
    log "ALL exact-length generation/repack/eval done"
    break
  fi
  sleep "$INTERVAL"
done
