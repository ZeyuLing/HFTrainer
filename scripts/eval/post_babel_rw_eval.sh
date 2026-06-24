#!/bin/bash
# Convert + evaluate the FAITHFUL-caption regenerated BABEL sequences (Table 3).
#   PRISM   : SMPLX npz  -> smpl_pred_to_272 (native, no frame fix) -> eval
#   MS      : motion_272 npz -> eval directly
#   KIMODO  : (T,22,3) joints -> joints_to_272_npz (src-fps 30) -> eval
# All rows use the new default (faithful rewrite caption) + --max-total 360 + humanml stats.
set -uo pipefail
ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
[ -d "$ROOT" ] || ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/ref_repo/MotionStreamer/MotionStreamer:${PYTHONPATH:-}"
export HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
B=outputs/evaluation/babel_seq
RES=$B/results
EV="python3 scripts/eval/eval_babel_seq_ms272.py --max-total 360 --mean-std humanml"

do_prism=${DO_PRISM:-1}; do_ms=${DO_MS:-1}; do_kimodo=${DO_KIMODO:-1}

if [ "$do_prism" = 1 ]; then
  echo "==== PRISM convert+eval $(date) ===="
  python3 scripts/eval/smpl_pred_to_272.py --in-dir "$B/prism_gen_rw" \
    --out-dir "$B/prism_272f_rw" --skip-existing > /tmp/prism_rw_272.log 2>&1
  echo "  prism_272f_rw: $(ls $B/prism_272f_rw/*.npz 2>/dev/null | wc -l) npz"
  $EV --pred-dir "$B/prism_272f_rw" --tag prism_rw --out-json "$RES/prism_rw.json" 2>&1 | grep -E "Subseq|Trans|\[pred\]"
fi

if [ "$do_ms" = 1 ]; then
  echo "==== MS eval $(date) ===="
  echo "  ms_gen_rw: $(ls $B/ms_gen_rw/*.npz 2>/dev/null | wc -l) npz"
  $EV --pred-dir "$B/ms_gen_rw" --tag ms_rw --out-json "$RES/ms_rw.json" 2>&1 | grep -E "Subseq|Trans|\[pred\]"
fi

if [ "$do_kimodo" = 1 ]; then
  echo "==== KIMODO convert+eval $(date) ===="
  python3 scripts/eval/joints_to_272_npz.py --in-dir "$B/kimodo_gen_rw" \
    --out "$B/kimodo_prep_rw" --input-kind joints --src-fps 30 --workers 16 > /tmp/kimodo_rw_272.log 2>&1
  echo "  kimodo_prep_rw: $(ls $B/kimodo_prep_rw/*.npz 2>/dev/null | wc -l) npz"
  $EV --pred-dir "$B/kimodo_prep_rw" --tag kimodo_rw --out-json "$RES/kimodo_rw.json" 2>&1 | grep -E "Subseq|Trans|\[pred\]"
fi
echo "[POST_BABEL_RW_DONE] $(date)"
