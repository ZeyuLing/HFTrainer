#!/bin/bash
# Canonical Table 3 (BABEL seq, MS-272) with BOTH fixes:
#   * per-segment canonicalization (sliced GT -> in-distribution)
#   * FlowMDM-style per-batch caption dedup
# Native 272 throughout (MS stays faithful, no lossy rots-FK roundtrip).
# Writes *_cd.json.
set -uo pipefail
ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
[ -d "$ROOT" ] || ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/ref_repo/MotionStreamer/MotionStreamer:${PYTHONPATH:-}"
export HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
RES="$ROOT/outputs/evaluation/babel_seq/results"
EV="python3 scripts/eval/eval_babel_seq_ms272.py --max-total 360 --mean-std humanml --dedup --per-seg-canon"

run() {
  tag="$1"; shift
  echo "==== [$tag] $(date) ===="
  $EV "$@" --tag "$tag" --out-json "$RES/$tag.json" 2>&1 \
    | grep -E "Subseq|Trans|\[pred\]|\[gt\]" || echo "  FAILED $tag"
}

run real_cd       --real
run prism_cd      --pred-dir outputs/evaluation/babel_seq/prism_272f_rw
run ms_cd         --pred-dir outputs/evaluation/babel_seq/ms_gen_rw
run flowmdm_cd    --pred-dir outputs/evaluation/babel_seq/flowmdm_272f
run doubletake_cd --pred-dir outputs/evaluation/babel_seq/doubletake_272f
run kimodo_cd     --pred-dir outputs/evaluation/babel_seq/kimodo_prep_rw
echo "[CANON_DEDUP_REEVAL_DONE] $(date)"
