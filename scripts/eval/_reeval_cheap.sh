#!/bin/bash
# Re-eval motion-unchanged rows (Real/FlowMDM/DoubleTake) with the faithful
# rewrite caption (new default in eval_babel_seq_ms272.py).
set -uo pipefail
ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
[ -d "$ROOT" ] || ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/ref_repo/MotionStreamer/MotionStreamer:${PYTHONPATH:-}"
export HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
RES="$ROOT/outputs/evaluation/babel_seq/results"

run() {
  tag="$1"; shift
  echo "==== [$tag] $(date) ===="
  python3 scripts/eval/eval_babel_seq_ms272.py --max-total 360 "$@" \
    --tag "$tag" --out-json "$RES/$tag.json" 2>&1 | grep -E "Subseq|Trans|\[pred\]" \
    || echo "  FAILED $tag"
}

run real_rw --real
run flowmdm_rw --pred-dir outputs/evaluation/babel_seq/flowmdm_272f
run doubletake_rw --pred-dir outputs/evaluation/babel_seq/doubletake_272f
echo "[CHEAP_REEVAL_DONE] $(date)"
