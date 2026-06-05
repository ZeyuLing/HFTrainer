#!/usr/bin/env bash
# Full MotionGPT evaluation: official checkpoint inference, SMPL retargeting,
# MotionCLIP135 remap, and rewritten-caption chunk_size=64 metrics.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:${PYTHONPATH:-}

NUM_SHARDS=${NUM_SHARDS:-8}
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/humanml3d/motiongpt_hml3d263}
EVAL_ROOT=${EVAL_ROOT:-outputs/evaluation/motiongpt_rw_c64_eval0605}
LOGDIR=${LOGDIR:-outputs/evaluation/motiongpt_0605/logs}
mkdir -p "$LOGDIR" "$OUT_ROOT/humanml3d" "$OUT_ROOT/motionhub"

python3 - <<'PY' > "$LOGDIR/install_check.log" 2>&1
missing = []
for mod in ("spacy",):
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
print("missing", missing)
PY
if ! grep -q "missing \\[\\]" "$LOGDIR/install_check.log"; then
  python3 -m pip install -q spacy >> "$LOGDIR/install_check.log" 2>&1
fi

run_split() {
  local split="$1"
  local anno="$2"
  local caption="$3"
  local out_dir="$4"

  echo "[$split-infer] $(date)" | tee -a "$LOGDIR/run.log"
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="$i" python3 scripts/eval/motiongpt_infer_hml3d263.py \
      --anno-file "$anno" \
      --caption-file "$caption" \
      --out-dir "$out_dir" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$i" \
      --batch-size 16 \
      --skip-existing \
      > "$LOGDIR/${split}_s${i}.log" 2>&1 &
  done
  wait
}

echo "[start] $(date)" | tee "$LOGDIR/run.log"
run_split h3d data/annotation/test_hml3d.json data/annotation/test_hml3d_rewritten.json "$OUT_ROOT/humanml3d"
run_split mh data/annotation/test_motionhub_t2m.json data/annotation/test_motionhub_t2m_rewritten.json "$OUT_ROOT/motionhub"

echo "[retarget-eval] $(date)" | tee -a "$LOGDIR/run.log"
METHOD=motiongpt \
  H3D_SRC="$OUT_ROOT/humanml3d" \
  MH_SRC="$OUT_ROOT/motionhub" \
  EVAL_ROOT="$EVAL_ROOT" \
  bash scripts/eval/run_hml263_method_rw_c64_eval_0605.sh \
  > "$LOGDIR/postprocess_eval.log" 2>&1

echo "[done] $(date)" | tee -a "$LOGDIR/run.log"
