#!/bin/bash
# Wait for the Taiji PRISM-BABEL generation to finish, then repack SMPLX->135
# and run the BABEL sequential MS-272 evaluator for the ours row of Table 3.
set -uo pipefail
ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$ROOT"
export PYTHONPATH=$PWD:${PYTHONPATH:-} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

GEN=outputs/evaluation/babel_seq/prism_gen
PREP=outputs/evaluation/babel_seq/prism_prep
RES=outputs/evaluation/babel_seq/results/prism.json
TARGET=${TARGET:-1700}     # ~1762 episodes; accept near-complete
STABLE_HITS=0

echo "[watch] $(date) waiting for $GEN to reach ~$TARGET npz"
for i in $(seq 1 240); do
  n=$(find "$GEN" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  echo "[watch $i $(date +%H:%M)] gen_npz=$n"
  if [ "$n" -ge "$TARGET" ]; then
    STABLE_HITS=$((STABLE_HITS+1))
  else
    STABLE_HITS=0
  fi
  # require 2 consecutive stable polls (writing finished)
  if [ "$STABLE_HITS" -ge 2 ]; then break; fi
  sleep 120
done

n=$(find "$GEN" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
echo "[watch] proceeding with n=$n npz at $(date)"

mkdir -p "$PREP" "$(dirname "$RES")"
echo "[repack] $(date)"
python3 scripts/eval/repack_pred_to_272ids.py \
  --npz-dir "$GEN" --id-passthrough \
  --anno-file data/annotation/test_hml3d.json \
  --out-dir "$PREP" --workers 16 2>&1 | tail -5

# cache the 272 evaluator ckpt for speed
if [ ! -f /dev/shm/eval272_epoch99.ckpt ]; then
  cp ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt /dev/shm/eval272_epoch99.ckpt 2>/dev/null || true
fi

echo "[eval] $(date)"
python3 scripts/eval/eval_babel_seq_ms272.py \
  --pred-dir "$PREP" --tag prism \
  --out-json "$RES" 2>&1 | tail -8

echo "PRISM_BABEL_EVAL_DONE $(date)"
python3 - <<PY
import json
d=json.load(open("$RES"))
s=d['subseq']; t=d['transition']
print("OURS subseq: R@3=%.4f FID=%.3f Div=%.3f MM-D=%.3f n=%d"%(s['r3'],s['fid'],s['diversity'],s['mm_dist'],s['nb']))
print("OURS trans:  FID=%.3f Div=%.3f PeakJerk=%.4f AreaJerk=%.3f"%(t['fid'],t['diversity'],t['peak_jerk'],t['area_jerk']))
PY
