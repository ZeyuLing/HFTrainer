#!/usr/bin/env bash
# Stage B (motion_135 -> 272) + Stage C (MotionStreamer-272 evaluation) for the
# MDM reproduction. Run after the IK shards have filled mdm_smpl135.
set -eo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD"

BASE=${BASE:-outputs/evaluation/mdm_h3d272_repro_1000s}
DIR135=$BASE/mdm_smpl135
DIR272=$BASE/mdm_272
DATA_ROOT=ref_repo/MotionStreamer/MotionStreamer/humanml3d_272
EVAL_CKPT=ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt
OUT_JSON=$BASE/eval_mdm_272.json

echo "==================== Stage B: motion_135 -> 272 ===================="
python3 -u scripts/data/convert_motion135_to_h3d272.py \
  --in-dir "$DIR135" --out-dir "$DIR272" --workers 8

echo "==================== Stage C: MotionStreamer-272 evaluation ===================="
CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} python3 -u ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py \
  --evaluator_ckpt "$EVAL_CKPT" \
  --data_root "$DATA_ROOT" \
  --pred_dir "$DIR272" \
  --n_repeats 20 --batch_size 32 \
  --out_json "$OUT_JSON"

echo "==================== DONE ===================="
cat "$OUT_JSON"
