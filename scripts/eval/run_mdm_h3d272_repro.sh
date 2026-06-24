#!/usr/bin/env bash
# End-to-end MDM reproduction under the MotionStreamer-272 evaluator.
#   Stage 0  hftrainer-native MDM T2M generation (HumanML3D-263, 20 fps)
#   Stage A  263 -> SMPL motion_135 (IK, 20->30 fps)
#   Stage B  135 -> 272
#   Stage C  MotionStreamer-272 evaluator
# Target (paper Table tab:eval_t2m, MDM row):
#   R-P T1 0.430 / T3 0.681 / FID 102.7 / MM-D 20.15 / Div 26.43
set -eo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"

# Overridable: the paper uses the 1000-step "best model" humanml-encoder-512
# (model000475000.pt, guidance 2.5), NOT the 50-step speed variant.
BASE=${BASE:-outputs/evaluation/mdm_h3d272_repro_1000s}
MODEL_PATH=${MODEL_PATH:-ref_repo/MDM/save/humanml_trans_enc_512/humanml_trans_enc_512/model000475000.pt}
GUIDANCE=${GUIDANCE:-2.5}
GEN_BS=${GEN_BS:-256}
DIR263=$BASE/mdm_263
DIR135=$BASE/mdm_smpl135
DIR272=$BASE/mdm_272
DATA_ROOT=ref_repo/MotionStreamer/MotionStreamer/humanml3d_272
EVAL_CKPT=ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt
OUT_JSON=$BASE/eval_mdm_272.json
mkdir -p "$BASE"

echo "==================== Stage 0: MDM generation ===================="
echo "model: $MODEL_PATH  guidance: $GUIDANCE"
python3 -u scripts/eval/mdm_t2m_humanml272.py \
  --out_dir "$DIR263" --model_path "$MODEL_PATH" \
  --batch_size "$GEN_BS" --device cuda --guidance_param "$GUIDANCE" --skip_existing

echo "==================== Stage A: 263 -> SMPL motion_135 ===================="
# NOTE: refine-iters 80 / lr 0.02 reproduces the paper pipeline. With refine-iters 0
# the analytic IK leaves a catastrophic-tail (mean MPJPE ~68 mm, p95 ~104 mm) that
# inflates the MS-272 FID to ~137; 80 iters of gradient refinement drops it to
# ~32 mm (paper-grade) and recovers FID ~102.7.
REFINE_ITERS=${REFINE_ITERS:-80}
REFINE_LR=${REFINE_LR:-0.02}
python3 -u scripts/eval/hml263_to_smpl_ik.py \
  --in-dir "$DIR263" --out-dir "$DIR135" \
  --model-dir ref_repo/MDM/body_models \
  --source-fps 20 --target-fps 30 --floor-align \
  --refine-iters "$REFINE_ITERS" --refine-lr "$REFINE_LR" \
  --rot6d-convention row \
  --device cuda --skip-existing

echo "==================== Stage B: motion_135 -> 272 ===================="
python3 -u scripts/data/convert_motion135_to_h3d272.py \
  --in-dir "$DIR135" --out-dir "$DIR272" --workers 8

echo "==================== Stage C: MotionStreamer-272 evaluation ===================="
python3 -u ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py \
  --evaluator_ckpt "$EVAL_CKPT" \
  --data_root "$DATA_ROOT" \
  --pred_dir "$DIR272" \
  --n_repeats 20 --batch_size 32 \
  --out_json "$OUT_JSON"

echo "==================== DONE ===================="
echo "result json: $OUT_JSON"
cat "$OUT_JSON"
