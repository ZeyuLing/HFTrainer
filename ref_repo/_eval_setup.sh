#!/bin/bash
# Download MoMask weights + MotionStreamer evaluator + 272-dim HumanML3D dataset.
# Run on lzy_debug_machine_1 in background.
# Outputs go under ref_repo/Momask/weights/ and ref_repo/MotionStreamer/MotionStreamer/...
set -uo pipefail

ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
LOG=$ROOT/ref_repo/_eval_setup_logs
mkdir -p $LOG

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120

# ---------------------------------------------------------------------------
# 1) MoMask t2m / kit weights from Google Drive (gdown)
# ---------------------------------------------------------------------------
{
  set -x
  cd $ROOT/ref_repo/Momask/weights
  rm -rf t2m kit
  mkdir -p t2m kit
  cd t2m
  echo "[$(date)] Downloading MoMask t2m weights..."
  gdown --fuzzy https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing -O humanml3d_models.zip
  unzip -q humanml3d_models.zip && rm -f humanml3d_models.zip
  cd ../kit
  echo "[$(date)] Downloading MoMask kit weights..."
  gdown --fuzzy https://drive.google.com/file/d/1FapdHNkxPouasVM8MWgg1f6sd_4Lua2q/view?usp=sharing -O kit_models.zip || echo "kit weights optional, skip on failure"
  if [ -f kit_models.zip ]; then unzip -q kit_models.zip && rm -f kit_models.zip; fi
  echo "[$(date)] MoMask weights done."
  ls $ROOT/ref_repo/Momask/weights/t2m | head -10
} > $LOG/momask_weights.log 2>&1 &
PID_MOMASK=$!
echo "MoMask weights PID=$PID_MOMASK"

# ---------------------------------------------------------------------------
# 2) MotionStreamer Causal TAE + Evaluator + t2m model checkpoints (HF)
# ---------------------------------------------------------------------------
{
  set -x
  cd $ROOT/ref_repo/MotionStreamer/MotionStreamer
  echo "[$(date)] Downloading MotionStreamer Causal TAE + Evaluator + t2m model..."
  huggingface-cli download lxxiao/MotionStreamer --local-dir ./MotionStreamer_HF --resume-download
  echo "[$(date)] MotionStreamer HF download done."
  ls ./MotionStreamer_HF
} > $LOG/motionstreamer_hf.log 2>&1 &
PID_MS=$!
echo "MotionStreamer HF PID=$PID_MS"

# ---------------------------------------------------------------------------
# 3) MotionStreamer 272-dim HumanML3D dataset (motion + texts + mean/std)
# ---------------------------------------------------------------------------
{
  set -x
  cd $ROOT/ref_repo/MotionStreamer/MotionStreamer
  echo "[$(date)] Downloading 272-dim HumanML3D dataset..."
  huggingface-cli download --repo-type dataset --resume-download lxxiao/272-dim-HumanML3D --local-dir ./humanml3d_272
  cd ./humanml3d_272
  for z in texts.zip motion_data.zip; do
    if [ -f "$z" ]; then
      echo "[$(date)] Unzipping $z..."
      unzip -q -o "$z"
    fi
  done
  echo "[$(date)] 272-dim HumanML3D ready."
  ls $ROOT/ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 | head -10
} > $LOG/h3d_272.log 2>&1 &
PID_H3D=$!
echo "272-dim HML3D PID=$PID_H3D"

echo
echo "All three downloads launched in background. Logs at $LOG"
echo "Monitor with:"
echo "  tail -f $LOG/momask_weights.log"
echo "  tail -f $LOG/motionstreamer_hf.log"
echo "  tail -f $LOG/h3d_272.log"

# Wait for all to finish (script itself runs in background)
wait $PID_MOMASK
RC1=$?
wait $PID_MS
RC2=$?
wait $PID_H3D
RC3=$?
echo "[$(date)] All done. RC: momask=$RC1 ms=$RC2 h3d=$RC3" | tee -a $LOG/done.log
