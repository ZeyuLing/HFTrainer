#!/bin/bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ROOT=outputs/evaluation/motionstreamer_272_rerun
CONV=$ROOT/hml263_to_ms272
MET=$ROOT/metrics
LOG=$ROOT/logs
PRISM_IDS=$ROOT/prism_kt_spectral_epoch3_h3d_none
mkdir -p "$CONV" "$MET" "$LOG"

MS=ref_repo/MotionStreamer/MotionStreamer
SHM=/dev/shm/ms272_data
CKPT=/dev/shm/eval272_epoch99.ckpt

echo "[start] $(date)"
echo "[cache] MotionStreamer HumanML3D-272 GT/texts/checkpoint"
mkdir -p "$SHM/motion_data" "$SHM/texts" "$SHM/split" "$SHM/mean_std"
cp "$MS/humanml3d_272/split/test.txt" "$SHM/split/test.txt"
cp "$MS/humanml3d_272/mean_std/Mean.npy" "$MS/humanml3d_272/mean_std/Std.npy" "$SHM/mean_std/"
if [ ! -f "$CKPT" ]; then
  cp "$MS/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt" "$CKPT"
fi
cat "$SHM/split/test.txt" | xargs -P 32 -I{} bash -c \
  '[ -f "$0/motion_data/$1.npy" ] || cp "$2/humanml3d_272/motion_data/$1.npy" "$0/motion_data/" 2>/dev/null || true; [ -f "$0/texts/$1.txt" ] || cp "$2/humanml3d_272/texts/$1.txt" "$0/texts/" 2>/dev/null || true' \
  "$SHM" {} "$MS"
echo "[cache] gt=$(ls "$SHM/motion_data" | wc -l) texts=$(ls "$SHM/texts" | wc -l) ckpt=$(du -h "$CKPT" | awk '{print $1}')"

convert_one() {
  local tag="$1"
  local src="$2"
  local dst="$CONV/$tag"
  mkdir -p "$dst"
  echo "[convert:$tag] src=$src dst=$dst"
  CUDA_VISIBLE_DEVICES="" python3 scripts/data/convert_hml263_pose_to_h3d272.py \
    --pred_dir_263 "$src" \
    --out_dir_272 "$dst" \
    --output_format npz \
    --skip_existing \
    > "$LOG/convert_${tag}.log" 2>&1
  echo "[convert:$tag] done count=$(ls "$dst"/*.npz 2>/dev/null | wc -l)"
}

convert_one momask outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix/momask &
convert_one mdm outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix/mdm &
convert_one motiongpt3 outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix/motiongpt3 &
convert_one mld_v1 outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix_mld_v1/mld &

echo "[repack:prism] epoch3 h3d/none -> canonical 272 ids"
python3 scripts/eval/repack_pred_to_272ids.py \
  --npz-dir outputs/evaluation/prism_kt_spectral_epoch3/h3d/none \
  --anno-file data/annotation/test_hml3d.json \
  --out-dir "$PRISM_IDS" \
  --workers 16 \
  > "$LOG/repack_prism_kt_spectral_epoch3_h3d_none.log" 2>&1 &

wait
echo "[prep done] $(date)"

eval_one() {
  local gpu="$1"
  local tag="$2"
  local pred_dir="${3:-}"
  local log="$LOG/eval_${tag}.log"
  local json="$MET/${tag}.json"
  echo "[eval:$tag] gpu=$gpu pred=${pred_dir:-GT_ONLY}"
  if [ -n "$pred_dir" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
      --pred-dir "$pred_dir" \
      --tag "$tag" \
      --out-json "$json" \
      > "$log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
      --tag "$tag" \
      --out-json "$json" \
      > "$log" 2>&1
  fi
  echo "[eval:$tag] done json=$json"
}

eval_one 0 gt_full "" &
eval_one 1 momask_hml263_ms272 "$CONV/momask" &
eval_one 2 mdm_hml263_ms272 "$CONV/mdm" &
eval_one 3 motiongpt3_hml263_ms272 "$CONV/motiongpt3" &
wait

eval_one 0 mld_v1_hml263_ms272 "$CONV/mld_v1" &
eval_one 1 prism_kt_spectral_epoch3_h3d_none "$PRISM_IDS" &
wait

echo "[done] $(date)"
