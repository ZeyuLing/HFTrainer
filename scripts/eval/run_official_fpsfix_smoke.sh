#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "$ROOT"

ANNO="outputs/evaluation/humanml3d/hml3d263_official_eval/hml3d_official_nonmirror_annotation.json"
REWRITTEN="outputs/evaluation/humanml3d/hml3d263_official_eval/hml3d_official_nonmirror_rewritten.json"
OUT="${OUT:-outputs/evaluation/humanml3d/_debug/official_fpsfix_0604}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"

mkdir -p "$OUT"/{_logs,momask,mdm,mld,motiongpt3}

pids=()
names=()

run_bg() {
  local name="$1"
  local gpu="$2"
  shift 2
  local log="$OUT/_logs/${name}.log"
  local status="$OUT/_logs/${name}.status"
  (
    set +e
    echo "[launch] name=${name} gpu=${gpu} started_at=$(date -Is)" > "$status"
    CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONUNBUFFERED=1 \
      TOKENIZERS_PARALLELISM=false \
      PYTHONPATH="$ROOT:${PYTHONPATH:-}" \
      "$@" > "$log" 2>&1
    local rc=$?
    echo "exit_code=${rc} finished_at=$(date -Is)" > "$status"
    exit "$rc"
  ) &
  local pid="$!"
  pids+=("$pid")
  names+=("$name")
  echo "[launch] ${name} pid=${pid} gpu=${gpu} log=${log}"
}

run_bg momask 0 \
  python3 scripts/eval/momask_infer_h3d_test.py \
    --momask_root ref_repo/Momask/momask-codes \
    --humanml3d_272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
    --out_dir "$OUT/momask" \
    --anno_file "$ANNO" \
    --rewritten_file "$REWRITTEN" \
    --data_dir data/motionhub \
    --caption_protocol rewritten \
    --max_samples "$MAX_SAMPLES" \
    --batch_size 16 \
    --gumbel_sample

run_bg mdm 1 \
  python3 scripts/eval/mdm_infer_hml3d263.py \
    --model_path ref_repo/MDM/save/humanml_enc_512_50steps/model000750000.pt \
    --out_dir "$OUT/mdm" \
    --anno_file "$ANNO" \
    --rewritten_file "$REWRITTEN" \
    --anno_data_dir data/motionhub \
    --caption_protocol rewritten \
    --max_samples "$MAX_SAMPLES" \
    --batch_size 16 \
    --device 0

run_bg mld 2 \
  python3 scripts/eval/mld_infer_hml3d263.py \
    --checkpoint ref_repo/MotionLCM/experiments_t2m/mld_humanml/mld_humanml_v1.ckpt \
    --out_dir "$OUT/mld" \
    --anno_file "$ANNO" \
    --rewritten_file "$REWRITTEN" \
    --anno_data_dir data/motionhub \
    --caption_protocol rewritten \
    --max_samples "$MAX_SAMPLES" \
    --batch_size 16

run_bg motiongpt3 3 \
  python3 scripts/eval/motiongpt3_infer_hml3d263.py \
    --out_dir "$OUT/motiongpt3" \
    --anno_file "$ANNO" \
    --rewritten_file "$REWRITTEN" \
    --anno_data_dir data/motionhub \
    --caption_protocol rewritten \
    --max_samples "$MAX_SAMPLES" \
    --batch_size 8 \
    --guidance_scale 3.0

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[fail] ${names[$i]} pid=${pids[$i]}"
    failed=1
  else
    echo "[done] ${names[$i]} pid=${pids[$i]}"
  fi
done

exit "$failed"
