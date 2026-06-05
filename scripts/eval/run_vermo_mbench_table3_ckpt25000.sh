#!/usr/bin/env bash
set -euo pipefail

ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
if [ ! -d "$ROOT" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

CKPT=work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager/checkpoint-iter_25000/model.pt
CONFIG=configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer.py
OUT=output/evaluation/table3_mbench/vermo_ckpt25000_full
LOG_DIR=logs/taiji

mkdir -p "$LOG_DIR"
rm -rf "$OUT"
mkdir -p "$OUT/shards"

if [ ! -f "$CKPT" ]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 2
fi

starts=(0 56 112 168 225 281 337 393)
ends=(55 111 167 224 280 336 392 449)

for i in 0 1 2 3 4 5 6 7; do
  (
    export CUDA_VISIBLE_DEVICES=$i
    python3 tools/export_vermo_mbench.py \
      --config "$CONFIG" \
      --checkpoint "$CKPT" \
      --output-dir "$OUT/shards/shard_$i" \
      --device cuda \
      --start-id "${starts[$i]}" \
      --end-id "${ends[$i]}" \
      --force \
      --max-extra-tokens 16 \
      2>&1 | tee "$OUT/shards/shard_$i/export.log"
  ) &
done
wait

python3 tools/merge_vermo_mbench_shards.py \
  --shard-root "$OUT/shards" \
  --output-dir "$OUT" \
  --expected-count 450 \
  --force \
  2>&1 | tee "$OUT/merge.log"

python3 tools/validate_mbench_eval_input.py \
  --eval-input-dir "$OUT/mbench_eval_input" \
  --output-json "$OUT/mbench_eval_input_manifest.json" \
  2>&1 | tee "$OUT/validate.log"

NON_VLM_DIMS=(
  Jitter_Degree
  Ground_Penetration
  Foot_Floating
  Foot_Sliding
  Dynamic_Degree
  Body_Penetration
  Pose_Quality
)

(
  cd ref_repo/ViMoGen
  python evaluate_mbench.py \
    --evaluation_path "$ROOT/$OUT/mbench_eval_input" \
    --output_path "$ROOT/$OUT/mbench_results_non_vlm" \
    --full_info_json "$ROOT/ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json" \
    --device cuda \
    --dimension "${NON_VLM_DIMS[@]}"
) 2>&1 | tee "$OUT/mbench_non_vlm.log"

if [ -n "${GEMINI_API_KEY:-}" ]; then
  (
    cd ref_repo/ViMoGen
    python evaluate_mbench.py \
      --evaluation_path "$ROOT/$OUT/mbench_eval_input" \
      --output_path "$ROOT/$OUT/mbench_results_vlm" \
      --full_info_json "$ROOT/ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json" \
      --device cuda \
      --gemini_api_key "$GEMINI_API_KEY" \
      --dimension Motion_Condition_Consistency Motion_Generalizability
  ) 2>&1 | tee "$OUT/mbench_vlm.log"
else
  echo "GEMINI_API_KEY is not set; VLM metrics are pending." | tee "$OUT/vlm_missing_key.txt"
fi

echo "DONE: $OUT"
