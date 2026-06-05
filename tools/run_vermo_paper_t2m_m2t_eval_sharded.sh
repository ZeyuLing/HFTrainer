#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-output/evaluation/vermo_paper_t2m_m2t_ckpt21000_20260605}"
CONFIG="${2:-configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer_paper_test_t2m_m2t.py}"
CHECKPOINT="${3:-work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager/checkpoint-iter_21000}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_EXTRA_TOKENS="${MAX_EXTRA_TOKENS:-32}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p "$ROOT/logs" "$ROOT/index_shards"

python3 - "$CONFIG" "$NUM_SHARDS" "$ROOT/index_shards" <<'PY'
import os
import sys

from mmengine.config import Config
import hftrainer  # noqa: F401
from hftrainer.registry import DATASETS

config, nshards, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = Config.fromfile(config)
dataset = DATASETS.build(cfg.train_dataloader.dataset)
indices = list(range(len(dataset.data_list)))
for shard in range(nshards):
    shard_indices = indices[shard::nshards]
    path = os.path.join(out_dir, f"shard_{shard}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(map(str, shard_indices)))
print(f"[shard] dataset={len(indices)} nshards={nshards} out={out_dir}", flush=True)
PY

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  indices="$(cat "$ROOT/index_shards/shard_${shard}.txt")"
  CUDA_VISIBLE_DEVICES="$shard" python3 tools/export_vermo_overfit_viewer.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$ROOT/shard_${shard}" \
    --indices "$indices" \
    --device cuda \
    --max-extra-tokens "$MAX_EXTRA_TOKENS" \
    --processor-optional-input-modal-mode all \
    --processor-task-template-mode first \
    --processor-shuffle-modal-parts false \
    --seed 42 \
    > "$ROOT/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "[launch] shard=${shard} pid=${pids[-1]} log=$ROOT/logs/shard_${shard}.log"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "[eval] one or more shards failed; partial manifests may still exist" >&2
  exit "$status"
fi

python3 tools/summarize_vermo_paper_t2m_m2t_eval.py --root "$ROOT"

CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --clip_pretrained checkpoints/clip-vit-base-patch32 \
  --stats_file data/statistic/smplx55_stats_hymotion_aug.json \
  --anno_file data/annotation/test_motionhub_t2m.json \
  --rewritten_caption_file data/annotation/test_motionhub_t2m_rewritten.json \
  --data_dir data/motionhub \
  --pred_dir "$ROOT/paper_t2m_pred_135d" \
  --out_json "$ROOT/t2m_motionclip_paper_metrics.json" \
  --n_repeats 20 \
  --chunk_size 64 \
  --seed 42 \
  > "$ROOT/logs/t2m_motionclip_paper_metrics.log" 2>&1

echo "[eval] done root=$ROOT"
