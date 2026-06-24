#!/bin/bash
# KIMODO BABEL regen with FAITHFUL rewritten captions (Table 3).
# Single host: (1) extract LLM2Vec features for the new corpus into a fresh
# namespace, then (2) run NUM_GPUS sharded T2M-concat generation reading that
# cache. One self-contained Taiji job.
set -uo pipefail
ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
[ -d "$ROOT" ] || ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/checkpoints/kimodo" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="$ROOT/checkpoints/kimodo/text_encoders"
export CHECKPOINT_DIR="$ROOT/checkpoints/kimodo/local_models" TEXT_ENCODER_MODE=local
PY=${PY:-python3}

NUM_GPUS=${NUM_GPUS:-8}
NS=${NS:-kimodo_soma_t2m_babel_val_llm2vec_rw}
CORPUS=${CORPUS:-data/babel/babel_kimodo_corpus_rw.jsonl}
OUT=${OUT:-outputs/evaluation/babel_seq/kimodo_gen_rw}
mkdir -p "$OUT/logs"

echo "[kimodo-rw] STEP1 extract features -> namespace=$NS  $(date)"
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/embodied/cursor_extract_kimodo_text_feature.py \
  --corpus "$CORPUS" --namespace "$NS" \
  --cache-dir data/kimodo_text_feature --hf-home checkpoints/kimodo \
  --text-encoder llm2vec --device cuda --batch-size 16 \
  > "$OUT/logs/feat.log" 2>&1
nfeat=$(ls data/kimodo_text_feature/"$NS"/*.npy 2>/dev/null | wc -l)
echo "[kimodo-rw] features extracted: $nfeat  $(date)"
if [ "$nfeat" -lt 100 ]; then echo "[kimodo-rw] FEATURE EXTRACTION FAILED"; tail -30 "$OUT/logs/feat.log"; exit 1; fi

echo "[kimodo-rw] STEP2 generate ($NUM_GPUS shards)  $(date)"
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" -u scripts/eval/gen_kimodo_babel_seq.py \
    --out-dir "$OUT" --min-total 24 --max-total 360 \
    --num-shards "$NUM_GPUS" --shard-index "$i" \
    --text-feature-namespace "$NS" --max-episodes "${MAX_EP:-0}" \
    --skip-existing --device cuda \
    > "$OUT/logs/gen_g$i.log" 2>&1 &
done
wait
n=$(find "$OUT" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
echo "[kimodo-rw DONE] total_npy=$n  $(date)"
