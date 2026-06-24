#!/usr/bin/env bash
# Generate ONE KIMODO-SMPLX M2M task's per-sid preds using all 8 A100 GPUs on a
# single idle Taiji host, 8-way sharded. Mirrors the full4042 T2M shard recipe
# (container python3 + KIMODO HF/cache env + light dep check), but drives the
# new SMPLX-native M2M generator. Shards write disjoint <sid>.npz into one shared
# --out-dir (no collision). Blocks until all shards finish (run inside a detached
# tmux from the caller). Reuses the cached qwen3/llm2vec caption features so no
# text encoder is loaded.
#
# Usage: bash scripts/eval/gen_kimodo_m2m_smplx_host8.sh <TASK> [OUT_DIR] [MAXS]
#   TASK in {inbetween,prediction,backcast,clip}
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"

TASK="${1:?usage: gen_kimodo_m2m_smplx_host8.sh <TASK> [OUT_DIR] [MAXS]}"
OUT_DIR="${2:-outputs/evaluation/kimodo_smplx_m2m_20260623/${TASK}/preds_npz}"
MAXS="${3:-0}"
CORPUS="$ROOT/outputs/evaluation/kimodo_smplx_hml3d_smpl_ms272_20260618_full4042/corpus.jsonl"
CACHE_DIR="$ROOT/data/kimodo_text_feature"
NAMESPACE="kimodo_smplx_t2m_hml3d_smpl_ms272_20260618_full4042"
GT272="$ROOT/data/evaluators/humanml3d_272"
LOGD="$ROOT/${OUT_DIR}/../genlogs"
mkdir -p "$OUT_DIR" "$LOGD"

export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/checkpoints/kimodo"
export HUGGINGFACE_HUB_CACHE="$ROOT/checkpoints/kimodo/hub"
export TRANSFORMERS_CACHE="$ROOT/checkpoints/kimodo/hub"
export TEXT_ENCODERS_DIR="$ROOT/checkpoints/kimodo/text_encoders"
export CHECKPOINT_DIR="$ROOT/checkpoints/kimodo/local_models"

# Light dependency check (container python3 may lack smplx).
python3 - <<'PY' > /tmp/kimodo_m2m_missing_deps.txt 2>/dev/null
import importlib
for mod, pkg in [("smplx", "smplx>=0.1.28")]:
    try:
        importlib.import_module(mod)
    except Exception:
        print(pkg)
PY
if [ -s /tmp/kimodo_m2m_missing_deps.txt ]; then
  echo "[deps] installing: $(tr '\n' ' ' < /tmp/kimodo_m2m_missing_deps.txt)"
  python3 -m pip install -q -i https://mirrors.tencent.com/pypi/simple \
    --trusted-host mirrors.tencent.com $(tr '\n' ' ' < /tmp/kimodo_m2m_missing_deps.txt) || true
fi

echo "[gen-host8] task=$TASK out=$OUT_DIR maxs=$MAXS corpus=$CORPUS"
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$s" python3 scripts/eval/gen_kimodo_m2m_smplx.py \
    --task "$TASK" \
    --humanml3d-272 "$GT272" \
    --corpus "$CORPUS" \
    --out-dir "$OUT_DIR" \
    --text-feature-cache-dir "$CACHE_DIR" \
    --text-feature-namespace "$NAMESPACE" \
    --num-shards 8 --shard-index "$s" \
    --diffusion-steps 100 --cfg 2.0 --max-samples "$MAXS" \
    --device cuda --skip-existing \
    > "$LOGD/${TASK}_shard${s}.log" 2>&1 &
done
wait
echo "ALL_SHARDS_DONE task=$TASK npz_count=$(ls "$OUT_DIR"/*.npz 2>/dev/null | wc -l)"
