#!/usr/bin/env bash
set +e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
PROMPT_BANK=configs/experiments/physflow_kimodo_g1/prompt_bank_humanml3d_overfit100.jsonl
OUT=output/physflow_kimodo_g1/overfit100_pool
mkdir -p "$OUT"
# Candidate python interpreters for KIMODO generation.
for PY in /root/physflow_isaacgym_py38_cu118/bin/python /usr/local/bin/python3 python3; do
  command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || continue
  echo "### trying python: $PY"
  "$PY" -c "import sys; print('  py', sys.version.split()[0])" 2>/dev/null
  "$PY" scripts/embodied/physflow_kimodo_g1_runner.py \
    --mode preflight \
    --output-dir "$OUT" \
    --prompt-bank "$PROMPT_BANK" \
    --prompt-split train \
    --max-prompts 100 \
    --kimodo-model Kimodo-G1-RP-v1 \
    --local-cache 2>&1 | tail -25
  echo "### exit=$?"
  break
done
echo "==== preflight report ===="
cat "$OUT/preflight.json" 2>/dev/null | head -60