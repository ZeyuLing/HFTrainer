#!/usr/bin/env bash
# Launch 8 parallel KIMODO-G1 generation workers (one per GPU) over the
# pre-split HumanML3D train shards, rolling into a shared .motion/.csv pool.
# Already-generated stems are skipped by the runner, so this is restart-safe.
#
# Run ON the dedicated PhysFlow Taiji node (physflow_oa_node_v1) where the
# env is provisioned. Each shard worker runs in its own detached tmux session
# (gen0..gen7) so it survives taiji_client exec disconnects.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

PY="${PHYSFLOW_KIMODO_PY:-/usr/local/bin/python3}"
SHARD_DIR="configs/experiments/physflow_kimodo_g1/oa_shards"
POOL="${PHYSFLOW_OA_POOL:-output/physflow_kimodo_g1/oa_pool_v1}"
NGPU="${PHYSFLOW_NGPU:-8}"
mkdir -p "$POOL"

for k in $(seq 0 $((NGPU-1))); do
  bank="$SHARD_DIR/train_shard${k}.jsonl"
  log="$POOL/gen_shard${k}.log"
  tmux kill-session -t "gen${k}" 2>/dev/null
  tmux new-session -d -s "gen${k}" \
    "PHYSFLOW_KIMODO_PY=$PY $PY scripts/embodied/physflow_kimodo_g1_runner.py \
       --mode generate --output-dir $POOL --prompt-bank $bank \
       --prompt-split train --max-prompts 100000 --samples-per-prompt 1 \
       --kimodo-model Kimodo-G1-RP-v1 --diffusion-steps 100 --seed 42 \
       --cfg-type separated --cfg-weight 2.0 2.0 --local-cache \
       --cuda-visible-devices $k --robot-json-subsample 1 \
       > $log 2>&1"
  echo "launched gen${k} on GPU $k -> $bank (log: $log)"
done
sleep 2
tmux ls 2>&1 | grep -E "^gen[0-9]" | sort
echo "OA_GEN_PARALLEL_LAUNCHED ngpu=$NGPU pool=$POOL"
