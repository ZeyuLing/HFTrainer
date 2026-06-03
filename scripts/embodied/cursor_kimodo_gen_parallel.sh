#!/usr/bin/env bash
# Launch 6 parallel KIMODO-G1 generate workers (shard k -> GPU 2+k) in tmux.
set +e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
P=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
OUT=output/physflow_kimodo_g1/overfit100_pool
SHARDS=configs/experiments/physflow_kimodo_g1/shards
mkdir -p "$OUT"

# stop the serial run
tmux kill-session -t gen100 2>/dev/null

for k in 0 1 2 3 4 5; do
  gpu=$((2 + k))
  bank="$SHARDS/overfit100_shard${k}.jsonl"
  log="$P/$OUT/gen_shard${k}.log"
  tmux kill-session -t gen_sh${k} 2>/dev/null
  tmux new-session -d -s gen_sh${k} \
    "PHYSFLOW_KIMODO_PY=python3 PHYSFLOW_PROMPT_BANK=$bank PHYSFLOW_OUT=$OUT PHYSFLOW_GPU=$gpu bash scripts/embodied/cursor_kimodo_gen_shard.sh > $log 2>&1"
  echo "launched gen_sh${k} on GPU ${gpu} bank=$bank"
done
sleep 2
tmux ls | grep gen_sh
echo "ALL_SHARDS_LAUNCHED"
