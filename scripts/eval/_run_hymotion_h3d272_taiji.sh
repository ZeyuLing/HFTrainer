#!/usr/bin/env bash
# Foreground 8-GPU HY-Motion-1.0 T2M -> MS-272 generation + scoring, for a
# DEDICATED Taiji job (blocks until all shards finish, then scores). Use with:
#   python3 tools/taiji_submit.py hy_t2m_eval \
#       --start-cmd "bash scripts/eval/_run_hymotion_h3d272_taiji.sh" \
#       --host_num 1 --gpu_name V100
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer || exit 1
export PYTHONPATH="$PWD"
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false

# NOTE: motion ODE/MMDiT now runs strictly in fp32 (see hymotion_t2m_h3d272.py);
# DTYPE here only selects the TEXT ENCODER precision (bf16 ok). New default OUT
# dir so --skip_existing does not reuse the old bf16+wrong-CFG outputs.
OUT=${OUT:-outputs/evaluation/hymotion_h3d272/hy_272_fp32cfg}
METRICS=${METRICS:-outputs/evaluation/hymotion_h3d272/metrics_fp32cfg.json}
LOGDIR=${LOGDIR:-outputs/evaluation/hymotion_h3d272/logs_fp32cfg}
N=${N:-8}
BATCH=${BATCH:-12}
GUIDANCE=${GUIDANCE:-5.0}
DTYPE=${DTYPE:-bf16}
LIMIT=${LIMIT:-0}
mkdir -p "$OUT" "$LOGDIR"

pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i nohup python3 -u scripts/eval/hymotion_t2m_h3d272.py \
    --out_dir "$OUT" --device cuda --guidance "$GUIDANCE" --batch_size "$BATCH" \
    --dtype "$DTYPE" --num_shards "$N" --shard_index "$i" --limit "$LIMIT" --skip_existing \
    > "$LOGDIR/shard_$i.log" 2>&1 &
  pids+=($!)
done
echo "[taiji] launched $N shards (batch=$BATCH dtype=$DTYPE guidance=$GUIDANCE)"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[taiji] generation done; #files=$(ls "$OUT"/*.npy 2>/dev/null | wc -l) (shard_fail=$fail)"

python3 -u scripts/eval/eval_ms_h3d272.py --pred_dir "$OUT" \
  --out_json "$METRICS" 2>&1 | tee "$LOGDIR/eval.log"
echo "HYMOTION_H3D272_ALL_DONE"
