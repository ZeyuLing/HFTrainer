#!/usr/bin/env bash
# Foreground 8-GPU MotionMillion / "Go to Zero" (7B) T2M -> MS-272 generation +
# scoring, for a DEDICATED Taiji job (blocks until all shards finish, then
# scores). Use with:
#   python3 tools/taiji_submit.py mm_t2m_eval \
#       --start-cmd "cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer && bash scripts/eval/_run_motionmillion_h3d272_taiji.sh" \
#       --host_num 1 --gpu_name V100
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer || exit 1
export PYTHONPATH="$PWD"
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

OUT=${OUT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/gotozero_7b_train}
LOGDIR=outputs/evaluation/motionmillion_h3d272/logs
N=${N:-8}
DTYPE=${DTYPE:-bf16}
STEPS=${STEPS:-50}
AR=${AR:-checkpoints/motionmillion/pretrained_models/motionmillion_7B.pth}
TEXT=${TEXT:-checkpoints/flan-t5-xl}
LIMIT=${LIMIT:-0}
mkdir -p "$OUT" "$LOGDIR"

pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i nohup python3 -u scripts/eval/motionmillion_h3d272.py \
    --out_dir "$OUT" --device cuda --dtype "$DTYPE" --ar_path "$AR" \
    --text_model_name "$TEXT" \
    --max_sample_steps "$STEPS" --num_shards "$N" --shard_index "$i" \
    --limit "$LIMIT" --skip_existing \
    > "$LOGDIR/shard_$i.log" 2>&1 &
  pids+=($!)
done
echo "[taiji] launched $N MotionMillion shards (dtype=$DTYPE steps=$STEPS ar=$AR)"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[taiji] generation done; #files=$(ls "$OUT"/*.npy 2>/dev/null | wc -l) (shard_fail=$fail)"

python3 -u scripts/eval/eval_ms_h3d272.py --pred_dir "$OUT" \
  --out_json outputs/evaluation/motionmillion_h3d272/metrics_ms272.json 2>&1 | tee "$LOGDIR/eval_ms272.log"
echo "MOTIONMILLION_H3D272_ALL_DONE"
