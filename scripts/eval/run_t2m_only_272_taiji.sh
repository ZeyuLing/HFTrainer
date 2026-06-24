#!/usr/bin/env bash
# HYMotion-M2M T2M-only (from-Lite finetune) -> MS-272 generation on the
# HumanML3D official test set, for a DEDICATED 1-host 8-GPU Taiji job.
#
# Sharded across 8 GPUs; outputs pred272 .npy + motion_135 .npy under the
# CLAUDE.md outputs/ contract. Scoring (eval_motionstreamer_272.py with the
# /dev/shm eval272 ckpt) is run separately on a small box afterwards.
#
# Env knobs: EPOCH (ckpt epoch, default 5), CFG (default 5.0), STEPS (50).
# Submit with:
#   python3 tools/taiji_submit.py t2m_only_ep5_272 \
#       --start-cmd "EPOCH=5 CFG=5.0 bash scripts/eval/run_t2m_only_272_taiji.sh" \
#       --host_num 1 -b AILab_DHC_DD
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD"
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false

N=${N:-8}
EPOCH=${EPOCH:-5}
CFG=${CFG:-5.0}
STEPS=${STEPS:-50}
LIMIT=${LIMIT:-0}
DATA_ROOT=${DATA_ROOT:-ref_repo/MotionStreamer/MotionStreamer/humanml3d_272}

CONFIG=configs/hymotion_m2m/hymotion_m2m_t2m_only_from_lite_046b.py
CKPT=work_dirs/hymotion_m2m_t2m_only_from_lite/checkpoint-epoch_${EPOCH}
ROOT=outputs/evaluation/t2m/humanml3d_official_test/ms272/t2m_only_from_lite_ep${EPOCH}_cfg${CFG}
out="$ROOT/pred272"; m135="$ROOT/m135"; logdir="$ROOT/logs"
mkdir -p "$out" "$m135" "$logdir"

echo "[taiji] T2M-only ep${EPOCH} cfg=${CFG} steps=${STEPS} ckpt=$CKPT -> $out"
if [ ! -e "$CKPT" ]; then echo "[taiji][error] ckpt missing: $CKPT"; exit 4; fi

extra=""; [ "$LIMIT" -gt 0 ] && extra="--max-samples $LIMIT"
pids=()
for i in $(seq 0 $((N-1))); do
  nohup python3 -u scripts/eval/gen_ours_m2m_272.py \
    --config "$CONFIG" --ckpt "$CKPT" \
    --data-root "$DATA_ROOT" --out "$out" --m135-dir "$m135" \
    --num-steps "$STEPS" --cfg-scale "$CFG" --rotation-space local \
    --gpu "$i" --num-shards "$N" --shard-index "$i" --skip-existing $extra \
    > "$logdir/shard_$i.log" 2>&1 &
  pids+=($!)
done
echo "[taiji] launched $N shards"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "T2M_ONLY_EP${EPOCH}_GEN_DONE n=$(ls "$out"/*.npy 2>/dev/null | wc -l) root=$ROOT (shard_fail=$fail)"
