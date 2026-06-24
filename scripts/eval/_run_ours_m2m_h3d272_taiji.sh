#!/usr/bin/env bash
# Foreground 8-GPU HYMotion-M2M (\ours) T2M -> MS-272 generation, for a DEDICATED
# Taiji job. Generates BOTH \ours variants (no-edit specialist + edit unified)
# sequentially, each sharded across the 8 GPUs. Scoring (native TMR evaluator)
# is run separately on a small box where distilbert is cached.
#
# Submit with:
#   python3 tools/taiji_submit.py ours_m2m_t2m272 \
#       --start-cmd "bash scripts/eval/_run_ours_m2m_h3d272_taiji.sh" \
#       --host_num 1 --gpu_name V100
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer || exit 1
export PYTHONPATH="$PWD"
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false

N=${N:-8}
CFG=${CFG:-5.0}
STEPS=${STEPS:-50}
LIMIT=${LIMIT:-0}            # 0 = all; >0 = smoke (max-samples per shard input)
DATA_ROOT=${DATA_ROOT:-ref_repo/MotionStreamer/MotionStreamer/humanml3d_272}
ROOT=${ROOT:-outputs/evaluation/ours_m2m_h3d272}

# model_tag | config | ckpt
MODELS=(
  "noedit_specialist_ep405|configs/hymotion_m2m/hymotion_m2m_smpl_caption_cleandata_ablation.py|work_dirs/hymotion_m2m_v2_smpl_caption_cleandata_ablation/checkpoint-epoch_405"
  "edit_unified_ep1710|configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py|work_dirs/hymotion_m2m_v2_smpl_caption_editfix_from870_20260528/checkpoint-epoch_1710"
)

run_model() {
  local tag="$1" config="$2" ckpt="$3"
  local out="$ROOT/$tag/pred272_cfg${CFG}"
  local m135="$ROOT/$tag/m135_cfg${CFG}"
  local logdir="$ROOT/$tag/logs_cfg${CFG}"
  mkdir -p "$out" "$m135" "$logdir"
  echo "[taiji] === $tag === config=$config ckpt=$ckpt out=$out (cfg=$CFG steps=$STEPS)"
  local extra=""
  if [ "$LIMIT" -gt 0 ]; then extra="--max-samples $LIMIT"; fi
  local pids=()
  for i in $(seq 0 $((N-1))); do
    # gen_ours_m2m_272.py sets CUDA_VISIBLE_DEVICES=<--gpu> itself (before torch
    # import), so pass --gpu $i and do NOT pre-set CUDA_VISIBLE_DEVICES here.
    nohup python3 -u scripts/eval/gen_ours_m2m_272.py \
      --config "$config" --ckpt "$ckpt" \
      --data-root "$DATA_ROOT" --out "$out" --m135-dir "$m135" \
      --num-steps "$STEPS" --cfg-scale "$CFG" --rotation-space local \
      --gpu "$i" --num-shards "$N" --shard-index "$i" --skip-existing $extra \
      > "$logdir/shard_$i.log" 2>&1 &
    pids+=($!)
  done
  echo "[taiji] $tag: launched $N shards"
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  echo "[taiji] $tag: done; #files=$(ls "$out"/*.npy 2>/dev/null | wc -l) (shard_fail=$fail)"
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag config ckpt <<< "$entry"
  run_model "$tag" "$config" "$ckpt"
done

echo "OURS_M2M_H3D272_GEN_ALL_DONE root=$ROOT cfg=$CFG"
