#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

mkdir -p outputs/evaluation/humanml3d/_monitor

run_shard() {
  local tag="$1"
  local in_tag="$2"
  local shard="$3"
  local nshards="$4"
  local gpu="$5"
  local log_dir="outputs/evaluation/humanml3d/${tag}/_logs"
  mkdir -p "${log_dir}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "outputs/evaluation/humanml3d/${in_tag}_hml3d263_rootfix/humanml3d" \
    --out-dir "outputs/evaluation/humanml3d/${tag}" \
    --num-shards "${nshards}" \
    --shard-index "${shard}" \
    --device cuda \
    --batch-size 512 \
    --floor-align \
    --refine-iters 80 \
    --refine-lr 0.02 \
    --skip-existing \
    > "${log_dir}/ik_rootfix_lzy2_s${shard}_of_${nshards}_gpu${gpu}.log" 2>&1
}

echo "[rootfix] starting MotionGPT3 and MDM retarget shards"
for shard in 0 1 2 3; do
  run_shard motiongpt3_rootfix motiongpt3 "${shard}" 4 "${shard}" &
done
for shard in 0 1 2 3; do
  run_shard mdm_rootfix mdm "${shard}" 4 "$((shard + 4))" &
done

wait
echo "[rootfix] all shards finished"
