#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"

run_shard() {
  local tag="$1"
  local gpu="$2"
  local num_shards="$3"
  local shard="$4"
  local log_dir="outputs/evaluation/humanml3d/${tag}/_logs"
  mkdir -p "${log_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "outputs/evaluation/humanml3d/${tag}_hml3d263/humanml3d" \
    --out-dir "outputs/evaluation/humanml3d/${tag}" \
    --num-shards "${num_shards}" \
    --shard-index "${shard}" \
    --device cuda \
    --batch-size 512 \
    --floor-align \
    --refine-iters 80 \
    --refine-lr 0.02 \
    --skip-existing \
    > "${log_dir}/ik_lzy2_s${shard}_of_${num_shards}_gpu${gpu}.log" 2>&1
}

python3 -c "import scipy, torch; import scripts.eval.hml263_to_smpl_ik as m; print('deps_ok', scipy.__version__, torch.__version__, torch.cuda.device_count(), m.smplx.__file__)"

pids=()

# lzy_debug_machine_2 has 8 free V100s. Skip MLD: current MLD HML3D output is noisy.
for shard in 0 1 2; do
  run_shard motiongpt3 "${shard}" 3 "${shard}" &
  pids+=("$!")
done

for shard in 0 1 2; do
  run_shard momask "$((shard + 3))" 3 "${shard}" &
  pids+=("$!")
done

for shard in 0 1; do
  run_shard mdm "$((shard + 6))" 2 "${shard}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
