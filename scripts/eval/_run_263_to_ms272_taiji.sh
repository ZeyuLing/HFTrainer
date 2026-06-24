#!/usr/bin/env bash
# DEDICATED 8-GPU Taiji job: 263-baseline (t2mgpt, momask) -> MS-272 eval.
# Stage A IK (refine-80, sharded across 8 GPUs) -> Stage B 135->272 -> Stage C MS-272.
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer || exit 1
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false

BASE=outputs/evaluation/ms272_from263
N=${N:-8}
declare -A SRC=(
  [t2mgpt]=outputs/evaluation/t2mgpt_h3d263_official/t2mgpt_263
  [momask]=outputs/evaluation/momask_h3d263_official/momask_263
)

for M in t2mgpt momask; do
  DIR263=${SRC[$M]}
  DIR135=$BASE/${M}_smpl135
  DIR272=$BASE/${M}_272
  LOGDIR=$BASE/logs_${M}
  mkdir -p "$DIR135" "$DIR272" "$LOGDIR"
  echo "==================== [$M] Stage A: 263->135 IK refine-80 ($N shards) ===================="
  pids=()
  for i in $(seq 0 $((N-1))); do
    CUDA_VISIBLE_DEVICES=$i nohup python3 -u scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "$DIR263" --out-dir "$DIR135" \
      --model-dir ref_repo/MDM/body_models \
      --source-fps 20 --target-fps 30 --floor-align \
      --refine-iters 80 --refine-lr 0.02 --rot6d-convention row \
      --num-shards "$N" --shard-index "$i" \
      --device cuda --skip-existing \
      > "$LOGDIR/ik_shard_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[$M] IK done: $(ls "$DIR135"/*.npz 2>/dev/null | wc -l) files"
  echo "==================== [$M] Stage B: 135->272 ===================="
  python3 -u scripts/data/convert_motion135_to_h3d272.py \
    --in-dir "$DIR135" --out-dir "$DIR272" --workers 16 --skip-existing \
    > "$LOGDIR/convert_272.log" 2>&1
  echo "[$M] 272 done: $(ls "$DIR272"/*.npy 2>/dev/null | wc -l) files"
done

echo "==================== Stage C: MS-272 evaluation ===================="
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/_ms272_eval_263baselines.py 2>&1 | tee "$BASE/ms272_eval_summary.log"
echo "==================== ALL DONE ===================="
