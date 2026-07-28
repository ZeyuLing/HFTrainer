#!/usr/bin/env bash
# One exact Table-4 position-baseline setting: generation -> IK -> NPZ -> metrics.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

METHOD=${METHOD:?Set METHOD=condmdi|maskcontrol|omnicontrol|projflow}
SETTING=${SETTING:?Set an E17_* Table-4 position setting}
PHASE=${PHASE:-all} # gen | post | all
RUN_ID=${RUN_ID:-official_20260720}
CANONICAL_ROOT=${CANONICAL_ROOT:-outputs/evaluation/body_part/humanml3d_official_test_4012}
BASE="$CANONICAL_ROOT/$SETTING/$METHOD/$RUN_ID"
if [ "$METHOD" = "projflow" ]; then
  PRED="$BASE/predictions_joints22"
else
  PRED="$BASE/predictions_hml263"
fi
SMPL="$BASE/smplx"
NPZ="$BASE/npz"
MET="$BASE/metrics"
LOG="$BASE/logs"
[ -d "$PRED" ] || mkdir -p "$PRED"
mkdir -p "$SMPL" "$NPZ" "$MET" "$LOG"

PYTHON=${PYTHON:-python3}
MOTIUS_ROOT=${MOTIUS_ROOT:-$(dirname "$ROOT")/Motius}
DATA_FILE=${DATA_FILE:-data/eval/m2m_v2/eval_hml3d_official_control_4012.json}
GT_HML263=${GT_HML263:-ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs}
MODEL_DIR=${MODEL_DIR:-ref_repo/MDM/body_models}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
MAX_SAMPLES=${MAX_SAMPLES:-0}
BATCH_SIZE=${BATCH_SIZE:-0}
DEVICE=${DEVICE:-cuda}
ALLOW_MASKCONTROL_EXTRA_AXES=${ALLOW_MASKCONTROL_EXTRA_AXES:-0}
PROJFLOW_REPO=${PROJFLOW_REPO:-$MOTIUS_ROOT/ref_repo/ProjFlow}
PROJFLOW_ARTIFACT=${PROJFLOW_ARTIFACT:-$MOTIUS_ROOT/outputs/checkpoints/projflow-official}
PROJFLOW_NUM_STEPS=${PROJFLOW_NUM_STEPS:-100}

extra=()
[ "$ALLOW_MASKCONTROL_EXTRA_AXES" = "1" ] && extra+=(--allow-maskcontrol-extra-axes)
[ "$BATCH_SIZE" != "0" ] && extra+=(--batch-size "$BATCH_SIZE")
[ "$MAX_SAMPLES" != "0" ] && extra+=(--max-samples "$MAX_SAMPLES")
if [ "$METHOD" = "projflow" ]; then
  extra+=(
    --artifact "$PROJFLOW_ARTIFACT"
    --projflow-repo "$PROJFLOW_REPO"
    --projflow-num-steps "$PROJFLOW_NUM_STEPS"
  )
fi

if [ "$PHASE" = "gen" ] || [ "$PHASE" = "all" ]; then
  "$PYTHON" scripts/eval/run_bodypart_position_baseline_4012.py \
    --method "$METHOD" --setting "$SETTING" --data-file "$DATA_FILE" \
    --gt-hml263-dir "$GT_HML263" --motius-root "$MOTIUS_ROOT" \
    --out-dir "$PRED" --device "$DEVICE" --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" --skip-existing "${extra[@]}" \
    2>&1 | tee "$LOG/gen_shard_${SHARD_INDEX}.log"
fi

[ "$PHASE" = "gen" ] && exit 0

EXPECTED=${EXPECTED_SAMPLES:-4012}
actual=$(find -L "$PRED" -maxdepth 1 -name '*.npy' | wc -l)
if [ "$actual" -ne "$EXPECTED" ]; then
  echo "incomplete generation: expected=$EXPECTED actual=$actual" >&2
  exit 3
fi

GPUS=${GPUS:-0}
IFS=',' read -r -a gpu_ids <<< "$GPUS"
IK_SHARDS_PER_GPU=${IK_SHARDS_PER_GPU:-2}
IK_BATCH_SIZE=${IK_BATCH_SIZE:-256}
total_ik_shards=$((${#gpu_ids[@]} * IK_SHARDS_PER_GPU))
pids=()
for shard in $(seq 0 $((total_ik_shards - 1))); do
  gpu=${gpu_ids[$((shard % ${#gpu_ids[@]}))]}
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$PRED" --out-dir "$SMPL" --model-dir "$MODEL_DIR" \
    --source-fps 20 --target-fps 30 --device cuda --batch-size "$IK_BATCH_SIZE" \
    --floor-align --refine-iters 0 --rotation-init position_ik --skip-existing \
    --target-length-anno "$DATA_FILE" \
    --num-shards "$total_ik_shards" --shard-index "$shard" \
    > "$LOG/ik_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON" scripts/eval/build_bodypart_baseline_eval_npz_4012.py \
  --ik-dir "$SMPL" --setting "$SETTING" --data-file "$DATA_FILE" \
  --out-dir "$NPZ" --expected-samples "$EXPECTED" | tee "$LOG/pack.log"
"$PYTHON" scripts/eval/score_bodypart_position_baseline_4012.py \
  --npz-dir "$NPZ" --setting "$SETTING" --method "$METHOD" \
  --expected-samples "$EXPECTED" --out "$MET/geometry.json" | tee "$LOG/geometry.log"
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "$PYTHON" scripts/eval/eval_npz_universal_tmr_fid.py \
  --pred-npz-dir "$NPZ" --tag "${METHOD}_${SETTING}" \
  --out-json "$MET/utmr.json" > "$LOG/utmr.log" 2>&1
touch "$BASE/DONE"
echo "[done] $BASE"
