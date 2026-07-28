#!/usr/bin/env bash
# ProjFlow dense-root and sparse-waypoint evaluation through the Motius adapter.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
MOTIUS_ROOT=${MOTIUS_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/Motius}
cd "$ROOT"
export PYTHONPATH="$MOTIUS_ROOT:$ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1

MODE=${MODE:?set MODE=dense or sparse}
AXES=${AXES:?set AXES=xz or xyz}
case "$MODE:$AXES" in
  dense:xz) SETTING=E5_A_xz_dense ;;
  sparse:xz) SETTING=E5_B_xz_sparse ;;
  dense:xyz) SETTING=E5_D_xyz_dense ;;
  sparse:xyz) SETTING=E5_E_xyz_sparse ;;
  *) echo "unsupported MODE=$MODE AXES=$AXES" >&2; exit 2 ;;
esac

TAG="projflow_${AXES}_${MODE}"
OUT=${OUT:-outputs/evaluation/humanml3d/trajectory_waypoint}
METHOD_ROOT="$OUT/$TAG"
JOINTS22="$METHOD_ROOT/joints22"
SMPLX="$METHOD_ROOT/smplx"
PACKED="$METHOD_ROOT/$SETTING"
LOG="$METHOD_ROOT/logs"
mkdir -p "$JOINTS22" "$SMPLX" "$PACKED" "$LOG"

PROTOCOL_DIR=${PROTOCOL_DIR:-data/evaluation/trajectory/humanml3d_official_test_4012}
IDS=${IDS:-$PROTOCOL_DIR/source_ids.json}
IDX2SID=${IDX2SID:-$PROTOCOL_DIR/idx2sid.json}
WAYPOINTS=${WAYPOINTS:-$PROTOCOL_DIR/sparse_waypoints.json}
CAPTIONS=${CAPTIONS:-data/eval/m2m_v2/eval_hml3d_official_control_4012.json}
GT_HML263=${GT_HML263:-ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs}
PROJFLOW_REPO=${PROJFLOW_REPO:-$MOTIUS_ROOT/ref_repo/ProjFlow}
ARTIFACT=${ARTIFACT:-$MOTIUS_ROOT/outputs/checkpoints/projflow-official}
MODEL_DIR=${MODEL_DIR:-ref_repo/MDM/body_models}
OURS_NPZ_BASE=${OURS_NPZ_BASE:-outputs/evaluation/humanml3d/trajectory_waypoint/motioncanvas_ep1500_20260716_official4012/merged/npz/smpl_caption_fulltasks_latest}
METRICS_DIR=${METRICS_DIR:-$OUT/_metrics}
EXPECTED_COUNT=${EXPECTED_COUNT:-$(python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); print(len(x.get("ids", x.get("source_ids", x.get("data_list", x))) if isinstance(x, dict) else x))' "$IDS")}

NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TOTAL_SHARDS=${TOTAL_SHARDS:-$NGPU}
SHARD_OFFSET=${SHARD_OFFSET:-0}
BATCH=${BATCH:-4}
NUM_STEPS=${NUM_STEPS:-100}
PHASE=${PHASE:-all}
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

if [ "$PHASE" = gen ] || [ "$PHASE" = all ]; then
  pids=()
  for local_rank in $(seq 0 $((NGPU - 1))); do
    gpu=${GPU_ARR[$local_rank]}
    shard=$((SHARD_OFFSET + local_rank))
    CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/run_projflow_trajectory_4012.py \
      --artifact "$ARTIFACT" --projflow-repo "$PROJFLOW_REPO" \
      --ids "$IDS" --waypoint-file "$WAYPOINTS" --data-file "$CAPTIONS" \
      --gt-hml263-dir "$GT_HML263" --out-dir "$JOINTS22" \
      --mode "$MODE" --axes "$AXES" --batch-size "$BATCH" \
      --num-steps "$NUM_STEPS" --num-shards "$TOTAL_SHARDS" \
      --shard-index "$shard" --skip-existing \
      > "$LOG/generate_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [ "$failed" = 0 ] || exit 3
fi

actual_count=$(find "$JOINTS22" -maxdepth 1 -name '*.npy' | wc -l)
echo "[generation] $TAG ${actual_count}/${EXPECTED_COUNT} files"
if [ "$actual_count" -ne "$EXPECTED_COUNT" ]; then
  echo "incomplete ProjFlow generation: ${actual_count}/${EXPECTED_COUNT}" >&2
  exit 4
fi
[ "$PHASE" = gen ] && exit 0

pids=()
IK_SHARDS_PER_GPU=${IK_SHARDS_PER_GPU:-2}
TOTAL_IK_SHARDS=$((NGPU * IK_SHARDS_PER_GPU))
for local_rank in $(seq 0 $((TOTAL_IK_SHARDS - 1))); do
  gpu=${GPU_ARR[$((local_rank % NGPU))]}
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$JOINTS22" --out-dir "$SMPLX" --model-dir "$MODEL_DIR" \
    --source-fps 20 --target-fps 30 --device cuda --batch-size 256 \
    --floor-align --refine-iters 0 --skip-existing \
    --num-shards "$TOTAL_IK_SHARDS" --shard-index "$local_rank" \
    > "$LOG/ik_shard_${local_rank}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

python3 scripts/eval/build_e5_baseline_eval_npz.py \
  --ours-npz-dir "$OURS_NPZ_BASE/$SETTING/npz" \
  --pred-sid-dir "$SMPLX" --idx2sid "$IDX2SID" --out-dir "$PACKED" \
  > "$LOG/package.log" 2>&1

METRICS_DIR="$METRICS_DIR" bash scripts/eval/run_e5_baseline_metrics.sh \
  "${TAG}_${SETTING}" "$PACKED" "${GPU_ARR[0]}" \
  > "$LOG/metrics.log" 2>&1
echo "[done] $TAG -> $METHOD_ROOT"
