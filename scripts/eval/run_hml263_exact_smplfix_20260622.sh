#!/usr/bin/env bash
# Clean rerun for HumanML3D-263 baselines whose saved SMPL/MS272 artifacts were
# contaminated by mixed/legacy conversion outputs.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HFTRAINER_SKIP_AUTOREGISTER=1

RUN_ID="${RUN_ID:-table1_hml263_smplfix_20260622}"
SUITE="outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/$RUN_ID"
LOG_DIR="$SUITE/logs"
ANNO="${ANNO:-data/annotation/test_hml3d_official272_gtlen.json}"
SPLIT="${SPLIT:-ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt}"
REFINE_ITERS="${REFINE_ITERS:-80}"
REFINE_LR="${REFINE_LR:-0.02}"
NUM_SHARDS="${NUM_SHARDS:-${NUM_GPUS:-${TJ_GPU_NUM:-8}}}"
WORKERS="${WORKERS:-16}"
mkdir -p "$LOG_DIR"

if [[ -z "${METHOD:-}" ]]; then
  echo "[error] set METHOD to one of: mdm motiongpt3 flowmdm motionlab mld momask t2mgpt mogents" >&2
  exit 2
fi

case "$METHOD" in
  mdm|motiongpt3|flowmdm|motionlab|mld|momask|t2mgpt)
    HML_DIR="outputs/evaluation/t2m/humanml3d_official_test/hml263/${METHOD}_official/predictions/hml263"
    ;;
  mogents)
    HML_DIR="outputs/evaluation/t2m/humanml3d_official_test/hml263/mogents"
    ;;
  *)
    echo "[error] unsupported METHOD=$METHOD" >&2
    exit 2
    ;;
esac

M135_DIR="$SUITE/motion135/$METHOD"
MS272_DIR="$SUITE/ms272_npy/$METHOD"
PREP_DIR="$SUITE/prep/$METHOD"
mkdir -p "$M135_DIR" "$MS272_DIR" "$PREP_DIR"

if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
else
  GPU_IDS=()
  for ((g=0; g<NUM_SHARDS; g++)); do
    GPU_IDS+=("$g")
  done
fi
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "[error] empty GPU list" >&2
  exit 2
fi

echo "[start] method=$METHOD run_id=$RUN_ID shards=$NUM_SHARDS gpus=${GPU_IDS[*]} refine_iters=$REFINE_ITERS $(date -Is)" | tee "$LOG_DIR/${METHOD}.run.log"
echo "[paths] hml=$HML_DIR m135=$M135_DIR ms272=$MS272_DIR prep=$PREP_DIR" | tee -a "$LOG_DIR/${METHOD}.run.log"

if [[ "${SKIP_IK:-0}" != "1" ]]; then
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPU_IDS[$((shard % ${#GPU_IDS[@]}))]}"
    log="$LOG_DIR/${METHOD}.ik_s${shard}_of_${NUM_SHARDS}.log"
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="$gpu"
      python3 scripts/eval/hml263_to_smpl_ik.py \
        --in-dir "$HML_DIR" \
        --out-dir "$M135_DIR" \
        --ids "$SPLIT" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard" \
        --source-fps 20 \
        --target-fps 30 \
        --target-length-anno "$ANNO" \
        --device cuda \
        --batch-size 1 \
        --floor-align \
        --refine-iters "$REFINE_ITERS" \
        --refine-lr "$REFINE_LR" \
        > "$log" 2>&1
      echo "exit_code=0 finished_at=$(date -Is)" > "$log.status"
    ) &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  if [[ "$rc" != "0" ]]; then
    echo "[fail] IK failed for $METHOD" | tee -a "$LOG_DIR/${METHOD}.run.log"
    exit "$rc"
  fi
fi

python3 scripts/eval/audit_table1_lengths.py \
  --out-dir "$SUITE/audits/${METHOD}_motion135_lengths" \
  --method "$METHOD=$M135_DIR" \
  > "$LOG_DIR/${METHOD}.length_audit.log" 2>&1

python3 scripts/data/convert_motion135_to_h3d272.py \
  --in-dir "$M135_DIR" \
  --out-dir "$MS272_DIR" \
  --rotation-space local \
  --workers "$WORKERS" \
  > "$LOG_DIR/${METHOD}.motion135_to_272.log" 2>&1

python3 scripts/eval/repack_pred_to_272ids.py \
  --motion272-dir "$MS272_DIR" \
  --anno-file "$ANNO" \
  --id-passthrough \
  --out-dir "$PREP_DIR" \
  --workers "$WORKERS" \
  > "$LOG_DIR/${METHOD}.repack.log" 2>&1

python3 scripts/eval/audit_table1_lengths.py \
  --out-dir "$SUITE/audits/${METHOD}_prep_lengths" \
  --method "$METHOD=$PREP_DIR" \
  >> "$LOG_DIR/${METHOD}.length_audit.log" 2>&1

echo "[done] method=$METHOD $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.run.log"
