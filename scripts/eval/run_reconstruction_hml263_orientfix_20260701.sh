#!/usr/bin/env bash
# Detached HumanML3D HML263 -> SMPL motion135 rebuild with root-heading lock.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false

SRC_BASE="${SRC_BASE:-outputs/evaluation/reconstruction/humanml3d_official_test}"
GT_HML263="${GT_HML263:-outputs/evaluation/t2m/humanml3d_official_test/hml263/gt}"
OUT_BASE="${OUT_BASE:-outputs/evaluation/reconstruction/humanml3d_official_test/_orientfix_20260701}"
LOG_DIR="${LOG_DIR:-logs/reconstruction_orientfix_20260701/retarget_motion135}"
METHODS="${METHODS:-gt_hml263_bridge t2mgpt momask mld mogents motiongpt3}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
SHARD_COUNT="${SHARD_COUNT:-$NUM_SHARDS}"
REFINE_ITERS="${REFINE_ITERS:-80}"
REFINE_LR="${REFINE_LR:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MODEL_DIR="${MODEL_DIR:-checkpoints/baselines/body_models}"
ANNO="${ANNO:-data/annotation/test_hml3d_official272_gtlen.json}"
EXPECTED="${EXPECTED:-4042}"

mkdir -p "$OUT_BASE/motion135" "$LOG_DIR"

method_in_dir() {
  local method="$1"
  if [[ "$method" == "gt_hml263_bridge" ]]; then
    echo "$GT_HML263"
  else
    echo "$SRC_BASE/hml263/$method"
  fi
}

launch() {
  echo "[controller-start] $(date -Is) host=$(hostname) out_base=$OUT_BASE" | tee "$LOG_DIR/controller.log"
  echo "[controller-start] methods=$METHODS shards=$NUM_SHARDS offset=$SHARD_OFFSET count=$SHARD_COUNT refine_iters=$REFINE_ITERS" | tee -a "$LOG_DIR/controller.log"
  : > "$LOG_DIR/pids.txt"
  for local_shard in $(seq 0 $((SHARD_COUNT - 1))); do
    local shard
    shard=$((SHARD_OFFSET + local_shard))
    local gpu log pid
    gpu=$((local_shard % 8))
    log="$LOG_DIR/shard_${shard}_of_${NUM_SHARDS}.log"
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="$gpu"
      for method in $METHODS; do
        in_dir="$(method_in_dir "$method")"
        out_dir="$OUT_BASE/motion135/$method"
        mkdir -p "$out_dir"
        echo "[method-start] $(date -Is) shard=$shard gpu=$gpu method=$method in=$in_dir out=$out_dir"
        python3 -u scripts/eval/hml263_to_smpl_ik.py \
          --in-dir "$in_dir" \
          --out-dir "$out_dir" \
          --model-dir "$MODEL_DIR" \
          --num-shards "$NUM_SHARDS" \
          --shard-index "$shard" \
          --source-fps 20 \
          --target-fps 30 \
          --target-length-anno "$ANNO" \
          --device cuda \
          --batch-size "$BATCH_SIZE" \
          --floor-align \
          --rotation-init hml263_init \
          --orientation-mode parent_frame \
          --refine-iters "$REFINE_ITERS" \
          --refine-lr "$REFINE_LR" \
          --restore-root-translation none \
          --lock-global-orient \
          --skip-existing
        echo "[method-done] $(date -Is) shard=$shard gpu=$gpu method=$method"
      done
      echo "[shard-done] $(date -Is) shard=$shard gpu=$gpu"
    ) > "$log" 2>&1 &
    pid=$!
    echo "$shard $gpu $pid $log" | tee -a "$LOG_DIR/pids.txt"
    sleep "${LAUNCH_STAGGER_SEC:-2}"
  done

  local failed=0
  while read -r shard gpu pid log; do
    if ! wait "$pid"; then
      echo "[shard-fail] shard=$shard gpu=$gpu pid=$pid log=$log" | tee -a "$LOG_DIR/controller.log"
      failed=1
    fi
  done < "$LOG_DIR/pids.txt"
  echo "[controller-done] $(date -Is) failed=$failed" | tee -a "$LOG_DIR/controller.log"
  return "$failed"
}

status() {
  echo "[status] $(date -Is) host=$(hostname) out_base=$OUT_BASE"
  echo "[status] shards=$NUM_SHARDS offset=$SHARD_OFFSET count=$SHARD_COUNT log_dir=$LOG_DIR"
  for method in $METHODS; do
    count="$(find "$OUT_BASE/motion135/$method" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l || true)"
    printf "[count] %-18s %4s/%s\n" "$method" "$count" "$EXPECTED"
  done
  echo "--- pids ---"
  if [[ -f "$LOG_DIR/pids.txt" ]]; then
    while read -r shard gpu pid log; do
      printf "shard=%s gpu=%s pid=%s " "$shard" "$gpu" "$pid"
      ps -p "$pid" -o pid,stat,etime,cmd --no-headers || echo dead
    done < "$LOG_DIR/pids.txt"
  fi
  echo "--- gpu ---"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  echo "--- logs ---"
  for log in "$LOG_DIR"/shard_*_of_"$NUM_SHARDS".log; do
    [[ -e "$log" ]] || continue
    echo "--- $(basename "$log") ---"
    tail -n 8 "$log" || true
  done
  echo "--- controller ---"
  tail -n 20 "$LOG_DIR/controller.log" 2>/dev/null || true
}

case "${1:-launch}" in
  launch) launch ;;
  status) status ;;
  *) echo "usage: $0 [launch|status]" >&2; exit 2 ;;
esac
