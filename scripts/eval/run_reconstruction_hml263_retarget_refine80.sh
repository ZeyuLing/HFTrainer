#!/usr/bin/env bash
# Rebuild HumanML3D reconstruction hml263-native rows as SMPL motion_135.
#
# The official bridge for hml263 -> SMPL/MS272 preserves the explicit HumanML
# root heading, maps the HumanML canonical-skeleton rotation block onto the SMPL
# rest skeleton, and only refines body pose/translation against joints.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

BASE="${BASE:-outputs/evaluation/reconstruction/humanml3d_official_test}"
METHODS="${METHODS:-t2mgpt momask mld mogents motiongpt3}"
NUM_SHARDS="${NUM_SHARDS:-8}"
REFINE_ITERS="${REFINE_ITERS:-80}"
REFINE_LR="${REFINE_LR:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MODEL_DIR="${MODEL_DIR:-checkpoints/baselines/body_models}"
ANNO="${ANNO:-outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json}"
LOG_DIR="${LOG_DIR:-logs/reconstruction_humanml3d_20260630/retarget_refine80}"
EXPECTED="${EXPECTED:-4042}"
mkdir -p "$LOG_DIR"

backup_invalid() {
  local ts backup moved=0
  ts="${TS:-$(date +%Y%m%d_%H%M%S)}"
  backup="$BASE/_invalid_hmlrot_refine0_${ts}"
  mkdir -p "$backup"
  for rep in motion135 ms272; do
    for method in $METHODS; do
      if [[ -d "$BASE/$rep/$method" ]]; then
        mkdir -p "$backup/$rep"
        mv "$BASE/$rep/$method" "$backup/$rep/$method"
        moved=$((moved + 1))
      fi
    done
  done
  for file in metrics_summary.json metrics_summary.tsv; do
    if [[ -f "$BASE/$file" ]]; then
      mv "$BASE/$file" "$backup/$file"
      moved=$((moved + 1))
    fi
  done
  echo "[backup] moved=$moved backup=$backup"
}

launch() {
  mkdir -p "$LOG_DIR"
  echo "[launch] $(date -Is) host=$(hostname) base=$BASE methods=[$METHODS]" | tee "$LOG_DIR/launch.log"
  echo "[launch] model_dir=$MODEL_DIR refine_iters=$REFINE_ITERS refine_lr=$REFINE_LR shards=$NUM_SHARDS" | tee -a "$LOG_DIR/launch.log"
  : > "$LOG_DIR/pids.txt"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu log pid
    gpu=$((shard % 8))
    log="$LOG_DIR/ik_s${shard}_of_${NUM_SHARDS}.log"
    (
      set -euo pipefail
      for method in $METHODS; do
        in_dir="$BASE/hml263/$method"
        out_dir="$BASE/motion135/$method"
        echo "[method-start] $(date -Is) shard=$shard gpu=$gpu method=$method in=$in_dir out=$out_dir"
        CUDA_VISIBLE_DEVICES="$gpu" python3 -u scripts/eval/hml263_to_smpl_ik.py \
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
        echo "[method-done] $(date -Is) shard=$shard method=$method"
      done
    ) > "$log" 2>&1 &
    pid=$!
    echo "$pid" > "$LOG_DIR/ik_s${shard}_of_${NUM_SHARDS}.pid"
    echo "$shard $gpu $pid $log" >> "$LOG_DIR/pids.txt"
    echo "[pid] shard=$shard gpu=$gpu pid=$pid log=$log" | tee -a "$LOG_DIR/launch.log"
    sleep "${LAUNCH_STAGGER_SEC:-2}"
  done
  if [[ "${WAIT_FOR_SHARDS:-1}" == "1" ]]; then
    echo "[wait] waiting for all shard workers" | tee -a "$LOG_DIR/launch.log"
    wait
    echo "[done] all shard workers finished at $(date -Is)" | tee -a "$LOG_DIR/launch.log"
  fi
}

launch_detached() {
  mkdir -p "$LOG_DIR"
  nohup setsid bash "$0" launch > "$LOG_DIR/controller.log" 2>&1 < /dev/null &
  echo "[controller] pid=$! log=$LOG_DIR/controller.log"
}

status() {
  echo "[status] $(date -Is) host=$(hostname)"
  for method in $METHODS motionstreamer; do
    hml_count=$(find "$BASE/hml263/$method" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l || true)
    m135_count=$(find "$BASE/motion135/$method" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
    ms272_count=$(find "$BASE/ms272/$method" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
    printf "[count] %-14s hml263=%4s motion135=%4s ms272=%4s\n" "$method" "$hml_count" "$m135_count" "$ms272_count"
  done
  echo "--- processes ---"
  for pid_file in "$LOG_DIR"/ik_s*_of_"$NUM_SHARDS".pid; do
    [[ -e "$pid_file" ]] || continue
    pid=$(cat "$pid_file")
    printf "%s pid=%s " "$(basename "$pid_file")" "$pid"
    ps -p "$pid" -o pid,stat,etime,cmd --no-headers || echo dead
  done
  echo "--- gpu ---"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
  for log in "$LOG_DIR"/ik_s*_of_"$NUM_SHARDS".log; do
    [[ -e "$log" ]] || continue
    echo "--- $(basename "$log") ---"
    tail -n 8 "$log" || true
  done
}

finalize() {
  local bad=0 count
  for method in $METHODS; do
    count=$(find "$BASE/motion135/$method" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
    if [[ "$count" -lt "$EXPECTED" ]]; then
      echo "[incomplete] motion135/$method $count/$EXPECTED"
      bad=1
    fi
  done
  if [[ "$bad" != 0 ]]; then
    exit 1
  fi
  python3 scripts/eval/materialize_leaderboard_canonical_paths.py --only reconstruction
  for method in $METHODS; do
    count=$(find "$BASE/ms272/$method" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
    echo "[finalize] ms272/$method $count/$EXPECTED"
  done
}

case "${1:-status}" in
  backup-invalid) backup_invalid ;;
  launch-detached) launch_detached ;;
  launch) launch ;;
  status) status ;;
  finalize) finalize ;;
  *)
    echo "usage: $0 {backup-invalid|launch-detached|launch|status|finalize}" >&2
    exit 2
    ;;
esac
