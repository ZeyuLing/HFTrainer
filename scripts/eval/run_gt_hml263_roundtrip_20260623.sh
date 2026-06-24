#!/usr/bin/env bash
# Round-trip diagnostic:
#   GT SMPL motion_135 -> HML263 (prebuilt, lossless control) -> SMPL motion_135.
#
# This measures the loss introduced by the HML263 -> SMPL retargeting bridge
# used for HML263-native baselines.
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

RUN_ROOT="${RUN_ROOT:-outputs/evaluation/t2m/humanml3d_official_test/motion135/gt_hml263_roundtrip_20260623}"
HML_DIR="${HML_DIR:-outputs/evaluation/t2m/humanml3d_official_test/hml263/gt_official_test_from_motion135/pred_hml263}"
META_DIR="${META_DIR:-outputs/evaluation/t2m/humanml3d_official_test/hml263/gt_official_test_from_motion135/canonical_meta}"
M135_DIR="${M135_DIR:-$RUN_ROOT/predictions/motion135}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
ANNO="${ANNO:-outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/test_hml3d_official272_gtlen_motionclip_selected_caption.json}"
NUM_SHARDS="${NUM_SHARDS:-8}"
REFINE_ITERS="${REFINE_ITERS:-80}"
REFINE_LR="${REFINE_LR:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESTORE_ROOT_TRANSLATION="${RESTORE_ROOT_TRANSLATION:-source_transl}"
mkdir -p "$M135_DIR" "$LOG_DIR"

launch() {
  echo "[launch] $(date -Is) host=$(hostname) hml=$(find "$HML_DIR" -maxdepth 1 -name '*.npy' | wc -l) out=$M135_DIR" | tee "$LOG_DIR/launch.log"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu=$((shard % 8))
    log="$LOG_DIR/ik_s${shard}_of_${NUM_SHARDS}.log"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 -u scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "$HML_DIR" \
      --out-dir "$M135_DIR" \
      --model-dir ref_repo/MDM/body_models_nochumpy \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      --source-fps 20 \
      --target-fps 30 \
      --target-length-anno "$ANNO" \
      --device cuda \
      --batch-size "$BATCH_SIZE" \
      --floor-align \
      --refine-iters "$REFINE_ITERS" \
      --refine-lr "$REFINE_LR" \
      --canonical-meta-dir "$META_DIR" \
      --restore-root-translation "$RESTORE_ROOT_TRANSLATION" \
      --skip-existing \
      > "$log" 2>&1 &
    pid=$!
    echo "$pid" > "$LOG_DIR/ik_s${shard}_of_${NUM_SHARDS}.pid"
    echo "[pid] shard=$shard gpu=$gpu pid=$pid" | tee -a "$LOG_DIR/launch.log"
    sleep "${LAUNCH_STAGGER_SEC:-2}"
  done
}

status() {
  echo "[status] $(date -Is) host=$(hostname)"
  echo "[count] npz=$(find "$M135_DIR" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)"
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
    tail -n 5 "$log" || true
  done
}

case "${1:-status}" in
  launch) launch ;;
  status) status ;;
  *)
    echo "usage: $0 {launch|status}" >&2
    exit 2
    ;;
esac
