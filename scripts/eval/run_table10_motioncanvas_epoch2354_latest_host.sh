#!/usr/bin/env bash
# Re-evaluate the complete Table 10 trajectory/waypoint grid with the audited
# latest checkpoint. Intended for an 8-host x 8-GPU Taiji job.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
cd "$ROOT"

NODE_RANK=${NODE_RANK:-${INDEX:-0}}
NUM_NODES=${NUM_NODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
PYTHON=${PYTHON:-python3}
MODEL=${MODEL:-smpl_caption_fulltasks_latest}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-work_dirs/hymotion_m2m_exact_sparse_rollout_excess_continue_from2200_to10000_lr1e5_tcp_20260722/checkpoint-epoch_2354}
OUT=${OUT:-outputs/evaluation/humanml3d/trajectory_waypoint/motioncanvas_epoch2354_table10_rerun_latest_20260726}
DATA_FILE_OVERRIDE=${DATA_FILE_OVERRIDE:-eval_hml3d_official_control_4012.json}
MOTION_DATA_DIR=${MOTION_DATA_DIR:-data/eval/m2m_v2}
SETTINGS=${SETTINGS:-"A_xz_dense B_xz_sparse C_xz_heading D_xyz_dense E_xyz_sparse F_xyz_heading"}
FINAL_SETTINGS=${FINAL_SETTINGS:-"E5_A_xz_dense E5_B_xz_sparse E5_C_xz_heading E5_D_xyz_dense E5_E_xyz_sparse E5_F_xyz_heading"}
WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS:-64800}
TOTAL_SHARDS=$((NUM_NODES * GPUS_PER_NODE))

if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES || GPUS_PER_NODE < 1 )); then
  echo "invalid rank/nodes/gpus: ${NODE_RANK}/${NUM_NODES}/${GPUS_PER_NODE}" >&2
  exit 2
fi
if [ ! -s "$CHECKPOINT_DIR/model.pt" ] && [ ! -s "$CHECKPOINT_DIR/model.safetensors" ]; then
  echo "missing model weights in exact checkpoint: $CHECKPOINT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT/logs/hosts"
trap 'date -Is > "$OUT/logs/hosts/node_${NODE_RANK}.FAILED"' ERR

specs=()
for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
  specs+=("${gpu}:$((NODE_RANK * GPUS_PER_NODE + gpu))")
done

ROOT="$ROOT" PYTHON="$PYTHON" MODEL="$MODEL" \
  WORK_DIR_OVERRIDE="$CHECKPOINT_DIR" OUT="$OUT" \
  TASKS=E5 SETTINGS="$SETTINGS" \
  SHARD_SPECS="${specs[*]}" NUM_SHARDS="$TOTAL_SHARDS" \
  MAX_SAMPLES=1000000 NUM_STEPS=50 \
  DATA_FILE_OVERRIDE="$DATA_FILE_OVERRIDE" MOTION_DATA_DIR="$MOTION_DATA_DIR" \
  USE_REWRITTEN=0 REPLACEMENT_GUIDANCE=all \
  SAVE_NPZ=1 SKIP_EXISTING_NPZ=1 \
  bash scripts/eval/run_table4_motioncanvas_shards.sh

date -Is > "$OUT/logs/hosts/node_${NODE_RANK}.DONE"
trap - ERR

if (( NODE_RANK != 0 )); then
  echo "[done] generation node=${NODE_RANK}/${NUM_NODES}"
  exit 0
fi

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while true; do
  failed=$(find "$OUT/logs/hosts" -maxdepth 1 -name 'node_*.FAILED' | wc -l)
  done_count=$(find "$OUT/logs/hosts" -maxdepth 1 -name 'node_*.DONE' | wc -l)
  echo "[wait] $(date -Is) done=${done_count}/${NUM_NODES} failed=${failed}"
  if (( failed > 0 )); then
    echo "at least one generation node failed" >&2
    exit 1
  fi
  if (( done_count == NUM_NODES )); then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for all generation nodes" >&2
    exit 1
  fi
  sleep 30
done

ROOT="$ROOT" OUT="$OUT" MODEL="$MODEL" SETTINGS="$FINAL_SETTINGS" GPU=0 \
  bash scripts/eval/finalize_table5_trajectory_metrics.sh

"$PYTHON" - "$OUT" "$MODEL" $FINAL_SETTINGS <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
model = sys.argv[2]
settings = sys.argv[3:]
errors = []
for setting in settings:
    npz_dir = out / "merged" / "npz" / model / setting / "npz"
    n_npz = len(list(npz_dir.glob("*.npz"))) if npz_dir.is_dir() else 0
    if n_npz != 4012:
        errors.append(f"{setting}: expected 4012 npz files, found {n_npz}")
    for suffix in ("ric", "new", "fid"):
        path = out / "metrics" / f"{setting}__{suffix}.json"
        if not path.is_file():
            errors.append(f"{setting}: missing {path.name}")
    fid_path = out / "metrics" / f"{setting}__fid.json"
    if fid_path.is_file():
        payload = json.loads(fid_path.read_text())
        if payload.get(setting, {}).get("n_records") != 4012:
            errors.append(f"{setting}: FID artifact does not contain 4012 records")
if errors:
    raise SystemExit("\n".join(errors))
(out / "TABLE10_COMPLETE").write_text("validated 6 settings x 4012 samples\n")
print("validated Table 10: 6 settings x 4012 samples")
PY

echo "[complete] $(date -Is) $OUT"
