#!/usr/bin/env bash
# KIMODO SOMA-30 global-rotation -> SMPL direct transfer (faithful "GMR"; CPU, no
# IK) for all 3 temporal protocols, then build (NO splice) + 272 metrics.
# Outputs to smplx_rot / eval_npz_rot / _metrics_rot so it can be compared against
# the guide-IK variant before picking the table numbers.
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
TU=output/evaluation/temporal_unified
LOG=$TU/logs; mkdir -p "$LOG"
NSH=${NSH:-8}

pids=()
for proto in pre20 post20 mid60; do
  SM=$TU/kimodo/$proto/smplx_rot; mkdir -p "$SM"
  RAWNPZ=$TU/kimodo/$proto/raw/E2_$proto/npz
  for s in $(seq 0 $((NSH-1))); do
    python3 scripts/eval/kimodo_soma_to_smpl_byid.py \
      --data-file data/eval/m2m_v2/eval_h3d_editing.json --max-samples 4012 \
      --raw-npz-dir "$RAWNPZ" --out-dir "$SM" --mode rotation --device cpu \
      --skip-existing --num-shards "$NSH" --shard-index "$s" \
      > "$LOG/kimodo_rot_${proto}_s${s}.log" 2>&1 &
    pids+=("$!")
  done
done
echo "[rot] launched ${#pids[@]} workers $(date)"
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[rot] retarget done $(date)"
for proto in pre20 post20 mid60; do
  echo "  $proto smplx_rot=$(ls $TU/kimodo/$proto/smplx_rot/*.npz 2>/dev/null | wc -l)"
done

for proto in pre20 post20 mid60; do
  SM=$TU/kimodo/$proto/smplx_rot; EN=$TU/kimodo/$proto/eval_npz_rot; MD=$TU/_metrics_rot
  mkdir -p "$EN" "$MD"
  python3 scripts/eval/build_baseline_eval_npz.py --ik-dir "$SM" --protocol "$proto" \
    --out-dir "$EN" > "$LOG/kimodo_rot_build_${proto}.log" 2>&1
  python3 scripts/eval/paper_npz_ric_mpjpe.py --npz-dir "$EN" --tag "kimodo_$proto" \
    --out-json "$MD/kimodo_${proto}__ric.json" > "$LOG/kimodo_rot_ric_${proto}.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_editing_272_fid.py --pred-npz-dir "$EN" \
    --tag "kimodo_$proto" --out-json "$MD/kimodo_${proto}__fid.json" > "$LOG/kimodo_rot_fid_${proto}.log" 2>&1
  echo "[metrics-rot] kimodo/$proto done $(date)"
done
echo "[rot] ALL DONE $(date)"
touch "$TU/kimodo/_ROT_DONE"
