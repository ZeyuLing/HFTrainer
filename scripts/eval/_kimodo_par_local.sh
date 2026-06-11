#!/usr/bin/env bash
# Local CPU-parallel KIMODO SOMA->SMPL retarget (library retargeter) + build
# (NO splice -> coherent metrics) + 272 metrics, for all 3 temporal protocols.
# Safe to re-run: --skip-existing resumes partial smplx dirs.
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
TU=output/evaluation/temporal_unified
LOG=$TU/logs; mkdir -p "$LOG"
NSH=${NSH:-5}

pids=()
for proto in pre20 post20 mid60; do
  SM=$TU/kimodo/$proto/smplx; mkdir -p "$SM"
  RAWNPZ=$TU/kimodo/$proto/raw/E2_$proto/npz
  for s in $(seq 0 $((NSH-1))); do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/kimodo_soma_to_smpl_byid.py \
      --data-file data/eval/m2m_v2/eval_h3d_editing.json --max-samples 4012 \
      --raw-npz-dir "$RAWNPZ" --out-dir "$SM" --model-dir ref_repo/MDM/body_models \
      --device cuda --refine-iters 5 --skip-existing --num-shards "$NSH" --shard-index "$s" \
      > "$LOG/kimodo_par_${proto}_s${s}.log" 2>&1 &
    pids+=("$!")
  done
done
echo "[par] launched ${#pids[@]} retarget workers $(date)"
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[par] retarget done $(date)"
for proto in pre20 post20 mid60; do
  echo "  $proto smplx=$(ls $TU/kimodo/$proto/smplx/*.npz 2>/dev/null | wc -l)"
done

for proto in pre20 post20 mid60; do
  SM=$TU/kimodo/$proto/smplx; EN=$TU/kimodo/$proto/eval_npz; MD=$TU/_metrics
  mkdir -p "$EN" "$MD"
  python3 scripts/eval/build_baseline_eval_npz.py --ik-dir "$SM" --protocol "$proto" \
    --out-dir "$EN" > "$LOG/kimodo_build_${proto}.log" 2>&1
  python3 scripts/eval/paper_npz_ric_mpjpe.py --npz-dir "$EN" --tag "kimodo_$proto" \
    --out-json "$MD/kimodo_${proto}__ric.json" > "$LOG/kimodo_ric_${proto}.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_editing_272_fid.py --pred-npz-dir "$EN" \
    --tag "kimodo_$proto" --out-json "$MD/kimodo_${proto}__fid.json" > "$LOG/kimodo_fid_${proto}.log" 2>&1
  echo "[metrics] kimodo/$proto done $(date)"
done
echo "[par] ALL DONE $(date)"
touch "$TU/kimodo/_LOCAL_DONE"
