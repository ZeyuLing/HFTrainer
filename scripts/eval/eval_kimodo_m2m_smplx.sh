#!/usr/bin/env bash
# Evaluate ONE KIMODO-SMPLX M2M task (per-sid preds from gen_kimodo_m2m_smplx.py):
#   (1) distribution metrics: motion_135 (row-major) -> motion135_to_272 ->
#       eval_motionstreamer_272.py vs NATIVE GT-272 on the FULL HumanML3D test
#       split (same protocol as output/evaluation/mib_ms272_ikfix/_full). Because
#       KIMODO-SMPLX outputs SMPL params natively, it is encoded DIRECTLY to 272
#       (no 263->IK lift, no 47.6 conversion floor) -- exactly the "ours" path.
#   (2) position metrics: UMO 272-ric MPJPE / [P]-MPJPE / jitter / foot.
#
# Usage: bash scripts/eval/eval_kimodo_m2m_smplx.sh <TASK> [BASE_DIR] [GPU]
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

TASK="${1:?usage: eval_kimodo_m2m_smplx.sh <TASK> [BASE_DIR] [GPU]}"
BASE="${2:-outputs/evaluation/kimodo_smplx_m2m_20260623/${TASK}}"
GPU="${3:-0}"
PRED="$BASE/preds_npz"
MET="$BASE/metrics"; mkdir -p "$MET"
LOG="$BASE/evallogs"; mkdir -p "$LOG"

echo "[eval] task=$TASK pred=$PRED npz=$(ls "$PRED"/*.npz 2>/dev/null | wc -l)"

# Warm the local /dev/shm cache of the 272 evaluator ckpt + GT (best effort).
bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true

# (1) distribution metrics (1 GPU). Direct motion_135 -> 272, full test split.
echo "[eval] distribution (MotionStreamer-272) ..."
CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/eval/eval_motionstreamer_272.py \
  --pred-dir "$PRED" --tag "kimodo_smplx_${TASK}" --also-refk \
  --out-json "$MET/ms272.json" > "$LOG/ms272.log" 2>&1
echo "[eval] dist -> $MET/ms272.json"

# (2) position metrics (UMO 272-ric).
echo "[eval] position (UMO 272-ric) ..."
CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/eval/eval_kimodo_m2m_smplx_positions.py \
  --pred-dir "$PRED" --out "$MET/positions.json" > "$LOG/positions.log" 2>&1
echo "[eval] pos  -> $MET/positions.json"

# compact summary
python3 - "$MET/ms272.json" "$MET/positions.json" "$TASK" <<'PY'
import json, sys
ms, pos, task = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(ms)); p = json.load(open(pos)); pr = d.get("pred", {})
rp = pr.get("r_precision", [float("nan")]*3)
print(f"\n=== KIMODO-SMPLX {task} SUMMARY ===")
print(f" dist n={d.get('ids_with_required_files')}  "
      f"FID={pr.get('fid_vs_gt_native',float('nan')):.2f}  "
      f"R@1/2/3={rp[0]:.3f}/{rp[1]:.3f}/{rp[2]:.3f}  "
      f"MM={pr.get('matching_score',float('nan')):.2f}  "
      f"Div={pr.get('diversity',float('nan')):.2f}")
print(f" pos  n={p['n_samples']}  MPJPE={p['mpjpe_full_cm']:.2f}cm  "
      f"[P]-MPJPE={p['p_mpjpe_cm']:.3f}cm  MPJPE(gen)={p['mpjpe_gen_cm']:.2f}cm  "
      f"jitter={p['jitter']:.1f}  foot={p['foot_skating']:.3f}")
PY
echo "EVAL_DONE task=$TASK"
