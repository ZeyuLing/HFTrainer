#!/usr/bin/env bash
# Single-mode MotionGPT smoke: infer -> retarget -> MotionCLIP eval.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BASE=${BASE:-outputs/evaluation/motiongpt_one_smoke_0605}
MODE=${MODE:-official_nolen}
ANNO_FILE=${ANNO_FILE:-data/annotation/test_hml3d.json}
CAPTION_FILE=${CAPTION_FILE:-data/annotation/test_hml3d_rewritten.json}
MAX_SAMPLES=${MAX_SAMPLES:-128}
GPU=${GPU:-5}
N_REPEATS=${N_REPEATS:-5}

OUT263="${BASE}/${MODE}/humanml3d263"
SMPL="${BASE}/${MODE}/smpl135"
MC135="${BASE}/${MODE}/motionclip135"
LOGDIR="${BASE}/logs"
mkdir -p "${OUT263}" "${SMPL}" "${MC135}" "${LOGDIR}"

echo "[infer] $(date)"
CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/motiongpt_infer_hml3d263.py \
  --anno-file "${ANNO_FILE}" \
  --caption-file "${CAPTION_FILE}" \
  --out-dir "${OUT263}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size 8 \
  --prompt-mode "${MODE}" \
  --debug-generations \
  --skip-existing \
  > "${LOGDIR}/infer.log" 2>&1

echo "[retarget] $(date)"
CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/hml263_to_smpl_ik.py \
  --in-dir "${OUT263}" \
  --out-dir "${SMPL}" \
  --model-dir ref_repo/MDM/body_models \
  --source-fps 20 \
  --target-fps 30 \
  --device cuda \
  --batch-size 512 \
  --floor-align \
  --refine-iters 0 \
  --skip-existing \
  > "${LOGDIR}/retarget.log" 2>&1

echo "[remap] $(date)"
python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
  --anno-file "${ANNO_FILE}" \
  --src-dir "${SMPL}" \
  --out-dir "${MC135}" \
  --include-mirrors \
  --key-fallback \
  --overwrite \
  --workers 4 \
  > "${LOGDIR}/remap.log" 2>&1

echo "[eval] $(date)"
CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file "${ANNO_FILE}" \
  --data_dir data/motionhub \
  --pred_dir "${MC135}" \
  --rewritten_caption_file data/annotation/test_hml3d_rewritten.json \
  --chunk_size 64 \
  --max_pairs "${MAX_SAMPLES}" \
  --out_json "${BASE}/${MODE}/motionclip_h3d_c64.json" \
  --n_repeats "${N_REPEATS}" \
  --seed 42 \
  > "${LOGDIR}/eval_motionclip.log" 2>&1

METRICS="${BASE}/${MODE}/motionclip_h3d_c64.json" python3 - <<'PY' | tee "${BASE}/summary.txt"
import json
import os

d = json.load(open(os.environ["METRICS"]))
print(
    "samples", d.get("samples"),
    "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
    "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
    "FID", f"{d.get('fid_mean', float('nan')):.4f}",
    "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
    "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
)
PY
