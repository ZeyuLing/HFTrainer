#!/usr/bin/env bash
# Small MotionGPT prompt-path A/B: infer -> HML263 retarget -> MotionCLIP eval.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BASE=${BASE:-outputs/evaluation/motiongpt_prompt_ab_0605}
MAX_SAMPLES=${MAX_SAMPLES:-128}
N_REPEATS=${N_REPEATS:-5}
mkdir -p "${BASE}/logs"

run_mode() {
  local mode="$1"
  local gpu="$2"
  local out263="${BASE}/${mode}/humanml3d263"
  local smpl="${BASE}/${mode}/smpl135"
  local mc135="${BASE}/${mode}/motionclip135"
  local eval_json="${BASE}/${mode}/motionclip_h3d_c64.json"
  mkdir -p "${out263}" "${smpl}" "${mc135}"

  {
    echo "[${mode}] infer $(date)"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/motiongpt_infer_hml3d263.py \
      --anno-file data/annotation/test_hml3d.json \
      --caption-file data/annotation/test_hml3d_rewritten.json \
      --out-dir "${out263}" \
      --max-samples "${MAX_SAMPLES}" \
      --batch-size 8 \
      --prompt-mode "${mode}" \
      --debug-generations \
      --skip-existing

    echo "[${mode}] retarget $(date)"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "${out263}" \
      --out-dir "${smpl}" \
      --model-dir ref_repo/MDM/body_models \
      --source-fps 20 \
      --target-fps 30 \
      --device cuda \
      --batch-size 512 \
      --floor-align \
      --refine-iters 0 \
      --skip-existing

    echo "[${mode}] remap $(date)"
    python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
      --anno-file data/annotation/test_hml3d.json \
      --src-dir "${smpl}" \
      --out-dir "${mc135}" \
      --include-mirrors \
      --key-fallback \
      --overwrite \
      --workers 4

    echo "[${mode}] eval $(date)"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
      --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
      --anno_file data/annotation/test_hml3d.json \
      --data_dir data/motionhub \
      --pred_dir "${mc135}" \
      --rewritten_caption_file data/annotation/test_hml3d_rewritten.json \
      --chunk_size 64 \
      --max_pairs "${MAX_SAMPLES}" \
      --out_json "${eval_json}" \
      --n_repeats "${N_REPEATS}" \
      --seed 42
    echo "[${mode}] done $(date)"
  } > "${BASE}/logs/${mode}.log" 2>&1
}

run_mode official_nolen 0 &
run_mode official_len 1 &
run_mode instruction 2 &
run_mode direct 3 &
wait

BASE_PATH="${BASE}" python3 - <<'PY' | tee "${BASE}/summary.txt"
import json
import os
from pathlib import Path

base = Path(os.environ["BASE_PATH"])
for mode in ("official_nolen", "official_len", "instruction", "direct"):
    p = base / mode / "motionclip_h3d_c64.json"
    if not p.exists():
        print(mode, "missing")
        continue
    d = json.load(open(p))
    print(
        mode,
        "samples", d.get("samples"),
        "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
        "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
        "FID", f"{d.get('fid_mean', float('nan')):.4f}",
        "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
        "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
    )
PY
