#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SRC=${SRC:-outputs/evaluation/motionhub_hml3d263_rewrite_0605/motionlab}
SMPL_DIR=${SMPL_DIR:-outputs/evaluation/motionhub_smpl135_rewrite_0605/motionlab}
MC_DIR=${MC_DIR:-outputs/evaluation/motionhub_motionclip135_rewrite_0605/motionlab}
LOGDIR=${LOGDIR:-outputs/evaluation/motionhub_motionclip135_rewrite_0605/logs_motionlab}
NUM_SHARDS=${NUM_SHARDS:-16}
DEVICE=${DEVICE:-cpu}
mkdir -p "${SMPL_DIR}" "${MC_DIR}" "${LOGDIR}"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "${SRC}" \
    --out-dir "${SMPL_DIR}" \
    --model-dir ref_repo/MDM/body_models \
    --source-fps 20 \
    --target-fps 30 \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard}" \
    --device "${DEVICE}" \
    --batch-size 512 \
    --floor-align \
    --orientation-mode parent_frame \
    --parent-ref-weight 0.25 \
    --refine-iters 0 \
    --skip-existing \
    > "${LOGDIR}/ik_s${shard}.log" 2>&1 &
done
wait

python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
  --anno-file data/annotation/test_motionhub_t2m.json \
  --src-dir "${SMPL_DIR}" \
  --out-dir "${MC_DIR}" \
  --data-dir data/motionhub \
  --align-to-gt-root \
  --align-root-mode yaw \
  --key-fallback \
  --overwrite \
  --workers 16 \
  > "${LOGDIR}/remap.log" 2>&1

CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file data/annotation/test_motionhub_t2m.json \
  --data_dir data/motionhub \
  --pred_dir "${MC_DIR}" \
  --chunk_size 64 \
  --out_json "${LOGDIR}/motionlab_motionhub_motionclip_orig_c64.json" \
  --n_repeats 20 \
  --seed 42 \
  > "${LOGDIR}/eval.log" 2>&1

python3 - <<PY | tee "${LOGDIR}/summary.txt"
import json
from pathlib import Path
out = Path("${LOGDIR}") / "motionlab_motionhub_motionclip_orig_c64.json"
d = json.load(open(out))
print(
    "samples", d.get("samples"),
    "R1", f"{d['r_precision_pred_top1_mean']:.4f}",
    "R3", f"{d['r_precision_pred_top3_mean']:.4f}",
    "FID", f"{d['fid_mean']:.4f}",
    "MM", f"{d['mm_dist_pred_mean']:.4f}",
    "Div", f"{d['diversity_pred_mean']:.4f}",
)
vals = []
for p in Path("${SMPL_DIR}").glob("_retarget_summary_s*_of_*.json"):
    vals.append(json.load(open(p)).get("mean_mpjpe_mm"))
vals = [v for v in vals if v is not None]
if vals:
    print("mean_shard_mpjpe_mm", f"{sum(vals)/len(vals):.2f}")
print("smpl_files", len(list(Path("${SMPL_DIR}").glob("*.npz"))))
print("motionclip_files", len(list(Path("${MC_DIR}").glob("*.npy"))))
PY
