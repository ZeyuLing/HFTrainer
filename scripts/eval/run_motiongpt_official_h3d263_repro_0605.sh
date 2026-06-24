#!/usr/bin/env bash
# Reproduce MotionGPT on the official HumanML3D-263 test ids, not the
# MotionHub-style HumanML3D annotation ids.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

NUM_SHARDS=${NUM_SHARDS:-8}
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/humanml3d/motiongpt_official_h3d263_repro_0605}
LOGDIR="${OUT_ROOT}/logs"
TMP_ANNO="${OUT_ROOT}/official_h3d263_test_anno.json"
CAPTION_SELECTION=${CAPTION_SELECTION:-first}
CAPTION_SEED=${CAPTION_SEED:-42}
TMP_CAPTION="${OUT_ROOT}/official_h3d263_${CAPTION_SELECTION}_caption.json"
RECON_ROOT=${RECON_ROOT:-work_dirs/h3d263_eval/h3d263_test_recon_fk}
SRC_H3D272=${SRC_H3D272:-ref_repo/MotionStreamer/MotionStreamer/humanml3d_272}
mkdir -p "${LOGDIR}" "${OUT_ROOT}/pred"

python3 - <<PY
import json
import math
import random
from pathlib import Path
import numpy as np

recon = Path("${RECON_ROOT}")
texts = Path("${SRC_H3D272}") / "texts"
anno = {}
captions = {}
rng = random.Random(int("${CAPTION_SEED}"))
caption_selection = "${CAPTION_SELECTION}"
for sid in [s.strip() for s in (recon / "test.txt").read_text().splitlines() if s.strip()]:
    m_path = recon / "new_joint_vecs" / f"{sid}.npy"
    t_path = texts / f"{sid}.txt"
    if not m_path.exists() or not t_path.exists():
        continue
    length = int(np.load(m_path, mmap_mode="r").shape[0])
    if length < 40 or length >= 200:
        continue
    full_captions = []
    for line in t_path.read_text().splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            f_tag = float(parts[2])
            to_tag = float(parts[3])
        except ValueError:
            continue
        if (math.isnan(f_tag) or f_tag == 0.0) and (math.isnan(to_tag) or to_tag == 0.0):
            caption = parts[0].strip()
            if caption:
                full_captions.append(caption)
    if not full_captions:
        continue
    if caption_selection == "first":
        caption = full_captions[0]
    elif caption_selection == "random":
        caption = rng.choice(full_captions)
    else:
        raise ValueError(f"unsupported CAPTION_SELECTION={caption_selection!r}")
    anno[sid] = {
        "fps": 20,
        "num_frames": length,
        "duration": length / 20.0,
    }
    captions[sid] = caption
Path("${TMP_ANNO}").write_text(json.dumps({"data_list": anno}, indent=2))
Path("${TMP_CAPTION}").write_text(json.dumps(captions, indent=2))
print({
    "jobs": len(captions),
    "anno": "${TMP_ANNO}",
    "caption": "${TMP_CAPTION}",
    "caption_selection": caption_selection,
    "caption_seed": int("${CAPTION_SEED}"),
})
PY

echo "[infer] $(date)" | tee "${LOGDIR}/run.log"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="${i}" python3 scripts/eval/motiongpt_infer_hml3d263.py \
    --anno-file "${TMP_ANNO}" \
    --caption-file "${TMP_CAPTION}" \
    --out-dir "${OUT_ROOT}/pred" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${i}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --gt-fps 20 \
    --model-fps 20 \
    --prompt-mode "${PROMPT_MODE:-official_nolen}" \
    --seed "${SEED:-42}" \
    --skip-existing \
    > "${LOGDIR}/infer_s${i}.log" 2>&1 &
done
wait

echo "[eval] $(date)" | tee -a "${LOGDIR}/run.log"
if [ -z "${EVAL_CAPTION_SELECTION:-}" ]; then
  if [ "${CAPTION_SELECTION}" = "random" ]; then
    EVAL_CAPTION_SELECTION=mapped
  else
    EVAL_CAPTION_SELECTION=first
  fi
fi
EVAL_CAPTION_MAP_ARGS=()
if [ "${EVAL_CAPTION_SELECTION}" = "mapped" ]; then
  EVAL_CAPTION_MAP_ARGS=(--caption_map "${TMP_CAPTION}")
fi
CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python3 scripts/eval/eval_momask_native_h3d263.py \
  --recon_root "${RECON_ROOT}" \
  --src_h3d272 "${SRC_H3D272}" \
  --momask_root ref_repo/Momask/momask-codes \
  --mode pred \
  --pred_dir "${OUT_ROOT}/pred" \
  --num_repeats "${NUM_REPEATS:-20}" \
  --drop_mirrored \
  --caption_selection "${EVAL_CAPTION_SELECTION}" \
  "${EVAL_CAPTION_MAP_ARGS[@]}" \
  --output "${OUT_ROOT}/eval_momask_native_${EVAL_CAPTION_SELECTION}_rep${NUM_REPEATS:-20}.json" \
  > "${LOGDIR}/eval_momask_native_${EVAL_CAPTION_SELECTION}.log" 2>&1

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
p = Path("${OUT_ROOT}/eval_momask_native_${EVAL_CAPTION_SELECTION}_rep${NUM_REPEATS:-20}.json")
d = json.load(open(p))
print(
    "samples", d.get("n_samples"),
    "R1", f"{d['r_precision']['mean'][0]:.4f}",
    "R3", f"{d['r_precision']['mean'][2]:.4f}",
    "FID", f"{d['fid']['mean']:.4f}",
    "MM", f"{d['matching_score']['mean']:.4f}",
    "Div", f"{d['diversity']['mean']:.4f}",
)
PY

touch "${OUT_ROOT}/_DONE"
echo "[done] $(date)" | tee -a "${LOGDIR}/run.log"
