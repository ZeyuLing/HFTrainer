#!/bin/bash
# Merge sharded ViMoGen outputs and run MotionCLIP + MotionStreamer evaluators.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:${PYTHONPATH:-}

DATASET=${DATASET:-h3d}       # h3d | mh
TAG=${TAG:-full0605}
NUM_SHARDS=${NUM_SHARDS:-8}
GPU=${GPU:-0}
CHUNK_SIZE=${CHUNK_SIZE:-256}
N_REPEATS=${N_REPEATS:-20}
ROOT=${ROOT:-outputs/evaluation/vimogen_t2m_0605}
VIMOGEN_ROOT=${VIMOGEN_ROOT:-ref_repo/ViMoGen}
M135_SRC_FPS=${M135_SRC_FPS:-20}

if [[ "$DATASET" == "h3d" ]]; then
  ANNO=data/annotation/test_hml3d.json
  MS_DATA=ref_repo/MotionStreamer/MotionStreamer/humanml3d_272
elif [[ "$DATASET" == "mh" ]]; then
  ANNO=data/annotation/test_motionhub_t2m.json
  MS_DATA=""
else
  echo "Unknown DATASET=$DATASET" >&2
  exit 2
fi

MERGED="$ROOT/${DATASET}_${TAG}_merged"
M135="$MERGED/motionclip135"
MS_NPZ="$MERGED/ms272_npz"
PRED272="$MERGED/pred272"
PRED272_MS="$MERGED/pred272_motionstreamer_ids"
CAPMAP="$MERGED/captions.json"
LOGDIR="$MERGED/logs"
mkdir -p "$M135" "$MS_NPZ" "$PRED272" "$PRED272_MS" "$LOGDIR"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  test -f "$ROOT/${DATASET}_${TAG}_s${i}of${NUM_SHARDS}/logs/_DONE"
done

find "$M135" -maxdepth 1 -type l -delete
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  src="$ROOT/${DATASET}_${TAG}_s${i}of${NUM_SHARDS}/motionclip135"
  find "$src" -maxdepth 1 -name '*.npy' -print0 \
    | while IFS= read -r -d '' f; do
        ln -sf "$(realpath "$f")" "$M135/$(basename "$f")"
      done
done

python3 - <<PY
import json
from pathlib import Path
dataset = "$DATASET"
tag = "$TAG"
n = int("$NUM_SHARDS")
root = Path("$VIMOGEN_ROOT") / "data" / "eval"
out = {}
for i in range(n):
    p = root / f"{dataset}_{tag}_s{i}of{n}" / f"vimogen_{dataset}_captions.json"
    out.update(json.load(open(p)))
dst = Path("$CAPMAP")
dst.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(dst, "w"), indent=2)
print({"caption_map": str(dst), "captions": len(out)})
PY

CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file "$ANNO" \
  --data_dir data/motionhub \
  --pred_dir "$M135" \
  --rewritten_caption_file "$CAPMAP" \
  --out_json "$MERGED/metrics_motionclip.json" \
  --forward_batch_size 64 \
  --chunk_size "$CHUNK_SIZE" \
  --n_repeats "$N_REPEATS" \
  > "$LOGDIR/eval_motionclip.log" 2>&1

if [[ "$DATASET" == "h3d" ]]; then
  python3 scripts/eval/joints_to_272_npz.py \
    --in-dir "$M135" \
    --out "$MS_NPZ" \
    --input-kind m135 \
    --src-fps "$M135_SRC_FPS" \
    --ext .npy \
    --workers 32 \
    > "$LOGDIR/convert_m135_to_272.log" 2>&1

  python3 scripts/eval/extract_motion272_npz.py \
    --in-dir "$MS_NPZ" \
    --out-dir "$PRED272" \
    > "$LOGDIR/extract_pred272.log" 2>&1

  python3 - <<PY > "$LOGDIR/remap_pred272_motionstreamer_ids.log" 2>&1
import json
import os
from pathlib import Path

anno = json.load(open("$ANNO"))["data_list"]
src = Path("$PRED272")
dst = Path("$PRED272_MS")
dst.mkdir(parents=True, exist_ok=True)
for old in dst.glob("*.npy"):
    old.unlink()

ok = skip = collision = 0
seen = set()
for name, entry in anno.items():
    pred = src / f"{name}.npy"
    if not pred.exists():
        skip += 1
        continue
    smplx = entry.get("smplx_path")
    if not smplx:
        skip += 1
        continue
    ms_id = Path(smplx).stem
    if ms_id in seen:
        collision += 1
        continue
    seen.add(ms_id)
    os.symlink(os.path.realpath(pred), dst / f"{ms_id}.npy")
    ok += 1
print({"ok": ok, "skip": skip, "collision": collision, "out": str(dst)}, flush=True)
PY

  CUDA_VISIBLE_DEVICES="$GPU" python3 ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py \
    --evaluator_ckpt ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt \
    --data_root "$MS_DATA" \
    --pred_dir "$PRED272_MS" \
    --n_repeats "$N_REPEATS" \
    --batch_size 32 \
    --out_json "$MERGED/metrics_motionstreamer272.json" \
    > "$LOGDIR/eval_motionstreamer272.log" 2>&1
else
  echo "[skip] MotionStreamer-272 evaluator is HumanML3D-native; skipping DATASET=$DATASET." \
    | tee "$LOGDIR/eval_motionstreamer272.log"
fi

python3 - <<PY | tee "$LOGDIR/summary.txt"
import json
from pathlib import Path
out = {"dataset": "$DATASET", "tag": "$TAG"}
for key, path in {
    "motionclip": Path("$MERGED/metrics_motionclip.json"),
    "motionstreamer272": Path("$MERGED/metrics_motionstreamer272.json"),
}.items():
    if path.exists():
        d = json.load(open(path))
        out[key] = {
            "samples": d.get("samples", d.get("n_samples_used")),
            "r1": d.get("r_precision_pred_top1_mean") or (d.get("r_precision_pred") or [None])[0],
            "r3": d.get("r_precision_pred_top3_mean") or (d.get("r_precision_pred") or [None, None, None])[2],
            "fid": d.get("fid_mean", d.get("fid")),
            "mm": d.get("mm_dist_pred_mean", d.get("matching_score_pred")),
            "div": d.get("diversity_pred_mean", d.get("diversity_pred")),
        }
print(json.dumps(out, indent=2))
PY
