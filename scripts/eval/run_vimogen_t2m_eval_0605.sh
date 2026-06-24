#!/bin/bash
# End-to-end ViMoGen T2M inference + MotionCLIP evaluation.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [[ ! -d "$ROOT" ]]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$ROOT"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
PY=${PY:-python3}

DATASET=${DATASET:-h3d}          # h3d | mh
MAX_SAMPLES=${MAX_SAMPLES:-64}   # 0 for full split
TAG=${TAG:-smoke}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_IDX=${SHARD_IDX:-0}
NPROC=${NPROC:-1}
TEST_BS=${TEST_BS:-4}
STEPS=${STEPS:-50}
CFG=${CFG:-5.0}
DENOISING_STRENGTH=${DENOISING_STRENGTH:-0.7}
DTYPE=${DTYPE:-fp16}
SKIP_EVAL=${SKIP_EVAL:-0}
EVAL_CHUNK_SIZE=${EVAL_CHUNK_SIZE:-256}
APPEND_DURATION_TO_PROMPT=${APPEND_DURATION_TO_PROMPT:-0}
CAPTION_STYLE=${CAPTION_STYLE:-first}
CAPTION_OVERRIDE_JSON=${CAPTION_OVERRIDE_JSON:-}
VIMOGEN_SRC_FPS=${VIMOGEN_SRC_FPS:-20}
VIMOGEN_DST_FPS=${VIMOGEN_DST_FPS:-20}
VIMOGEN_TEXT_BATCH_SIZE=${VIMOGEN_TEXT_BATCH_SIZE:-4}
VIMOGEN_TEXT_OVERWRITE=${VIMOGEN_TEXT_OVERWRITE:-1}
EVAL_REWRITTEN_CAPTIONS=${EVAL_REWRITTEN_CAPTIONS:-0}
VIMOGEN_ROOT=${VIMOGEN_ROOT:-ref_repo/ViMoGen}
RUN_TAG="${DATASET}_${TAG}"
if [[ "$NUM_SHARDS" -gt 1 ]]; then
  RUN_TAG="${RUN_TAG}_s${SHARD_IDX}of${NUM_SHARDS}"
fi
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/vimogen_t2m_0605/${RUN_TAG}}
LOGDIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOGDIR"

if [[ "$DATASET" == "h3d" ]]; then
  ANNO=data/annotation/test_hml3d.json
elif [[ "$DATASET" == "mh" ]]; then
  ANNO=data/annotation/test_motionhub_t2m.json
elif [[ "$DATASET" == "mbench" ]]; then
  ANNO=data/annotation/mbench_450_hml263_prompts.json
  CAPTION_OVERRIDE_JSON=${CAPTION_OVERRIDE_JSON:-data/annotation/mbench_450_hml263_captions.json}
else
  echo "Unknown DATASET=$DATASET" >&2
  exit 2
fi
ANNO=${ANNO_OVERRIDE:-$ANNO}

EVAL_DIR="$VIMOGEN_ROOT/data/eval/${RUN_TAG}"
JSON="$EVAL_DIR/vimogen_${DATASET}.json"
CAPMAP="$EVAL_DIR/vimogen_${DATASET}_captions.json"
EMBED_DIR="$EVAL_DIR/text_embeddings"
CONFIG="$EVAL_DIR/t2m_infer_${DATASET}.yaml"
MOTIONCLIP_DIR="$OUT_ROOT/motionclip135"
METRIC_JSON="$OUT_ROOT/metrics_motionclip.json"
VIMOGEN_VIS="$OUT_ROOT/vimogen_exp/test_visualization/${RUN_TAG}"

VIMOGEN_ROOT_ABS=$(realpath "$VIMOGEN_ROOT")
OUT_ROOT_ABS=$(realpath "$OUT_ROOT")
LOGDIR_ABS=$(realpath "$LOGDIR")

BUILD_JSON_ARGS=(
  --anno-file "$ANNO"
  --data-dir data/motionhub
  --out-json "$JSON"
  --caption-map-json "$CAPMAP"
  --embedding-dir "$EMBED_DIR"
  --max-samples "$MAX_SAMPLES"
  --num-shards "$NUM_SHARDS"
  --shard-idx "$SHARD_IDX"
  --caption-style "$CAPTION_STYLE"
)
if [[ -n "$CAPTION_OVERRIDE_JSON" ]]; then
  BUILD_JSON_ARGS+=(--caption-override-json "$CAPTION_OVERRIDE_JSON")
fi
if [[ "$APPEND_DURATION_TO_PROMPT" == "1" ]]; then
  BUILD_JSON_ARGS+=(--append-duration-to-prompt)
fi
"$PY" scripts/eval/build_vimogen_eval_json.py "${BUILD_JSON_ARGS[@]}" \
  > "$LOGDIR/build_json.log" 2>&1

mkdir -p "$EMBED_DIR"
JSON_ABS=$(realpath "$JSON")
CAPMAP_ABS=$(realpath "$CAPMAP")
EMBED_DIR_ABS=$(realpath "$EMBED_DIR")
CONFIG_ABS=$(realpath -m "$CONFIG")
MOTIONCLIP_DIR_ABS=$(realpath -m "$MOTIONCLIP_DIR")
VIMOGEN_VIS_ABS=$(realpath -m "$VIMOGEN_VIS")
METRIC_JSON_ABS=$(realpath -m "$METRIC_JSON")

(
  flock 9
  "$PY" - <<'PY' > "$LOGDIR/install_check.log" 2>&1
mods = ["omegaconf", "easydict", "diffusers", "transformers", "safetensors", "ftfy", "natsort", "smplx", "torchgeometry", "mmengine"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
print("missing", missing)
PY
  if grep -q "missing \\[\\]" "$LOGDIR/install_check.log"; then
    :
  else
    "$PY" -m pip install -q omegaconf easydict 'diffusers>=0.31.0' 'transformers>=4.40.0' safetensors ftfy natsort smplx torchgeometry mmengine gdown \
      >> "$LOGDIR/install_check.log" 2>&1
  fi
) 9>/tmp/vimogen_eval_pip_install.lock

mkdir -p "$VIMOGEN_ROOT_ABS/checkpoints"
ln -sfn "$(pwd)/checkpoints/Wan2.1-T2V-1.3B" "$VIMOGEN_ROOT_ABS/checkpoints/Wan2.1-T2V-1.3B"
MODEL_BYTES=0
if [[ -f "$VIMOGEN_ROOT_ABS/checkpoints/model.pt" ]]; then
  MODEL_BYTES=$(stat -c%s "$VIMOGEN_ROOT_ABS/checkpoints/model.pt")
fi
if [[ "$MODEL_BYTES" -lt 4400000000 ]]; then
  gdown --fuzzy 'https://drive.google.com/file/d/10rOvlIwH_vMpHLuvqQTYl7sOMuYyJs_u/view?usp=sharing' \
    -O "$VIMOGEN_ROOT_ABS/checkpoints/model.pt" \
    > "$LOGDIR/download_model.log" 2>&1
fi
ls -lh "$VIMOGEN_ROOT_ABS/checkpoints/model.pt" > "$LOGDIR/model_file.log"

pushd "$VIMOGEN_ROOT_ABS" >/dev/null
TEXT_CUDA_VISIBLE_DEVICES=${ENC_GPU:-${CUDA_VISIBLE_DEVICES:-0}}
TEXT_ENCODING_ARGS=(
  --json_file "$JSON_ABS"
  --text_key prompt
  --save_dir "$EMBED_DIR_ABS"
  --batch_size "$VIMOGEN_TEXT_BATCH_SIZE"
)
if [[ "$VIMOGEN_TEXT_OVERWRITE" == "1" ]]; then
  TEXT_ENCODING_ARGS+=(--overwrite)
fi
VIMOGEN_TEXT_DTYPE=${VIMOGEN_TEXT_DTYPE:-float16} \
CUDA_VISIBLE_DEVICES=${TEXT_CUDA_VISIBLE_DEVICES} \
"$PY" ./models/transformer/wan/text_encoding_batch.py "${TEXT_ENCODING_ARGS[@]}" \
  > "$LOGDIR_ABS/text_encoding.log" 2>&1
popd >/dev/null

"$PY" - <<PY
from pathlib import Path
from omegaconf import OmegaConf
cfg = OmegaConf.load("$VIMOGEN_ROOT/configs/t2m_infer.yaml")
cfg.experiment.result_dir = "$OUT_ROOT_ABS/vimogen_exp"
cfg.experiment.auto_resume = False
cfg.experiment.render_vis = False
cfg.experiment.validation_steps = int("$STEPS")
cfg.experiment.cfg_scale = float("$CFG")
cfg.experiment.denoising_strength = float("$DENOISING_STRENGTH")
cfg.precision.mixed_precision = "$DTYPE"
cfg.precision.text_precision = "$DTYPE"
cfg.dataloader.test_local_batch = int("$TEST_BS")
cfg.dataloader.num_workers = 4
cfg.dataset.test_json_file_list = ["$JSON_ABS"]
cfg.dataset.text_key = "prompt"
Path("$CONFIG_ABS").parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, "$CONFIG_ABS")
PY

pushd "$VIMOGEN_ROOT_ABS" >/dev/null
INFER_ARGS=(
  -m torch.distributed.run
  --nproc_per_node "$NPROC"
  --master_port "${MASTER_PORT:-29615}"
  train_eval_vimogen.py
  --mode eval
  --config "$CONFIG_ABS"
  --mbench_name "$RUN_TAG"
)
"$PY" "${INFER_ARGS[@]}" \
  > "$LOGDIR_ABS/infer.log" 2>&1
popd >/dev/null

if [[ "$DATASET" == "mbench" ]]; then
  MBENCH_OUT="$OUT_ROOT/mbench"
  mkdir -p "$MBENCH_OUT"
  pushd "$VIMOGEN_ROOT_ABS" >/dev/null
  "$PY" scripts/organize_mbench_results.py \
    --input_dir "$VIMOGEN_VIS_ABS" \
    --output_dir "$OUT_ROOT_ABS/mbench/mbench_eval_input" \
    > "$LOGDIR_ABS/organize_mbench.log" 2>&1
  popd >/dev/null
  echo "[mbench] organized shard to $MBENCH_OUT" | tee "$LOGDIR/summary.txt"
  touch "$LOGDIR/_DONE"
  exit 0
fi

"$PY" scripts/eval/convert_vimogen276_to_motionclip135.py \
  --vimogen-root "$VIMOGEN_ROOT_ABS" \
  --input-root "$VIMOGEN_VIS_ABS" \
  --out-dir "$MOTIONCLIP_DIR_ABS" \
  --src-fps "$VIMOGEN_SRC_FPS" \
  --dst-fps "$VIMOGEN_DST_FPS" \
  --coord-conversion mbench \
  --overwrite \
  > "$LOGDIR/convert.log" 2>&1

if [[ "$SKIP_EVAL" != "1" ]]; then
EVAL_ARGS=(
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq
  --anno_file "$ANNO"
  --data_dir data/motionhub
  --pred_dir "$MOTIONCLIP_DIR_ABS"
  --out_json "$METRIC_JSON_ABS"
  --forward_batch_size 64
  --chunk_size "$EVAL_CHUNK_SIZE"
  --n_repeats 20
)
if [[ "$EVAL_REWRITTEN_CAPTIONS" == "1" ]]; then
  EVAL_ARGS+=(--rewritten_caption_file "$CAPMAP_ABS")
fi
"$PY" scripts/eval/eval_with_motionclip_evaluator.py \
  "${EVAL_ARGS[@]}" \
  > "$LOGDIR/eval_motionclip.log" 2>&1

"$PY" - <<PY | tee "$LOGDIR/summary.txt"
import json
d=json.load(open("$METRIC_JSON_ABS"))
print({
  "dataset": "$DATASET",
  "tag": "$TAG",
  "run_tag": "$RUN_TAG",
  "num_shards": int("$NUM_SHARDS"),
  "shard_idx": int("$SHARD_IDX"),
  "samples": d.get("samples"),
  "r1": d.get("r_precision_pred_top1_mean"),
  "r3": d.get("r_precision_pred_top3_mean"),
  "fid": d.get("fid_mean"),
  "mm": d.get("mm_dist_pred_mean"),
  "div": d.get("diversity_pred_mean"),
})
PY
else
  echo "[skip] SKIP_EVAL=1, metric step skipped." | tee "$LOGDIR/summary.txt"
fi
touch "$LOGDIR/_DONE"
