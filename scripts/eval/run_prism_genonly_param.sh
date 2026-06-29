#!/usr/bin/env bash
# Generalized generation-ONLY shard worker (fan-out across Taiji hosts).
# Parameterized by env: CKPT, KAFS_MODE, OUT, CONFIG, TOTAL_SHARDS, SHARD_BASE, NGPU.
# Each invocation runs NGPU local GPUs as global shards [SHARD_BASE..SHARD_BASE+NGPU-1]
# of TOTAL_SHARDS, all writing the SAME $OUT/$KAFS_MODE dir with --skip-existing.
# Repack + MS-272 eval are done separately by a local watcher.
#
#   CKPT=.../checkpoint-epoch_16 KAFS_MODE=depth_driven \
#   OUT=outputs/evaluation/prism_kt_spectral_epoch16_rw/h3d \
#   TOTAL_SHARDS=48 SHARD_BASE=0 bash scripts/eval/run_prism_genonly_param.sh
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"
pick_py() {
  for c in "${PY:-}" python3 python3.10 python3.9 python3.8 python3.11 \
      /usr/local/bin/python3 /usr/bin/python3.10 /usr/bin/python3.9 /usr/bin/python3.8 \
      /opt/conda/bin/python /root/miniconda3/bin/python /opt/miniconda3/bin/python \
      /usr/local/miniconda3/bin/python "$HOME/miniconda3/bin/python"; do
    [ -n "$c" ] || continue
    command -v "$c" >/dev/null 2>&1 || [ -x "$c" ] || continue
    "$c" -c 'import sys; import numpy; import torch; sys.exit(0 if sys.version_info[:2]>=(3,8) else 1)' 2>/dev/null && { echo "$c"; return 0; }
  done
  return 1
}
PY=$(pick_py) || { echo "[error] Python >=3.8 with numpy+torch not found"; exit 2; }
echo "[python] $(command -v "$PY") $("$PY" --version 2>&1)"

ensure_py_dep() {
  local import_name="$1"
  local package_spec="$2"
  "$PY" -c "import ${import_name}" >/dev/null 2>&1 && return 0
  echo "[deps] installing missing ${package_spec}"
  PIP_ROOT_USER_ACTION=ignore "$PY" -m pip install --quiet "$package_spec"
  "$PY" -c "import ${import_name}" >/dev/null 2>&1
}
ensure_py_dep mmengine "mmengine>=0.10"
ensure_py_dep einops "einops>=0.7"
ensure_py_dep smplx "smplx>=0.1.28"

NGPU=${NGPU:-8}
TOTAL_SHARDS=${TOTAL_SHARDS:-}
SHARD_BASE=${SHARD_BASE:-}
CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_16}
KAFS_MODE=${KAFS_MODE:-depth_driven}
KAFS_ALPHA=${KAFS_ALPHA:-}
OUT_SUBDIR=${OUT_SUBDIR:-}
OUT=${OUT:-outputs/evaluation/prism_kt_spectral_epoch16_rw/h3d}
ANNO=${ANNO:-data/annotation/test_hml3d.json}
REWRITTEN=${REWRITTEN:-}
DATA_DIR=${DATA_DIR:-data/motionhub}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-5.0}
SEED=${SEED:-42}
SMOOTH_OUTPUT=${SMOOTH_OUTPUT:-0}
TRANSLATION_DECODE_MODE=${TRANSLATION_DECODE_MODE:-rollout}
LENGTH_POLICY=${LENGTH_POLICY:-pad360_crop}
PAD_TO_FRAMES=${PAD_TO_FRAMES:-360}
SKIP_MOTION_EXISTENCE_CHECK=${SKIP_MOTION_EXISTENCE_CHECK:-0}
MIN_FRAMES=${MIN_FRAMES:-24}
MAX_FRAMES=${MAX_FRAMES:-360}
ID_FILE=${ID_FILE:-}

# Sharding modes:
#   1. Explicit single-host mode: pass TOTAL_SHARDS + SHARD_BASE.
#   2. Taiji multi-host mode: Taiji sets INDEX and NODE_LIST; optional
#      JOB_RANK/JOB_COUNT partitions several independent multi-host jobs.
if [ -z "$TOTAL_SHARDS" ] || [ -z "$SHARD_BASE" ]; then
  if [ -n "${NODE_LIST:-}" ] || [ -n "${INDEX:-}" ] || [ -n "${JOB_COUNT:-}" ]; then
    HOST_RANK=${INDEX:-0}
    if [ -n "${NODE_LIST:-}" ]; then
      MACHINE_NUM=$("$PY" - <<'PY' 2>/dev/null || echo 1
import os
print(len([x for x in os.environ.get("NODE_LIST", "").split(",") if x]))
PY
)
    else
      MACHINE_NUM=${MACHINE_NUM:-1}
    fi
    JOB_RANK=${JOB_RANK:-0}
    JOB_COUNT=${JOB_COUNT:-1}
    TOTAL_SHARDS=${TOTAL_SHARDS:-$((JOB_COUNT * MACHINE_NUM * NGPU))}
    GLOBAL_HOST=$((JOB_RANK * MACHINE_NUM + HOST_RANK))
    SHARD_BASE=${SHARD_BASE:-$((GLOBAL_HOST * NGPU))}
  else
    TOTAL_SHARDS=${TOTAL_SHARDS:-48}
    SHARD_BASE=${SHARD_BASE:-0}
  fi
fi

# custom alpha -> mode=none + --kafs-alpha; subdir defaults to OUT_SUBDIR or mode
alpha_flag=""
if [ -n "$KAFS_ALPHA" ]; then alpha_flag="--kafs-alpha $KAFS_ALPHA"; KAFS_MODE=none; fi
smooth_flag=""
[ "$SMOOTH_OUTPUT" = "1" ] && smooth_flag="--smooth-output"
skip_motion_flag=""
[ "$SKIP_MOTION_EXISTENCE_CHECK" = "1" ] && skip_motion_flag="--skip-motion-existence-check"
id_file_flag=()
[ -n "$ID_FILE" ] && id_file_flag=(--id-file "$ID_FILE")
rewrite_flag=()
[ -n "$REWRITTEN" ] && rewrite_flag=(--rewritten-caption-file "$REWRITTEN")
SUB=${OUT_SUBDIR:-$KAFS_MODE}
mkdir -p "$OUT/$SUB" "$OUT/_logs"
echo "[genonly] $(date) CKPT=$CKPT SUB=$SUB mode=$KAFS_MODE alpha=[$KAFS_ALPHA] seed=$SEED TOTAL_SHARDS=$TOTAL_SHARDS SHARD_BASE=$SHARD_BASE min/max=$MIN_FRAMES/$MAX_FRAMES rewritten=[$REWRITTEN] -> $OUT/$SUB"
echo "[genonly] translation_decode_mode=$TRANSLATION_DECODE_MODE length_policy=$LENGTH_POLICY pad_to_frames=$PAD_TO_FRAMES smooth=$SMOOTH_OUTPUT"

pids=()
for g in $(seq 0 $((NGPU-1))); do
  gidx=$((SHARD_BASE + g))
  [ "$gidx" -ge "$TOTAL_SHARDS" ] && continue
  CUDA_VISIBLE_DEVICES=$g "$PY" scripts/eval/eval_prism_kafs_ablation.py \
    --config "$CONFIG" --checkpoint "$CKPT" --kafs-mode "$KAFS_MODE" $alpha_flag \
    --out-subdir "$SUB" \
    --anno-file "$ANNO" "${rewrite_flag[@]}" \
    --data-dir "$DATA_DIR" --output-dir "$OUT" \
    "${id_file_flag[@]}" \
    --min-frames "$MIN_FRAMES" --max-frames "$MAX_FRAMES" \
	    --num-inference-steps $STEPS --guidance-scale $GUIDANCE \
	    --translation-decode-mode "$TRANSLATION_DECODE_MODE" \
	    --length-policy "$LENGTH_POLICY" --pad-to-frames "$PAD_TO_FRAMES" \
	    --seed "$SEED" \
    --num-shards $TOTAL_SHARDS --shard-idx $gidx --skip-existing \
    $smooth_flag $skip_motion_flag \
    > "$OUT/_logs/${SUB}_g${gidx}of${TOTAL_SHARDS}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
n=$("$PY" -c "import os;d='$OUT/$SUB';print(sum(1 for e in os.scandir(d) if e.name.endswith('.npz')))")
echo "[genonly done] $(date) base=$SHARD_BASE $SUB total now=$n"
