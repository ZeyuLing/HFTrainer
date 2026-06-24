#!/usr/bin/env bash
# MotionMillion (Go-To-Zero) 7B-all T2M generation on the HumanML3D test set.
# Each host runs NGPU local GPUs as shards [SHARD_BASE..SHARD_BASE+NGPU-1] of
# TOTAL_SHARDS, all writing RAW vector_272 .npy (canonical-id) into the same OUT
# dir with skip-existing. Repack(--gt272-dir)+MS-272 eval done by local watcher.
set -uo pipefail

HFT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
[ -d "$HFT" ] || HFT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
MM="$HFT/ref_repo/MotionMillion-Codes"
cd "$MM"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

NGPU=${NGPU:-8}
TOTAL_SHARDS=${TOTAL_SHARDS:-8}
SHARD_BASE=${SHARD_BASE:-0}
MODEL=${MODEL:-7B}
SRC_TRANS=${TRANS:-$MM/checkpoints/pretrained_models/motionmillion_7B_all.pth}
SRC_FSQ="$MM/checkpoints/pretrained_models/fsq_net_6000000.pth"

# By default copy the big checkpoints to node-local /dev/shm ONCE so every shard
# loads from RAM-backed storage (fast on multi-GPU nodes with lots of RAM).
# Set SHM_COPY=0 for single-GPU backfill jobs: those get a small per-job RAM quota
# (~1/8 node) and shm(17G)+torch.load(17G)=34G OOM-kills them, so load straight
# from CephFS instead (taiji CephFS read is fast enough).
SHM_COPY=${SHM_COPY:-1}
if [ "$SHM_COPY" = "1" ]; then
  LOCAL=/dev/shm/mm_gtz
  mkdir -p "$LOCAL"
  TRANS="$LOCAL/$(basename "$SRC_TRANS")"
  FSQ="$LOCAL/$(basename "$SRC_FSQ")"
  for pair in "$SRC_TRANS|$TRANS" "$SRC_FSQ|$FSQ"; do
    s="${pair%%|*}"; d="${pair##*|}"
    if [ ! -s "$d" ] || [ "$(stat -c%s "$s")" != "$(stat -c%s "$d" 2>/dev/null || echo 0)" ]; then
      echo "[mm-gen] caching $(basename "$s") -> $d"; cp -f "$s" "$d"
    fi
  done
  echo "[mm-gen] local ckpts ready: $(ls -lh "$LOCAL" | tr -s ' ')"
else
  TRANS="$SRC_TRANS"; FSQ="$SRC_FSQ"
  echo "[mm-gen] SHM_COPY=0: loading ckpts directly from CephFS"
fi

# PROMPTS_FILE/IDS_FILE/OUT_DIR can override defaults (e.g. for MotionHub).
PROMPTS_FILE="${PROMPTS_FILE:-$MM/run_hml3d/prompts.txt}"
IDS_FILE="${IDS_FILE:-$MM/run_hml3d/ids.txt}"
OUT="${OUT_DIR:-$HFT/outputs/evaluation/motionmillion_gtz/raw272}"
mkdir -p "$OUT" "$HFT/outputs/evaluation/motionmillion_gtz/_logs"
echo "[mm-gen] $(date) MODEL=$MODEL TOTAL_SHARDS=$TOTAL_SHARDS SHARD_BASE=$SHARD_BASE -> $OUT"

pids=()
for g in $(seq 0 $((NGPU-1))); do
  gidx=$((SHARD_BASE + g))
  [ "$gidx" -ge "$TOTAL_SHARDS" ] && continue
  MM_PROMPTS="$PROMPTS_FILE" \
  MM_IDS="$IDS_FILE" \
  MM_OUT="$OUT" \
  MM_PRETRAINED="$MODEL" \
  MM_RESUME_TRANS="$TRANS" \
  MM_RESUME_PTH="$FSQ" \
  MM_SHARD_IDX="$gidx" MM_NUM_SHARDS="$TOTAL_SHARDS" MM_GPU="$g" \
  python3 mm_infer_batch_272.py \
    > "$HFT/outputs/evaluation/motionmillion_gtz/_logs/${JOB_TAG:-h3d}_g${gidx}of${TOTAL_SHARDS}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
n=$(python3 -c "import os;d='$OUT';print(sum(1 for e in os.scandir(d) if e.name.endswith('.npy')))")
echo "[mm-gen done] $(date) raw272 total now=$n"
