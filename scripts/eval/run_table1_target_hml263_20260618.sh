#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

METHOD="${METHOD:?set METHOD=mld|motionlab|ik_<method>}"
NUM_GPUS="${NUM_GPUS:-${TJ_GPU_NUM:-8}}"
TOTAL_SHARDS="${TOTAL_SHARDS:-8}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
LOCAL_SHARDS="${LOCAL_SHARDS:-$NUM_GPUS}"

ANNO="data/annotation/test_hml3d_official272_gtlen.json"
DATA_DIR="."
RUN_ROOT="outputs/evaluation/t2m/humanml3d_official_test/_runs/table1_remaining_hml263_20260618"
LOG_DIR="$RUN_ROOT/logs"
PREP_DIR="$RUN_ROOT/prep"
AUDIT_DIR="$RUN_ROOT/audits"
mkdir -p "$LOG_DIR" "$PREP_DIR" "$AUDIT_DIR"

HML_BASE="outputs/evaluation/t2m/humanml3d_official_test/hml263"
M135_BASE="outputs/evaluation/t2m/humanml3d_official_test/motion135"

caption_file="$PREP_DIR/official_first_caption.json"
full_anno="$PREP_DIR/official_4042_anno.json"

declare -A HML_DIR=(
  [momask]="$HML_BASE/momask_official/predictions/hml263"
  [t2mgpt]="$HML_BASE/t2mgpt_official/predictions/hml263"
  [mdm]="$HML_BASE/mdm_official/predictions/hml263"
  [mld]="$HML_BASE/mld_official/predictions/hml263"
  [flowmdm]="$HML_BASE/flowmdm_official/predictions/hml263"
  [motionlab]="$HML_BASE/motionlab_official/predictions/hml263"
  [motiongpt3]="$HML_BASE/motiongpt3_official/predictions/hml263"
)

declare -A M135_DIR=(
  [momask]="$M135_BASE/momask_official/predictions/motion135"
  [t2mgpt]="$M135_BASE/t2mgpt_official/predictions/motion135"
  [mdm]="$M135_BASE/mdm_official/predictions/motion135"
  [mld]="$M135_BASE/mld_official/predictions/motion135"
  [flowmdm]="$M135_BASE/flowmdm_official/predictions/motion135"
  [motionlab]="$M135_BASE/motionlab_official/predictions/motion135"
  [motiongpt3]="$M135_BASE/motiongpt3_official/predictions/motion135"
)

echo "[target-start] method=$METHOD total=$TOTAL_SHARDS offset=$SHARD_OFFSET local=$LOCAL_SHARDS root=$ROOT $(date -Is)" | tee -a "$LOG_DIR/target_${METHOD}.log"

ensure_python_deps() {
  local missing="$PREP_DIR/missing_python_deps_target_${METHOD}.txt"
  python3 - <<'PY' > "$missing"
mods = {
    "einops": "einops",
    "omegaconf": "omegaconf>=2.3",
    "hydra": "hydra-core>=1.3",
    "smplx": "smplx>=0.1.28",
    "chumpy": "chumpy>=0.70",
    "sentence_transformers": "sentence-transformers",
    "rotary_embedding_torch": "rotary-embedding-torch",
    "roma": "roma",
}
for mod, pkg in mods.items():
    try:
        __import__(mod)
    except Exception:
        print(pkg)
try:
    __import__("clip")
except Exception:
    print("git+https://github.com/openai/CLIP.git")
PY
  if [[ -s "$missing" ]]; then
    echo "[deps] installing $(tr '\n' ' ' < "$missing")" | tee -a "$LOG_DIR/target_${METHOD}.log"
    python3 -m pip install -q \
      -i https://mirrors.tencent.com/pypi/simple \
      --trusted-host mirrors.tencent.com \
      -r "$missing"
  else
    echo "[deps] python deps ok" | tee -a "$LOG_DIR/target_${METHOD}.log"
  fi
}

run_mld() {
  local shards="$1" shard="$2"
  python3 scripts/eval/mld_infer_hml3d263.py \
    --anno_file "$AUDIT_DIR/mld_missing_anno.json" \
    --anno_data_dir "$DATA_DIR" \
    --caption_protocol original \
    --out_dir "${HML_DIR[mld]}" \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --batch_size 16 \
    --skip_existing
}

run_motionlab() {
  local shards="$1" shard="$2"
  python3 scripts/eval/motionlab_infer_hml3d263.py \
    --anno-file "$full_anno" \
    --caption-file "$caption_file" \
    --data-dir "$DATA_DIR" \
    --out-dir "${HML_DIR[motionlab]}" \
    --source-id-file "$AUDIT_DIR/motionlab_missing.txt" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --device cuda \
    --min-length 1 \
    --max-length 196 \
    --stage demo \
    --skip-existing
}

run_ik_method() {
  local method="$1" shards="$2" shard="$3"
  python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "${HML_DIR[$method]}" \
    --out-dir "${M135_DIR[$method]}" \
    --ids "$AUDIT_DIR/${method}_m135_missing.txt" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --source-fps 20 \
    --target-fps 30 \
    --device cuda \
    --batch-size 1 \
    --floor-align \
    --refine-iters 80 \
    --refine-lr 0.02 \
    --skip-existing
}

run_one() {
  local shard="$1"
  case "$METHOD" in
    mld) run_mld "$TOTAL_SHARDS" "$shard" ;;
    motionlab) run_motionlab "$TOTAL_SHARDS" "$shard" ;;
    ik_*) run_ik_method "${METHOD#ik_}" "$TOTAL_SHARDS" "$shard" ;;
    *) echo "unknown METHOD=$METHOD" >&2; return 2 ;;
  esac
}

ensure_python_deps

pids=()
for ((local_idx=0; local_idx<LOCAL_SHARDS; local_idx++)); do
  shard=$((SHARD_OFFSET + local_idx))
  if (( shard >= TOTAL_SHARDS )); then
    continue
  fi
  gpu=$((local_idx % NUM_GPUS))
  log="$LOG_DIR/target_${METHOD}_g${shard}of${TOTAL_SHARDS}.log"
  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" run_one "$shard" > "$log" 2>&1
    code=$?
    echo "exit_code=$code finished_at=$(date -Is)" > "${log}.status"
    exit "$code"
  ) &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[target-fail] method=$METHOD offset=$SHARD_OFFSET $(date -Is)" | tee -a "$LOG_DIR/target_${METHOD}.log"
  exit 1
fi

echo "[target-done] method=$METHOD offset=$SHARD_OFFSET $(date -Is)" | tee -a "$LOG_DIR/target_${METHOD}.log"
