#!/usr/bin/env bash
# Rerun the HumanML3D official-test MotionLab / MotionGPT3 predictions into
# versioned directories, then retarget HML263 outputs to SMPL motion_135.
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
export HFTRAINER_SKIP_AUTOREGISTER=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

METHOD="${METHOD:?set METHOD=motionlab|motiongpt3|all}"
NUM_GPUS="${NUM_GPUS:-${TJ_GPU_NUM:-8}}"
TOTAL_SHARDS="${TOTAL_SHARDS:-8}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
LOCAL_SHARDS="${LOCAL_SHARDS:-$NUM_GPUS}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  NUM_GPUS=1
fi

ANNO="data/annotation/test_hml3d_official272_gtlen.json"
DATA_DIR="."
RUN_ROOT="outputs/evaluation/t2m/humanml3d_official_test/_runs/rerun_motionlab_motiongpt3_20260619"
LOG_DIR="$RUN_ROOT/logs"
PREP_DIR="$RUN_ROOT/prep"
mkdir -p "$LOG_DIR" "$PREP_DIR"

caption_file="$PREP_DIR/official_first_caption.json"
full_anno="$PREP_DIR/official_4042_anno.json"
ids_file="$PREP_DIR/official_4042_ids.txt"

HML_BASE="outputs/evaluation/t2m/humanml3d_official_test/hml263"
M135_BASE="outputs/evaluation/t2m/humanml3d_official_test/motion135"

declare -A HML_DIR=(
  [motionlab]="$HML_BASE/motionlab_official_demo201_rerun_20260619/predictions/hml263"
  [motiongpt3]="$HML_BASE/motiongpt3_official_gs3_rerun_20260619/predictions/hml263"
)

declare -A M135_DIR=(
  [motionlab]="$M135_BASE/motionlab_official_demo201_rerun_20260619/predictions/motion135"
  [motiongpt3]="$M135_BASE/motiongpt3_official_gs3_rerun_20260619/predictions/motion135"
)

ensure_python_deps() {
  local missing="$PREP_DIR/missing_python_deps_${METHOD}.txt"
  python3 - <<'PY' > "$missing"
mods = {
    "einops": "einops",
    "ftfy": "ftfy",
    "regex": "regex",
    "tqdm": "tqdm",
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
    echo "[deps] installing $(tr '\n' ' ' < "$missing")" | tee -a "$LOG_DIR/run_${METHOD}.log"
    python3 -m pip install -q \
      -i https://mirrors.tencent.com/pypi/simple \
      --trusted-host mirrors.tencent.com \
      -r "$missing"
  else
    echo "[deps] python deps ok" | tee -a "$LOG_DIR/run_${METHOD}.log"
  fi
}

prepare_official_inputs() {
  python3 - <<'PY' "$ANNO" "$full_anno" "$caption_file" "$ids_file" "$DATA_DIR"
import json
import sys
from pathlib import Path

anno_path = Path(sys.argv[1])
out_anno = Path(sys.argv[2])
out_caption = Path(sys.argv[3])
out_ids = Path(sys.argv[4])
data_dir = Path(sys.argv[5])

raw = json.loads(anno_path.read_text())
data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
if not isinstance(data, dict):
    data = {
        str(item.get("source_id") or item.get("motion_id") or idx): item
        for idx, item in enumerate(data)
    }

def first_caption(entry):
    rel = entry.get("hierarchical_caption_path")
    if not rel:
        return None
    path = data_dir / rel
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None
    stack = [obj]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
        if isinstance(cur, dict):
            for key in ("caption", "text", "sentence", "raw_caption"):
                val = cur.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            for val in cur.values():
                if isinstance(val, (dict, list, str)):
                    stack.append(val)
        elif isinstance(cur, list):
            stack[:0] = cur
    return None

captions = {}
kept = {}
for sid, entry in data.items():
    sid = str(sid)
    captions[sid] = first_caption(entry) or "a person is moving"
    kept[sid] = entry

out_anno.parent.mkdir(parents=True, exist_ok=True)
out_anno.write_text(json.dumps({"meta": raw.get("meta", {}), "data_list": kept}, indent=2))
out_caption.write_text(json.dumps(captions, indent=2))
out_ids.write_text("\n".join(kept.keys()) + "\n")
print(f"[prep] ids={len(kept)} anno={out_anno} captions={out_caption}", flush=True)
PY
}

write_run_meta() {
  local method="$1"
  local method_root
  method_root="$(dirname "$(dirname "${M135_DIR[$method]}")")"
  mkdir -p "$method_root/logs" "$method_root/metrics" \
    "$method_root/conversions/hml263_to_motion135"
  cat > "$method_root/command.txt" <<EOF
ROOT=$ROOT METHOD=$method NUM_GPUS=$NUM_GPUS TOTAL_SHARDS=$TOTAL_SHARDS bash scripts/eval/run_t2m_rerun_motionlab_motiongpt3_20260619.sh
EOF
  python3 - <<'PY' "$method_root/run_config.json" "$method" "${HML_DIR[$method]}" "${M135_DIR[$method]}" "$ANNO" "$RUN_ROOT"
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "method": sys.argv[2],
    "dataset": "humanml3d_official_test",
    "task": "t2m",
    "native_representation": "hml263",
    "target_representation": "motion135_smpl",
    "hml263_dir": sys.argv[3],
    "motion135_dir": sys.argv[4],
    "annotation": sys.argv[5],
    "runner": sys.argv[6],
    "created_by": "scripts/eval/run_t2m_rerun_motionlab_motiongpt3_20260619.sh",
}, indent=2))
PY
}

run_shards() {
  local phase="$1"
  local shards="$2"
  shift 2
  echo "[phase-start] $phase shards=$shards offset=$SHARD_OFFSET local=$LOCAL_SHARDS $(date -Is)" | tee -a "$LOG_DIR/run_${METHOD}.log"
  local pids=()
  local local_idx shard gpu log
  for ((local_idx=0; local_idx<LOCAL_SHARDS; local_idx++)); do
    shard=$((SHARD_OFFSET + local_idx))
    if (( shard >= TOTAL_SHARDS )); then
      continue
    fi
    gpu=$((local_idx % NUM_GPUS))
    log="$LOG_DIR/${phase}_s${shard}_of_${TOTAL_SHARDS}.log"
    (
      set +e
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$@" "$TOTAL_SHARDS" "$shard" > "$log" 2>&1
      code=$?
      echo "exit_code=$code finished_at=$(date -Is)" > "${log}.status"
      exit "$code"
    ) &
    pids+=("$!")
  done

  local fail=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[phase-fail] $phase $(date -Is)" | tee -a "$LOG_DIR/run_${METHOD}.log"
    return 1
  fi
  echo "[phase-done] $phase $(date -Is)" | tee -a "$LOG_DIR/run_${METHOD}.log"
}

run_motionlab() {
  local shards="$1" shard="$2"
  python3 scripts/eval/motionlab_infer_hml3d263.py \
    --anno-file "$full_anno" \
    --caption-file "$caption_file" \
    --data-dir "$DATA_DIR" \
    --out-dir "${HML_DIR[motionlab]}" \
    --source-id-file "$ids_file" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --device cuda \
    --min-length 1 \
    --max-length 196 \
    --stage demo \
    --skip-existing
}

run_motiongpt3() {
  local shards="$1" shard="$2"
  python3 scripts/eval/motiongpt3_infer_hml3d263.py \
    --anno_file "$full_anno" \
    --anno_data_dir "$DATA_DIR" \
    --caption_protocol original \
    --out_dir "${HML_DIR[motiongpt3]}" \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --batch_size 8 \
    --guidance_scale 3.0 \
    --skip_existing
}

run_ik_method() {
  local method="$1"
  local shards="$2" shard="$3"
  python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "${HML_DIR[$method]}" \
    --out-dir "${M135_DIR[$method]}" \
    --ids "$ids_file" \
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

run_method() {
  local method="$1"
  mkdir -p "${HML_DIR[$method]}" "${M135_DIR[$method]}"
  write_run_meta "$method"
  case "$method" in
    motionlab) run_shards "infer_motionlab" "$TOTAL_SHARDS" run_motionlab ;;
    motiongpt3) run_shards "infer_motiongpt3" "$TOTAL_SHARDS" run_motiongpt3 ;;
    *) echo "unknown method=$method" >&2; return 2 ;;
  esac
  run_shards "ik_${method}" "$TOTAL_SHARDS" run_ik_method "$method"
}

echo "[start] method=$METHOD root=$ROOT num_gpus=$NUM_GPUS total_shards=$TOTAL_SHARDS $(date -Is)" | tee "$LOG_DIR/run_${METHOD}.log"
ensure_python_deps
prepare_official_inputs

case "$METHOD" in
  motionlab) run_method motionlab ;;
  motiongpt3) run_method motiongpt3 ;;
  all)
    run_method motionlab
    run_method motiongpt3
    ;;
  *) echo "unknown METHOD=$METHOD" >&2; exit 2 ;;
esac

echo "[done] method=$METHOD $(date -Is)" | tee -a "$LOG_DIR/run_${METHOD}.log"
