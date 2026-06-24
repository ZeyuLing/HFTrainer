#!/usr/bin/env bash
# Fill missing HumanML3D official-test motion135 samples in the normalized T2M
# result directories. Heavy generation is shardable for Taiji; final outputs are
# written directly into outputs/evaluation/t2m/humanml3d_official_test/motion135.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HFTRAINER_SKIP_AUTOREGISTER=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

pick_python() {
  local candidates=()
  if [[ -n "${PY:-}" ]]; then
    candidates+=("$PY")
  fi
  candidates+=(
    python3
    python
    /opt/conda/bin/python
    /root/miniconda3/bin/python
    /opt/miniconda3/bin/python
    /usr/local/miniconda3/bin/python
    "$HOME/miniconda3/bin/python"
  )
  while IFS= read -r candidate; do
    candidates+=("$candidate")
  done < <(find /opt /root /usr/local -maxdepth 6 -type f \( -name python -o -name python3 \) 2>/dev/null | sort -u)
  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    if ! command -v "$candidate" >/dev/null 2>&1 && [[ ! -x "$candidate" ]]; then
      continue
    fi
    if "$candidate" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
PY
    then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

PY_BIN="$(pick_python)" || {
  echo "[error] could not find a Python with torch" >&2
  echo "[python-debug] searched common python paths under /opt /root /usr/local" >&2
  exit 2
}
echo "[python] $PY_BIN $("$PY_BIN" --version 2>&1)"

ensure_python_deps() {
  local stamp="${DEPS_STAMP:-/tmp/hftrainer_t2m_fill_missing_deps_v6.stamp}"
  if [[ -f "$stamp" ]]; then
    return 0
  fi
  local missing
  missing="$("$PY_BIN" - <<'PY'
import importlib.util
print("numpy<2")
checks = [
    ("einops", "einops>=0.7"),
    ("roma", "roma>=1.4"),
    ("rotary_embedding_torch", "rotary-embedding-torch>=0.8"),
    ("sentence_transformers", "sentence-transformers>=2.2"),
    ("mmengine", "mmengine>=0.7"),
    ("hydra", "hydra-core>=1.3"),
    ("omegaconf", "omegaconf>=2.3"),
    ("peft", "peft>=0.12"),
    ("boto3", "boto3"),
    ("chumpy", "chumpy>=0.70"),
    ("smplx", "smplx>=0.1.28"),
    ("torchgeometry", "torchgeometry>=0.1.2"),
]
for module, package in checks:
    if importlib.util.find_spec(module) is None:
        print(package)
PY
)"
  if [[ -n "$missing" ]]; then
    echo "[deps] installing: $(tr '\n' ' ' <<<"$missing")"
    "$PY_BIN" -m pip install -q \
      -i https://mirrors.tencent.com/pypi/simple \
      --trusted-host mirrors.tencent.com \
      $missing
  else
    echo "[deps] all required optional packages importable"
  fi
  touch "$stamp"
}

ensure_python_deps

METHOD="${METHOD:?set METHOD=flowmdm|motionlab|mdm|mld|motiongpt3|ik_<method>|kimodo_smplx_cache|kimodo_smplx|kimodo_smplx_convert|gotozero_raw272|gotozero_convert}"
NUM_GPUS="${NUM_GPUS:-${TJ_GPU_NUM:-8}}"
TOTAL_SHARDS="${TOTAL_SHARDS:-64}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
LOCAL_SHARDS="${LOCAL_SHARDS:-$NUM_GPUS}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  NUM_GPUS=1
fi

DATA_DIR="."
ANNO="data/annotation/test_hml3d_official272_gtlen.json"
H3D_ROOT="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272"
MISSING_ROOT="outputs/evaluation/t2m/humanml3d_official_test/_missing_audits/normalized_motion135_20260619"
RUN_TAG="${RUN_TAG:-fill_missing_motion135_20260620_atomic}"
RUN_ROOT="outputs/evaluation/t2m/humanml3d_official_test/_runs/$RUN_TAG"
LOG_DIR="$RUN_ROOT/logs"
PREP_DIR="$RUN_ROOT/prep"
HML_FILL="$RUN_ROOT/hml263"
KIMODO_WORK="outputs/evaluation/t2m/humanml3d_official_test/motion135/kimodo_official/intermediate"
KIMODO_SMPLX_DEBUG_SRC="${KIMODO_SMPLX_DEBUG_SRC:-$KIMODO_WORK/debug_npz}"
mkdir -p "$LOG_DIR" "$PREP_DIR" "$HML_FILL" "$KIMODO_WORK"

M135_BASE="outputs/evaluation/t2m/humanml3d_official_test/motion135"
declare -A M135_DIR=(
  [flowmdm]="$M135_BASE/flowmdm_official/predictions/motion135"
  [motionlab]="$M135_BASE/motionlab_official/predictions/motion135"
  [mdm]="$M135_BASE/mdm_official/predictions/motion135"
  [mld]="$M135_BASE/mld_official/predictions/motion135"
  [motiongpt3]="$M135_BASE/motiongpt3_official/predictions/motion135"
  [kimodo]="$M135_BASE/kimodo_official/predictions/motion135"
  [gotozero]="$M135_BASE/gotozero_official/predictions/motion135"
)

full_anno="$PREP_DIR/official_4042_anno.json"
caption_file="$PREP_DIR/official_first_caption.json"
ids_file="$PREP_DIR/official_4042_ids.txt"
kimodo_corpus="$PREP_DIR/kimodo_missing_corpus.jsonl"

prepare_inputs() {
  "$PY_BIN" - <<'PY' "$ANNO" "$DATA_DIR" "$H3D_ROOT" "$MISSING_ROOT" "$full_anno" "$caption_file" "$ids_file" "$kimodo_corpus"
import json
import os
import sys
from pathlib import Path

anno_path = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
h3d_root = Path(sys.argv[3])
missing_root = Path(sys.argv[4])
out_anno = Path(sys.argv[5])
out_caption = Path(sys.argv[6])
out_ids = Path(sys.argv[7])
kimodo_corpus = Path(sys.argv[8])

raw = json.loads(anno_path.read_text())
data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
if not isinstance(data, dict):
    data = {str(item.get("source_id") or item.get("motion_id") or i): item for i, item in enumerate(data)}

def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)

def first_caption_from_entry(entry):
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
            stack.extend(v for v in cur.values() if isinstance(v, (dict, list, str)))
        elif isinstance(cur, list):
            stack[:0] = cur
    return None

def first_caption_from_h3d(sid):
    path = h3d_root / "texts" / f"{sid}.txt"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
            t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
        except ValueError:
            continue
        if parts[0].strip() and f_tag == 0.0 and t_tag == 0.0:
            return parts[0].strip()
    return None

captions = {}
for sid, entry in data.items():
    sid = str(sid)
    captions[sid] = first_caption_from_entry(entry) or first_caption_from_h3d(sid) or "a person is moving"

out_anno.parent.mkdir(parents=True, exist_ok=True)
atomic_write_text(out_anno, json.dumps({"meta": raw.get("meta", {}), "data_list": data}, indent=2))
atomic_write_text(out_caption, json.dumps(captions, indent=2))
atomic_write_text(out_ids, "\n".join(data.keys()) + "\n")

for method in ("flowmdm", "motionlab", "mdm", "mld", "motiongpt3", "kimodo"):
    miss_path = missing_root / f"{method}_missing.txt"
    missing = [x.strip() for x in miss_path.read_text().splitlines() if x.strip()] if miss_path.exists() else []
    subset = {sid: data[sid] for sid in missing if sid in data}
    atomic_write_text(out_anno.parent / f"{method}_missing_anno.json", json.dumps({
        "meta": raw.get("meta", {}),
        "data_list": subset,
    }, indent=2))

kimodo_missing = [x.strip() for x in (missing_root / "kimodo_missing.txt").read_text().splitlines() if x.strip()]
kimodo_lines = []
for sid in kimodo_missing:
    motion = h3d_root / "motion_data" / f"{sid}.npy"
    if not motion.exists():
        continue
    import numpy as np
    length = int(np.load(str(motion), mmap_mode="r").shape[0])
    kimodo_lines.append(json.dumps({
        "id": sid,
        "split": "test",
        "prompt": captions.get(sid) or first_caption_from_h3d(sid) or "a person is moving",
        "length": length,
    }, ensure_ascii=False))
atomic_write_text(kimodo_corpus, "\n".join(kimodo_lines) + ("\n" if kimodo_lines else ""))

print(f"[prep] ids={len(data)} captions={len(captions)} kimodo_missing={len(kimodo_missing)}", flush=True)
PY
}

run_shards() {
  local phase="$1"
  shift
  echo "[phase-start] $phase method=$METHOD total=$TOTAL_SHARDS offset=$SHARD_OFFSET local=$LOCAL_SHARDS $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
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
    echo "[phase-fail] $phase method=$METHOD $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
    return 1
  fi
  echo "[phase-done] $phase method=$METHOD $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
}

run_flowmdm() {
  local shards="$1" shard="$2"
  "$PY_BIN" scripts/eval/flowmdm_infer_hml3d263.py \
    --anno-file "$full_anno" \
    --caption-file "$caption_file" \
    --data-dir "$DATA_DIR" \
    --out-dir "$HML_FILL/flowmdm" \
    --only-ids "$MISSING_ROOT/flowmdm_missing.txt" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --device 0 \
    --min-length 1 \
    --max-length 196 \
    --clip-download-root "${FLOWMDM_CLIP_DOWNLOAD_ROOT:-$ROOT/checkpoints/clip}" \
    --skip-existing
}

run_motionlab() {
  local shards="$1" shard="$2"
  "$PY_BIN" scripts/eval/motionlab_infer_hml3d263.py \
    --anno-file "$full_anno" \
    --caption-file "$caption_file" \
    --data-dir "$DATA_DIR" \
    --out-dir "$HML_FILL/motionlab" \
    --source-id-file "$MISSING_ROOT/motionlab_missing.txt" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --device cuda \
    --min-length 1 \
    --max-length 196 \
    --stage demo \
    --skip-existing
}

run_mdm() {
  local shards="$1" shard="$2"
  "$PY_BIN" scripts/eval/mdm_infer_hml3d263.py \
    --model_path ref_repo/MDM/save/humanml_enc_512_50steps/model000750000.pt \
    --anno_file "$PREP_DIR/mdm_missing_anno.json" \
    --anno_data_dir "$DATA_DIR" \
    --caption_protocol original \
    --out_dir "$HML_FILL/mdm" \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --batch_size 16 \
    --device 0 \
    --skip_existing
}

run_mld() {
  local shards="$1" shard="$2"
  local mld_hf_home="${MLD_HF_HOME:-$ROOT/checkpoints/huggingface_mld}"
  HF_HOME="$mld_hf_home" TRANSFORMERS_CACHE="$mld_hf_home/hub" \
  "$PY_BIN" scripts/eval/mld_infer_hml3d263.py \
    --anno_file "$PREP_DIR/mld_missing_anno.json" \
    --anno_data_dir "$DATA_DIR" \
    --caption_protocol original \
    --out_dir "$HML_FILL/mld" \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --batch_size 16 \
    --skip_existing
}

run_motiongpt3() {
  local shards="$1" shard="$2"
  "$PY_BIN" scripts/eval/motiongpt3_infer_hml3d263.py \
    --anno_file "$PREP_DIR/motiongpt3_missing_anno.json" \
    --anno_data_dir "$DATA_DIR" \
    --caption_protocol original \
    --out_dir "$HML_FILL/motiongpt3" \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --batch_size 8 \
    --guidance_scale 3.0 \
    --skip_existing
}

run_ik_method() {
  local method="$1" shards="$2" shard="$3"
  "$PY_BIN" scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$HML_FILL/$method" \
    --out-dir "${M135_DIR[$method]}" \
    --ids "$MISSING_ROOT/${method}_missing.txt" \
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

run_kimodo_smplx_cache() {
  local cache_dir="data/kimodo_text_feature"
  local namespace="kimodo_smplx_t2m_hml3d_smpl_ms272_20260616"
  local ns_dir="$cache_dir/$namespace"
  local sentinel="$ns_dir/.extracting"
  local meta="$ns_dir/meta.json"
  local manifest="$ns_dir/manifest.jsonl"
  mkdir -p "$ns_dir"
  if [[ -f "$meta" ]]; then
    echo "[kimodo-cache] already complete: $meta"
    return 0
  fi
  if [[ -f "$sentinel" ]]; then
    echo "[kimodo-cache] another extractor is active; waiting for $meta"
    local waited=0
    while [[ ! -f "$meta" && "$waited" -lt "${KIMODO_CACHE_WAIT_SECONDS:-21600}" ]]; do
      if [[ -f "$manifest" ]]; then
        echo "[kimodo-cache] waiting ${waited}s manifest_lines=$(wc -l < "$manifest" || echo 0)"
      else
        echo "[kimodo-cache] waiting ${waited}s manifest_lines=0"
      fi
      sleep 60
      waited=$((waited + 60))
    done
    if [[ -f "$meta" ]]; then
      echo "[kimodo-cache] completed by another extractor: $meta"
      return 0
    fi
    echo "[kimodo-cache] wait timed out; continuing extraction"
  fi
  touch "$sentinel"
  trap 'rm -f "$sentinel"' RETURN
  HF_HOME="$ROOT/checkpoints/kimodo" \
  HUGGINGFACE_HUB_CACHE="$ROOT/checkpoints/kimodo/hub" \
  TRANSFORMERS_CACHE="$ROOT/checkpoints/kimodo/hub" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TEXT_ENCODERS_DIR="$ROOT/checkpoints/kimodo/text_encoders" \
  CHECKPOINT_DIR="$ROOT/checkpoints/kimodo/local_models" \
  TEXT_ENCODER_MODE=local \
  LOCAL_CACHE=true \
  "$PY_BIN" scripts/embodied/cursor_extract_kimodo_text_feature.py \
    --corpus "$kimodo_corpus" \
    --namespace "$namespace" \
    --cache-dir "$cache_dir" \
    --hf-home "$ROOT/checkpoints/kimodo" \
    --text-encoder llm2vec \
    --device cuda \
    --batch-size "${KIMODO_FEATURE_BATCH_SIZE:-8}"
}

run_kimodo_smplx() {
  local shards="$1" shard="$2"
  "$PY_BIN" scripts/eval/gen_kimodo_t2m_positions.py \
    --humanml3d-272 "$H3D_ROOT" \
    --corpus "$kimodo_corpus" \
    --out-dir "$KIMODO_WORK/positions22" \
    --debug-npz-dir "$KIMODO_WORK/debug_npz" \
    --model-path checkpoints/kimodo/hftrainer_smplx_rp \
    --model-name Kimodo-SMPLX-RP-v1 \
    --diffusion-steps "${KIMODO_DIFFUSION_STEPS:-100}" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --device cuda \
    --skip-existing \
    --min-len 1 \
    --max-len 100000 \
    --text-feature-cache-dir data/kimodo_text_feature \
    --text-feature-namespace kimodo_smplx_t2m_hml3d_smpl_ms272_20260616
}

run_kimodo_smplx_convert() {
  local shards="$1" shard="$2"
  local ids="$PREP_DIR/kimodo_missing_s${shard}_of_${shards}.txt"
  awk -v n="$shards" -v s="$shard" 'NF { if (((NR - 1) % n) == s) print $0 }' \
    "$MISSING_ROOT/kimodo_missing.txt" > "$ids"

  "$PY_BIN" scripts/eval/kimodo_smplx_to_motion135.py \
    --in-dir "$KIMODO_SMPLX_DEBUG_SRC" \
    --out-dir "${M135_DIR[kimodo]}" \
    --ids "$ids" \
    --skip-existing
}

run_kimodo_smplx_convert_serial() {
  local local_idx shard
  echo "[phase-start] kimodo_smplx_convert_serial method=$METHOD total=$TOTAL_SHARDS offset=$SHARD_OFFSET local=$LOCAL_SHARDS $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
  for ((local_idx=0; local_idx<LOCAL_SHARDS; local_idx++)); do
    shard=$((SHARD_OFFSET + local_idx))
    if (( shard >= TOTAL_SHARDS )); then
      continue
    fi
    set +e
    run_kimodo_smplx_convert "$TOTAL_SHARDS" "$shard" \
      > "$LOG_DIR/kimodo_smplx_convert_s${shard}_of_${TOTAL_SHARDS}.log" 2>&1
    code=$?
    set -e
    echo "exit_code=$code finished_at=$(date -Is)" > "$LOG_DIR/kimodo_smplx_convert_s${shard}_of_${TOTAL_SHARDS}.log.status"
    if [[ "$code" -ne 0 ]]; then
      echo "[phase-fail] kimodo_smplx_convert_serial shard=$shard method=$METHOD $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
      return "$code"
    fi
  done
  echo "[phase-done] kimodo_smplx_convert_serial method=$METHOD $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
}

run_gotozero_raw272() {
  local shards="$1" shard="$2"
  "$PY_BIN" scripts/eval/motionmillion_h3d272.py \
    --out_dir outputs/evaluation/t2m/humanml3d_official_test/ms272/motionmillion_exactlen_0617/raw272 \
    --device cuda \
    --dtype bf16 \
    --artifact checkpoints/gotozero/hftrainer_7b_humanml272 \
    --text_model_name checkpoints/flan-t5-xl \
    --max_sample_steps 50 \
    --pair_source official_split \
    --canonical_output \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --skip_existing
}

run_gotozero_convert() {
  "$PY_BIN" scripts/eval/h3d_272_to_135.py \
    --gt-dir outputs/evaluation/t2m/humanml3d_official_test/ms272/motionmillion_exactlen_0617/raw272 \
    --id-file "$MISSING_ROOT/gotozero_missing.txt" \
    --out-dir "${M135_DIR[gotozero]}"
}

prepare_inputs

case "$METHOD" in
  flowmdm) run_shards infer_flowmdm run_flowmdm; run_shards ik_flowmdm run_ik_method flowmdm ;;
  motionlab) run_shards infer_motionlab run_motionlab; run_shards ik_motionlab run_ik_method motionlab ;;
  mdm) run_shards infer_mdm run_mdm; run_shards ik_mdm run_ik_method mdm ;;
  mld) run_shards infer_mld run_mld; run_shards ik_mld run_ik_method mld ;;
  motiongpt3) run_shards infer_motiongpt3 run_motiongpt3; run_shards ik_motiongpt3 run_ik_method motiongpt3 ;;
  ik_*) run_shards "$METHOD" run_ik_method "${METHOD#ik_}" ;;
  kimodo_smplx_cache) run_kimodo_smplx_cache ;;
  kimodo_smplx) run_shards kimodo_smplx run_kimodo_smplx; run_kimodo_smplx_convert_serial ;;
  kimodo_smplx_convert) run_shards kimodo_smplx_convert run_kimodo_smplx_convert ;;
  gotozero_raw272) run_shards gotozero_raw272 run_gotozero_raw272 ;;
  gotozero_convert) run_gotozero_convert ;;
  *) echo "unknown METHOD=$METHOD" >&2; exit 2 ;;
esac

echo "[done] method=$METHOD $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
