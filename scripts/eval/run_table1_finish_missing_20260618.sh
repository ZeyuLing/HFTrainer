#!/usr/bin/env bash
# Finish missing PRISM Table-1 HumanML3D official-test results.
#
# This job is intentionally eval-focused: it consumes already materialized
# 4042-case predictions and writes a fresh MotionStreamer-272 suite.  Expensive
# generation gaps such as ViMoGen/KIMODO are launched by their own pipelines.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HFTRAINER_SKIP_AUTOREGISTER=1

OUT_ROOT="${OUT_ROOT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/table1_finish_missing_20260618}"
PREP="$OUT_ROOT/prep"
LOG="$OUT_ROOT/logs"
RES="$OUT_ROOT/results"
mkdir -p "$PREP" "$LOG" "$RES"

NGPU="${NGPU:-${TJ_GPU_NUM:-8}}"
if [[ "$NGPU" -lt 1 ]]; then
  NGPU=1
fi

ANNO_OFFICIAL="data/annotation/test_hml3d_official272_gtlen.json"
MS_BASE="outputs/evaluation/t2m/humanml3d_official_test/ms272"

echo "[start] $(date -Is) root=$ROOT out=$OUT_ROOT ngpu=$NGPU" | tee "$LOG/run.log"

ensure_deps() {
  local missing="$OUT_ROOT/missing_python_deps.txt"
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
    "scipy": "scipy",
    "mmengine": "mmengine>=0.7",
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
    echo "[deps] installing $(tr '\n' ' ' < "$missing")" | tee -a "$LOG/run.log"
    python3 -m pip install -q \
      -i https://mirrors.tencent.com/pypi/simple \
      --trusted-host mirrors.tencent.com \
      -r "$missing"
  else
    echo "[deps] python deps ok" | tee -a "$LOG/run.log"
  fi
}

count_npz() {
  local d="$1"
  find "$d" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l
}

count_npy() {
  local d="$1"
  find "$d" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l
}

cache_ms272() {
  bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true
  local ms_rel="ref_repo/MotionStreamer/MotionStreamer"
  if [[ ! -f /dev/shm/eval272_epoch99.ckpt ]]; then
    cp "$ms_rel/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt" /dev/shm/eval272_epoch99.ckpt 2>/dev/null || true
  fi
}

repack_motion272() {
  local name="$1"
  local src="$2"
  local dst="$PREP/$name"
  mkdir -p "$dst"
  if [[ -f "$dst/_DONE" ]]; then
    echo "$dst"
    return 0
  fi
  echo "[repack] $name src=$src -> $dst" | tee -a "$LOG/run.log" >&2
  python3 scripts/eval/repack_pred_to_272ids.py \
    --motion272-dir "$src" \
    --id-passthrough \
    --anno-file "$ANNO_OFFICIAL" \
    --out-dir "$dst" \
    --workers "${REPACK_WORKERS:-32}" \
    > "$LOG/repack_${name}.log" 2>&1
  touch "$dst/_DONE"
  echo "$dst"
}

repack_gt272() {
  local name="$1"
  local src="$2"
  local dst="$PREP/$name"
  mkdir -p "$dst"
  if [[ -f "$dst/_DONE" ]]; then
    echo "$dst"
    return 0
  fi
  echo "[repack] $name gt272=$src -> $dst" | tee -a "$LOG/run.log" >&2
  python3 scripts/eval/repack_pred_to_272ids.py \
    --gt272-dir "$src" \
    --id-passthrough \
    --out-dir "$dst" \
    --workers "${REPACK_WORKERS:-32}" \
    > "$LOG/repack_${name}.log" 2>&1
  touch "$dst/_DONE"
  echo "$dst"
}

eval_one() {
  local name="$1"
  local pred="$2"
  local gpu="$3"
  local oj="$RES/${name}.json"
  if [[ -s "$oj" && "${FORCE_EVAL:-0}" != "1" ]]; then
    return 0
  fi
  echo "[eval] $name gpu=$gpu pred=$pred" | tee -a "$LOG/run.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$pred" \
    --tag "$name" \
    --also-refk \
    --out-json "$oj" \
    > "$LOG/eval_${name}.log" 2>&1
}

ensure_deps
cache_ms272

declare -a EVAL_ENTRIES=()
add_motion272_method() {
  local name="$1"
  local src="$MS_BASE/${name}_official_from_motion135/predictions/ms272"
  local n
  n="$(count_npy "$src")"
  echo "[source] $name motion272_npy=$n src=$src" | tee -a "$LOG/run.log"
  if [[ "$n" -eq 0 ]]; then
    echo "[skip] missing source for $name" | tee -a "$LOG/run.log"
    return 0
  fi
  local prep
  prep="$(repack_motion272 "$name" "$src")"
  local pn
  pn="$(count_npz "$prep")"
  echo "[prep] $name npz=$pn dir=$prep" | tee -a "$LOG/run.log"
  EVAL_ENTRIES+=("$name|$prep|gt272")
}

for method in motiongpt3 mld momask mdm t2mgpt flowmdm motionlab; do
  add_motion272_method "$method"
done

GOTOZERO_RAW="$MS_BASE/motionmillion_exactlen_0617/raw272"
if [[ -d "$GOTOZERO_RAW" ]]; then
  n="$(count_npy "$GOTOZERO_RAW")"
  echo "[source] gotozero raw272_npy=$n src=$GOTOZERO_RAW" | tee -a "$LOG/run.log"
  if [[ "$n" -gt 0 ]]; then
    gotozero_prep="$(repack_gt272 gotozero "$GOTOZERO_RAW")"
    pn="$(count_npz "$gotozero_prep")"
    echo "[prep] gotozero npz=$pn dir=$gotozero_prep" | tee -a "$LOG/run.log"
    EVAL_ENTRIES+=("gotozero|$gotozero_prep|gt272")
  fi
else
  echo "[skip] gotozero missing raw dir=$GOTOZERO_RAW" | tee -a "$LOG/run.log"
fi

declare -a EXISTING_ENTRIES=(
  "hymotion_1b|$MS_BASE/hymotion_1b_exactlen_0617_vermo/prep/hymotion|m135"
  "motionstreamer|$MS_BASE/motionstreamer_exactlen_0617_vermo/prep|m135"
  "ours_prism_e31_smooth|$MS_BASE/prism_epoch31_smooth_exactlen_0617_vermo/prep/ours_e31_smooth|m135"
)

for entry in "${EXISTING_ENTRIES[@]}"; do
  IFS='|' read -r name pred mode <<< "$entry"
  if [[ ! -d "$pred" ]]; then
    echo "[skip] $name missing dir=$pred" | tee -a "$LOG/run.log"
    continue
  fi
  n="$(count_npz "$pred")"
  echo "[source] $name npz=$n mode=$mode dir=$pred" | tee -a "$LOG/run.log"
  EVAL_ENTRIES+=("$name|$pred|$mode")
done

echo "[eval-start] methods=${#EVAL_ENTRIES[@]} $(date -Is)" | tee -a "$LOG/run.log"
pids=()
idx=0
for entry in "${EVAL_ENTRIES[@]}"; do
  IFS='|' read -r name pred mode <<< "$entry"
  gpu=$((idx % NGPU))
  eval_one "$name" "$pred" "$gpu" &
  pids+=("$!")
  idx=$((idx + 1))
  if (( idx % NGPU == 0 )); then
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    pids=()
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[phys-start] $(date -Is)" | tee -a "$LOG/run.log"
PHYS_MANIFEST="$OUT_ROOT/phys_manifest.tsv"
: > "$PHYS_MANIFEST"
for entry in "${EVAL_ENTRIES[@]}"; do
  IFS='|' read -r name pred mode <<< "$entry"
  printf "%s\t%s\t%s\n" "$name" "$mode" "$pred" >> "$PHYS_MANIFEST"
done
python3 scripts/eval/compute_phys_h3d.py \
  --manifest "$PHYS_MANIFEST" \
  --workers "${PHYS_WORKERS:-32}" \
  --out-json "$RES/phys.json" \
  > "$LOG/phys.log" 2>&1

python3 scripts/eval/_agg_ms272_tables.py \
  --res-dir "$RES" \
  --out "$OUT_ROOT/summary.json" \
  | tee "$OUT_ROOT/summary.txt"

python3 - <<'PY' "$OUT_ROOT"
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
summary = json.load(open(root / "summary.json"))
phys = json.load(open(root / "results" / "phys.json"))
audit = {}
for name, row in summary.get("methods", {}).items():
    audit[name] = {
        "samples": row.get("samples"),
        "fid_native": row.get("FID_native"),
        "r1": row.get("R1"),
        "r3": row.get("R3"),
        "phys_n": (phys.get(name) or {}).get("n"),
    }
(root / "completion_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
PY

touch "$OUT_ROOT/_DONE"
echo "[done] $(date -Is) out=$OUT_ROOT" | tee -a "$LOG/run.log"
