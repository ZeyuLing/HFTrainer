#!/usr/bin/env bash
# Recompute every current Table-1 HumanML3D method with the MotionStreamer-272
# evaluator and the shared MBench physical metric implementation.
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

STAMP="${STAMP:-20260622}"
OUT_ROOT="${OUT_ROOT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/table1_full_rerun_${STAMP}}"
LOG="$OUT_ROOT/logs"
RES="$OUT_ROOT/results"
mkdir -p "$LOG" "$RES"

NGPU="${NGPU:-${TJ_GPU_NUM:-1}}"
if [[ "$NGPU" -lt 1 ]]; then
  NGPU=1
fi
if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
  NGPU="${#GPU_IDS[@]}"
else
  GPU_IDS=()
  for ((g=0; g<NGPU; g++)); do
    GPU_IDS+=("$g")
  done
fi
MIN_MOTION_LEN="${MIN_MOTION_LEN:-60}"
MAX_MOTION_LENGTH="${MAX_MOTION_LENGTH:-300}"
SEED="${SEED:-0}"
FORCE_EVAL="${FORCE_EVAL:-1}"
RUN_PHYS="${RUN_PHYS:-1}"
PHYS_WORKERS="${PHYS_WORKERS:-32}"
IO_WORKERS="${IO_WORKERS:-32}"

MS_BASE="outputs/evaluation/t2m/humanml3d_official_test/ms272"
LATEST="outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_latest_epoch42_20260622"
OLD_PREP="outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/table1_finish_missing_20260618/prep"

METHODS_ALL=(
  "mdm|gt272|$OLD_PREP/mdm"
  "mld|gt272|$OLD_PREP/mld"
  "momask|gt272|$OLD_PREP/momask"
  "motiongpt3|gt272|$OLD_PREP/motiongpt3"
  "t2mgpt|gt272|$OLD_PREP/t2mgpt"
  "flowmdm|gt272|$OLD_PREP/flowmdm"
  "motionlab|gt272|$OLD_PREP/motionlab"
  "gotozero|gt272|$OLD_PREP/gotozero"
  "motionstreamer|m135|$MS_BASE/motionstreamer_exactlen_0617_vermo/prep"
  "hymotion_1b|m135|$MS_BASE/hymotion_1b_exactlen_0617_vermo/prep/hymotion"
  "mogents|gt272|$LATEST/prep/mogents"
  "ours_epoch42_abs|m135|$LATEST/prep/ours_epoch42_abs"
)

echo "[start] $(date -Is) root=$ROOT out=$OUT_ROOT ngpu=$NGPU min_len=$MIN_MOTION_LEN max_len=$MAX_MOTION_LENGTH seed=$SEED" | tee "$LOG/run.log"

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

cache_ms272() {
  bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true
  local ms_rel="ref_repo/MotionStreamer/MotionStreamer"
  if [[ ! -f /dev/shm/eval272_epoch99.ckpt ]]; then
    cp "$ms_rel/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt" /dev/shm/eval272_epoch99.ckpt 2>/dev/null || true
  fi
}

count_npz() {
  find "$1" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l
}

select_methods() {
  if [[ -z "${METHODS:-}" ]]; then
    printf "%s\n" "${METHODS_ALL[@]}"
    return
  fi
  python3 - "$METHODS" "${METHODS_ALL[@]}" <<'PY'
import sys
want = {x.strip() for x in sys.argv[1].split(",") if x.strip()}
for entry in sys.argv[2:]:
    name = entry.split("|", 1)[0]
    if name in want:
        print(entry)
PY
}

mapfile -t METHOD_ENTRIES < <(select_methods)
if [[ "${#METHOD_ENTRIES[@]}" -eq 0 ]]; then
  echo "[error] no methods selected; METHODS=${METHODS:-<unset>}" | tee -a "$LOG/run.log"
  exit 2
fi

COVERAGE_JSON="$OUT_ROOT/coverage.json"
python3 - "$COVERAGE_JSON" "${METHOD_ENTRIES[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = {}
for entry in sys.argv[2:]:
    name, mode, pred = entry.split("|", 2)
    p = Path(pred)
    n = sum(1 for _ in p.glob("*.npz")) if p.is_dir() else 0
    rows[name] = {"mode": mode, "pred_dir": pred, "npz": n, "exists": p.is_dir()}
out.write_text(json.dumps(rows, indent=2))
for name, row in rows.items():
    print(f"[coverage] {name:16s} mode={row['mode']:5s} npz={row['npz']:4d} exists={row['exists']} dir={row['pred_dir']}")
PY

bad=0
for entry in "${METHOD_ENTRIES[@]}"; do
  IFS='|' read -r name mode pred <<< "$entry"
  n="$(count_npz "$pred")"
  if [[ ! -d "$pred" || "$n" -lt 4042 ]]; then
    echo "[coverage-bad] $name npz=$n dir=$pred" | tee -a "$LOG/run.log"
    bad=1
  fi
done
if [[ "$bad" -ne 0 && "${ALLOW_INCOMPLETE:-0}" != "1" ]]; then
  echo "[error] coverage is incomplete; set ALLOW_INCOMPLETE=1 only for diagnostics" | tee -a "$LOG/run.log"
  exit 3
fi
if [[ "${COVERAGE_ONLY:-0}" == "1" ]]; then
  echo "[coverage-only] ok $(date -Is)" | tee -a "$LOG/run.log"
  exit 0
fi

ensure_deps
cache_ms272

eval_one() {
  local name="$1"
  local pred="$2"
  local gpu="$3"
  local out_json="$RES/${name}.json"
  if [[ -s "$out_json" && "$FORCE_EVAL" != "1" ]]; then
    echo "[eval-skip] $name exists=$out_json" | tee -a "$LOG/run.log"
    return 0
  fi
  echo "[eval-start] $name gpu=$gpu pred=$pred $(date -Is)" | tee -a "$LOG/run.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$pred" \
    --tag "$name" \
    --also-refk \
    --seed "$SEED" \
    --min-motion-len "$MIN_MOTION_LEN" \
    --max-motion-length "$MAX_MOTION_LENGTH" \
    --out-json "$out_json" \
    > "$LOG/eval_${name}.log" 2>&1
  echo "[eval-done] $name $(date -Is)" | tee -a "$LOG/run.log"
}

pids=()
idx=0
for entry in "${METHOD_ENTRIES[@]}"; do
  IFS='|' read -r name mode pred <<< "$entry"
  gpu="${GPU_IDS[$((idx % NGPU))]}"
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

python3 scripts/eval/_agg_ms272_tables.py \
  --res-dir "$RES" \
  --out "$OUT_ROOT/summary_ms_eval.json" \
  | tee "$OUT_ROOT/summary_ms_eval.txt"

if [[ "$RUN_PHYS" == "1" ]]; then
  PHYS_MANIFEST="$OUT_ROOT/phys_manifest.tsv"
  : > "$PHYS_MANIFEST"
  for entry in "${METHOD_ENTRIES[@]}"; do
    IFS='|' read -r name mode pred <<< "$entry"
    printf "%s\t%s\t%s\n" "$name" "$mode" "$pred" >> "$PHYS_MANIFEST"
  done
  echo "[phys-start] $(date -Is)" | tee -a "$LOG/run.log"
  python3 scripts/eval/compute_phys_h3d.py \
    --manifest "$PHYS_MANIFEST" \
    --workers "$PHYS_WORKERS" \
    --out-json "$RES/phys.json" \
    > "$LOG/phys.log" 2>&1
  echo "[phys-done] $(date -Is)" | tee -a "$LOG/run.log"
fi

python3 - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.load(open(root / "summary_ms_eval.json"))
coverage = json.load(open(root / "coverage.json"))
phys_path = root / "results" / "phys.json"
phys = json.load(open(phys_path)) if phys_path.exists() else {}

rows = {}
for name, row in summary.get("methods", {}).items():
    prow = phys.get(name) or {}
    rows[name] = {
        **row,
        "coverage_npz": (coverage.get(name) or {}).get("npz"),
        "pred_dir": (coverage.get(name) or {}).get("pred_dir"),
        "phys_n": prow.get("n"),
        "phys_raw": prow,
    }
out = {"gt_real_reference": summary.get("gt_real_reference"), "methods": rows}
(root / "summary_with_phys.json").write_text(json.dumps(out, indent=2))

print("[summary-with-phys]")
for name in sorted(rows):
    r = rows[name]
    print(
        f"{name:16s} cov={r.get('coverage_npz')} samples={r.get('samples')} "
        f"nb={r.get('nb_pred')} FIDnat={r.get('FID_native')} "
        f"FIDrefk={r.get('FID_refk')} R1={r.get('R1')} R3={r.get('R3')} "
        f"MM={r.get('MM')} phys_n={r.get('phys_n')}"
    )
PY

echo "[done] $(date -Is) out=$OUT_ROOT" | tee -a "$LOG/run.log"
