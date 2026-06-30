#!/usr/bin/env bash
# Evaluate PRISM epoch-43 TP2M Table-2 outputs after generation.
# Computes MotionStreamer-272 semantic metrics and shared physical metrics.
set -uo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "${ROOT}" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "${ROOT}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

PY=${PY:-python3}
GPU=${GPU:-0}
WORKERS=${WORKERS:-16}
CONDS=${CONDS:-"1 5 9"}
GEN=${GEN:-outputs/evaluation/tp2m/humanml3d_official_test/motion135/prism_epoch43_pad360crop_selected_20260628}
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/tp2m/humanml3d_official_test/_suites/table2_prism_epoch43_pad360crop_selected_20260628_ms272}
ANNO=${ANNO:-outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/test_hml3d_official272_gtlen_motionclip_selected_caption.json}
TEXT_DIR=${TEXT_DIR:-outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/texts}
SKIP_CACHE=${SKIP_CACHE:-0}

PREP="$OUT_ROOT/prep"
LOG="$OUT_ROOT/logs"
RES="$OUT_ROOT/results"
mkdir -p "$PREP" "$LOG" "$RES"

if [ "$SKIP_CACHE" = "1" ]; then
  echo "[cache] skipped by SKIP_CACHE=1" > "$LOG/cache.log"
else
  bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true
fi
MS_REL="ref_repo/MotionStreamer/MotionStreamer"
if [ ! -f /dev/shm/eval272_epoch99.ckpt ]; then
  cp "$MS_REL/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt" /dev/shm/eval272_epoch99.ckpt 2>/dev/null || true
fi

cat > "$OUT_ROOT/command.txt" <<EOF
ROOT=$ROOT
GEN=$GEN
OUT_ROOT=$OUT_ROOT
ANNO=$ANNO
TEXT_DIR=$TEXT_DIR
CONDS=$CONDS
WORKERS=$WORKERS
EOF

echo "[start] $(date) gen=$GEN out=$OUT_ROOT conds=$CONDS" | tee "$LOG/run.log"

for cond in $CONDS; do
  name="prism_c${cond}"
  src="$GEN/cond${cond}_depth_driven"
  prep="$PREP/$name"
  if [ ! -d "$src" ]; then
    echo "[MISS] cond=$cond src=$src" | tee -a "$LOG/run.log"
    continue
  fi

  src_n=$(find "$src" -maxdepth 1 -name '*.npz' | wc -l)
  echo "[repack] $name src_n=$src_n src=$src" | tee -a "$LOG/run.log"
  mkdir -p "$prep"
  "$PY" scripts/eval/repack_pred_to_272ids.py \
    --npz-dir "$src" \
    --anno-file "$ANNO" \
    --out-dir "$prep" \
    --workers "$WORKERS" \
    > "$LOG/repack_${name}.log" 2>&1
  prep_n=$(find "$prep" -maxdepth 1 -name '*.npz' | wc -l)
  echo "[repack] $name prep_n=$prep_n" | tee -a "$LOG/run.log"

  echo "[eval-ms272] $name $(date)" | tee -a "$LOG/run.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$prep" \
    --tag "$name" \
    --also-refk \
    --text-dir "$TEXT_DIR" \
    --out-json "$RES/${name}_ms272.json" \
    > "$LOG/eval_ms272_${name}.log" 2>&1

  echo "[eval-phys] $name $(date)" | tee -a "$LOG/run.log"
  "$PY" scripts/eval/compute_phys_h3d.py \
    --m135-dir "$prep" \
    --tag "$name" \
    --out-json "$RES/${name}_physics.json" \
    --workers "$WORKERS" \
    > "$LOG/eval_phys_${name}.log" 2>&1
  grep "TABLE" "$LOG/eval_phys_${name}.log" | tee -a "$LOG/run.log" || true
done

"$PY" - "$RES" "$OUT_ROOT/summary.json" <<'PY'
import json
import sys
from pathlib import Path

res = Path(sys.argv[1])
out = Path(sys.argv[2])
summary = {}
for ms_path in sorted(res.glob("*_ms272.json")):
    name = ms_path.name.replace("_ms272.json", "")
    ms = json.loads(ms_path.read_text()).get("pred", {})
    phys_path = res / f"{name}_physics.json"
    phys = {}
    if phys_path.exists():
        phys = json.loads(phys_path.read_text()).get(name, {})
    rp = ms.get("r_precision") or [None, None, None]
    summary[name] = {
        "n": ms.get("nb"),
        "R@1": rp[0],
        "R@2": rp[1],
        "R@3": rp[2],
        "FID": ms.get("fid_vs_gt_native"),
        "FID_refk": ms.get("fid_vs_gt_refk"),
        "MM_Dist": ms.get("matching_score"),
        "Diversity": ms.get("diversity"),
        "Slide": phys.get("Slide"),
        "Float": phys.get("Float"),
        "Jitter": phys.get("Jitter"),
        "Dynamic": phys.get("Dynamic"),
        "phys_n": phys.get("n"),
    }
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
for name, row in summary.items():
    print(name, json.dumps(row, sort_keys=True))
PY

touch "$OUT_ROOT/_DONE"
echo "[done] $(date) -> $OUT_ROOT/summary.json" | tee -a "$LOG/run.log"
