#!/usr/bin/env bash
# Metrics for the clean T2M-GPT HumanML3D official-test rerun.
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

BASE="outputs/evaluation/t2m/humanml3d_official_test"
RUN_ROOT="$BASE/ms272/_suites/t2mgpt_clean_20260624_metrics"
RESULTS="$RUN_ROOT/results"
LOGS="$RUN_ROOT/logs"
MOTION135="$BASE/motion135/t2mgpt"
MS272="$BASE/ms272/t2mgpt"
ROUNDTRIP_MS272="$BASE/_runs/noncanonical_legacy_20260623/ms272/gt_hml263_roundtrip_20260623_rootfix/predictions/ms272"
ROUNDTRIP_MC135="$BASE/ms272/_suites/motionclip_gt_hml263_roundtrip_20260623/motionclip135/roundtrip"
CAPTION_TEXTS="$BASE/captions/humanml3d_official_corrected/texts"
CAPTION_ANNO="$BASE/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json"
MC_REAL="$ROUNDTRIP_MC135"
MC_PRED="$RUN_ROOT/motionclip135/t2mgpt"
MC_MANIFEST="$RUN_ROOT/motionclip_manifest.tsv"

mkdir -p "$RESULTS" "$LOGS" "$MC_PRED"

echo "[start] T2M-GPT clean metrics $(date -Is)"
echo "[paths] motion135=$MOTION135"
echo "[paths] ms272=$MS272"
echo "[paths] roundtrip_ms272=$ROUNDTRIP_MS272"
echo "[paths] roundtrip_motionclip135=$MC_REAL"

python3 scripts/eval/eval_motionstreamer_272.py \
  --pred-dir "$MOTION135" \
  --gt-272-dir "$ROUNDTRIP_MS272" \
  --tag t2mgpt_clean_hmlroundtrip \
  --text-dir "$CAPTION_TEXTS" \
  --min-motion-len 1 \
  --out-json "$RESULTS/t2mgpt_motionstreamer272_hmlroundtrip.json" \
  2>&1 | tee "$LOGS/motionstreamer272_hmlroundtrip.log"

python3 scripts/eval/eval_motionstreamer_272.py \
  --pred-dir "$MOTION135" \
  --tag t2mgpt_clean_raw_refk \
  --real-encoding refk \
  --also-refk \
  --text-dir "$CAPTION_TEXTS" \
  --min-motion-len 1 \
  --out-json "$RESULTS/t2mgpt_motionstreamer272_raw_refk.json" \
  2>&1 | tee "$LOGS/motionstreamer272_raw_refk.log"

python3 scripts/eval/eval_mbench_physics_dir.py \
  --src "$MOTION135" \
  --mode m135 \
  --workers "${PHYS_WORKERS:-32}" \
  --out-json "$RESULTS/t2mgpt_physics.json" \
  2>&1 | tee "$LOGS/physics.log"

python3 scripts/eval/convert_row135_npz_to_motionclip_col.py \
  --anno-file "$CAPTION_ANNO" \
  --src-dir "$MOTION135" \
  --out-dir "$MC_PRED" \
  --overwrite \
  2>&1 | tee "$LOGS/convert_motionclip135.log"

printf "T2M-GPT\t%s\n" "$MC_PRED" > "$MC_MANIFEST"
python3 scripts/eval/eval_motionclip_table1_dirs.py \
  --anno-file "$CAPTION_ANNO" \
  --data-dir "." \
  --caption-key hierarchical_caption \
  --real-dir "$MC_REAL" \
  --pred-manifest "$MC_MANIFEST" \
  --out-dir "$RUN_ROOT/motionclip" \
  --min-frames 1 \
  --max-frames 300 \
  2>&1 | tee "$LOGS/motionclip_hmlroundtrip.log"

echo "[done] T2M-GPT clean metrics $(date -Is)"
