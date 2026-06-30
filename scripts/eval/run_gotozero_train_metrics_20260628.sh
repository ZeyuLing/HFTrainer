#!/usr/bin/env bash
# Evaluate a GoToZero / MotionMillion train-only checkpoint on HumanML3D.
#
# Input predictions are canonical HumanML3D ids stored as MS272 .npy files:
#   outputs/evaluation/t2m/humanml3d_official_test/ms272/gotozero_7b_train
#
# This script intentionally mirrors the existing GoToZero leaderboard protocol:
# MotionStreamer-272 raw/refk metrics, MotionCLIP raw/no-L2 metrics against the
# HML263-roundtrip GT reference, and MBench-style physical metrics on MS272.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HFTRAINER_SKIP_AUTOREGISTER=1
export PYTHONPATH="$ROOT:$ROOT/tools:$ROOT/scripts/eval:${PYTHONPATH:-}"

BASE="outputs/evaluation/t2m/humanml3d_official_test"
METHOD_SLUG="${METHOD_SLUG:-gotozero_7b_train}"
METHOD_LABEL="${METHOD_LABEL:-GoToZero}"
MS272_DIR="${MS272_DIR:-$BASE/ms272/$METHOD_SLUG}"
RUN_ROOT="${RUN_ROOT:-$BASE/ms272/_suites/gotozero_7b_train_20260628}"
RESULTS="$RUN_ROOT/results"
LOGS="$RUN_ROOT/logs"
MC_PRED="$RUN_ROOT/motionclip135/$METHOD_SLUG"
MC_MANIFEST="$RUN_ROOT/motionclip_manifest.tsv"
VERSION_LABEL="${VERSION_LABEL:-7B-train}"

CAPTION_ROOT="$BASE/captions/humanml3d_official_corrected"
CAPTION_TEXTS="$CAPTION_ROOT/texts"
CAPTION_ANNO="$CAPTION_ROOT/test_hml3d_official272_gtlen_motionclip_selected_caption.json"
ROUNDTRIP_MC135="$BASE/ms272/_suites/motionclip_gt_hml263_roundtrip_20260623/motionclip135/roundtrip"

mkdir -p "$RESULTS" "$LOGS" "$MC_PRED"

echo "[start] GoToZero train-only metrics $(date -Is)"
echo "[paths] ms272=$MS272_DIR"
echo "[paths] run_root=$RUN_ROOT"
echo "[paths] captions=$CAPTION_ANNO"

count="$(find "$MS272_DIR" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l)"
if [[ "$count" -lt 4042 && "${ALLOW_INCOMPLETE:-0}" != "1" ]]; then
  echo "[error] expected 4042 MS272 .npy files, found $count under $MS272_DIR" >&2
  exit 3
fi
echo "[coverage] ms272_npy=$count"

python3 scripts/eval/eval_motionstreamer_272.py \
  --pred-dir "$MS272_DIR" \
  --tag "$METHOD_SLUG" \
  --also-refk \
  --text-dir "$CAPTION_TEXTS" \
  --min-motion-len 1 \
  --out-json "$RESULTS/${METHOD_SLUG}_motionstreamer272_raw_refk.json" \
  2>&1 | tee "$LOGS/${METHOD_SLUG}_motionstreamer272_raw_refk.log"

python3 scripts/eval/eval_mbench_physics_dir.py \
  --src "$MS272_DIR" \
  --mode gt272 \
  --workers "${PHYS_WORKERS:-32}" \
  --out-json "$RESULTS/${METHOD_SLUG}_physics.json" \
  2>&1 | tee "$LOGS/${METHOD_SLUG}_physics.log"

python3 scripts/eval/convert_ms272_dir_for_t2m_eval.py \
  --src-dir "$MS272_DIR" \
  --anno-file "$CAPTION_ANNO" \
  --data-dir "." \
  --motionclip-dir "$MC_PRED" \
  --overwrite \
  --workers "${CONVERT_WORKERS:-32}" \
  2>&1 | tee "$LOGS/${METHOD_SLUG}_convert_motionclip135.log"

printf "%s\t%s\n" "$METHOD_LABEL" "$MC_PRED" > "$MC_MANIFEST"

python3 scripts/eval/eval_motionclip_table1_dirs.py \
  --anno-file "$CAPTION_ANNO" \
  --data-dir "." \
  --caption-key hierarchical_caption \
  --real-dir "$ROUNDTRIP_MC135" \
  --pred-manifest "$MC_MANIFEST" \
  --out-dir "$RUN_ROOT/motionclip_no_l2" \
  --min-frames 1 \
  --max-frames 300 \
  --no-l2-normalize \
  2>&1 | tee "$LOGS/${METHOD_SLUG}_motionclip_no_l2.log"

python3 - "$RUN_ROOT" "$METHOD_SLUG" "$METHOD_LABEL" "$VERSION_LABEL" "$MS272_DIR" "$MC_PRED" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
slug = sys.argv[2]
label = sys.argv[3]
version = sys.argv[4]
ms272_dir = sys.argv[5]
mc_pred = sys.argv[6]
results = run_root / "results"

ms = json.loads((results / f"{slug}_motionstreamer272_raw_refk.json").read_text())
mc_summary = json.loads((run_root / "motionclip_no_l2" / "summary.json").read_text())
mc = mc_summary[label]
phys = json.loads((results / f"{slug}_physics.json").read_text())
pred = ms["pred"]

summary = {
    "method": label,
    "version": version,
    "samples": int(mc["samples"]),
    "ms_r1": pred["r_precision"][0],
    "ms_r2": pred["r_precision"][1],
    "ms_r3": pred["r_precision"][2],
    "ms_fid": pred["fid_vs_gt_native"],
    "ms_mm": pred["matching_score"],
    "ms_div": pred["diversity"],
    "ms_fid_refk": pred.get("fid_vs_gt_refk"),
    "mc_r1": mc["r_precision_pred"][0],
    "mc_r2": mc["r_precision_pred"][1],
    "mc_r3": mc["r_precision_pred"][2],
    "mc_fid": mc["fid_mean"],
    "mc_mm": mc["mm_dist_pred_mean"],
    "mc_div": mc["diversity_pred_mean"],
    "slide": phys["table"]["Slide"],
    "float": phys["table"]["Float"],
    "jitter": phys["table"]["Jitter"],
    "dynamic": phys["table"]["Dynamic"],
    "penet": phys["table"].get("Penet", 0.0),
    "paths": {
        "ms272": ms272_dir,
        "motionclip135": mc_pred,
    },
    "protocol": {
        "caption": "humanml3d_official_corrected / motionclip-selected full-clip captions",
        "motionstreamer_text_dir": "outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected/texts",
        "motionclip_reference": "outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/motionclip_gt_hml263_roundtrip_20260623/motionclip135/roundtrip",
        "motionclip_l2_normalize": False,
    },
}
(run_root / "summary.json").write_text(json.dumps(summary, indent=2))

header = [
    "method", "version", "N",
    "MS_R1", "MS_R2", "MS_R3", "MS_FID", "MS_MM", "MS_Div", "MS_FID_refk",
    "MC_R1", "MC_R2", "MC_R3", "MC_FID", "MC_MM", "MC_Div",
    "Slide", "Float", "Jitter", "Dynamic",
]
values = [
    label, version, summary["samples"],
    summary["ms_r1"], summary["ms_r2"], summary["ms_r3"], summary["ms_fid"],
    summary["ms_mm"], summary["ms_div"], summary["ms_fid_refk"],
    summary["mc_r1"], summary["mc_r2"], summary["mc_r3"], summary["mc_fid"],
    summary["mc_mm"], summary["mc_div"],
    summary["slide"], summary["float"], summary["jitter"], summary["dynamic"],
]
(run_root / "summary.tsv").write_text(
    "\t".join(header) + "\n" + "\t".join(str(x) for x in values) + "\n"
)
print((run_root / "summary.tsv").read_text())
PY

echo "[done] GoToZero train-only metrics $(date -Is)"
