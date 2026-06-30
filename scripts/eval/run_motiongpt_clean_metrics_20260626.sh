#!/usr/bin/env bash
# Metrics for framework-native MotionGPT on HumanML3D official test.
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
RUN_ROOT="${RUN_ROOT:-$BASE/ms272/_suites/motiongpt_framework_native_20260626_metrics}"
RESULTS="$RUN_ROOT/results"
LOGS="$RUN_ROOT/logs"
MC_PRED_ROOT="$RUN_ROOT/motionclip135"
MC_MANIFEST="$RUN_ROOT/motionclip_manifest.tsv"

MOTION135="$BASE/motion135/motiongpt"
MS272="$BASE/ms272/motiongpt"
ROUNDTRIP_MS272="$BASE/_runs/noncanonical_legacy_20260623/ms272/gt_hml263_roundtrip_20260623_rootfix/predictions/ms272"
ROUNDTRIP_MC135="$BASE/ms272/_suites/motionclip_gt_hml263_roundtrip_20260623/motionclip135/roundtrip"
CAPTION_TEXTS="$BASE/captions/humanml3d_official_corrected/texts"
CAPTION_ANNO="$BASE/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json"
MC_PRED="$MC_PRED_ROOT/motiongpt"

mkdir -p "$RESULTS" "$LOGS" "$MC_PRED"

echo "[start] MotionGPT clean metrics $(date -Is)"
echo "[paths] motion135=$MOTION135"
echo "[paths] ms272=$MS272"
echo "[paths] roundtrip_ms272=$ROUNDTRIP_MS272"
echo "[paths] roundtrip_motionclip135=$ROUNDTRIP_MC135"
echo "[paths] captions=$CAPTION_ANNO"

if [[ ! -d "$MOTION135" ]]; then
  echo "[error] missing motion135 dir: $MOTION135" >&2
  exit 2
fi

python3 scripts/eval/eval_motionstreamer_272.py \
  --pred-dir "$MOTION135" \
  --gt-272-dir "$ROUNDTRIP_MS272" \
  --tag motiongpt_framework_hmlroundtrip \
  --text-dir "$CAPTION_TEXTS" \
  --min-motion-len 1 \
  --out-json "$RESULTS/motiongpt_motionstreamer272_hmlroundtrip.json" \
  2>&1 | tee "$LOGS/motionstreamer272_hmlroundtrip.log"

python3 scripts/eval/eval_motionstreamer_272.py \
  --pred-dir "$MOTION135" \
  --tag motiongpt_framework_raw_refk \
  --real-encoding refk \
  --also-refk \
  --text-dir "$CAPTION_TEXTS" \
  --min-motion-len 1 \
  --out-json "$RESULTS/motiongpt_motionstreamer272_raw_refk.json" \
  2>&1 | tee "$LOGS/motionstreamer272_raw_refk.log"

python3 scripts/eval/eval_mbench_physics_dir.py \
  --src "$MOTION135" \
  --mode m135 \
  --workers "${PHYS_WORKERS:-32}" \
  --out-json "$RESULTS/motiongpt_physics.json" \
  2>&1 | tee "$LOGS/physics.log"

python3 scripts/eval/convert_row135_npz_to_motionclip_col.py \
  --anno-file "$CAPTION_ANNO" \
  --src-dir "$MOTION135" \
  --out-dir "$MC_PRED" \
  --overwrite \
  2>&1 | tee "$LOGS/convert_motionclip135.log"

printf "MotionGPT\t%s\n" "$MC_PRED" > "$MC_MANIFEST"

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
  2>&1 | tee "$LOGS/motionclip_no_l2_hmlroundtrip.log"

python3 - <<'PY'
import json
from pathlib import Path

base = Path("outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/motiongpt_framework_native_20260626_metrics")
results = base / "results"
mc_summary = json.loads((base / "motionclip_no_l2" / "summary.json").read_text())

ms = json.loads((results / "motiongpt_motionstreamer272_hmlroundtrip.json").read_text())
raw = json.loads((results / "motiongpt_motionstreamer272_raw_refk.json").read_text())
phys = json.loads((results / "motiongpt_physics.json").read_text())
mc = mc_summary["MotionGPT"]
pred = ms["pred"]
pred_raw = raw["pred"]

row = {
    "method": "MotionGPT",
    "samples": int(mc["samples"]),
    "motion135_dir": "outputs/evaluation/t2m/humanml3d_official_test/motion135/motiongpt",
    "ms_hmlroundtrip": {
        "R1": pred["r_precision"][0],
        "R2": pred["r_precision"][1],
        "R3": pred["r_precision"][2],
        "FID": pred["fid_vs_gt_native"],
        "MM": pred["matching_score"],
        "Diversity": pred["diversity"],
    },
    "ms_raw_refk": {
        "R1": pred_raw["r_precision"][0],
        "R2": pred_raw["r_precision"][1],
        "R3": pred_raw["r_precision"][2],
        "FID_refk": pred_raw.get("fid_vs_gt_refk"),
        "MM": pred_raw["matching_score"],
        "Diversity": pred_raw["diversity"],
    },
    "motionclip_no_l2_hmlroundtrip": {
        "R1": mc["r_precision_pred"][0],
        "R2": mc["r_precision_pred"][1],
        "R3": mc["r_precision_pred"][2],
        "FID": mc["fid_mean"],
        "MM": mc["mm_dist_pred_mean"],
        "Diversity": mc["diversity_pred_mean"],
    },
    "physics": phys["table"],
}

summary = {
    "protocol": {
        "caption": "humanml3d_official_corrected",
        "semantic_reference": "GT SMPL -> HML263 -> SMPL roundtrip",
        "motionstreamer_reference": "outputs/evaluation/t2m/humanml3d_official_test/_runs/noncanonical_legacy_20260623/ms272/gt_hml263_roundtrip_20260623_rootfix/predictions/ms272",
        "motionclip_reference": "outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/motionclip_gt_hml263_roundtrip_20260623/motionclip135/roundtrip",
        "motionclip_l2_normalize": False,
    },
    "rows": [row],
}
(base / "summary.json").write_text(json.dumps(summary, indent=2))

header = [
    "method", "N",
    "MS_R1", "MS_R2", "MS_R3", "MS_FID", "MS_MM", "MS_Div",
    "MS_raw_FID_refk",
    "MC_R1", "MC_R2", "MC_R3", "MC_FID", "MC_MM", "MC_Div",
    "Slide", "Float", "Jitter", "Dynamic",
]
ms = row["ms_hmlroundtrip"]
raw = row["ms_raw_refk"]
mc = row["motionclip_no_l2_hmlroundtrip"]
ph = row["physics"]
vals = [
    row["method"], row["samples"],
    ms["R1"], ms["R2"], ms["R3"], ms["FID"], ms["MM"], ms["Diversity"],
    raw["FID_refk"],
    mc["R1"], mc["R2"], mc["R3"], mc["FID"], mc["MM"], mc["Diversity"],
    ph["Slide"], ph["Float"], ph["Jitter"], ph["Dynamic"],
]
(base / "summary.tsv").write_text("\t".join(header) + "\n" + "\t".join(str(v) for v in vals) + "\n")
print((base / "summary.tsv").read_text())
PY

echo "[done] MotionGPT clean metrics $(date -Is)"
