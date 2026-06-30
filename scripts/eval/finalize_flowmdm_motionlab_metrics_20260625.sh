#!/usr/bin/env bash
# Finish FlowMDM / MotionLab metrics after the core MS272 + physics metrics exist.
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
RUN_ROOT="${RUN_ROOT:-$BASE/ms272/_suites/flowmdm_motionlab_clean_20260625_metrics}"
RESULTS="$RUN_ROOT/results"
LOGS="$RUN_ROOT/logs"
MC_PRED_ROOT="$RUN_ROOT/motionclip135"
MC_MANIFEST="$RUN_ROOT/motionclip_manifest.tsv"

ROUNDTRIP_MC135="$BASE/ms272/_suites/motionclip_gt_hml263_roundtrip_20260623/motionclip135/roundtrip"
CAPTION_ANNO="$BASE/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json"

mkdir -p "$LOGS" "$RUN_ROOT/motionclip"
printf "FlowMDM\t%s\nMotionLab\t%s\n" "$MC_PRED_ROOT/flowmdm" "$MC_PRED_ROOT/motionlab" > "$MC_MANIFEST"

python3 scripts/eval/eval_motionclip_table1_dirs.py \
  --anno-file "$CAPTION_ANNO" \
  --data-dir "." \
  --caption-key hierarchical_caption \
  --real-dir "$ROUNDTRIP_MC135" \
  --pred-manifest "$MC_MANIFEST" \
  --out-dir "$RUN_ROOT/motionclip" \
  --min-frames 1 \
  --max-frames 300 \
  --no-l2-normalize \
  2>&1 | tee "$LOGS/motionclip_hmlroundtrip.log"

python3 - <<'PY'
import json
from pathlib import Path

base = Path("outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/flowmdm_motionlab_clean_20260625_metrics")
results = base / "results"
mc_summary = json.loads((base / "motionclip" / "summary.json").read_text())

labels = {"flowmdm": "FlowMDM", "motionlab": "MotionLab"}
rows = []
for method, label in labels.items():
    ms = json.loads((results / f"{method}_motionstreamer272_hmlroundtrip.json").read_text())
    raw = json.loads((results / f"{method}_motionstreamer272_raw_refk.json").read_text())
    phys = json.loads((results / f"{method}_physics.json").read_text())
    mc = mc_summary[label]
    pred = ms["pred"]
    pred_raw = raw["pred"]
    rows.append({
        "method": label,
        "samples": int(mc["samples"]),
        "motion135_dir": f"outputs/evaluation/t2m/humanml3d_official_test/motion135/{method}",
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
        "motionclip_hmlroundtrip": {
            "R1": mc["r_precision_pred"][0],
            "R2": mc["r_precision_pred"][1],
            "R3": mc["r_precision_pred"][2],
            "FID": mc["fid_mean"],
            "MM": mc["mm_dist_pred_mean"],
            "Diversity": mc["diversity_pred_mean"],
        },
        "physics": phys["table"],
    })

summary = {
    "protocol": {
        "caption": "humanml3d_official_corrected",
        "semantic_reference": "GT SMPL -> HML263 -> SMPL roundtrip",
        "motionstreamer_reference": "outputs/evaluation/t2m/humanml3d_official_test/_runs/noncanonical_legacy_20260623/ms272/gt_hml263_roundtrip_20260623_rootfix/predictions/ms272",
        "motionclip_reference": "outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/motionclip_gt_hml263_roundtrip_20260623/motionclip135/roundtrip",
    },
    "rows": rows,
}
(base / "summary.json").write_text(json.dumps(summary, indent=2))

header = [
    "method", "N",
    "MS_R1", "MS_R2", "MS_R3", "MS_FID", "MS_MM", "MS_Div",
    "MC_R1", "MC_R2", "MC_R3", "MC_FID", "MC_MM", "MC_Div",
    "Slide", "Float", "Jitter", "Dynamic",
]
lines = ["\t".join(header)]
for row in rows:
    ms = row["ms_hmlroundtrip"]
    mc = row["motionclip_hmlroundtrip"]
    ph = row["physics"]
    vals = [
        row["method"], row["samples"],
        ms["R1"], ms["R2"], ms["R3"], ms["FID"], ms["MM"], ms["Diversity"],
        mc["R1"], mc["R2"], mc["R3"], mc["FID"], mc["MM"], mc["Diversity"],
        ph["Slide"], ph["Float"], ph["Jitter"], ph["Dynamic"],
    ]
    lines.append("\t".join(str(v) for v in vals))
(base / "summary.tsv").write_text("\n".join(lines) + "\n")
print((base / "summary.tsv").read_text())
PY

echo "[done] finalized FlowMDM/MotionLab metrics $(date -Is)"
