#!/usr/bin/env bash
# Metrics for framework-native MLD / MotionLCM HumanML3D runs.
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
RUN_ROOT="${RUN_ROOT:-$BASE/ms272/_suites/framework_native_mld_motionlcm_20260625_metrics}"
RESULTS="$RUN_ROOT/results"
LOGS="$RUN_ROOT/logs"
MC_PRED_ROOT="$RUN_ROOT/motionclip135"
MC_MANIFEST="$RUN_ROOT/motionclip_manifest.tsv"

ROUNDTRIP_MS272="$BASE/_runs/noncanonical_legacy_20260623/ms272/gt_hml263_roundtrip_20260623_rootfix/predictions/ms272"
ROUNDTRIP_MC135="$BASE/ms272/_suites/motionclip_gt_hml263_roundtrip_20260623/motionclip135/roundtrip"
CAPTION_TEXTS="$BASE/captions/humanml3d_official_corrected/texts"
CAPTION_ANNO="$BASE/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json"

METHODS=(mld motionlcm)

label_for() {
  case "$1" in
    mld) echo "MLD" ;;
    motionlcm) echo "MotionLCM" ;;
    *) echo "$1" ;;
  esac
}

mkdir -p "$RESULTS" "$LOGS" "$MC_PRED_ROOT"

DEPS_STAMP="$RUN_ROOT/_deps_ok_$(hostname).stamp"
if [ ! -f "$DEPS_STAMP" ]; then
  python3 - <<'PY' > /tmp/mld_motionlcm_metrics_missing_deps.txt
mods = {
    "mmengine": "mmengine>=0.10",
    "smplx": "smplx>=0.1.28",
    "chumpy": "chumpy>=0.70",
    "scipy": "scipy",
    "tqdm": "tqdm",
    "einops": "einops",
    "sentence_transformers": "sentence-transformers",
    "transformers": "transformers",
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
  if [ -s /tmp/mld_motionlcm_metrics_missing_deps.txt ]; then
    echo "[deps] installing: $(tr '\n' ' ' < /tmp/mld_motionlcm_metrics_missing_deps.txt)"
    python3 -m pip install -q -i https://mirrors.tencent.com/pypi/simple --trusted-host mirrors.tencent.com -r /tmp/mld_motionlcm_metrics_missing_deps.txt
  else
    echo "[deps] all required python packages already importable"
  fi
  touch "$DEPS_STAMP"
fi

echo "[start] framework-native MLD/MotionLCM metrics $(date -Is)"
echo "[paths] run_root=$RUN_ROOT"
echo "[paths] roundtrip_ms272=$ROUNDTRIP_MS272"
echo "[paths] roundtrip_motionclip135=$ROUNDTRIP_MC135"
echo "[paths] captions=$CAPTION_ANNO"

for method in "${METHODS[@]}"; do
  motion135="$BASE/motion135/$method"
  if [[ ! -d "$motion135" ]]; then
    echo "[error] missing motion135 dir for $method: $motion135" >&2
    exit 2
  fi
  count=$(find "$motion135" -maxdepth 1 -type f -name '*.npz' | wc -l)
  echo "[method] $method motion135=$motion135 count=$count"
  if [[ "$count" -lt 4042 ]]; then
    echo "[error] incomplete motion135 dir for $method: count=$count expected=4042" >&2
    exit 3
  fi

  python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$motion135" \
    --gt-272-dir "$ROUNDTRIP_MS272" \
    --tag "${method}_framework_hmlroundtrip" \
    --text-dir "$CAPTION_TEXTS" \
    --min-motion-len 1 \
    --out-json "$RESULTS/${method}_motionstreamer272_hmlroundtrip.json" \
    2>&1 | tee "$LOGS/${method}_motionstreamer272_hmlroundtrip.log"

  python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$motion135" \
    --tag "${method}_framework_raw_refk" \
    --real-encoding refk \
    --also-refk \
    --text-dir "$CAPTION_TEXTS" \
    --min-motion-len 1 \
    --out-json "$RESULTS/${method}_motionstreamer272_raw_refk.json" \
    2>&1 | tee "$LOGS/${method}_motionstreamer272_raw_refk.log"

  python3 scripts/eval/eval_mbench_physics_dir.py \
    --src "$motion135" \
    --mode m135 \
    --workers "${PHYS_WORKERS:-32}" \
    --out-json "$RESULTS/${method}_physics.json" \
    2>&1 | tee "$LOGS/${method}_physics.log"

  mc_pred="$MC_PRED_ROOT/$method"
  mkdir -p "$mc_pred"
  python3 scripts/eval/convert_row135_npz_to_motionclip_col.py \
    --anno-file "$CAPTION_ANNO" \
    --src-dir "$motion135" \
    --out-dir "$mc_pred" \
    --overwrite \
    2>&1 | tee "$LOGS/${method}_convert_motionclip135.log"
done

: > "$MC_MANIFEST"
for method in "${METHODS[@]}"; do
  printf "%s\t%s\n" "$(label_for "$method")" "$MC_PRED_ROOT/$method" >> "$MC_MANIFEST"
done

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

RUN_ROOT="$RUN_ROOT" python3 - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["RUN_ROOT"])
results = base / "results"
mc_summary = json.loads((base / "motionclip_no_l2" / "summary.json").read_text())

labels = {
    "mld": "MLD",
    "motionlcm": "MotionLCM",
}
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
        "motionclip_no_l2_hmlroundtrip": {
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
        "motionclip_l2_normalize": False,
    },
    "rows": rows,
}
(base / "summary.json").write_text(json.dumps(summary, indent=2))

header = [
    "method", "N",
    "MS_R1", "MS_R2", "MS_R3", "MS_FID", "MS_MM", "MS_Div",
    "MS_raw_FID_refk",
    "MC_R1", "MC_R2", "MC_R3", "MC_FID", "MC_MM", "MC_Div",
    "Slide", "Float", "Jitter", "Dynamic",
]
lines = ["\t".join(header)]
for row in rows:
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
    lines.append("\t".join(str(v) for v in vals))
(base / "summary.tsv").write_text("\n".join(lines) + "\n")
print((base / "summary.tsv").read_text())
PY

echo "[done] framework-native MLD/MotionLCM metrics $(date -Is)"
