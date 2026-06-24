#!/usr/bin/env bash
# Re-evaluate PRISM Table 1 with the MotionCLIP evaluator on the official
# HumanML3D-272 test ids.  Inputs are existing Table-1 MotionStreamer-272
# predictions; outputs are MotionCLIP135 conversions plus metrics/compare TSVs.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

OUT_ROOT="${OUT_ROOT:-outputs/evaluation/t2m/humanml3d_official_test/motionclip_table1_20260619}"
ANNO="${ANNO:-data/annotation/test_hml3d_official272_gtlen.json}"
DATA_DIR="${DATA_DIR:-.}"
WORKERS="${WORKERS:-32}"
GPU="${GPU:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-32}"
N_REPEATS="${N_REPEATS:-20}"

MC_ROOT="$OUT_ROOT/motionclip135"
LOG="$OUT_ROOT/logs"
RES="$OUT_ROOT/eval"
mkdir -p "$MC_ROOT" "$LOG" "$RES"

declare -A SRC=(
  [real]="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data"
  [motiongpt3]="outputs/evaluation/t2m/humanml3d_official_test/motion135/motiongpt3_official/predictions/motion135"
  [mld]="outputs/evaluation/t2m/humanml3d_official_test/motion135/mld_official/predictions/motion135"
  [momask]="outputs/evaluation/t2m/humanml3d_official_test/motion135/momask_official/predictions/motion135"
  [mogents]="outputs/evaluation/t2m/humanml3d_official_test/motion135/mogents_ts10_cfg4_rescfg5_seed0_ik80"
  [mdm]="outputs/evaluation/t2m/humanml3d_official_test/motion135/mdm_official/predictions/motion135"
  [t2mgpt]="outputs/evaluation/t2m/humanml3d_official_test/motion135/t2mgpt_official/predictions/motion135"
  [flowmdm]="outputs/evaluation/t2m/humanml3d_official_test/motion135/flowmdm_official/predictions/motion135"
  [motionlab]="outputs/evaluation/t2m/humanml3d_official_test/motion135/motionlab_official/predictions/motion135"
  [kimodo]="outputs/evaluation/t2m/humanml3d_official_test/motion135/kimodo_official/predictions/motion135"
  [vimogen]="outputs/evaluation/ms272_tables_h3d_0607/prep/vimogen"
  # HYMotion raw files are 135D SMPL row-convention 6D.  The directory name is
  # historical; these files are not safe to feed to the MotionCLIP evaluator
  # without the HY-specific row->matrix->column conversion below.
  [hymotion_1b]="outputs/evaluation/t2m/humanml3d_official_test/ms272/hymotion_1b_exactlen_0617_vermo/h3d/motionclip135"
  [gotozero]="outputs/evaluation/t2m/humanml3d_official_test/motion135/gotozero_official/predictions/motion135"
  [motionstreamer]="outputs/evaluation/t2m/humanml3d_official_test/ms272/motionstreamer_exactlen_0617_vermo/prep"
  [ours_best_live]="outputs/evaluation/t2m/humanml3d_official_test/ms272/prism_epoch31_smooth_reseed_badcases_20260618/best_of_ms_eval_live/prep/ours_best_ms_eval"
)

# These inputs contain raw 135D SMPL motions.  They still need an explicit
# convention conversion before MotionCLIP evaluation.
declare -A HYMOTION_RAW135=(
  [hymotion_1b]=1
)

declare -A ROW135_NPZ=(
  [motiongpt3]=1
  [mld]=1
  [momask]=1
  [mogents]=1
  [mdm]=1
  [t2mgpt]=1
  [flowmdm]=1
  [motionlab]=1
  [kimodo]=1
  [vimogen]=1
  [gotozero]=1
  [motionstreamer]=1
  [ours_best_live]=1
)

# Inputs already verified to be annotation-keyed MotionCLIP-compatible 135D.
declare -A DIRECT_135=()

ORDER=(
  real
  motiongpt3 mld momask mogents mdm t2mgpt flowmdm motionlab
  kimodo vimogen hymotion_1b gotozero motionstreamer ours_best_live
)

convert_one() {
  local name="$1"
  local src="${SRC[$name]}"
  local out="$MC_ROOT/$name"
  if [[ ! -d "$src" ]]; then
    echo "[skip-convert] $name missing src=$src" | tee -a "$LOG/run.log"
    return 0
  fi
  if [[ "${HYMOTION_RAW135[$name]:-0}" == "1" ]]; then
    mkdir -p "$out"
    if [[ "${FORCE_CONVERT:-0}" != "1" && -s "$out/conversion_summary.json" ]]; then
      n="$(find "$out" -maxdepth 1 -name '*.npy' | wc -l)"
      echo "[skip-convert-ok] $name hy_raw135 files=$n dir=$out" | tee -a "$LOG/run.log"
      return 0
    fi
    echo "[convert-hy-raw135] $name src=$src -> $out" | tee -a "$LOG/run.log"
    python3 scripts/eval/convert_hylite135_to_motionclip_col.py \
      --src-dir "$src" \
      --anno-file "$ANNO" \
      --data-dir "$DATA_DIR" \
      --out-dir "$out" \
      --workers "$WORKERS" \
      --overwrite \
      > "$LOG/convert_${name}.log" 2>&1
    return 0
  fi
  if [[ "${ROW135_NPZ[$name]:-0}" == "1" ]]; then
    mkdir -p "$out"
    if [[ "${FORCE_CONVERT:-0}" != "1" && -s "$out/_convert_row135_to_motionclip_col_summary.json" ]]; then
      n="$(find "$out" -maxdepth 1 -name '*.npy' | wc -l)"
      echo "[skip-convert-ok] $name row135_npz files=$n dir=$out" | tee -a "$LOG/run.log"
      return 0
    fi
    echo "[convert-row135-npz] $name src=$src -> $out" | tee -a "$LOG/run.log"
    python3 scripts/eval/convert_row135_npz_to_motionclip_col.py \
      --src-dir "$src" \
      --anno-file "$ANNO" \
      --out-dir "$out" \
      --overwrite \
      > "$LOG/convert_${name}.log" 2>&1
    return 0
  fi
  if [[ "${DIRECT_135[$name]:-0}" == "1" ]]; then
    echo "[direct-135] $name src=$src" | tee -a "$LOG/run.log"
    return 0
  fi
  mkdir -p "$out"
  if [[ "${FORCE_CONVERT:-0}" != "1" && -s "$out/_convert_ms272_summary.json" ]]; then
    if python3 - "$out/_convert_ms272_summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ok = int(d.get("failed", 1)) == 0 and int(d.get("motionclip_files", 0)) > 0
raise SystemExit(0 if ok else 1)
PY
    then
      n="$(find "$out" -maxdepth 1 -name '*.npy' | wc -l)"
      echo "[skip-convert-ok] $name files=$n dir=$out" | tee -a "$LOG/run.log"
      return 0
    fi
  fi
  local extra=()
  if [[ "${FORCE_CONVERT:-0}" == "1" ]]; then
    extra+=(--overwrite)
  fi
  echo "[convert] $name src=$src -> $out" | tee -a "$LOG/run.log"
  python3 scripts/eval/convert_ms272_dir_for_t2m_eval.py \
    --src-dir "$src" \
    --anno-file "$ANNO" \
    --data-dir "$DATA_DIR" \
    --motionclip-dir "$out" \
    --workers "$WORKERS" \
    "${extra[@]}" \
    > "$LOG/convert_${name}.log" 2>&1
}

echo "[start] $(date -Is) out=$OUT_ROOT" | tee "$LOG/run.log"

for name in "${ORDER[@]}"; do
  convert_one "$name"
done

MANIFEST="$OUT_ROOT/pred_manifest.tsv"
: > "$MANIFEST"
for name in "${ORDER[@]}"; do
  if [[ "${DIRECT_135[$name]:-0}" == "1" ]]; then
    out="${SRC[$name]}"
  else
    out="$MC_ROOT/$name"
  fi
  if [[ -d "$out" ]]; then
    n_npy="$(find "$out" -maxdepth 1 -name '*.npy' | wc -l)"
    n_npz="$(find "$out" -maxdepth 1 -name '*.npz' | wc -l)"
    n=$((n_npy + n_npz))
    echo "[manifest] $name files=$n (npy=$n_npy npz=$n_npz) dir=$out" | tee -a "$LOG/run.log"
    if [[ "$n" -gt 0 ]]; then
      printf "%s\t%s\n" "$name" "$out" >> "$MANIFEST"
    fi
  fi
done

echo "[eval] $(date -Is)" | tee -a "$LOG/run.log"
CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/eval/eval_motionclip_table1_dirs.py \
  --evaluator-ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno-file "$ANNO" \
  --data-dir "$DATA_DIR" \
  --real-dir "$MC_ROOT/real" \
  --pred-manifest "$MANIFEST" \
  --out-dir "$RES" \
  --min-frames 60 \
  --max-frames 300 \
  --chunk-size "$CHUNK_SIZE" \
  --forward-batch-size "$FORWARD_BATCH_SIZE" \
  --n-repeats "$N_REPEATS" \
  --seed 0 \
  > "$LOG/eval_motionclip_table1.log" 2>&1

python3 - <<'PY' "$RES/summary.json" "$OUT_ROOT/compare_vs_ms272.tsv"
import json
import sys

mc = json.load(open(sys.argv[1]))
out = sys.argv[2]

ms = {
    "real": (0.7124496, 0.9059980, 0.0, 15.6634, 27.7278),
    "motiongpt3": (0.0580, 0.1419, 326.4260, 25.6228, 20.8866),
    "mld": (0.4767, 0.7192, 175.1785, 20.8541, 24.3881),
    "momask": (0.5975, 0.8256, 106.5409, 18.5589, 25.2023),
    "mogents": (0.4760, 0.7100, 113.1, 19.47, 25.30),
    "mdm": (0.2098, 0.3299, 267.3709, 24.2454, 21.4669),
    "t2mgpt": (0.5282, 0.7523, 116.8928, 19.3020, 25.5367),
    "flowmdm": (0.1099, 0.2331, 506.7680, 25.4901, 13.1702),
    "motionlab": (0.0988, 0.2248, 575.8519, 25.7751, 11.1444),
    "kimodo": (0.3230, 0.5410, 143.9, 21.71, 25.32),
    "vimogen": (0.2620, 0.4700, 187.6, 22.99, 24.57),
    "hymotion_1b": (0.7593, 0.9246, 14.6784, 15.3085, 27.3526),
    "gotozero": (0.5895, 0.7979, 20.2582, 17.9307, 27.2927),
    "motionstreamer": (0.6303, 0.8498, 11.1830, 16.5810, 27.4637),
    "ours_best_live": (0.7404, 0.9123, 17.7513, 15.7052, 27.4599),
}

lines = [
    "method\tsamples\tmc_R1\tms_R1\td_R1\tmc_R3\tms_R3\td_R3\t"
    "mc_FID\tms_FID\td_FID\tmc_MM\tms_MM\td_MM\tmc_Div\tms_Div\td_Div"
]
for name, row in mc.items():
    if name not in ms:
        continue
    rp = row["r_precision_pred"]
    vals = (rp[0], rp[2], row["fid_mean"], row["mm_dist_pred_mean"], row["diversity_pred_mean"])
    ref = ms[name]
    pieces = [name, str(row["samples"])]
    for a, b in zip(vals, ref):
        pieces.extend([f"{a:.4f}", f"{b:.4f}", f"{a-b:+.4f}"])
    lines.append("\t".join(pieces))

open(out, "w").write("\n".join(lines) + "\n")
print(open(out).read())
PY

touch "$OUT_ROOT/_DONE"
echo "[done] $(date -Is)" | tee -a "$LOG/run.log"
