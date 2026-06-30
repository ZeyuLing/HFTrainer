#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
cd "$ROOT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HFTRAINER_SKIP_AUTOREGISTER=1

SUITE=${SUITE:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_epoch43_translation_decode_t2m_20260629}
ANNO=${ANNO:-outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json}
TEXT_DIR=${TEXT_DIR:-outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected/texts}
CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
CHECKPOINT=${CHECKPOINT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_43}
DATA_DIR=${DATA_DIR:-data/motionhub}
SCHEMES=${SCHEMES:-"rollout absolute xz_rollout_y_absolute"}
DECODE_MODE=${DECODE_MODE:-all}
NGPU=${NGPU:-8}
N_SHARDS=${N_SHARDS:-8}
SHARD_BASE=${SHARD_BASE:-0}
STEPS=${STEPS:-50}
CFG=${CFG:-5.0}
KAFS_MODE=${KAFS_MODE:-depth_driven}
PAD_TO_FRAMES=${PAD_TO_FRAMES:-360}
EXPECTED=${EXPECTED:-4042}
RUN_POSTPROCESS=${RUN_POSTPROCESS:-1}
POSTPROCESS_WAIT_SECONDS=${POSTPROCESS_WAIT_SECONDS:-72000}

mkdir -p "$SUITE"/{raw,prep,results,analysis,logs}

cat > "$SUITE/run_config_base${SHARD_BASE}.json" <<JSON
{
  "suite": "$SUITE",
  "annotation": "$ANNO",
  "text_dir": "$TEXT_DIR",
  "config": "$CONFIG",
  "checkpoint": "$CHECKPOINT",
  "schemes": "$SCHEMES",
  "decode_mode": "$DECODE_MODE",
  "ngpu": $NGPU,
  "num_shards": $N_SHARDS,
  "shard_base": $SHARD_BASE,
  "num_inference_steps": $STEPS,
  "guidance_scale": $CFG,
  "kafs_mode": "$KAFS_MODE",
  "min_frames": 1,
  "length_policy": "pad360_crop",
  "pad_to_frames": $PAD_TO_FRAMES,
  "expected_count": $EXPECTED,
  "run_postprocess": "$RUN_POSTPROCESS"
}
JSON

{
  echo "cd $ROOT"
  echo "bash scripts/eval/run_prism_t2m_translation_decode_ablation_20260629.sh"
  echo "SCHEMES=$SCHEMES DECODE_MODE=$DECODE_MODE NGPU=$NGPU N_SHARDS=$N_SHARDS SHARD_BASE=$SHARD_BASE STEPS=$STEPS CFG=$CFG KAFS_MODE=$KAFS_MODE RUN_POSTPROCESS=$RUN_POSTPROCESS"
} > "$SUITE/command_base${SHARD_BASE}.txt"

echo "[driver] start $(date)"
echo "[driver] suite=$SUITE"
echo "[driver] schemes=$SCHEMES"
echo "[driver] decode_mode=$DECODE_MODE"
echo "[driver] ngpu=$NGPU n_shards=$N_SHARDS shard_base=$SHARD_BASE run_postprocess=$RUN_POSTPROCESS"

for gpu in $(seq 0 $((NGPU - 1))); do
  shard_idx=$((SHARD_BASE + gpu))
  if [[ "$shard_idx" -ge "$N_SHARDS" ]]; then
    echo "[driver] skip gpu=$gpu shard=$shard_idx >= n_shards=$N_SHARDS"
    continue
  fi
  (
    set -euo pipefail
    if [[ "$DECODE_MODE" == "all" ]]; then
      log="$SUITE/logs/gen_all_shard${shard_idx}of${N_SHARDS}.log"
      echo "[driver] launch decode=all shard=$shard_idx gpu=$gpu log=$log"
      CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/eval_prism_kafs_ablation.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --anno-file "$ANNO" \
        --data-dir "$DATA_DIR" \
        --skip-motion-existence-check \
        --output-dir "$SUITE/raw" \
        --kafs-mode "$KAFS_MODE" \
        --translation-decode-mode all \
        --length-policy pad360_crop \
        --pad-to-frames "$PAD_TO_FRAMES" \
        --num-inference-steps "$STEPS" \
        --guidance-scale "$CFG" \
        --min-frames 1 \
        --num-shards "$N_SHARDS" \
        --shard-idx "$shard_idx" \
        --skip-existing \
        > "$log" 2>&1
      echo "[driver] done decode=all shard=$shard_idx gpu=$gpu"
    else
      for scheme in $SCHEMES; do
        log="$SUITE/logs/gen_${scheme}_shard${shard_idx}of${N_SHARDS}.log"
        echo "[driver] launch scheme=$scheme shard=$shard_idx gpu=$gpu log=$log"
        CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/eval_prism_kafs_ablation.py \
          --config "$CONFIG" \
          --checkpoint "$CHECKPOINT" \
          --anno-file "$ANNO" \
          --data-dir "$DATA_DIR" \
          --skip-motion-existence-check \
          --output-dir "$SUITE/raw" \
          --out-subdir "$scheme" \
          --kafs-mode "$KAFS_MODE" \
          --translation-decode-mode "$scheme" \
          --length-policy pad360_crop \
          --pad-to-frames "$PAD_TO_FRAMES" \
          --num-inference-steps "$STEPS" \
          --guidance-scale "$CFG" \
          --min-frames 1 \
          --num-shards "$N_SHARDS" \
          --shard-idx "$shard_idx" \
          --skip-existing \
          > "$log" 2>&1
        echo "[driver] done scheme=$scheme shard=$shard_idx gpu=$gpu"
      done
    fi
  ) &
done

wait
echo "[driver] generation done for shard_base=$SHARD_BASE $(date)"
touch "$SUITE/_GEN_DONE_SHARDS_${SHARD_BASE}_${NGPU}"

if [[ "$RUN_POSTPROCESS" != "1" ]]; then
  echo "[driver] RUN_POSTPROCESS=$RUN_POSTPROCESS; worker exits after generation"
  exit 0
fi

for scheme in $SCHEMES; do
  deadline=$((SECONDS + POSTPROCESS_WAIT_SECONDS))
  while true; do
    count=$(find "$SUITE/raw/$scheme" -maxdepth 1 -name '*.npz' | wc -l)
    echo "[driver] raw_count scheme=$scheme count=$count expected=$EXPECTED"
    if [[ "$count" -eq "$EXPECTED" ]]; then
      break
    fi
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "[driver][error] raw coverage timeout for $scheme: $count/$EXPECTED" >&2
      exit 10
    fi
    sleep 120
  done
done

touch "$SUITE/_GEN_DONE"

for scheme in $SCHEMES; do
  echo "[driver] repack scheme=$scheme $(date)"
  python3 scripts/eval/repack_pred_to_272ids.py \
    --npz-dir "$SUITE/raw/$scheme" \
    --anno-file "$ANNO" \
    --id-passthrough \
    --out-dir "$SUITE/prep/$scheme" \
    --workers 16 \
    > "$SUITE/logs/repack_${scheme}.log" 2>&1

  prep_count=$(find "$SUITE/prep/$scheme" -maxdepth 1 -name '*.npz' | wc -l)
  echo "[driver] prep_count scheme=$scheme count=$prep_count expected=$EXPECTED"
  if [[ "$prep_count" -ne "$EXPECTED" ]]; then
    echo "[driver][error] prep coverage mismatch for $scheme: $prep_count/$EXPECTED" >&2
    exit 11
  fi

  echo "[driver] evaluator scheme=$scheme $(date)"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$SUITE/prep/$scheme" \
    --tag "prism_epoch43_t2m_${scheme}_pad360_corrected_caption" \
    --also-refk \
    --min-motion-len 1 \
    --text-dir "$TEXT_DIR" \
    --out-json "$SUITE/results/$scheme.json" \
    > "$SUITE/logs/eval_${scheme}.log" 2>&1

  echo "[driver] height drift scheme=$scheme $(date)"
  CUDA_VISIBLE_DEVICES= python3 scripts/eval/compute_motion135_height_drift.py \
    --m135-dir "$SUITE/prep/$scheme" \
    --anno-file "$ANNO" \
    --out-json "$SUITE/analysis/height_drift_${scheme}.json" \
    > "$SUITE/logs/height_drift_${scheme}.log" 2>&1
done

python3 - <<'PY'
import json
from pathlib import Path

suite = Path("outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_epoch43_translation_decode_t2m_20260629")
schemes = ["rollout", "absolute", "xz_rollout_y_absolute"]
summary = {}
for scheme in schemes:
    row = {}
    metric_path = suite / "results" / f"{scheme}.json"
    drift_path = suite / "analysis" / f"height_drift_{scheme}.json"
    if metric_path.exists():
        metric = json.loads(metric_path.read_text())
        row["coverage_ids"] = metric.get("ids_with_required_files")
        pred = metric.get("pred", {})
        row["fid_vs_gt_native"] = pred.get("fid_vs_gt_native")
        row["fid_vs_gt_refk"] = pred.get("fid_vs_gt_refk")
        row["r_precision"] = pred.get("r_precision")
        row["matching_score"] = pred.get("matching_score")
        row["diversity"] = pred.get("diversity")
    if drift_path.exists():
        drift = json.loads(drift_path.read_text())
        row["drift_coverage"] = drift.get("coverage")
        row["root_y_abs_drift"] = drift.get("root_y_abs_drift")
        row["foot_min_y_abs_drift"] = drift.get("foot_min_y_abs_drift")
        row["root_y_signed_drift_mean_cm"] = drift.get("root_y_signed_drift_mean_cm")
        row["foot_min_y_signed_drift_mean_cm"] = drift.get("foot_min_y_signed_drift_mean_cm")
    summary[scheme] = row
(suite / "summary_translation_decode.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

touch "$SUITE/_EVAL_DONE"
echo "[driver] all done $(date)"
