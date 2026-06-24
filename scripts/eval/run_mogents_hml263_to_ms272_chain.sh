#!/usr/bin/env bash
# MoGenTS HumanML3D-263 -> SMPL motion_135 -> MotionStreamer-272 bridge/eval.
#
# This script is restartable: all conversion stages use --skip-existing, so it
# can resume after preemption or reuse partial local/Taiji outputs.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false

PRED263=${PRED263:-outputs/evaluation/t2m/humanml3d_official_test/hml263/mogents_ts10_cfg4_rescfg5_seed0}
SMPL135=${SMPL135:-outputs/evaluation/t2m/humanml3d_official_test/motion135/mogents_ts10_cfg4_rescfg5_seed0_ik80}
PRED272=${PRED272:-outputs/evaluation/t2m/humanml3d_official_test/ms272/mogents_ts10_cfg4_rescfg5_seed0_ik80}
METRICS=${METRICS:-$PRED272/metrics}
LOGDIR=${LOGDIR:-$PRED272/_logs}
N_REPEATS=${N_REPEATS:-20}
CONVERT_WORKERS=${CONVERT_WORKERS:-8}

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(python3 - <<'PY'
import torch
print(max(1, torch.cuda.device_count()))
PY
)
fi
NUM_SHARDS=${NUM_SHARDS:-$NUM_GPUS}

mkdir -p "$SMPL135" "$PRED272" "$METRICS" "$LOGDIR"

python3 - <<'PY' > /tmp/mogents_ms272_missing_deps.txt
import importlib.util
checks = [
    ("mmengine", "mmengine>=0.7"),
    ("safetensors", "safetensors"),
    ("transformers", "transformers"),
    ("smplx", "smplx>=0.1.28"),
    ("chumpy", "chumpy>=0.70"),
    ("scipy", "scipy"),
]
for module, package in checks:
    if importlib.util.find_spec(module) is None:
        print(package)
PY
if [ -s /tmp/mogents_ms272_missing_deps.txt ]; then
  python3 -m pip install -q -i https://mirrors.tencent.com/pypi/simple --trusted-host mirrors.tencent.com \
    $(tr '\n' ' ' < /tmp/mogents_ms272_missing_deps.txt)
fi

cat > "$PRED272/run_config.json" <<EOF
{
  "pred263": "$PRED263",
  "smpl135": "$SMPL135",
  "pred272": "$PRED272",
  "metrics": "$METRICS",
  "num_gpus": $NUM_GPUS,
  "num_shards": $NUM_SHARDS,
  "n_repeats": $N_REPEATS,
  "bridge": "hml263_to_smpl135_ik_refine80_to_ms272"
}
EOF

printf '%s\n' \
  "PRED263=$PRED263" \
  "SMPL135=$SMPL135" \
  "PRED272=$PRED272" \
  "METRICS=$METRICS" \
  "NUM_GPUS=$NUM_GPUS" \
  "NUM_SHARDS=$NUM_SHARDS" \
  "N_REPEATS=$N_REPEATS" \
  > "$PRED272/command.txt"

echo "[mogents-ms272] Stage A HML263 -> SMPL135 IK refine-80 shards=$NUM_SHARDS gpus=$NUM_GPUS $(date)"
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((i % NUM_GPUS))
  CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$PRED263" \
    --out-dir "$SMPL135" \
    --model-dir ref_repo/MDM/body_models \
    --source-fps 20 \
    --target-fps 30 \
    --floor-align \
    --refine-iters 80 \
    --refine-lr 0.02 \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$i" \
    --device cuda \
    --skip-existing \
    > "$LOGDIR/ik_shard_${i}_of_${NUM_SHARDS}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do
  wait "$p"
done
n135=$(find "$SMPL135" -maxdepth 1 -name '*.npz' | wc -l)
echo "[mogents-ms272] smpl135=$n135"
test "$n135" -gt 0

echo "[mogents-ms272] Stage B SMPL135 -> MS272 $(date)"
python3 -u scripts/data/convert_motion135_to_h3d272.py \
  --in-dir "$SMPL135" \
  --out-dir "$PRED272" \
  --workers "$CONVERT_WORKERS" \
  --skip-existing \
  > "$LOGDIR/convert_272.log" 2>&1
n272=$(find "$PRED272" -maxdepth 1 -name '*.npy' | wc -l)
echo "[mogents-ms272] pred272=$n272"
test "$n272" -gt 0

echo "[mogents-ms272] Stage C MotionStreamer-272 evaluator $(date)"
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/verify_evaluators.py \
  --which ms272 \
  --ms272-pred "$PRED272" \
  --n-repeats "$N_REPEATS" \
  --out-dir "$METRICS" \
  > "$LOGDIR/ms272_eval.log" 2>&1

python3 - "$PRED263" "$SMPL135" "$PRED272" "$METRICS" "$NUM_SHARDS" "$N_REPEATS" <<'PY'
import json
import sys
from pathlib import Path

pred263, smpl135, pred272, metrics = map(Path, sys.argv[1:5])
summary = {
    "method": "MoGenTS",
    "bridge": "hml263_to_smpl135_ik_refine80_to_ms272",
    "pred263": str(pred263),
    "smpl135": str(smpl135),
    "pred272": str(pred272),
    "num_shards": int(sys.argv[5]),
    "n_repeats": int(sys.argv[6]),
    "n_pred263": len(list(pred263.glob("*.npy"))),
    "n_smpl135": len(list(smpl135.glob("*.npz"))),
    "n_pred272": len(list(pred272.glob("*.npy"))),
    "metric_json_ms272": str(metrics / "verify_ms272.json"),
}
(metrics / "run_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "[mogents-ms272] done $(date)"
