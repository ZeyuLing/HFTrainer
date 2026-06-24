#!/usr/bin/env bash
# Convert the 30 official HumanML3D short MoGenTS fallback samples
# from HML263 -> SMPL135 -> MotionStreamer 272.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HFTRAINER_SKIP_AUTOREGISTER=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PRED263=${PRED263:-outputs/evaluation/t2m/humanml3d_official_test/hml263/mogents_ts10_cfg4_rescfg5_seed0}
SMPL135=${SMPL135:-outputs/evaluation/t2m/humanml3d_official_test/motion135/mogents_ts10_cfg4_rescfg5_seed0_ik80}
PRED272=${PRED272:-outputs/evaluation/t2m/humanml3d_official_test/ms272/mogents_ts10_cfg4_rescfg5_seed0_ik80}
ID_FILE=${ID_FILE:-$PRED263/_missing_official30.txt}
NUM_GPUS=${NUM_GPUS:-8}

mkdir -p "$SMPL135" "$PRED272/_logs"

python3 - <<'PY' > /tmp/mogents_convert30_missing_deps.txt
import importlib.util
for module, package in [
    ("chumpy", "chumpy>=0.70"),
    ("smplx", "smplx>=0.1.28"),
    ("scipy", "scipy"),
]:
    if importlib.util.find_spec(module) is None:
        print(package)
PY
if [ -s /tmp/mogents_convert30_missing_deps.txt ]; then
  python3 -m pip install -q -i https://mirrors.tencent.com/pypi/simple \
    --trusted-host mirrors.tencent.com \
    $(tr '\n' ' ' < /tmp/mogents_convert30_missing_deps.txt)
fi

echo "[convert30] IK start $(date) ids=$ID_FILE gpus=$NUM_GPUS"
pids=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i python3 -u scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$PRED263" \
    --out-dir "$SMPL135" \
    --ids "$ID_FILE" \
    --model-dir ref_repo/MDM/body_models \
    --source-fps 20 \
    --target-fps 30 \
    --floor-align \
    --refine-iters 80 \
    --refine-lr 0.02 \
    --num-shards "$NUM_GPUS" \
    --shard-index "$i" \
    --device cuda \
    > "$PRED272/_logs/debug_convert30_ik_shard_${i}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do
  wait "$p"
done

echo "[convert30] 272 encode start $(date)"
python3 -u scripts/data/convert_motion135_to_h3d272.py \
  --in-dir "$SMPL135" \
  --out-dir "$PRED272" \
  --workers 16 \
  --skip-existing \
  > "$PRED272/_logs/debug_convert30_272.log" 2>&1

echo "[convert30] length trim/check start $(date)"
python3 - <<'PY'
from pathlib import Path
import json
import numpy as np

pred263 = Path("outputs/evaluation/t2m/humanml3d_official_test/hml263/mogents_ts10_cfg4_rescfg5_seed0")
smpl135 = Path("outputs/evaluation/t2m/humanml3d_official_test/motion135/mogents_ts10_cfg4_rescfg5_seed0_ik80")
pred272 = Path("outputs/evaluation/t2m/humanml3d_official_test/ms272/mogents_ts10_cfg4_rescfg5_seed0_ik80")
ids = [x.strip() for x in (pred263 / "_missing_official30.txt").read_text().splitlines() if x.strip()]
anno = json.loads(Path("data/annotation/test_hml3d_official272_gtlen.json").read_text())["data_list"]


def fit_len(arr, n):
    arr = np.asarray(arr)
    if arr.shape[0] == n:
        return arr
    if arr.shape[0] < 2 or n < 2:
        return np.repeat(arr[:1], n, axis=0)
    grid = np.linspace(0, arr.shape[0] - 1, n)
    lo = np.floor(grid).astype(np.int64)
    hi = np.minimum(lo + 1, arr.shape[0] - 1)
    w = (grid - lo).astype(np.float64)
    shape = (n,) + (1,) * (arr.ndim - 1)
    return arr[lo] * (1 - w.reshape(shape)) + arr[hi] * w.reshape(shape)


for sid in ids:
    target = int(anno[sid]["num_frames"])
    p = smpl135 / f"{sid}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        out = {}
        for k in z.files:
            a = z[k]
            if getattr(a, "ndim", 0) >= 1 and a.shape[0] > 1 and a.shape[0] != target:
                out[k] = fit_len(a, target).astype(a.dtype, copy=False)
            else:
                out[k] = a
        np.savez(p, **out)
    q = pred272 / f"{sid}.npy"
    if q.exists():
        arr = np.load(q)
        if arr.shape[0] != target:
            np.save(q, fit_len(arr, target).astype(np.float32))

missing135 = [sid for sid in ids if not (smpl135 / f"{sid}.npz").exists()]
missing272 = [sid for sid in ids if not (pred272 / f"{sid}.npy").exists()]
print({"missing135": missing135, "missing272": missing272})
if missing135 or missing272:
    raise SystemExit(1)
PY

echo "[convert30] done $(date)"
