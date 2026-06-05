#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

LOG_DIR="outputs/evaluation/humanml3d/mdm/_logs"
mkdir -p "${LOG_DIR}" outputs/evaluation/humanml3d/_monitor

echo "[restart] stopping old MDM IK shards"
pids="$(ps -eo pid,args | awk '/[h]ml263_to_smpl_ik.py/ && /mdm_hml3d263\/humanml3d/ {print $1}')"
if [[ -n "${pids}" ]]; then
  kill -TERM ${pids} || true
  sleep 8
  pids="$(ps -eo pid,args | awk '/[h]ml263_to_smpl_ik.py/ && /mdm_hml3d263\/humanml3d/ {print $1}')"
  if [[ -n "${pids}" ]]; then
    kill -KILL ${pids} || true
  fi
fi

echo "[restart] validating existing MDM npz files"
python3 - <<'PY'
from pathlib import Path

import numpy as np

out_dir = Path("outputs/evaluation/humanml3d/mdm")
bad = []
for path in out_dir.glob("*.npz"):
    try:
        with np.load(path) as item:
            required = ["motion_135", "target_joints", "fitted_joints", "fit_mpjpe_mm"]
            if any(key not in item.files for key in required):
                raise KeyError("missing required key")
            if not all(np.isfinite(item[key]).all() for key in required):
                raise ValueError("non-finite values")
    except Exception:
        bad.append(path)

for path in bad:
    path.unlink()
print(f"[restart] removed_bad={len(bad)}")
PY

echo "[restart] launching MDM IK as 8 shards"
for shard in 0 1 2 3 4 5 6 7; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    python3 scripts/eval/hml263_to_smpl_ik.py \
      --in-dir outputs/evaluation/humanml3d/mdm_hml3d263/humanml3d \
      --out-dir outputs/evaluation/humanml3d/mdm \
      --num-shards 8 \
      --shard-index "${shard}" \
      --device cuda \
      --batch-size 512 \
      --floor-align \
      --refine-iters 80 \
      --refine-lr 0.02 \
      --skip-existing
  ) > "${LOG_DIR}/ik_lzy2_mdm_re8_s${shard}_of_8_gpu${shard}.log" 2>&1 &
done

wait
echo "[restart] all MDM re8 shards finished"
