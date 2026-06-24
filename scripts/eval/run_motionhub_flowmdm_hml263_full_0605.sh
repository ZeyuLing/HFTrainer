#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_DIR=${OUT_DIR:-outputs/evaluation/motionhub_hml3d263_rewrite_0605/flowmdm}
LOGDIR=${LOGDIR:-outputs/evaluation/motionhub_hml3d263_rewrite_0605/logs_flowmdm}
NUM_SHARDS=${NUM_SHARDS:-16}
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3,4,5,6,7}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "GPUS must contain at least one GPU id" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}" "${LOGDIR}"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/flowmdm_infer_hml3d263.py \
    --anno-file data/annotation/test_motionhub_t2m.json \
    --caption-file data/annotation/test_motionhub_t2m_rewritten.json \
    --data-dir data/motionhub \
    --out-dir "${OUT_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${i}" \
    --device 0 \
    --skip-existing \
    > "${LOGDIR}/shard_${i}_gpu${gpu}.log" 2>&1 &
done
wait

pred_count=$(find "${OUT_DIR}" -maxdepth 1 -name '*.npy' | wc -l)
echo "[infer done] files=${pred_count}"
if [ "${pred_count}" -eq 0 ]; then
  echo "[error] no FlowMDM MotionHub outputs; abort" >&2
  exit 2
fi

python3 - <<PY | tee "${LOGDIR}/summary.txt"
from pathlib import Path
out = Path("${OUT_DIR}")
logs = Path("${LOGDIR}")
print("files", len(list(out.glob("*.npy"))), "out", out)
for p in sorted(logs.glob("shard_*.log")):
    tail = p.read_text(errors="ignore").splitlines()[-5:]
    print(p.name, " | ".join(tail))
PY
