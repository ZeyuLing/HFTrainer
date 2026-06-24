#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_DIR=${OUT_DIR:-outputs/evaluation/motionhub_hml3d263_rewrite_0605/motionlab}
LOGDIR=${LOGDIR:-outputs/evaluation/motionhub_hml3d263_rewrite_0605/logs_motionlab}
NUM_SHARDS=${NUM_SHARDS:-8}
mkdir -p "${OUT_DIR}" "${LOGDIR}"

python3 -c "import rotary_embedding_torch, roma" || \
  python3 -m pip install --user rotary-embedding-torch==0.8.5 roma==1.5.1

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="${i}" python3 scripts/eval/motionlab_infer_hml3d263.py \
    --anno-file data/annotation/test_motionhub_t2m.json \
    --caption-file data/annotation/test_motionhub_t2m_rewritten.json \
    --data-dir data/motionhub \
    --out-dir "${OUT_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${i}" \
    --stage demo \
    --batch-size "${BATCH_SIZE:-8}" \
    --device cuda \
    --skip-existing \
    > "${LOGDIR}/shard_${i}.log" 2>&1 &
done
wait

pred_count=$(find "${OUT_DIR}" -maxdepth 1 -name '*.npy' | wc -l)
echo "[infer done] files=${pred_count}"
if [ "${pred_count}" -eq 0 ]; then
  echo "[error] no MotionLab MotionHub outputs; abort" >&2
  exit 2
fi

python3 - <<PY | tee "${LOGDIR}/summary.txt"
from pathlib import Path
out = Path("${OUT_DIR}")
logs = Path("${LOGDIR}")
print("files", len(list(out.glob("*.npy"))), "out", out)
for p in sorted(logs.glob("shard_*.log")):
    tail = p.read_text(errors="ignore").splitlines()[-3:]
    print(p.name, " | ".join(tail))
PY
