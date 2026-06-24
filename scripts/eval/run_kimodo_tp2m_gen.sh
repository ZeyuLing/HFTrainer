#!/bin/bash
# KIMODO TP2M (prefix-pose + text) generation on HumanML3D test, multi-host sharded.
# Each host runs NUM_GPUS parallel shards of gen_kimodo_tp2m_positions.py per cond;
# shards are split across (machines x gpus). Outputs SMPL-22 joints to
# outputs/evaluation/kimodo_tp2m/cond{C}/<id>.npy on shared CephFS (--skip-existing,
# so it co-operates with / resumes any partial local run). Repack + eval are run
# afterwards on the head node (run_kimodo_tp2m_eval).
set -uo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "${ROOT}" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "${ROOT}"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=${PY:-python3}

OUT=${OUT:-outputs/evaluation/kimodo_tp2m}
NUM_GPUS=${NUM_GPUS:-8}
CONDS=${CONDS:-"5 9"}
MAX_SAMPLES=${MAX_SAMPLES:-600}
mkdir -p "$OUT/logs"

# Read GT prefix poses straight from CephFS (each shard only needs ~600/Nshard
# clips, so the /dev/shm pre-copy of all 4384 files is pure overhead on a cold
# Taiji node and was blocking startup). Keep the path overridable.
MS_REL="ref_repo/MotionStreamer/MotionStreamer"
H3D_ROOT=${H3D_ROOT:-$MS_REL/humanml3d_272}
if [ ! -f "$H3D_ROOT/split/test.txt" ]; then
  H3D_ROOT="$MS_REL/humanml3d_272"
fi

# --- Sharding ---
# Within a job: Taiji sets INDEX=node rank, NODE_LIST=comma node list.
# Across independent (single-host) jobs: pass JOB_RANK / JOB_COUNT so several
# 1-host x 8-GPU jobs partition the work without multi-host MPI/RDMA.
HOST_RANK=${INDEX:-0}
if [ -n "${NODE_LIST:-}" ]; then
  MACHINE_NUM=$(python3 -c "import os;print(len(os.environ['NODE_LIST'].split(',')))" 2>/dev/null || echo 1)
else
  MACHINE_NUM=${MACHINE_NUM:-1}
fi
JOB_RANK=${JOB_RANK:-0}
JOB_COUNT=${JOB_COUNT:-1}
TOTAL_SHARDS=$((JOB_COUNT * MACHINE_NUM * NUM_GPUS))
GLOBAL_HOST=$((JOB_RANK * MACHINE_NUM + HOST_RANK))

echo "[start] out=$OUT conds=$CONDS host_rank=$HOST_RANK machines=$MACHINE_NUM gpus/node=$NUM_GPUS total_shards=$TOTAL_SHARDS H3D_ROOT=$H3D_ROOT"

for cond in $CONDS; do
  echo "[gen cond$cond] host=$HOST_RANK $(date)"
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD=$((GLOBAL_HOST * NUM_GPUS + i))
    CUDA_VISIBLE_DEVICES=$i "$PY" -u scripts/eval/gen_kimodo_tp2m_positions.py \
      --humanml3d-272 "$H3D_ROOT" --out-dir "$OUT" --condition-num-frames "$cond" \
      --max-samples "$MAX_SAMPLES" --num-shards "$TOTAL_SHARDS" --shard-index "$SHARD" \
      --skip-existing --device cuda \
      > "$OUT/logs/gen_cond${cond}_gh${GLOBAL_HOST}_g$i.log" 2>&1 &
  done
  wait
  n=$(ls "$OUT/cond${cond}"/*.npy 2>/dev/null | wc -l)
  echo "[gen cond$cond done] host=$HOST_RANK n_total=$n"
done
echo "[ALL DONE host=$HOST_RANK]"
