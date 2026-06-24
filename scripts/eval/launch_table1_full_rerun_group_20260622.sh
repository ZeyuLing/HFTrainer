#!/usr/bin/env bash
# Launch one Table-1 metric rerun shard. Intended for lzy_debug_machine_1/2.
set -euo pipefail

GROUP="${1:?usage: $0 g1|g2|g3}"
ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

BASE="outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/table1_full_rerun_20260622"
case "$GROUP" in
  g1)
    METHODS="mdm,mld,momask,motiongpt3"
    GPU_LIST="${GPU_LIST:-0,1,2,3}"
    ;;
  g2)
    METHODS="t2mgpt,flowmdm,motionlab,gotozero"
    GPU_LIST="${GPU_LIST:-0,1,2,3}"
    ;;
  g3)
    METHODS="motionstreamer,hymotion_1b,mogents,ours_epoch42_abs"
    GPU_LIST="${GPU_LIST:-4,5,6,7}"
    ;;
  *)
    echo "unknown group: $GROUP" >&2
    exit 2
    ;;
esac

OUT_ROOT="$BASE/shards/$GROUP"
mkdir -p "$OUT_ROOT/logs"
echo "[launcher] group=$GROUP methods=$METHODS gpu_list=$GPU_LIST out=$OUT_ROOT start=$(date -Is)"

OUT_ROOT="$OUT_ROOT" \
METHODS="$METHODS" \
GPU_LIST="$GPU_LIST" \
RUN_PHYS=0 \
FORCE_EVAL=1 \
bash scripts/eval/run_table1_full_rerun_20260622.sh
