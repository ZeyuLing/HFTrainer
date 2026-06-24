#!/bin/bash
# Drive versatilemotion's eval_babel_baseline.py (FlowMDM / DoubleTake / MotionStreamer)
# on OUR BABEL manifest text dir, sharded across local GPUs. Independent single-host
# jobs set JOB_RANK / JOB_COUNT. Outputs SMPL pred.npz per episode.
set -uo pipefail
HF=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
[ -d "$HF" ] || HF=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
VM=/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion
[ -d "$VM" ] || VM=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion
PIPELINE=${PIPELINE:-flowmdm}
# DoubleTake's SlimSMPLTransform loads BABEL stats via relative ./data_loaders/...,
# so it must run with CWD = priormdm root. FlowMDM/MS use absolute stat paths.
if [ "$PIPELINE" = "doubletake" ]; then
  RUNDIR="$VM/third_party/priormdm"
else
  RUNDIR="$VM"
fi
cd "$RUNDIR"
export PYTHONPATH="$VM:$VM/third_party/priormdm:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=${PY:-python3}
SCRIPT="$VM/scripts/evaluation/eval_babel_baseline.py"
TEXT_DIR=${TEXT_DIR:-$HF/data/babel/babel_seq_text_ours}
NPY_DIR=${NPY_DIR:-$HF/data/babel_272_stream/val_stream}
OUT_ROOT=${OUT_ROOT:-$HF/outputs/evaluation/babel_seq/vm_baselines}
NUM_GPUS=${NUM_GPUS:-8}

if [ -n "${JOB_COUNT:-}" ]; then
  HOST_RANK=${JOB_RANK:-0}; MACHINE_NUM=${JOB_COUNT}
else
  HOST_RANK=${INDEX:-0}; MACHINE_NUM=${MACHINE_NUM:-1}
fi
TOTAL_SHARDS=$((MACHINE_NUM * NUM_GPUS))

mkdir -p "$OUT_ROOT/logs"
echo "[vm-$PIPELINE] host_rank=$HOST_RANK machines=$MACHINE_NUM gpus/node=$NUM_GPUS total_shards=$TOTAL_SHARDS text=$TEXT_DIR"
for i in $(seq 0 $((NUM_GPUS - 1))); do
  SHARD=$((HOST_RANK * NUM_GPUS + i))
  RANK=$SHARD WORLD_SIZE=$TOTAL_SHARDS LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=$i \
    "$PY" -u "$SCRIPT" main \
      --pipeline_type="$PIPELINE" \
      --babel_text_dir="$TEXT_DIR" \
      --babel_npy_dir="$NPY_DIR" \
      --output_root="$OUT_ROOT" \
      > "$OUT_ROOT/logs/${PIPELINE}_h${HOST_RANK}_g$i.log" 2>&1 &
done
wait
n=$(find "$OUT_ROOT/$PIPELINE/default/default" -name 'pred.npz' 2>/dev/null | wc -l)
echo "[vm-$PIPELINE host=$HOST_RANK done] pred_npz=$n"
