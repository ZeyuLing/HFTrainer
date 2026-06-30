#!/usr/bin/env bash
# Per-node launcher for the 32x A100 (4 node x 8) VerMo bf16/sdpa resume run.
# Usage: setsid bash tools/_a100_resume_launch.sh <NODE_RANK> > log 2>&1 < /dev/null &
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
# NOTE: RoCE/IB is present (mlx5_0/1 @100Gb/s, RoCE v2 gid 3) BUT the inter-node
# fabric rejects RDMA -- the first NCCL allgather fails with ncclRemoteError
# (switch PFC/ECN not configured for lossless RoCE; not fixable from container).
# This is why the platform default disables IB.  Force TCP over bond1.
# To retry RoCE if the fabric is ever fixed: VERMO_IB_DISABLE=0
#   (then NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3).
export NCCL_IB_DISABLE="${VERMO_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# NOTE: expandable_segments is unsupported on this platform's CUDA driver and
# triggers "CUDA driver error: invalid argument" during NCCL memory registration
# (DDP _broadcast_coalesced). Force the default native allocator instead.
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
export PYTHONFAULTHANDLER=1

export NNODES="${NNODES:-4}"
export NODE_RANK="${1:?need node rank}"
export MASTER_ADDR="${MASTER_ADDR:-30.72.67.8}"
export MASTER_PORT="${MASTER_PORT:-29501}"

# NOTE: eager is FASTER than sdpa on this cluster -- the host CUDA driver lacks
# `cuLaunchKernelEx`, so SDPA's mem-efficient/flash kernels cannot launch and
# fall back to a slow path (measured: eager 5.1s/step vs sdpa 9.0s/step).
export CONFIG="${CONFIG:-configs/vermo/vermo_pretrain_16k_llama1b_a100_fp16_eager_resume.py}"
echo "[a100-resume] starting NODE_RANK=${NODE_RANK} NNODES=${NNODES} MASTER=${MASTER_ADDR}:${MASTER_PORT} CONFIG=${CONFIG} at $(date)"
exec bash tools/dist_train.sh "${CONFIG}" 8
