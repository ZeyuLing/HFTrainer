#!/bin/bash
# Per-node driver to reproduce the PAPER main-table "ours" using the ORIGINAL
# (vanilla, sequential) checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000.
# Paper protocol: GENERATE on rewritten captions, mode=none (standard sampler).
#
# Env: SHARD_START (this node's first global shard), NSHARDS (global total), NGPU.
# Each node runs H3D shards then MH shards over its [SHARD_START, SHARD_START+NGPU).
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:${PYTHONPATH:-}

# NOTE: iter_15000 was trained with the wanmo_vae2d_aug VAE; the *_iter15k config
# swaps the VAE accordingly (vermo_vae -> wanmo_vae2d_aug). Using the plain
# prism_1b_tp2m_multiframe.py here decodes in the WRONG latent space -> garbage.
CONFIG=configs/prism/prism_1b_tp2m_multiframe_iter15k.py
CKPT=work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000
ROOT=outputs/evaluation/prism_paper_iter15000_nomask
NSHARDS=${NSHARDS:-32}
SHARD_START=${SHARD_START:-0}
NGPU=${NGPU:-8}
MODE=${MODE:-none}
# Text cross-attention is now always mask-free (Wan-style: text padded with
# zeros, context_lens=None), hardcoded in PrismARPipeline. No flag needed.
EXTRA_ARGS=${EXTRA_ARGS:-}
export CONFIG CKPT NSHARDS NGPU MODE EXTRA_ARGS

mkdir -p "$ROOT/_logs"
echo "[iter15k] NODE BEGIN $(date) SHARD_START=$SHARD_START NSHARDS=$NSHARDS NGPU=$NGPU" | tee -a "$ROOT/_logs/driver_s${SHARD_START}.log"

run_one () {
    local ds=$1
    if [ "$ds" = "h3d" ]; then
        ANNO=data/annotation/test_hml3d.json
        REWRITTEN=data/annotation/test_hml3d_rewritten.json
        OUT=$ROOT/h3d
    else
        ANNO=data/annotation/test_motionhub_t2m.json
        REWRITTEN=data/annotation/test_motionhub_t2m_rewritten.json
        OUT=$ROOT/mh
    fi
    export ANNO REWRITTEN OUT
    echo "[iter15k] >>> $ds/$MODE s=$SHARD_START start $(date)" | tee -a "$ROOT/_logs/driver_s${SHARD_START}.log"
    bash scripts/eval/run_gen_node.sh
    local n=$(ls "$OUT/$MODE/"*.npz 2>/dev/null | wc -l)
    echo "[iter15k] <<< $ds/$MODE s=$SHARD_START end $(date) npz=$n" | tee -a "$ROOT/_logs/driver_s${SHARD_START}.log"
}

run_one h3d
run_one mh

touch "$ROOT/_logs/_NODE_DONE_s${SHARD_START}"
echo "[iter15k] NODE ALL DONE $(date) SHARD_START=$SHARD_START" | tee -a "$ROOT/_logs/driver_s${SHARD_START}.log"
