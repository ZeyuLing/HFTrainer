#!/bin/bash
# Epoch-4 rewritten-protocol generation driver (single 8-GPU node).
# Generates motions conditioned on REWRITTEN captions (main-table input protocol);
# metrics are computed downstream against the ORIGINAL captions in --anno-file.
#
# Priority order (so the main-table "ours" numbers land first):
#   1. H3D depth_driven   (= ours, KAFS on)
#   2. MH  depth_driven   (= ours, KAFS on)
#   3. H3D none           (no-KAFS reference / KAFS ablation)
#   4. MH  none
# Resumable via --skip-existing inside run_gen_node.sh.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py
CKPT=work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_4
ROOT=outputs/evaluation/prism_kt_spectral_epoch4_rw
NSHARDS=8
NGPU=8
export CONFIG CKPT NSHARDS NGPU

mkdir -p "$ROOT/_logs"
echo "[epoch4-rw] DRIVER BEGIN $(date)" | tee -a "$ROOT/_logs/driver.log"

run_one () {
    local ds=$1 mode=$2
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
    echo "[epoch4-rw] >>> $ds/$mode start $(date)" | tee -a "$ROOT/_logs/driver.log"
    MODE=$mode SHARD_START=0 bash scripts/eval/run_gen_node.sh
    local n=$(ls "$OUT/$mode/"*.npz 2>/dev/null | wc -l)
    echo "[epoch4-rw] <<< $ds/$mode end $(date) npz=$n" | tee -a "$ROOT/_logs/driver.log"
}

run_one h3d depth_driven
run_one mh  depth_driven
run_one h3d none
run_one mh  none

touch "$ROOT/_logs/_DRIVER_DONE"
echo "[epoch4-rw] DRIVER ALL DONE $(date)" | tee -a "$ROOT/_logs/driver.log"
