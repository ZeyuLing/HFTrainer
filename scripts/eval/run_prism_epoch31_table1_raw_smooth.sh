#!/usr/bin/env bash
# Run PRISM epoch31 Table-1 raw evaluation, then smooth the same generations and
# evaluate the smoothed copy.  The first argument is an output suffix, e.g.
# ``debug2`` or ``dd4``; the second argument is the number of generation GPUs.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

SUFFIX=${1:-debug2}
NGPU=${2:-8}

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HFTRAINER_SKIP_AUTOREGISTER=1

CKPT_PATH=work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_31
ANNO=${ANNO:-data/annotation/test_hml3d_official272_gtlen.json}
REWRITTEN=${REWRITTEN:-data/annotation/test_hml3d_rewritten.json}
RAW_GEN=outputs/evaluation/prism_kt_spectral_epoch31_rw_${SUFFIX}/h3d
SMOOTH_GEN=outputs/evaluation/prism_kt_spectral_epoch31_smooth_rw_${SUFFIX}/h3d
RAW_EVAL=outputs/evaluation/prism_epoch31_ms272_h3d_${SUFFIX}
SMOOTH_EVAL=outputs/evaluation/prism_epoch31_smooth_ms272_h3d_${SUFFIX}

mkdir -p "$RAW_EVAL" "$SMOOTH_EVAL" "outputs/evaluation/prism_epoch31_${SUFFIX}_launcher"
echo "[driver] $(date) suffix=$SUFFIX ngpu=$NGPU"

CKPT="$CKPT_PATH" PHASE=t2m NGPU="$NGPU" RUN_PHYS=1 SKIP_CACHE=1 \
  ANNO="$ANNO" REWRITTEN="$REWRITTEN" \
  OUT_ROOT="$RAW_EVAL" GEN_OUT="$RAW_GEN" T2M_TAG=ours_e31 \
  bash scripts/eval/run_prism_epoch15_ms272_h3d.sh

python3 scripts/eval/smooth_smplx_npz_dir.py \
  --src-dir "$RAW_GEN/depth_driven" \
  --out-dir "$SMOOTH_GEN/depth_driven" \
  --workers 16 --skip-existing \
  > "$SMOOTH_EVAL/smooth.log" 2>&1

CKPT="$CKPT_PATH" PHASE=t2m NGPU="$NGPU" RUN_PHYS=1 SKIP_CACHE=1 \
  ANNO="$ANNO" REWRITTEN="$REWRITTEN" \
  OUT_ROOT="$SMOOTH_EVAL" T2M_GEN_DIR="$SMOOTH_GEN/depth_driven" \
  T2M_TAG=ours_e31_smooth \
  bash scripts/eval/run_prism_epoch15_ms272_h3d.sh

echo "[driver done] $(date) suffix=$SUFFIX"
