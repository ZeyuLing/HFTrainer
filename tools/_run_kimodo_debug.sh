#!/usr/bin/env bash
# Mini debug run on lzy_debug_machine_1 to verify KIMODO aux loss in real training.
# Uses GPU 1 (empty), uncond_local config (smaller per-GPU batch), early stop after a few steps.
set -e

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/_kimodo_debug.log
mkdir -p "$(dirname "$OUT")"
: > "$OUT"

# Override: small batch + max_iters + no checkpoint save to debug fast.
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=. \
accelerate launch --num_processes=1 --main_process_port=29501 \
  tools/train.py configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py \
  --cfg-options \
    train_dataloader.batch_size=2 \
    train_dataloader.num_workers=0 \
    train_cfg.by_epoch=False \
    train_cfg.max_iters=8 \
    train_cfg.val_interval=100 \
    default_hooks.checkpoint=None \
    work_dir=/tmp/kimodo_debug_work_dir \
  > "$OUT" 2>&1 || echo "[debug] training script returned non-zero (expected on early stop)"

echo "===== TAIL OF $OUT ====="
tail -60 "$OUT"
