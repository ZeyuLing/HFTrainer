# PRISM 1B text+pose-to-motion, T5 cached — SPEED OPTIMIZED v4
#
# Strategy: Maximize throughput WITHOUT fp16 (which causes NaN on this model).
#
# Why fp16 doesn't work on this model:
#   - FSDP mixed_precision='fp16': blanket casts all ops → NaN from step 1
#   - torch.cuda.amp.autocast(fp16): selective casting still overflows because
#     spectral positional encoding (scale=22.0) and deep 30-layer transformer
#     produce intermediate values exceeding fp16 range (65504)
#   - V100 has NO bf16 tensor cores, so bf16 runs as emulated fp32 anyway
#
# Optimizations in this config:
#   1. batch_size=16: V100-32GB has ~20GB headroom (baseline uses ~11.4GB at bs=6).
#      With gradient checkpointing + FSDP FULL_SHARD, bs=16 fits in ~18-20GB.
#   2. pin_memory=True + prefetch_factor=4: proven to reduce data_time 1.35s → 0.65s
#   3. persistent_workers=True: avoids worker respawn overhead per epoch
#
# Expected performance (8 GPU V100-32GB, no tensor cores):
#   - Baseline (bs=6, 32 GPU): train_time=17s, data_time=1.35s → 11.3 samples/s
#   - This config (bs=16, 8 GPU): train_time=~38-42s, data_time=~0.6s
#     → 8*16/40 = 3.2 samples/s per step (fewer GPUs, so lower total throughput)
#     → But per-GPU throughput: 16/40 = 0.4 vs 6/17 = 0.35 → ~14% improvement
#
# Note: effective global batch = 8 GPUs x 16 = 128 samples/step
# Learning rate unchanged — monitor convergence.

_base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py'

# ---- NO fp16 — keep bf16 params with fp32 compute (V100 safe) ----
# Explicitly no autocast — the model's spectral pos enc overflows fp16.
trainer = dict(use_fp16_autocast=False)

# ---- Larger batch + faster data loading ----
train_dataloader = dict(
    batch_size=16,
    num_workers=12,
    persistent_workers=True,
    shuffle=True,
    pin_memory=True,
    prefetch_factor=4,
)

work_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_v4'
