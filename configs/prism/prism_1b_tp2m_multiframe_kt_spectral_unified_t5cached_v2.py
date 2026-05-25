# PRISM 1B text+pose-to-motion, T5 cached — SPEED OPTIMIZED v2
#
# Key optimizations over base t5cached config:
#   1. Trainer-level fp16 autocast: V100 has fp16 Tensor Cores (NOT bf16!)
#      - torch.amp.autocast(fp16) selectively uses fp16 for matmuls → Tensor Cores
#      - Keeps LayerNorm/softmax/loss in fp32 → no NaN overflow
#      - FSDP-level mixed_precision='fp16' was too aggressive (NaN from step 1)
#   2. batch_size=12: utilizing freed GPU memory (11GB from T5 removal)
#   3. pin_memory=True + prefetch_factor=4: async CPU→GPU data transfer
#
# Memory budget (V100 32GB):
#   - Without T5: ~11.4GB used at batch_size=6 (bf16 emulated)
#   - With batch_size=12 + autocast: ~20-22GB (safe margin)
#
# Performance results:
#   - bf16 emulated (baseline):     train_time=33.4s, data_time=2.9s → 11.5 samples/s
#   - fp16 autocast + prefetch:     train_time=~9.6s, data_time=~0.02s → ~40 samples/s (8 GPU)
#   Expected 32 GPU: train_time=~10s → ~38 samples/s per GPU
#
# Note: effective global batch = 32 GPUs × 12 = 384 samples/step
# (doubled from 192). Learning rate unchanged — monitor convergence.

_base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py'

# ---- Trainer-level fp16 autocast for V100 Tensor Core acceleration ----
# use_fp16_autocast wraps the forward pass in torch.amp.autocast(fp16),
# which selectively runs matmuls in fp16 (Tensor Cores) while keeping
# sensitive ops (LayerNorm, softmax) in fp32.
# FSDP stays in mixed_precision='no' — parameters kept in bf16.
trainer = dict(use_fp16_autocast=True)

# ---- Larger batch + faster data loading ----
train_dataloader = dict(
    batch_size=12,
    num_workers=12,
    persistent_workers=True,
    shuffle=True,
    pin_memory=True,
    prefetch_factor=4,
)

work_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_v2'
