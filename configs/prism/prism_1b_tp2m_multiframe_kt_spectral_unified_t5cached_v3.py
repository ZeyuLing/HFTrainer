# PRISM 1B text+pose-to-motion, T5 cached — SPEED OPTIMIZED v3
#
# Key optimizations:
#   1. fp16 AUTOCAST (not FSDP mixed precision): Uses torch.cuda.amp.autocast
#      to selectively apply fp16 tensor cores for matmuls while keeping
#      sensitive ops (layernorm, softmax, attention scores) in fp32.
#      - v2 used FSDP mixed_precision='fp16' which forces ALL ops to fp16 → NaN
#      - v3 uses autocast which lets PyTorch decide per-op → stable + fast
#   2. module_dtype='bf16': Parameters stored in bf16 (same as base config).
#      Autocast casts matmul inputs to fp16 at compute time → V100 tensor cores.
#      NOTE: fp32 master weights caused OOM during pre-FSDP model loading
#      (1.4B × 4 bytes = 5.6GB per rank, 8 ranks concurrent → OOM).
#   3. batch_size=12: utilizing freed GPU memory (T5 removed + gradient checkpointing)
#   4. pin_memory=True + prefetch_factor=4: async CPU→GPU data transfer
#
# Performance:
#   - v1 baseline (bf16 emulated): train_time=33.4s, 11.5 samples/s
#   - v2 FSDP fp16: train_time=9.6s but NaN losses (overflow)
#   - v3 autocast fp16: expected ~10-12s (same tensor cores, stable numerics)
#
# Memory budget (V100 32GB per GPU, FSDP FULL_SHARD):
#   - bf16 params sharded: 1.4B*2/N_GPUs ≈ 0.35GB (8 GPU)
#   - Optimizer (fp32 states): 2x full param size / N_GPUs
#   - Activations in fp16 (via autocast + gradient checkpointing): ~2-3GB
#   - Total: ~5-6GB per GPU → safe margin for batch_size=12-16

_base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py'

# ---- NO FSDP mixed precision (autocast handles fp16 in trainer) ----
accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
    fsdp_plugin=dict(
        sharding_strategy='FULL_SHARD',
        backward_prefetch='BACKWARD_PRE',
        auto_wrap_policy='TRANSFORMER_BASED_WRAP',
        transformer_cls_names_to_wrap=['WanTransformerBlockWithMask'],
        state_dict_type='FULL_STATE_DICT',
        sync_module_states=True,
        use_orig_params=True,
        cpu_offload=False,
    ),
)

# ---- Keep bf16 params (base config default) ----
# NOTE: module_dtype='fp32' caused OOM during pre-FSDP loading (5.6GB/rank × 8 ranks).
# bf16 params (2.8GB/rank) load fine. Autocast handles fp16 tensor core compute.
# model = dict(transformer=dict(module_dtype='fp32'))  # DON'T USE - OOM

# ---- Trainer: enable fp16 autocast for V100 tensor cores ----
trainer = dict(
    use_fp16_autocast=True,
)

# ---- Larger batch + faster data loading ----
train_dataloader = dict(
    batch_size=12,
    num_workers=12,
    persistent_workers=True,
    shuffle=True,
    pin_memory=True,
    prefetch_factor=4,
)

work_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_v3'
