# G1-native Text-to-Motion fine-tune (path b'): warm-start HY-Motion-1.0-Lite
# and re-target its output head to the 38-d G1 robot representation, then
# supervise on the robot-suitable 456k clips retargeted into ``data/g1/``.
#
# Representation (physflow/g1_repr.py, G1_MOTION_DIM=38):
#   [0:3] pelvis transl | [3:9] pelvis rot6d | [9:38] 29 DOF angles
# Decoding to MuJoCo qpos(36) is exact -> the online PhysFlow loop stays in G1
# space (no SMPL->G1 retarget, lossless, real-time).
#
# Warm-start: input_dim/output_dim change 201->38, so the transformer's
# ``input_encoder`` / ``final_layer`` (and the 201-d mean/std buffers) are
# shape-mismatched and skipped by ``load_state_dict_selective``; the MMDiT
# backbone + text cross-attention + null embeddings all load from Lite.

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/hymotion_g1_t2m_38dim'

_motion_dim = 38

# ----- Model -----
model = dict(
    type='HyMotionT2MBundle',
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        trainable=True,
        input_dim=_motion_dim,         # 38 (G1; reinit input_encoder)
        feat_dim=1024,
        output_dim=_motion_dim,        # 38 (G1; reinit final_layer)
        ctxt_input_dim=4096,
        vtxt_input_dim=768,
        num_layers=18,
        num_heads=16,
        mlp_ratio=4.0,
        mlp_act_type='gelu_tanh',
        norm_type='layer',
        qk_norm_type='rms',
        qkv_bias=True,
        dropout=0.0,
        text_refiner_cfg=dict(num_layers=2),
        final_layer_cfg=dict(act_type='silu'),
        mask_mode='narrowband',
        apply_rope_to_single_branch=False,
        insert_start_token=False,
        with_long_skip_connection=False,
        time_factor=1000.0,
    ),
    # Empty on purpose: training never runs a text encoder -- captions are fed
    # as PRE-EXTRACTED embeddings (HyMotionG1Dataset reads the ``qwen3_*/*.pt``
    # dirs == Qwen3-8B CausalLM 4096 + CLIP-L 768, same as HYMotion-M2M).
    # The 8B encoder is only needed offline (extract) or at inference.
    text_encoder=dict(),
    # G1 stats computed by scripts/embodied/compute_g1_motion_stats.py.
    mean_std_dir='data/g1_t2m_stats/',
    motion_type='g1_29dof',
    pred_type='velocity',
    uncondition_mode=False,
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
        # Per-modality reduction: split the 38-d velocity loss at the G1 layout
        # boundaries -- transl(0:3) / pelvis rot6d(3:9) / 29 joint angles(9:38) --
        # take each component's own mean, then average the three equally.  This
        # gives translation and root rotation ~1/3 of the gradient each instead
        # of the 3/38~8% they get under a flat element-mean, without the brittle
        # hand-tuned ``trans_dim_weight`` up-weighting.
        velocity_loss_reduction='component_mean',
        trans_dims=3,
    ),
    noise_scheduler_cfg=dict(method='euler'),
    infer_noise_scheduler_cfg=dict(validation_steps=50),
    cond_mask_prob=0.1,
    vtxt_input_dim=768,
    ctxt_input_dim=4096,
    body_model_path=None,
)

trainer = dict(
    type='HyMotionT2MTrainer',
    val_num_steps=10,
)

# ----- Data -----
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        type='HyMotionG1Dataset',
        # Pre-filtered embeddable list (164,771 clips whose annotation caption
        # has a pre-extracted qwen3 .pt). Slim fields + precomputed emb_rel ->
        # fast load. Built by joining train_g1_t2m.json with CAPTION_TO_QWEN3_DIR.
        anno_file='data/annotation/train_g1_t2m_emb.json',
        g1_dir='data/g1',
        data_dir='data/hymotion_data',
        clip_len=300,
        min_frames=30,
        # Do not randomly sample caption variants.  Wrong-caption augmentation
        # corrupts T2M supervision; classifier-free dropout is already handled
        # by cond_mask_prob above.
        random_caption=False,
        require_embedding=True,
    ),
    # collate_fn auto-detected from HyMotionG1Dataset.collate_fn by the runner.
)

# ----- Optimizer (fine-tune lr) -----
optimizer = dict(
    type='AdamW',
    lr=2e-5,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

lr_scheduler = None

accelerator = dict(
    mixed_precision='bf16',
    gradient_accumulation_steps=1,
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=40,         # warm-start fine-tune; ~2.5k iter/epoch @ 1x8 bs8 -> stop early via ckpts
    val_interval=100000,   # no val (decode is SMPL-specific); rely on infer
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    # by_epoch=False so ``interval`` counts ITERS, not epochs.  Without this the
    # hook inherits train_cfg.by_epoch=True and "interval=2000" means every 2000
    # *epochs* -> no checkpoint ever saved within a 40-epoch run.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=5, save_last=True),
)

# ----- Warm-start from HY-Motion-1.0-Lite -----
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
    # 201-d mean/std must NOT overwrite the 38-d G1 buffers.
    exclude_bundle_keys=['mean', 'std'],
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
