# HyMotion M2M v2 0.46B base config — 198-dim motion (rot6d + position).
#
# Key differences from v1 (_base_hymotion_m2m_046b.py):
#   - motion_dim = 198 (3 trans + 132 rot6d + 63 position)
#   - VACE: no_inactive mode → input = [x_t(198), reactive(198), mask(198)] = 594-dim
#   - Condition sampler v2: two-tier architecture with per-dim position control
#   - FK consistency loss: enforces rotation/position self-consistency
#   - Data pipeline: Compute198DimPosition transform inserts position channels
#   - T2M 1.0 pretrained weights: 18 transformer blocks loaded, input/output re-initialized
#
# 198-dim layout:
#   [0:3]      translation (SMPL trans)
#   [3:135]    22 joints × 6D rot6d (row-major)
#   [135:198]  21 joints × 3D position (XZ rel pelvis, Y absolute, pelvis excluded)

_base_ = '../_base_/default_runtime.py'

work_dir = 'work_dirs/hymotion_m2m_v2_046b'

# ----- Model -----
_motion_dim = 198

model = dict(
    type='HyMotionM2MBundle',
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        trainable=True,
        # v2 slim VACE: drops the `inactive` channel because mask-aware noise
        # already carries known-region clean values through x_t. VACE then
        # only needs `reactive` (pre-edit value in mask=1 regions, zero in
        # completion) + `mask`. Total input = x_t + reactive + mask = 3 * D.
        input_dim=_motion_dim * 3,  # 594
        feat_dim=1024,
        output_dim=_motion_dim,     # 198
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
    text_encoder=dict(),
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    motion_type='smpl_22',
    pred_type='velocity',
    uncondition_mode=True,
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        x1_weight=0.0,
        keypoints3d_weight=0.0,
        translation_weight=0.0,
        trans_dim_weight=5.0,
        motion_smoothness_weight=0.5,
        # NOTE: legacy fk_consistency in M2MLoss is disabled when KIMODO-style
        # aux loss is enabled below (see kimodo_aux_loss_cfg) — they compute
        # the same quantity, so we keep only one path active.
        fk_consistency_weight=0.0,
        fk_consistency_warmup_steps=2000,
    ),
    # KIMODO-style auxiliary losses (Eq. 1 in KIMODO paper):
    #   - joint_pos      ≈ γ₃: smooth-L1 on global (FK-derived) joint
    #     positions in metres, vs GT joint positions.  Directly supervises
    #     world-space leg/foot positions, suppressing "pelvis cheating" /
    #     slipping that the relative-pelvis 198-dim representation otherwise
    #     allows.
    #   - joint_vel      ≈ γ₄: smooth-L1 on global joint velocities (from
    #     the temporal derivative of FK positions, NOT 198-dim d/dt).  A
    #     slipping pose immediately incurs a velocity error at every joint.
    #   - fk_consistency ≈ γ₇: pos-channel ↔ FK(pred_rot/trans) consistency,
    #     **purely intra-prediction** (no GT involved).  Its primary
    #     purpose is NOT slip-suppression but rather to teach the model an
    #     *explicit FK equivalence map* inside the 198-dim representation,
    #     so that at inference time a position-only condition (e.g. E1 hand
    #     trajectory, E4 end-effector) can be imputed into pred[135:198]
    #     and the model can generate self-consistent rot/trans (pred[:135])
    #     without needing IK.  The model has no built-in FK operator; this
    #     loss is the *only* supervision signal that pred[:135] and
    #     pred[135:198] must satisfy FK(pred[:135])_rel_pelvis ≡ pred[135:198].
    #     Without it, FK-equivalence is only induced indirectly via main
    #     loss + joint_pos through GT correlation, which generalises poorly
    #     to OOD inference settings (pure position condition).
    #
    # === Weight magnitudes (≠ KIMODO γ values; reason below) ===
    # KIMODO computes Eq.1 in *normalised unit-variance* space where typical
    # smooth_l1 base values are O(1e-3), so γ in the 1–10 range produces
    # weighted losses ~1e-2.  Our aux block runs in *denormalised metres*
    # (FK world coords).  For the already-converged ckpt that resumes here
    # (~1 cm joint error, ~mm-level intra-pred consistency), smooth_l1
    # quadratic-region base values are roughly:
    #   joint_pos:        O(1e-4)        (cm-level pred-vs-GT joint pos)
    #   joint_vel:        O(1e-6)        (mm/frame; T=360 sequences)
    #   fk_consistency:   O(1.4e-6)      (mm-level intra-pred consistency)
    # Combined with t² re-weighting (E[t²]=1/3) the raw base is ~3× weaker.
    # The weights below target a meaningful fraction of loss_velocity
    # (≈ 0.025 in normalised space):
    #   joint_pos:       50      ⇒ ≈ 5.0e-3   (~14% of loss_velocity)
    #   joint_vel:       500     ⇒ ≈ 1.0e-3   (~ 4% of loss_velocity)
    #   fk_consistency:  1500    ⇒ ≈ 2.1e-3   (~ 7% of loss_velocity)
    # fk_consistency uses a much larger nominal weight than joint_pos
    # because its base value is ~70× smaller (intra-pred consistency on a
    # already-FK-consistent representation is naturally tighter than pred-
    # vs-GT joint-position error).  The point of this re-weight is *not* to
    # mechanically follow KIMODO's γ₃:γ₇ = 10:5 ratio, but to make the
    # explicit FK-equivalence supervision strong enough to actually shape
    # the rot↔pos mapping (rather than relying on indirect correlation).
    kimodo_aux_loss_cfg=dict(
        joint_pos_weight=50.0,
        joint_vel_weight=500.0,
        fk_consistency_weight=1500.0,
        loss_type='smooth_l1',
        timestep_squared_weighting=True,
        fk_consistency_warmup_steps=2000,
        joint_pos_warmup_steps=2000,
        joint_vel_warmup_steps=2000,
    ),
    noise_scheduler_cfg=dict(method='euler'),
    infer_noise_scheduler_cfg=dict(validation_steps=50),
    cond_mask_prob=0.0,
    # no_inactive: VACE = [reactive, mask]; model input = x_t + reactive + mask = 3*D
    vace_condition_mode='no_inactive',
    vtxt_input_dim=768,
    ctxt_input_dim=4096,
    body_model_path=None,
)

trainer = dict(
    type='HyMotionM2MTrainer',
    val_num_steps=10,
    mask_aware_noise=True,  # v2 uses MAN by default
)

# ----- Data -----
train_dataloader = dict(
    batch_size=28,  # V100-32GB peak ~30GB at bs=28 (uncond); caption configs override to 20
    num_workers=4,
    persistent_workers=False,
    shuffle=True,
    dataset=dict(
        type='MotionhubMultiTaskMultiAgentDataset',
        motion_key='smplx',
        data_dir='data/motionhub',
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        task_mode='auto',
        num_person=1,
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=True),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            # Compute 198-dim position channels from 135-dim motion via FK
            # MUST come BEFORE LocalToGlobalRotation (FK requires local rotation)
            dict(type='Compute198DimPosition', key='motion'),
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            # v2 condition sampler: two-tier architecture with per-dim control
            dict(
                type='PrepareM2Mv2Condition',
                key='motion',
                tier2_prob=0.4,
                editing_prob=0.15,
                corruptor_names=[
                    'jitter', 'joint_jump', 'sliding',
                    'limb_candy_wrapper', 'wrist_candy_wrapper',
                ],
                max_corruptions=2,
            ),
            dict(
                type='PackInputs',
                keys=[
                    'src_motion', 'tgt_motion', 'src_mask',
                    'tgt_length', 'src_length', 'edit_mode',
                ],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
        verbose=True,
        refetch=True,
    ),
)

# ----- Optimizer -----
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

lr_scheduler = None

# ----- Accelerator -----
accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)

# ----- Train cfg -----
train_cfg = dict(
    by_epoch=True,
    max_epochs=10000,
    val_interval=10,
    max_grad_norm=2.0,
)

# ----- Hooks -----
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=100, save_last=True),
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)

# ----- Load T2M pretrained weights -----
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

val_dataloader = None
val_evaluator = None
val_visualizer = None
