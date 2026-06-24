"""Smoke config for HyMotion-V2M (pre-extracted feature -> motion).

Builds a *tiny* vendored ``MotionGenerationV2M`` (few layers, small feat_dim,
2 ODE steps) with no checkpoint, so the feature-to-motion path can be exercised
end-to-end on CPU in seconds.  Stage 1 has no trainer; the dedicated smoke test
``tests/smoke/test_hymotion_v2m.py`` builds the bundle from ``model`` here and
runs ``HyMotionV2MPipeline.infer_from_feature`` on synthetic features.
"""

_base_ = '../_base_/default_runtime.py'

train_frames = 40
feature_dim = 1024
mean_std_path = 'assets/v2m_wv_mean_std_1200h_step10.json'

model = dict(
    type='HyMotionV2MBundle',
    network_module='hymotion/network/hymotion_mmdit_for_v2m.HunyuanMotionMMDiT',
    network_module_args=dict(
        apply_rope_to_single_branch=True,
        ctxt_input_dim=dict(camera_R=9, camera_T=3, feature=feature_dim),
        dropout=0.0,
        feat_dim=64,
        final_layer_cfg=dict(act_type='silu', zero_init=True),
        input_dim=349,
        insert_start_token=False,
        mask_mode='narrowband_v2m',
        mlp_ratio=2.0,
        narrowband_v2m_length=3.0,
        num_heads=4,
        num_layers=3,
        time_factor=1000.0,
        vtxt_input_dim=64,
    ),
    pipeline_args=dict(
        mean_std=mean_std_path,
        motion_rep='wvrot6d_transl_shape_stationary_std',
        pred_type='x1',
        noise_scheduler_cfg=dict(method='euler'),
        infer_noise_scheduler_cfg=dict(validation_steps=2),
        train_frames=train_frames,
        # The wv decode path uses ``mesh_model`` (SMPLMesh), which is only built
        # when a ``vertex`` loss is declared -> keep it (assets/model.npz).
        losses_cfg=dict(
            recons=dict(name='SmoothL1Loss', weight=1.0),
            vertex=dict(name='SmoothL1Loss', overlap_step=10000, start_step=10000, weight=100.0),
            transroll=dict(name='SmoothL1Loss', overlap_step=10000, start_step=10000, weight=10.0),
        ),
        train_cfg=dict(cond_mask_prob=0.1),
        test_cfg=dict(mean_std_dir=mean_std_path, text_guidance_scale=1.0),
    ),
    ckpt_path=None,
)

# Stage 1 is inference-only: no trainer.  Provided for completeness / future
# training; the smoke test drives the pipeline directly.
trainer = None

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    shuffle=False,
    dataset=dict(
        type='HyMotionV2MSyntheticDataset',
        num_samples=2,
        num_frames=train_frames,
        feature_dim=feature_dim,
    ),
)

optimizer = None
lr_scheduler = None
val_dataloader = None
val_evaluator = None
val_visualizer = None
