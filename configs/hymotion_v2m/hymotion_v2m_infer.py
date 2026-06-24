"""Inference config for HyMotion-V2M with the released checkpoint.

Loads the original V2M ``config.yml`` (network + pipeline args) and
``epoch100.ckpt`` verbatim through the vendored ``MotionGenerationV2M``, so
results are numerically identical to the source repository.

Usage::

    python tools/infer.py \\
        --config configs/hymotion_v2m/hymotion_v2m_infer.py \\
        --checkpoint none \\
        --input <feature.pt|feature.npz> \\
        --output outputs/inference/hymotion_v2m/output.npz \\
        --device cuda

``--input`` carries ``feature`` (T, 1024) and optionally ``camera_RT`` /
``camera_K``.  When omitted, a random feature stream is used for a sanity run.
The bundle loads ``ckpt_path`` itself, so ``--checkpoint`` can be any
placeholder (a missing path just warns and is ignored).
"""

_base_ = '../_base_/default_runtime.py'

# Local copy of the released V2M config + checkpoint (727MB).
_V2M_CKPT_DIR = 'checkpoints/hymotion_v2m'

model = dict(
    type='HyMotionV2MBundle',
    v2m_config_path=f'{_V2M_CKPT_DIR}/config.yml',
    ckpt_path=f'{_V2M_CKPT_DIR}/epoch100.ckpt',
    mean_std_path='assets/v2m_wv_mean_std_1200h_step10.json',
    strict_load=True,
)

trainer = None
train_dataloader = None
optimizer = None
lr_scheduler = None
val_dataloader = None
val_evaluator = None
val_visualizer = None
