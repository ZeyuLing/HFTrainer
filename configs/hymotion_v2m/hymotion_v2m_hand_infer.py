"""Inference config for HyMotion-V2M with the **hand-enabled** checkpoint.

Same as ``hymotion_v2m_infer.py`` but loads the ``load_hand=True`` variant
(``base046B_render2000_hand``): the network consumes the full 3072-dim SAM-3D
feature (body 1024 + left/right hand 2x1024) so finger motion is data-driven.

The output motion representation is identical (``wvrot6d_transl_shape_stationary_std``,
52 SMPL-H joints), so decode / viewer code is unchanged.

Usage (pre-extracted feature -> motion, stage 1)::

    python tools/infer.py \\
        --config configs/hymotion_v2m/hymotion_v2m_hand_infer.py \\
        --checkpoint none \\
        --input <feature.npz with full 3072-dim feature> \\
        --output outputs/inference/hymotion_v2m/output.npz \\
        --device cuda

``--input`` (feature mode) must carry the **full 3072-dim** feature stream (do
NOT slice to 1024), plus optional ``camera_RT`` / ``camera_K`` / ``movement_type``.

Usage (end-to-end video -> motion, stage 2)::

    # --input ending in .mp4/.mov/... triggers HyMotionV2MPipeline.infer_v2m:
    #   ffmpeg transcode -> YOLOX + ByteTrack -> SAM-3D-Body tokens -> motion
    python tools/infer.py \\
        --config configs/hymotion_v2m/hymotion_v2m_hand_infer.py \\
        --checkpoint none --input <video.mp4> \\
        --output outputs/inference/hymotion_v2m/output.npz --device cuda

Stage-2 needs external deps/weights NOT bundled here (resolve via the
``HYMOTION_V2M_*`` env vars or ``preprocessor_kwargs``):
``ffmpeg``; ``yolox`` + ``yolox_l.pth``; ``supervision``; the ``sam_3d_body``
package + the **gated** ``facebook/sam-3d-body-dinov3`` weights
(``model.ckpt`` + ``assets/mhr_model.pt``).

Programmatic loading (any pipeline supports ``from_pretrained``)::

    from hftrainer.pipelines.motion.hymotion_v2m_pipeline import HyMotionV2MPipeline
    pipe = HyMotionV2MPipeline.from_pretrained(
        'checkpoints/hymotion_v2m_hand', bundle_kwargs={'device': 'cuda'})
    out = pipe.infer_v2m('video.mp4')          # end-to-end
    out = pipe.infer_from_feature(feature=...)  # pre-extracted feature
"""

_base_ = '../_base_/default_runtime.py'

# Local copy of the released hand V2M config + checkpoint (769MB).
_V2M_CKPT_DIR = 'checkpoints/hymotion_v2m_hand'

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
