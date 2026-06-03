# Inference config for reproducing the ORIGINAL iter_15000 checkpoint
# (work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000), i.e. the paper "ours".
#
# CRITICAL FIX vs prism_1b_tp2m_multiframe.py:
#   iter_15000 was trained in *versatilemotion* with the `wanmo_vae2d_aug` VAE
#   (see versatilemotion/configs/prism/prism_1b_tp2m_hq_t5xxl_256text_aug_multiframe.py).
#   The default hf_trainer config decodes with `vermo_vae`, whose latent space
#   (different latents_mean/std + weights) does NOT match -> garbage output.
#   Swapping to wanmo_vae2d_aug restores the correct latent space.
#
# RoPE: joint_pos_mode="sequential" (inherited) is byte-for-byte identical to the
#   original versatilemotion MotionWanRotaryPosEmbed, so no RoPE change is needed.
_base_ = './prism_1b_tp2m_multiframe.py'

model = dict(
    vae=dict(
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/wanmo_vae2d_aug',
        ),
    ),
)
