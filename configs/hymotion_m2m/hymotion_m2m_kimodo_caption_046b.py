# HyMotion M2M v2 — PHASE 0 E4: KIMODO Root + Caption Conditioning
#
# **Experiment E4** from next-gen proposal: SMPL motion converted to KIMODO
# Root representation (with online ADMM smoothing) and text caption conditioning.
#
# Key overrides from base config (_base_hymotion_m2m_046b.py):
#   - keypoints3d_weight: 0.0 → 10.0 (enable keypoint supervision)
#   - timestep_squared_weighting: True (t² weighting to suppress noisy-FK spikes)
#   - cond_mask_prob: 0.0 → 0.1 (enable CFG during training)
#   - Data pipeline: Add SmplTransToKimodoRootOnline transform (ADMM smoothing)
#   - Text encoding: enabled (caption_local)
#   - Rotation space: local (SMPL frame)
#   - Mean/std: KIMODO Root 198-dim stats
#
# KIMODO Root representation (198-dim):
#   [0:3]      ADMM smoothed pelvis translation (online smoothing during load)
#   [3:9]      root joint 6D rotation (continuous)
#   [9:135]    body (21 non-root joints) 6D rotations
#   [135:198]  FK-derived joint positions relative to pelvis (21 × 3)
#
# ADMM smoothing: Applied online during dataset __getitem__ with margin ≤ 6cm
# on XZ plane (horizontal), Y-axis unchanged (vertical is unsmoothed).
#
# E4 combines KIMODO Root with text guidance (T2M with structured root):
#   - Text-driven generation with smoothed root trajectory
#   - Better consistency for embodied tasks (robot deployment)
#   - Position-to-root relative coordinates for text conditioning
#
# Prerequisite: KIMODO Root mean/std must be computed and placed at
#   data/hymotion_m2m_data/_stats_198dim_kimodo_root/
#
# Launch (local):
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_kimodo_caption_046b.py 8 --auto-resume
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_kimodo_caption_E4 configs/hymotion_m2m/hymotion_m2m_kimodo_caption_046b.py --host_num 8

_base_ = './_base_hymotion_m2m_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_kimodo_caption_E4'

# Resume from M2M v2 caption_local_phase2 checkpoint (epoch 3370).
# caption_local_phase2 has CORRECT null_ctxt values (non-zero).
# null_embedding_source added as safety net regardless.
# exclude_bundle_keys: prevent SMPL Root mean/std from overwriting
# KIMODO Root mean/std (different statistical distributions).
# load_scope='model' resets optimizer/scheduler (loss config changed:
# keypoints3d 0→10, t² weighting on, sampler_v3, KIMODO root).
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    exclude_bundle_keys=['mean', 'std'],
)

model = dict(
    pred_type='velocity',
    uncondition_mode=False,  # Enable text conditioning
    cond_mask_prob=0.1,       # CFG: 10% unconditional during training
    motion_cond_mask_prob=0.0,  # Keep motion/source conditions intact during M2M training
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root',
    rotation_space='local',
    text_encoder=dict(),  # Use default QWEN3 + CLIP-L
    losses_cfg=dict(
        # Override base: enable keypoint supervision (E4 baseline)
        keypoints3d_weight=10.0,
        # Decompose velocity loss into trans/root_rot/body_rot/joint_pos
        # for per-component monitoring in training logs.
        velocity_loss_reduction='component_mean',
    ),
)

train_dataloader = dict(
    batch_size=20,  # Reduce batch size for caption config (higher memory)
    num_workers=8,  # Increase from 4 to keep DataLoader prefetch ahead of train_step
    persistent_workers=True,  # Avoid per-epoch worker restart overhead
    dataset=dict(
        anno_file='data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260514.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),  # Require captions
            dict(
                type='LoadPreExtractedTextEmbedding',
                key='caption',
                allow_none=True,
                text_emb_augment_dir='qwen3_augmented',
            ),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs',
                smpl_type='smpl_22',
            ),
            # Compute198DimPosition MUST come before SmplTransToKimodoRootOnline:
            # LoadSmplx55 outputs 135-dim, Compute198DimPosition → 198-dim,
            # then SmplTransToKimodoRootOnline smooths translation on 198-dim.
            dict(type='Compute198DimPosition', key='motion'),
            # **KEY DIFFERENCE FROM E2/E4**: Convert SMPL Root → KIMODO Root
            # ADMM smoothing applied online during __getitem__
            dict(
                type='SmplTransToKimodoRootOnline',
                key='motion',
                admm_margin_m=0.06,  # 6cm margin on XZ plane
            ),
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
            ),
            dict(
                type='PrepareM2Mv2Condition',
                key='motion',
                sampler_version='v3',
                editing_prob=0.0,
                corruptor_names=[],
            ),
            # Override synthetic corruption with real Neutral source for
            # PerMo editing pairs (source_motion_path present in annotation).
            # Pass-through for regular T2M / completion samples.
            # kimodo_root_cfg: Apply same ADMM smoothing to source motion
            # so both src and tgt use KIMODO Root representation.
            dict(
                type='LoadEditingSourceMotion',
                kimodo_root_cfg=dict(admm_margin_m=0.06),
            ),
            dict(
                type='PackInputs',
                keys=[
                    'src_motion', 'tgt_motion', 'src_mask',
                    'tgt_length', 'src_length', 'edit_mode',
                    'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length',
                ],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=True,
                dummy_value=None,
            ),
        ],
    ),
)
