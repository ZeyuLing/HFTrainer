"""HYMotion M2M — T2M-only specialist training (continue from T2M Lite backbone).

Goal
----
Diagnose / fix the weak pure-text-to-motion result of the unified M2M model
(paper tab:t2m FID ~91, R@1 ~0.33) by training a *single-task* T2M specialist
under the SAME 198-dim M2M architecture and evaluator pipeline.

How "T2M-only" is enforced
--------------------------
The v3 Rank-K mask sampler builds ``lock = OR of K atoms`` and emits
``mask = 1 - lock`` with the convention ``1 = generate, 0 = known``. With
``k_weights = [1, 0, 0, 0, 0]`` the number of atoms is forced to ``K = 0``,
so ``lock = ∅`` and ``src_mask`` is **all-generate** on every sample: no motion
coordinate is ever observed and the reactive channel is zero. Combined with
``editing_prob = 0`` (no corruptor branch) this is exactly pure text-to-motion
(text condition only), expressed through the same completion interface.

Data source (PRISM / HYMotion-T2M aligned)
-------------------------------------------
``train_hq_motionhub_hymotion.json`` = HYMotion 400h (549K) + MotionHub HQ
(275K). Its MotionHub HQ slice carries the ``human_checked_augmented_caption``
hierarchical captions, whose style is much closer to the HumanML3D T2M
evaluator than the M2M ``improved_simple_augmented_caption`` source. Qwen3+CLIP
features for these captions were pre-extracted into ``data/motionhub_qwen3``.

Initialization
--------------
Continues from the HY-Motion-T2M-1.0-Lite backbone (the pretrained T2M
foundation model MotionCanvas itself is fine-tuned from), NOT an M2M editing
checkpoint, to inherit the strong unconditional T2M prior without
editing-task interference. The 594-dim VACE input projection is re-initialized
by the bundle when loading the T2M-only checkpoint.

Launch (Taiji, 64 V100 = 8 host x 8 gpu)
----------------------------------------
    python3 tools/taiji_submit.py m2m_t2m_only_from_lite \
        configs/hymotion_m2m/hymotion_m2m_t2m_only_from_lite_046b.py \
        --host_num 8
"""

_base_ = './hymotion_m2m_smpl_caption_cleandata_ablation.py'

work_dir = 'work_dirs/hymotion_m2m_t2m_only_from_lite'

# Save every epoch (base ablation uses interval=5) so we can track the
# per-epoch T2M metric trend during this short fine-tune.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=100, save_last=True),
)

# Continue from the pretrained T2M backbone (not an M2M editing ckpt).
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

train_dataloader = dict(
    # 32-GPU (4 host x 8 A100-40GB) multinode run. fp32 sdpa uses the
    # memory-efficient backend (Flash kernel needs fp16/bf16). bs=20 already
    # peaks ~32GB/40GB, so bs=24 (~36GB) is the safe fp32 max with NCCL
    # multinode buffers; global batch = 32*24 = 768 (close to the original
    # 64-GPU design of 1280, so lr is unchanged).
    batch_size=24,
    num_workers=8,
    dataset=dict(
        anno_file='data/annotation/train_hq_motionhub_hymotion.json',
        pipeline=[
            dict(allow_none=False, type='LoadCompatibleCaption'),
            dict(
                allow_none=True,
                key='caption',
                type='LoadPreExtractedTextEmbedding'),
            dict(
                key='motion',
                rot_type='rotation_6d',
                smpl_type='smpl_22',
                transl_type='abs',
                type='LoadSmplx55'),
            dict(key='motion', type='Compute198DimPosition'),
            dict(
                allow_shorter=True,
                clip_len=360,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
                pad_mode='replicate',
                type='RandomCropPadding'),
            # T2M-only: force K=0 -> lock=empty -> src_mask all-generate.
            # No corruptor / editing branch.
            dict(
                type='PrepareM2Mv2Condition',
                key='motion',
                sampler_version='v3',
                editing_prob=0.0,
                corruptor_names=[],
                max_corruptions=0,
                v3_config=dict(
                    k_weights=[1.0, 0.0, 0.0, 0.0, 0.0],
                    editing_prob=0.0,
                ),
            ),
            dict(type='LoadEditingSourceMotion'),
            dict(
                dummy_value=None,
                keys=[
                    'src_motion',
                    'tgt_motion',
                    'src_mask',
                    'tgt_length',
                    'src_length',
                    'edit_mode',
                    'text_vec_raw',
                    'text_ctxt_raw',
                    'text_ctxt_raw_length',
                ],
                meta_keys=[
                    'motion_path',
                    'fps',
                ],
                set_dummy_value=True,
                type='PackInputs'),
        ],
    ),
)
