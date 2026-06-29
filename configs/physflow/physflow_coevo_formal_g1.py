# PhysFlow FORMAL co-evolution config (G1-native generator), 4x8 V100.
#
# Goal: demonstrate the *key capability* -- the co-evolving tracker learns to
# track agile/dynamic motions that the frozen baseline tracker drops -- by
# training the generator on the FULL diverse G1 prompt bank (locomotion +
# gestures + agile) while the trainee tracker co-trains on the accepted
# generations.  The 80 held-out AGILE clips (_heldout_agile.json) are EXCLUDED
# from this prompt bank so the tracker key-capability eval is truly held out.
#
# Driven by scripts/embodied/physflow_coevolve_orchestrator.py, which overrides
# per-round max_iters / tracker_pool_dir / checkpoint interval / load_from.

_base_ = 'physflow_online_adv_g1_38dim.py'

work_dir = 'work_dirs/physflow_coevo_formal_g1'

# Full diverse prompt bank MINUS the held-out agile eval clips and high-
# confidence scene-interaction clips.
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(anno_file='data/annotation/train_g1_t2m_emb_minus_heldout_scene_clean.json'),
)

# Formal online fine-tune lr (same as the validated v6 generator run).
optimizer = dict(type='AdamW', lr=5e-6, betas=[0.9, 0.99], weight_decay=0.0)

trainer = dict(
    # Stream judge-accepted GT + generations into the trainee pool so the
    # co-evolving tracker sees a steady diverse + agile diet.
    export_gt_to_pool=True,
    gt_pool_freq=2,
    pool_max_motions=8000,
    tracker_pool_dir='work_dirs/physflow_coevo_formal_g1/tracker_motion_pool',
)
