# PhysFlow formal multi-node run on the filtered HYMotion physical-text corpus.
#
# This is the current training configuration requested on 2026-06-04:
#   - train prompts: filtered HYMotion no-scene/physically-real text corpus
#   - base checkpoint: previous formal MN checkpoint through physflow_online_adv_mn.py
#   - tracker pool: isolated from earlier HML3D/HumanML3D runs
#
# Text features are intentionally stored in a HYMotion-specific namespace. Use
# tools/physflow_hymotion_mn_start.sh so the cache is precomputed before DDP.

_base_ = './physflow_online_adv_mn.py'

work_dir = 'work_dirs/physflow_online_adv_mn_hymotion_real'

# Continue from the latest completed formal MN checkpoint. This loads model
# weights only because the HYMotion corpus changes the training distribution
# and this run owns a separate work_dir/checkpoint schedule.
load_from = dict(
    _delete_=True,
    path='work_dirs/physflow_online_adv_mn/checkpoint-iter_1500',
    load_scope='model',
)

_corpus = 'configs/experiments/physflow_kimodo_g1/physflow_text_hymotion_g1_real_train.jsonl'
_feature_dir = 'data/kimodo_text_feature/kimodo_g1_llm2vec_hymotion_real_train'

trainer = dict(
    tracker_pool_dir='work_dirs/physflow_online_adv_mn_hymotion_real/tracker_motion_pool',
)

train_dataloader = dict(
    dataset=dict(
        corpus_file=_corpus,
        feature_dir=_feature_dir,
        split='train',
    ),
)
