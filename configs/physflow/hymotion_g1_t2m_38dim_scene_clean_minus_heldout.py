# G1-native T2M continuation on the scene-clean, heldout-excluded PhysFlow set.
#
# This is the supervised base checkpoint that should initialize the final
# Table-2 generator before/after experiments.  It excludes the flat-ground
# held-out generator/eval clips and high-confidence scene-interaction clips.

_base_ = "hymotion_g1_t2m_38dim_long.py"

work_dir = "work_dirs/hymotion_g1_t2m_38dim_scene_clean_minus_heldout"

train_dataloader = dict(
    dataset=dict(
        anno_file="data/annotation/train_g1_t2m_emb_minus_heldout_scene_clean.json",
        random_caption=False,
        require_embedding=True,
    ),
)
