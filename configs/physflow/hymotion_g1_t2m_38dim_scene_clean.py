# G1-native T2M continuation on the scene-clean PhysFlow training set.
#
# Build the annotations with:
#   python3 scripts/embodied/filter_g1_t2m_training_set.py \
#     --anno data/annotation/train_g1_t2m_emb.json \
#     --out data/annotation/train_g1_t2m_emb_scene_clean.json \
#     --allowed-caption-dirs '' --caption-source json --allow-empty-caption \
#     --min-words 1 --max-words 0 --max-chars 0 --skip-quality \
#     --scene-filter-mode hard --num-workers 32
#
# For paper before/after rows, use a checkpoint trained from the corresponding
# minus-heldout scene-clean set, or at minimum verify that heldout clips are not
# present in the supervised base used to initialize GenTrack.

_base_ = "hymotion_g1_t2m_38dim_long.py"

work_dir = "work_dirs/hymotion_g1_t2m_38dim_scene_clean"

train_dataloader = dict(
    dataset=dict(
        anno_file="data/annotation/train_g1_t2m_emb_scene_clean.json",
        random_caption=False,
        require_embedding=True,
    ),
)
