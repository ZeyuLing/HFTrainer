# G1-native T2M continuation on the cleaned simple-caption / quality-filtered set.
#
# Build the annotation with:
#   python3 scripts/embodied/filter_g1_t2m_training_set.py \
#     --anno data/annotation/train_g1_t2m_emb.json \
#     --out data/annotation/train_g1_t2m_clean_simple_emb.json \
#     --num-workers 32

_base_ = "hymotion_g1_t2m_38dim_long.py"

work_dir = "work_dirs/hymotion_g1_t2m_38dim_clean_simple"

train_dataloader = dict(
    dataset=dict(
        anno_file="data/annotation/train_g1_t2m_clean_simple_emb.json",
        random_caption=False,
        require_embedding=True,
    ),
)

# Resume from the current broad generator unless explicitly overridden.
load_from = dict(
    _delete_=True,
    path="work_dirs/hymotion_g1_t2m_38dim/checkpoint-iter_107000",
    load_scope="full",
)
