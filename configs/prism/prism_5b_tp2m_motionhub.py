_base_ = './prism_1b_tp2m_motionhub.py'

model = dict(
    transformer=dict(
        ffn_dim=14336,
        num_attention_heads=24,
    ),
)

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        anno_file='data/motionhub/train.json',
    ),
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=10000,
    val_interval=10000,
)
