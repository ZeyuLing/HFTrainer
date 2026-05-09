_base_ = '../hymotion_m2m_v2_uncond_local_046b.py'

# Main-flow loss ablation: follow KIMODO's component-normalized spirit for
# the velocity target.  Each semantic block gets its own mean before the
# blocks are averaged, and padding / known MAN cells are still excluded by
# M2MLoss.  The old 5x translation compensation is disabled because
# translation now receives a full semantic slot.
model = dict(
    losses_cfg=dict(
        velocity_loss_reduction='component_mean',
        trans_dim_weight=1.0,
    ),
)
