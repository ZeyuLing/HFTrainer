"""Main VerMo checkpoint test-set viewer export config with 2s clips."""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager.py'

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    shuffle=False,
    dataset=dict(
        anno_file='data/annotation/vermo_test_mainpipeline_18tasks_36_short2s_20260604.json',
        task_mode='auto',
        log_task_iter=20,
    ),
)
