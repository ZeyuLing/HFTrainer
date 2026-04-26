# VerMo SFT: 64k codebook, Qwen3-1.7B backbone
# Launch: bash tools/taiji_dist_train.sh configs/vermo/vermo_sft_64k_qwen1.7b_wavtokenizer.py 8

_base_ = './_base_vermo_sft_wavtokenizer.py'

model = dict(
    processor=dict(
        pretrained_text_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/Qwen3-1.7B',
            ),
        ),
        motion_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/vermo_vqvae2d_64k_rescale_iter47k',
            ),
        ),
    ),
    lm=dict(
        type='VermoQwen3ForCausalLM',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/Qwen3-1.7B',
        ),
    ),
)

load_from = dict(
    path='work_dirs/vermo_pretrain_64k_qwen1.7b_wavtokenizer/FILL_IN_PRETRAIN_CHECKPOINT',
    load_scope='model',
)
