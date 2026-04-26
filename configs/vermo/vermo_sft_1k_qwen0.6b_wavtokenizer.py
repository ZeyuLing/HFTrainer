# VerMo SFT: 1k codebook, Qwen3-0.6B backbone
# Launch: bash tools/taiji_dist_train.sh configs/vermo/vermo_sft_1k_qwen0.6b_wavtokenizer.py 8

_base_ = './_base_vermo_sft_wavtokenizer.py'

model = dict(
    processor=dict(
        pretrained_text_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/Qwen3-0.6B',
            ),
        ),
        motion_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/vermo_vqvae2d_1k_rescale_iter47k',
            ),
        ),
    ),
    lm=dict(
        type='VermoQwen3ForCausalLM',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/Qwen3-0.6B',
        ),
    ),
)

load_from = dict(
    path='work_dirs/vermo_pretrain_1k_qwen0.6b_wavtokenizer/FILL_IN_PRETRAIN_CHECKPOINT',
    load_scope='model',
)
