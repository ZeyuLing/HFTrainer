# VerMo SFT: 4k codebook, Qwen3-4B-Instruct-2507 backbone
# Launch: bash tools/taiji_dist_train.sh configs/vermo/vermo_sft_4k_qwen4b_wavtokenizer.py 8

_base_ = './_base_vermo_sft_wavtokenizer.py'

model = dict(
    processor=dict(
        pretrained_text_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/Qwen3-4B-Instruct-2507',
            ),
        ),
        motion_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/vermo_vqvae2d_4k_rescale_iter47k',
            ),
        ),
    ),
    lm=dict(
        type='VermoQwen3ForCausalLM',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/Qwen3-4B-Instruct-2507',
        ),
    ),
)

load_from = dict(
    path='work_dirs/vermo_pretrain_4k_qwen4b_wavtokenizer/FILL_IN_PRETRAIN_CHECKPOINT',
    load_scope='model',
)
