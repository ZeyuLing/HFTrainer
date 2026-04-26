# VerMo Pretrain: 16k codebook, Qwen3-0.6B backbone
# Launch: bash tools/taiji_dist_train.sh configs/vermo/vermo_pretrain_16k_qwen0.6b_wavtokenizer.py 8

_base_ = './_base_vermo_pretrain_wavtokenizer.py'

model = dict(
    processor=dict(
        pretrained_text_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/Qwen3-0.6B',
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
