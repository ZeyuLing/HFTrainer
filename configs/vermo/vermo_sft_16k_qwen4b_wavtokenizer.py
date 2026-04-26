# VerMo SFT: 16k codebook, Qwen3-4B-Instruct-2507 backbone
# Launch: bash tools/taiji_dist_train.sh configs/vermo/vermo_sft_16k_qwen4b_wavtokenizer.py 8

_base_ = './_base_vermo_sft_wavtokenizer.py'

model = dict(
    processor=dict(
        pretrained_text_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/Qwen3-4B-Instruct-2507',
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
    path='work_dirs/vermo_pretrain_16k_qwen4b_wavtokenizer/FILL_IN_PRETRAIN_CHECKPOINT',
    load_scope='model',
)
