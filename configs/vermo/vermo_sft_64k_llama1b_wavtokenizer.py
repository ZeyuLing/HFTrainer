# VerMo SFT: 64k codebook, Llama-3.2-1B-Instruct backbone
# Launch: bash tools/taiji_dist_train.sh configs/vermo/vermo_sft_64k_llama1b_wavtokenizer.py 8

_base_ = './_base_vermo_sft_wavtokenizer.py'

model = dict(
    processor=dict(
        motion_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/vermo_vqvae2d_64k_rescale_iter47k',
            ),
        ),
    ),
    lm=dict(
        type='VermoLlamaForCausalLM',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/Llama-3.2-1B-Instruct',
        ),
    ),
)

load_from = dict(
    path='work_dirs/vermo_pretrain_64k_llama1b_wavtokenizer/FILL_IN_PRETRAIN_CHECKPOINT',
    load_scope='model',
)
