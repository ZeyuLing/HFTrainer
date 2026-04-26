# VerMo Pretrain: 1k codebook, Llama-3.2-3B-Instruct backbone
# Launch: bash tools/taiji_dist_train.sh configs/vermo/vermo_pretrain_1k_llama3b_wavtokenizer.py 8

_base_ = './_base_vermo_pretrain_wavtokenizer.py'

model = dict(
    processor=dict(
        pretrained_text_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/Llama-3.2-3B-Instruct',
            ),
        ),
        motion_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/vermo_vqvae2d_1k_rescale_iter47k',
            ),
        ),
    ),
    lm=dict(
        type='VermoLlamaForCausalLM',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/Llama-3.2-3B-Instruct',
        ),
    ),
)
