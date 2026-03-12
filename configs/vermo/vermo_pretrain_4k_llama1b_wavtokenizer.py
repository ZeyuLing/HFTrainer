_base_ = './_base_vermo_pretrain_wavtokenizer.py'

model = dict(
    processor=dict(
        motion_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/vermo_vqvae2d_4k_rescale_iter47k',
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
