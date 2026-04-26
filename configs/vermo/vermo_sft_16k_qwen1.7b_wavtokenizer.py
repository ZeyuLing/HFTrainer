# VerMo SFT: 16k codebook, Qwen3-1.7B backbone
# Launch: bash tools/taiji_dist_train.sh configs/vermo/vermo_sft_16k_qwen1.7b_wavtokenizer.py 8

_base_ = './_base_vermo_sft_wavtokenizer.py'

model = dict(
    processor=dict(
        pretrained_text_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path='checkpoints/Qwen3-1.7B',
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

train_dataloader = dict(batch_size=2)

# Legacy .pth converted to checkpoint-iter_35000/ for auto-resume compatibility.
# load_from not needed — use --auto-resume instead.
