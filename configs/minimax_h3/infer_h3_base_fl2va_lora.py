"""MiniMax-H3 Base FL2VA inference with a local HFTrainer LoRA checkpoint."""

# ruff: noqa: C408 - MMEngine config files conventionally use dict(...).

_base_ = ["./infer_h3_base_fl2va.py"]

# This adapter schema must match train_h3_base_lora.py. Pass its model.pt (or
# enclosing checkpoint directory) with --checkpoint, then use --merge-lora to
# fold the adapter into the repository-local transformer before inference.
model = dict(
    transformer=dict(
        trainable="lora",
        save_ckpt=True,
        checkpoint_format="lora",
        lora_cfg=dict(
            rank=16,
            alpha=16,
            dropout=0.0,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
    )
)
