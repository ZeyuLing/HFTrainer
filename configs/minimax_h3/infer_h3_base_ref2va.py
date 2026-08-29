"""MiniMax-H3 Base ordered-reference synchronized A/V inference."""

# ruff: noqa: C408 - MMEngine config files conventionally use dict(...).

_base_ = ["./infer_h3_base_fl2va.py"]

root = __import__("os").environ.get("MINIMAX_H3_ROOT", "checkpoints/MiniMax-H3")
load_device = __import__("os").environ.get("MINIMAX_H3_LOAD_DEVICE", "cpu")
transformer_device = __import__("os").environ.get(
    "MINIMAX_H3_TRANSFORMER_DEVICE", load_device
)

model = dict(
    variant="ref2va",
    transformer=dict(
        type="MiniMaxH3Transformer3DModel",
        from_pretrained=dict(
            pretrained_model_name_or_path=root,
            subfolder="transformer_ref",
            torch_dtype="bf16",
            low_cpu_mem_usage=True,
            device=transformer_device,
            strict=True,
        ),
        trainable=False,
        save_ckpt=False,
    ),
)
