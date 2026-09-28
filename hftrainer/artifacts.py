"""Checkpoint-only dispatch for repository-owned model implementations.

Hub metadata selects a member of a fixed allowlist; it cannot import remote
Python or supply executable configuration. Architecture stays in the native
component configs next to the weights.
"""

from __future__ import annotations

import json
from pathlib import Path

FAMILIES = {
    "vit": (
        "ViTBundle",
        "ClassificationPipeline",
        "image_classification",
        "hftrainer.tasks.image_classification",
    ),
    "llama": (
        "LlamaBundle",
        "CausalLMPipeline",
        "text_generation",
        "hftrainer.tasks.causal_language_modeling",
    ),
    "sd15": ("SD15Bundle", "SD15Pipeline", "text_to_image", "hftrainer.pipelines.sd15"),
    "dmd": ("DMDBundle", "DMDPipeline", "text_to_image", "hftrainer.pipelines.dmd"),
    "stylegan2": (
        "StyleGAN2Bundle",
        "StyleGAN2Pipeline",
        "unconditional_image_generation",
        "hftrainer.pipelines.stylegan2",
    ),
    "wan": ("WanBundle", "WanPipeline", "text_to_video", "hftrainer.pipelines.wan"),
    "ltx_video": (
        "LTXVideoBundle",
        "LTXVideoPipeline",
        "text_to_video",
        "hftrainer.pipelines.ltx_video",
    ),
    "minimax_h3": (
        "MiniMaxH3Bundle",
        "MiniMaxH3Pipeline",
        "multimodal_to_audio_video",
        "hftrainer.pipelines.minimax_h3",
    ),
}


def write_artifact_config(directory, bundle_name):
    """Add dispatch metadata while preserving any root architecture config."""
    family = next((k for k, v in FAMILIES.items() if v[0] == bundle_name), None)
    if family is None:
        raise ValueError(f"Unknown native bundle: {bundle_name}")
    path = Path(directory) / "config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    config["_hftrainer"] = {
        "format_version": 1,
        "family": family,
        "bundle": bundle_name,
        "pipeline": FAMILIES[family][1],
        "task": FAMILIES[family][2],
    }
    if family == "minimax_h3":
        root = Path(directory)
        config["_hftrainer"]["inference_ready"] = all(
            (root / name).is_dir() for name in ("text_encoder", "vae", "audio_vae")
        )
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


def config_from_pretrained(model, *, revision=None, device=None):
    """Resolve a local directory or pinned Hub snapshot to a safe runtime config."""
    import torch
    from mmengine.config import Config

    root = Path(model).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(str(model), revision=revision))
    path = root / "config.json"
    if not path.is_file():
        raise ValueError(
            f"{root} has no config.json; export a complete HFTrainer checkpoint first."
        )
    metadata = json.loads(path.read_text()).get("_hftrainer", {})
    family = metadata.get("family")
    if metadata.get("format_version") != 1 or family not in FAMILIES:
        raise ValueError("Unsupported HFTrainer checkpoint metadata in config.json")
    if metadata.get("inference_ready") is False:
        raise ValueError(
            "This checkpoint contains training-only components. Export the complete "
            "H3 bundle with configs/pretrained/export_minimax_h3.py before inference."
        )
    bundle, pipeline, task, _pipeline_module = FAMILIES[family]
    if family == "vit":
        import hftrainer.models.vit
        import hftrainer.tasks.image_classification
    elif family == "llama":
        import hftrainer.models.llama
        import hftrainer.tasks.causal_language_modeling
    elif family == "sd15":
        import hftrainer.models.sd15
        import hftrainer.pipelines.sd15
    elif family == "dmd":
        import hftrainer.models.dmd
        import hftrainer.pipelines.dmd
    elif family == "stylegan2":
        import hftrainer.models.stylegan2
        import hftrainer.pipelines.stylegan2
    elif family == "wan":
        import hftrainer.models.wan
        import hftrainer.pipelines.wan
    elif family == "ltx_video":
        import hftrainer.models.ltx_video
        import hftrainer.pipelines.ltx_video
    elif family == "minimax_h3":
        import hftrainer.models.minimax_h3
        import hftrainer.pipelines.minimax_h3  # noqa: F401 — registers the local implementation
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = {"type": bundle, "from_pretrained": str(root)}
    pipeline_cfg = {"type": pipeline}
    if family == "minimax_h3":
        # Place complete components sequentially; all weights remain local.
        pipeline_cfg["cpu_offload_device"] = device
    elif family == "wan":
        model_cfg["shared_pretrained_kwargs"] = {"torch_dtype": "bf16"}
    elif family == "ltx_video":
        model_cfg["device"] = device
        pipeline_cfg.update(
            height=512,
            width=768,
            num_frames=121,
            num_inference_steps=30,
            negative_prompt="blurry, distorted, low quality, artifacts",
        )
    return Config(
        {"model": model_cfg, "pipeline": pipeline_cfg, "inference": {"task": task}}
    )
