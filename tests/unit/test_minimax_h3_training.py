"""Executable loss/backward contracts for MiniMax-H3 cached training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config

from hftrainer.models.lora import LoRALinear
from hftrainer.models.minimax_h3 import MiniMaxH3Bundle
from hftrainer.pipelines.builder import build_pipeline_from_cfg
from hftrainer.pipelines.minimax_h3 import MiniMaxH3Pipeline
from hftrainer.trainers.minimax_h3 import MiniMaxH3Trainer


def _bundle(*, lora: bool = False) -> MiniMaxH3Bundle:
    trainable = "lora" if lora else True
    adapter = (
        {
            "lora_cfg": {
                "rank": 2,
                "alpha": 2,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            },
            "checkpoint_format": "lora",
        }
        if lora
        else {}
    )
    return MiniMaxH3Bundle(
        transformer={
            "type": "MiniMaxH3Transformer3DModel",
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "hidden_size": 24,
            "num_layers": 2,
            "num_refiner_layers": 2,
            "ffn_dim": 32,
            "in_channels": 4,
            "audio_in_channels": 6,
            "patch_size": (1, 2, 2),
            "text_dim": 8,
            "freq_dim": 8,
            "time_embed_hidden_dim": 24,
            "time_embed_dim": 16,
            "rope_freq_dim": 2,
            "trainable": trainable,
            "save_ckpt": lora,
            **adapter,
        },
        scheduler={
            "type": "MiniMaxH3Scheduler",
            "shift": 12.0,
            "trainable": False,
            "save_ckpt": False,
        },
        audio_scheduler={
            "type": "MiniMaxH3Scheduler",
            "shift": 3.0,
            "trainable": False,
            "save_ckpt": False,
        },
    )


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator("cpu").manual_seed(17)
    return {
        "video_latents": torch.randn(batch_size, 4, 2, 4, 4, generator=generator),
        "audio_latents": torch.randn(batch_size, 2, 6, 3, generator=generator),
        "prompt_embeds": torch.randn(batch_size, 4, 8, generator=generator),
        "text_token_tags": torch.ones(4, dtype=torch.long),
    }


def test_cached_training_runs_joint_loss_and_backward():
    torch.manual_seed(23)
    bundle = _bundle()
    trainer = MiniMaxH3Trainer(bundle, mode="t2va")
    result = trainer.train_step(_batch())

    assert torch.isfinite(result["loss"])
    assert result["loss_video"] > 0
    assert result["loss_audio"] > 0
    result["loss"].backward()
    gradients = [
        parameter.grad
        for parameter in bundle.transformer.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )


def test_clean_visual_condition_rows_are_noised_to_condition_timestep():
    torch.manual_seed(29)
    bundle = _bundle()
    trainer = MiniMaxH3Trainer(bundle, mode="fl2va", keyframe_noise_aug=0.999)
    trainer._sample_timestep = lambda device: torch.tensor(0.5, device=device)
    batch = _batch()
    batch["keyframe_anchors"] = ("first",)
    batch["condition_video_rows"] = torch.randn(2, 4, 16)

    captured = {}
    original = bundle.predict_velocity

    def capture(video_rows, audio_rows, prompt_embeds, layout, timesteps, indices):
        captured["video_rows"] = video_rows.detach().clone()
        captured["timesteps"] = timesteps.detach().clone()
        return original(
            video_rows,
            audio_rows,
            prompt_embeds,
            layout,
            timesteps,
            indices,
        )

    bundle.predict_velocity = capture
    result = trainer.train_step(batch)
    assert torch.isfinite(result["loss"])
    conditioned = captured["video_rows"][:, :4]
    clean = batch["condition_video_rows"]
    assert not torch.equal(conditioned, clean)
    assert (conditioned - clean).abs().max() < 0.01
    assert torch.isclose(captured["timesteps"], torch.tensor(0.999), atol=1e-6).any()


def test_training_rejects_a_zero_objective():
    with pytest.raises(ValueError, match="At least one"):
        MiniMaxH3Trainer(_bundle(), video_loss_weight=0.0, audio_loss_weight=0.0)


def test_local_lora_recipe_has_trainable_adapters_and_backward():
    torch.manual_seed(31)
    bundle = _bundle(lora=True)
    assert bundle.is_lora_module("transformer")
    trainable_names = [
        name
        for name, parameter in bundle.transformer.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)

    result = MiniMaxH3Trainer(bundle).train_step(_batch(batch_size=1))
    result["loss"].backward()
    gradients = {
        name: parameter.grad
        for name, parameter in bundle.transformer.named_parameters()
        if parameter.requires_grad
    }
    assert all(gradient is not None for gradient in gradients.values())
    assert any(
        gradient.abs().sum() > 0
        for name, gradient in gradients.items()
        if "lora_B" in name
    )


def test_training_adapter_checkpoint_loads_and_merges_for_inference(
    tmp_path: Path,
):
    assert MiniMaxH3Pipeline.__name__ == "MiniMaxH3Pipeline"
    torch.manual_seed(47)
    training_bundle = _bundle(lora=True)
    adapter_layer = training_bundle.transformer.transformer_blocks[0].attn.to_q
    assert isinstance(adapter_layer, LoRALinear)
    with torch.no_grad():
        adapter_layer.lora_B.weight.normal_(mean=0.0, std=0.03)
        expected_weight = adapter_layer.merged_weight().clone()
    checkpoint = tmp_path / "model.pt"
    torch.save(training_bundle.state_dict_to_save(), checkpoint)

    torch.manual_seed(47)
    config = SimpleNamespace(
        model={
            "type": "MiniMaxH3Bundle",
            "variant": "fl2va",
            "transformer": {
                "type": "MiniMaxH3Transformer3DModel",
                "num_attention_heads": 2,
                "attention_head_dim": 16,
                "hidden_size": 24,
                "num_layers": 2,
                "num_refiner_layers": 2,
                "ffn_dim": 32,
                "in_channels": 4,
                "audio_in_channels": 6,
                "patch_size": (1, 2, 2),
                "text_dim": 8,
                "freq_dim": 8,
                "time_embed_hidden_dim": 24,
                "time_embed_dim": 16,
                "rope_freq_dim": 2,
                "trainable": "lora",
                "save_ckpt": True,
                "checkpoint_format": "lora",
                "lora_cfg": {
                    "rank": 2,
                    "alpha": 2,
                    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
                },
            },
            "scheduler": {
                "type": "MiniMaxH3Scheduler",
                "shift": 12.0,
                "trainable": False,
                "save_ckpt": False,
            },
            "audio_scheduler": {
                "type": "MiniMaxH3Scheduler",
                "shift": 3.0,
                "trainable": False,
                "save_ckpt": False,
            },
        },
        pipeline={"type": "MiniMaxH3Pipeline", "num_inference_steps": 2},
    )
    pipeline = build_pipeline_from_cfg(
        config,
        checkpoint_path=str(checkpoint),
        merge_lora=True,
        strict_checkpoint=True,
    )
    assert not pipeline.bundle.is_lora_module("transformer")
    merged_layer = pipeline.bundle.transformer.transformer_blocks[0].attn.to_q
    assert isinstance(merged_layer, torch.nn.Linear)
    torch.testing.assert_close(merged_layer.weight, expected_weight)


def test_lora_inference_recipe_matches_the_training_adapter_schema(
    repo_root: Path,
):
    training = Config.fromfile(repo_root / "configs/minimax_h3/train_h3_base_lora.py")
    inference = Config.fromfile(
        repo_root / "configs/minimax_h3/infer_h3_base_fl2va_lora.py"
    )
    assert training.model.transformer.trainable == "lora"
    assert inference.model.transformer.trainable == "lora"
    assert inference.model.transformer.checkpoint_format == "lora"
    assert inference.model.transformer.lora_cfg.to_dict() == (
        training.model.transformer.lora_cfg.to_dict()
    )
