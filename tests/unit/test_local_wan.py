"""CPU tests for the dependency-isolated local Wan implementation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _tiny_bundle_config():
    return {
        "text_encoder": {
            "type": "UMT5EncoderModel",
            "vocab_size": 300,
            "d_model": 16,
            "d_kv": 8,
            "d_ff": 32,
            "num_layers": 1,
            "num_heads": 2,
            "dropout_rate": 0.0,
            "trainable": False,
            "save_ckpt": False,
        },
        "vae": {
            "type": "AutoencoderKLWan",
            "base_dim": 8,
            "z_dim": 4,
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "latents_mean": [0.0] * 4,
            "latents_std": [1.0] * 4,
            "trainable": False,
            "save_ckpt": False,
        },
        "transformer": {
            "type": "WanTransformer3DModel",
            "patch_size": (1, 2, 2),
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 16,
            "ffn_dim": 32,
            "num_layers": 1,
            "gradient_checkpointing": True,
            "trainable": True,
            "save_ckpt": True,
        },
        "scheduler": {
            "type": "FlowMatchEulerDiscreteScheduler",
            "num_train_timesteps": 20,
            "shift": 3.0,
            "trainable": False,
            "save_ckpt": False,
        },
        "tokenizer_path": None,
        "max_token_length": 12,
    }


@pytest.fixture()
def tiny_bundle():
    from hftrainer.models.wan.bundle import WanBundle

    torch.manual_seed(7)
    bundle = WanBundle(**_tiny_bundle_config())
    bundle.train()
    return bundle


def test_import_blocker_proves_wan_has_no_external_model_runtime(repo_root: Path):
    script = r"""import importlib.abc
import sys

class ForbiddenModelPackages(importlib.abc.MetaPathFinder):
    roots = {"transformers", "diffusers", "peft"}

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in self.roots or root.startswith("ltx_"):
            raise AssertionError(f"forbidden model-package import: {fullname}")
        return None

sys.meta_path.insert(0, ForbiddenModelPackages())
from hftrainer.models.wan.bundle import WanBundle
from hftrainer.models.wan.network import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UMT5EncoderModel,
    WanTokenizer,
    WanTransformer3DModel,
)
assert WanBundle._LOCAL_TYPES["text_encoder"] is UMT5EncoderModel
assert WanBundle._LOCAL_TYPES["vae"] is AutoencoderKLWan
assert WanBundle._LOCAL_TYPES["transformer"] is WanTransformer3DModel
assert WanBundle._LOCAL_TYPES["scheduler"] is FlowMatchEulerDiscreteScheduler
assert WanTokenizer(vocab_size=259)(["local"], return_tensors="pt").input_ids.shape[0] == 1
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tiny_tokenizer_text_vae_and_transformer_shapes(tiny_bundle):
    tokens = tiny_bundle.tokenizer(
        ["a cat", "海浪"],
        padding="max_length",
        truncation=True,
        max_length=12,
        return_tensors="pt",
    )
    assert tokens.input_ids.shape == (2, 12)
    assert tokens.attention_mask.dtype == torch.long

    text = tiny_bundle.encode_text(["a cat", "海浪"])
    assert text.shape == (2, 12, 16)
    assert torch.count_nonzero(text[tokens.attention_mask == 0]) == 0

    video_bcthw = torch.randn(2, 3, 5, 16, 16)
    latents = tiny_bundle.encode_video(video_bcthw)
    assert latents.shape == (2, 4, 2, 2, 2)
    decoded = tiny_bundle.decode_latent(latents)
    assert decoded.shape == video_bcthw.shape
    assert decoded.min() >= -1 and decoded.max() <= 1

    prediction = tiny_bundle.predict_noise(
        latents,
        torch.tensor([2, 7]),
        text,
    )
    assert prediction.shape == latents.shape
    assert torch.isfinite(prediction).all()


def test_tiny_flow_denoise_loss_and_backward(tiny_bundle):
    text = tiny_bundle.encode_text(["one", "two"])
    latents = tiny_bundle.encode_video(torch.randn(2, 5, 3, 16, 16))
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([3.0, 14.0])
    fraction = timesteps / tiny_bundle.scheduler.config.num_train_timesteps
    fraction = fraction.view(2, 1, 1, 1, 1)
    noisy = fraction * noise + (1.0 - fraction) * latents
    target = noise - latents
    prediction = tiny_bundle.predict_noise(noisy, timesteps.long(), text)
    loss = F.mse_loss(prediction.float(), target.float())
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in tiny_bundle.transformer.parameters()
        if parameter.requires_grad
    ]
    assert torch.isfinite(loss)
    assert any(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )
    assert all(
        parameter.grad is None for parameter in tiny_bundle.text_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in tiny_bundle.vae.parameters())

    scheduler = tiny_bundle.scheduler
    scheduler.set_timesteps(3)
    sample = latents.detach()
    for timestep in scheduler.timesteps:
        with torch.no_grad():
            velocity = tiny_bundle.predict_noise(
                sample,
                timestep.expand(sample.shape[0]),
                text,
            )
        sample = scheduler.step(velocity, timestep, sample).prev_sample
    assert sample.shape == latents.shape
    assert scheduler.step_index == 3
    assert torch.isfinite(sample).all()


def test_strict_local_bundle_save_load_and_manifest(tiny_bundle, tmp_path: Path):
    tiny_bundle.eval()
    text = tiny_bundle.encode_text(["round trip"])
    latents = torch.randn(1, 4, 2, 2, 2)
    expected = tiny_bundle.predict_noise(latents, torch.tensor([5]), text)

    artifact = tmp_path / "wan-local"
    manifest_path = Path(tiny_bundle.save_pretrained(str(artifact)))
    assert manifest_path.name == "wan_bundle_manifest.json"
    for component in ("text_encoder", "vae", "transformer"):
        assert (artifact / component / "wan_local_manifest.json").is_file()
    assert (artifact / "tokenizer" / "wan_tokenizer_manifest.json").is_file()
    assert (artifact / "scheduler" / "wan_scheduler_manifest.json").is_file()

    from hftrainer.models.wan.bundle import WanBundle

    restored = WanBundle.from_pretrained(str(artifact), strict=True)
    restored.eval()
    actual_text = restored.encode_text(["round trip"])
    actual = restored.predict_noise(latents, torch.tensor([5]), actual_text)
    assert restored.max_token_length == 12
    assert torch.equal(text, actual_text)
    assert torch.equal(expected, actual)
    assert restored.transformer._load_report["parameter_coverage"] == 1.0

    transformer_config = artifact / "transformer" / "config.json"
    transformer_config.write_text(
        transformer_config.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        WanBundle.from_pretrained(str(artifact), strict=True)
