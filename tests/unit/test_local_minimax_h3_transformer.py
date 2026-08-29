"""CPU contracts for the repository-local MiniMax-H3 transformer/scheduler."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _tiny_config() -> dict:
    # As in the upstream test, inner attention width deliberately differs from
    # the residual width and MM-RoPE covers only part of every head.
    return {
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
    }


def _tiny_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator("cpu").manual_seed(1234)
    num_text, num_audio, num_video = 4, 6, 8
    sequence_length = num_text + num_audio + num_video
    text_indices = torch.arange(num_text)
    audio_indices = torch.arange(num_text, num_text + num_audio)
    video_indices = torch.arange(num_text + num_audio, sequence_length)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0
    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    timestep_indices[audio_indices] = 1
    position_ids = torch.zeros(sequence_length, 3)
    position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32)
    position_ids[video_indices, 1] = torch.arange(num_video, dtype=torch.float32) % 4
    position_ids[video_indices, 2] = torch.arange(num_video, dtype=torch.float32) % 2
    return {
        "hidden_states": torch.randn(batch_size, num_video, 16, generator=generator),
        "audio_hidden_states": torch.randn(
            batch_size, num_audio, 6, generator=generator
        ),
        "encoder_hidden_states": torch.randn(
            batch_size, num_text, 8, generator=generator
        ),
        "timestep": torch.tensor([0.7, 0.3]),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }


def test_tiny_forward_output_shape_and_checkpointed_backward():
    from hftrainer.models.minimax_h3.network import MiniMaxH3Transformer3DModel

    torch.manual_seed(9)
    model = MiniMaxH3Transformer3DModel(**_tiny_config())
    model.train()
    model.gradient_checkpointing_enable()
    inputs = _tiny_inputs()
    inputs["hidden_states"].requires_grad_(True)
    inputs["audio_hidden_states"].requires_grad_(True)

    output = model(**inputs)
    tuple_output = model(**inputs, return_dict=False)
    assert output.sample.shape == (2, 8, 16)
    assert output.audio_sample is not None
    assert output.audio_sample.shape == (2, 6, 6)
    # Frozen against the official Diffusers implementation at c1bf18c9 with
    # the same seed/state/input.  The local PyTorch SDPA path is exactly equal
    # on the reference CPU runtime; a small tolerance keeps the test portable.
    torch.testing.assert_close(
        output.sample[0, -1, -8:],
        torch.tensor(
            [
                -0.0508790761,
                0.4136232436,
                0.1779966950,
                -0.1134405881,
                0.1245228797,
                -0.0208541695,
                -0.6877627373,
                -0.6232486963,
            ]
        ),
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        output.audio_sample[1, -1],
        torch.tensor(
            [
                0.2042890489,
                1.1924262047,
                -0.1194638610,
                -0.6689125299,
                -1.0743861198,
                -0.1414645463,
            ]
        ),
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(output.sample, tuple_output[0])
    torch.testing.assert_close(output.audio_sample, tuple_output[1])

    loss = output.sample.float().square().mean()
    loss = loss + output.audio_sample.float().square().mean()
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )
    assert inputs["hidden_states"].grad is not None
    assert inputs["audio_hidden_states"].grad is not None
    assert model.token_refiner.gradient_checkpointing is True


def test_official_state_dict_names_and_original_key_conversion():
    from hftrainer.models.minimax_h3.network import MiniMaxH3Transformer3DModel

    model = MiniMaxH3Transformer3DModel(**_tiny_config())
    keys = set(model.state_dict())
    expected = {
        "proj_in.weight",
        "audio_proj_in.weight",
        "context_embedder.weight",
        "time_embedder.linear_1.weight",
        "time_embedder.linear_2.weight",
        "token_refiner.refiner_blocks.0.attn.to_q.weight",
        "token_refiner.refiner_blocks.0.ff.net.0.proj.weight",
        "transformer_blocks.0.attn.to_k.weight",
        "transformer_blocks.0.attn.to_v.weight",
        "transformer_blocks.0.attn.to_out.0.weight",
        "transformer_blocks.0.ff.net.2.weight",
        "transformer_blocks.0.adaln_proj.linear.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
        "audio_proj_out.weight",
    }
    assert expected <= keys
    assert not any("qkv_proj" in key or ".mlp." in key for key in keys)

    raw_config = {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 24,
        "num_layers": 2,
        "token_refiner_num_layers": 2,
        "ffn_hidden_size": 32,
        "latents_dim": 4,
        "audio_latents_dim": 6,
        "patch_size": [1, 2, 2],
        "text_dim": 8,
        "timestep_input_dim": 8,
        "time_embed_hidden_size": 24,
        "time_embed_dim": 16,
        "rope_inv_freq_len": 2,
        "adaln_out_features": 432,
        "final_adaln_out_features": 48,
    }
    converted_config = MiniMaxH3Transformer3DModel._convert_config(raw_config)
    assert converted_config["num_refiner_layers"] == 2
    assert converted_config["ffn_dim"] == 32
    assert converted_config["in_channels"] == 4
    assert "adaln_out_features" not in converted_config

    # Raw shards interleave q/k/v inside every head.  Verify the exact reorder.
    raw = torch.arange(3 * 2 * 16 * 24, dtype=torch.float32).reshape(96, 24)
    converted = MiniMaxH3Transformer3DModel._convert_checkpoint_tensor(
        "blocks.0.attn.qkv_proj.weight", raw, converted_config
    )
    grouped = raw.reshape(2, 48, 24)
    q0, k0, v0 = grouped.split(16, dim=1)
    torch.testing.assert_close(
        converted["transformer_blocks.0.attn.to_q.weight"],
        q0.reshape(32, 24),
    )
    torch.testing.assert_close(
        converted["transformer_blocks.0.attn.to_k.weight"],
        k0.reshape(32, 24),
    )
    torch.testing.assert_close(
        converted["transformer_blocks.0.attn.to_v.weight"],
        v0.reshape(32, 24),
    )


def test_scheduler_official_golden_values_and_roundtrip(tmp_path: Path):
    from hftrainer.models.minimax_h3.network import MiniMaxH3Scheduler

    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(5)
    torch.testing.assert_close(
        scheduler.sigmas,
        torch.tensor([1.0, 0.9729729891, 0.9230769277, 0.8, 0.0]),
        rtol=0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        scheduler.timesteps,
        torch.tensor([0.0, 0.0270270109, 0.0769230723, 0.2]),
        rtol=0,
        atol=1e-7,
    )
    assert scheduler.num_inference_steps == 4
    assert scheduler.init_noise_sigma == 1.0

    sample = torch.tensor([1.0])
    velocity = torch.tensor([0.5])
    first = scheduler.step(velocity, scheduler.timesteps[0], sample).prev_sample
    torch.testing.assert_close(first, torch.tensor([1.0135135651]), atol=1e-7, rtol=0)
    noised = scheduler.scale_noise(torch.tensor([[2.0]]), 0.25, torch.tensor([[-2.0]]))
    torch.testing.assert_close(noised, torch.tensor([[-1.0]]))

    artifact = tmp_path / "scheduler"
    manifest = Path(scheduler.save_pretrained(artifact))
    assert manifest.name == "minimax_h3_scheduler_manifest.json"
    restored = MiniMaxH3Scheduler.from_pretrained(artifact)
    assert restored.shift == 12.0


def test_randn_tensor_matches_cpu_generator_and_generator_list_semantics():
    from hftrainer.models.minimax_h3.network.common import randn_tensor

    expected_generator = torch.Generator("cpu").manual_seed(42)
    actual_generator = torch.Generator("cpu").manual_seed(42)
    expected = torch.randn((2, 3), generator=expected_generator, dtype=torch.float64)
    actual = randn_tensor(
        (2, 3),
        generator=actual_generator,
        device="cpu",
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    expected_rows = torch.cat(
        [
            torch.randn(
                (1, 3),
                generator=torch.Generator("cpu").manual_seed(seed),
            )
            for seed in (7, 11)
        ]
    )
    actual_rows = randn_tensor(
        (2, 3),
        generator=[
            torch.Generator("cpu").manual_seed(7),
            torch.Generator("cpu").manual_seed(11),
        ],
        device="cpu",
    )
    torch.testing.assert_close(actual_rows, expected_rows, atol=0, rtol=0)

    expected_single = torch.randn(
        (2, 3), generator=torch.Generator("cpu").manual_seed(19)
    )
    actual_single = randn_tensor(
        (2, 3),
        generator=[torch.Generator("cpu").manual_seed(19)],
        device="cpu",
    )
    torch.testing.assert_close(actual_single, expected_single, atol=0, rtol=0)

    # A non-CPU target available on every test machine exercises the same
    # route as CPU-generator -> CUDA without requiring accelerator hardware.
    meta_noise = randn_tensor(
        (2, 3),
        generator=torch.Generator("cpu").manual_seed(42),
        device="meta",
    )
    assert meta_noise.is_meta
    assert meta_noise.shape == (2, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_randn_tensor_routes_cpu_generators_to_cuda_without_device_mismatch():
    from hftrainer.models.minimax_h3.network.common import randn_tensor

    expected = torch.randn(
        (2, 3), generator=torch.Generator("cpu").manual_seed(42)
    ).cuda()
    actual = randn_tensor(
        (2, 3),
        generator=torch.Generator("cpu").manual_seed(42),
        device="cuda",
    )
    assert actual.is_cuda
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    expected_rows = torch.cat(
        [
            torch.randn(
                (1, 3),
                generator=torch.Generator("cpu").manual_seed(seed),
            )
            for seed in (7, 11)
        ]
    ).cuda()
    actual_rows = randn_tensor(
        (2, 3),
        generator=[
            torch.Generator("cpu").manual_seed(7),
            torch.Generator("cpu").manual_seed(11),
        ],
        device="cuda",
    )
    assert actual_rows.is_cuda
    torch.testing.assert_close(actual_rows, expected_rows, atol=0, rtol=0)


def test_sharded_safetensors_artifact_roundtrip_and_schema_guard(tmp_path: Path):
    from hftrainer.models.minimax_h3.network import MiniMaxH3Transformer3DModel

    torch.manual_seed(31)
    model = MiniMaxH3Transformer3DModel(**_tiny_config()).eval()
    inputs = _tiny_inputs(batch_size=1)
    with torch.no_grad():
        expected = model(**inputs)

    artifact = tmp_path / "transformer"
    manifest_path = Path(
        model.save_pretrained(artifact, safe_serialization=True, max_shard_size="20KB")
    )
    assert manifest_path.name == "minimax_h3_local_manifest.json"
    index_path = artifact / "diffusion_pytorch_model.safetensors.index.json"
    assert index_path.is_file()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(set(index["weight_map"].values())) > 1

    restored = MiniMaxH3Transformer3DModel.from_pretrained(
        artifact, low_cpu_mem_usage=True
    ).eval()
    with torch.no_grad():
        actual = restored(**inputs)
    torch.testing.assert_close(expected.sample, actual.sample, atol=0, rtol=0)
    torch.testing.assert_close(
        expected.audio_sample, actual.audio_sample, atol=0, rtol=0
    )
    assert restored._load_report["parameter_coverage"] == 1.0

    config_path = artifact / "config.json"
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MiniMaxH3Transformer3DModel.from_pretrained(artifact)


def test_partial_foreign_checkpoint_requires_explicit_opt_in(tmp_path: Path):
    from hftrainer.models.minimax_h3.network import MiniMaxH3Transformer3DModel

    model = MiniMaxH3Transformer3DModel(**_tiny_config())
    artifact = tmp_path / "partial"
    artifact.mkdir()
    config = model.config.to_dict()
    config["_class_name"] = type(model).__name__
    (artifact / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save(
        {"proj_in.bias": model.proj_in.bias.detach().clone()},
        artifact / "diffusion_pytorch_model.bin",
    )
    with pytest.raises(RuntimeError, match="Strict.*load failed"):
        MiniMaxH3Transformer3DModel.from_pretrained(artifact)
    with pytest.raises(RuntimeError, match="Only .* parameters are covered"):
        MiniMaxH3Transformer3DModel.from_pretrained(artifact, strict=False)
    accepted = MiniMaxH3Transformer3DModel.from_pretrained(
        artifact,
        strict=False,
        allow_partial_load=True,
        low_cpu_mem_usage=False,
    )
    assert accepted._load_report["parameter_coverage"] < 0.5


def test_repeated_component_save_removes_stale_formats_and_manifest_limits_load(
    tmp_path: Path,
):
    from hftrainer.models.minimax_h3.network import MiniMaxH3Transformer3DModel

    torch.manual_seed(37)
    model = MiniMaxH3Transformer3DModel(**_tiny_config()).eval()
    artifact = tmp_path / "repeated-save"

    model.save_pretrained(artifact, safe_serialization=True, max_shard_size="20KB")
    assert (artifact / "diffusion_pytorch_model.safetensors.index.json").is_file()
    assert list(artifact.glob("diffusion_pytorch_model-*.safetensors"))

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.125)
    model.save_pretrained(artifact, safe_serialization=True, max_shard_size="1GB")
    assert (artifact / "diffusion_pytorch_model.safetensors").is_file()
    assert not (artifact / "diffusion_pytorch_model.safetensors.index.json").exists()
    assert not list(artifact.glob("diffusion_pytorch_model-*.safetensors"))
    restored_safe = MiniMaxH3Transformer3DModel.from_pretrained(artifact)
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(restored_safe.state_dict()[key], expected)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.sub_(0.25)
    model.save_pretrained(artifact, safe_serialization=False, max_shard_size="1GB")
    assert (artifact / "diffusion_pytorch_model.bin").is_file()
    assert not list(artifact.glob("*.safetensors"))

    # An unmanifested, higher-priority filename must not shadow the checkpoint
    # selected by the component's immutable inventory.
    (artifact / "diffusion_pytorch_model.safetensors").write_bytes(b"stale")
    restored_bin = MiniMaxH3Transformer3DModel.from_pretrained(artifact)
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(restored_bin.state_dict()[key], expected)


def test_component_loader_rejects_unsupported_loading_options(tmp_path: Path):
    from hftrainer.models.minimax_h3.network import MiniMaxH3Transformer3DModel

    artifact = tmp_path / "transformer"
    MiniMaxH3Transformer3DModel(**_tiny_config()).save_pretrained(artifact)
    for option, value in (("device_map", "auto"), ("variant", "fp16")):
        with pytest.raises(TypeError, match=option):
            MiniMaxH3Transformer3DModel.from_pretrained(artifact, **{option: value})


def test_import_blocker_proves_no_external_model_runtime(repo_root: Path):
    script = r"""import importlib.abc
import sys

class ForbiddenModelPackages(importlib.abc.MetaPathFinder):
    roots = {"transformers", "diffusers", "peft"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.roots:
            raise AssertionError(f"forbidden model-package import: {fullname}")
        return None

sys.meta_path.insert(0, ForbiddenModelPackages())
from hftrainer.models.minimax_h3.network import (
    MiniMaxH3Scheduler,
    MiniMaxH3Transformer3DModel,
)
model = MiniMaxH3Transformer3DModel(
    num_attention_heads=1,
    attention_head_dim=8,
    hidden_size=8,
    num_layers=0,
    num_refiner_layers=0,
    ffn_dim=16,
    in_channels=2,
    audio_in_channels=2,
    patch_size=(1, 1, 1),
    text_dim=4,
    freq_dim=4,
    time_embed_hidden_dim=8,
    time_embed_dim=4,
    rope_freq_dim=1,
)
assert model.proj_in.in_features == 2
assert MiniMaxH3Scheduler(shift=3.0).shift == 3.0
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


def test_public_exports_and_cached_bundle_artifact_roundtrip(tmp_path: Path):
    from hftrainer.models.minimax_h3 import MiniMaxH3Bundle
    from hftrainer.models.minimax_h3.network import (
        AutoencoderKLMiniMaxH3,
        AutoencoderKLMiniMaxH3Audio,
        MiniMaxH3Processor,
        MiniMaxH3Qwen3VLEncoder,
        MiniMaxH3Scheduler,
        MiniMaxH3Tokenizer,
        MiniMaxH3Transformer3DModel,
    )

    config = _tiny_config()
    bundle = MiniMaxH3Bundle(
        transformer={
            "type": "MiniMaxH3Transformer3DModel",
            **config,
            "trainable": False,
            "save_ckpt": True,
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
        vae={
            "type": "AutoencoderKLMiniMaxH3",
            "latent_channels": 2,
            "block_out_channels": (4,),
            "layers_per_block": 1,
            "spatial_downsample_factors": (2,),
            "temporal_downsample_factors": (4,),
            "norm_num_groups": 1,
            "decoder_num_layers": 1,
            "decoder_num_attention_heads": 1,
            "decoder_attention_head_dim": 8,
            "decoder_num_register_tokens": 1,
            "decoder_ffn_mult": 1,
            "decoder_rope_dim_ratio": 0.75,
            "clip_length": 17,
            "token_drop": 3,
            "latents_mean": (0.0, 0.0),
            "latents_std": (1.0, 1.0),
            "trainable": False,
            "save_ckpt": True,
        },
        audio_vae={
            "type": "AutoencoderKLMiniMaxH3Audio",
            "encoder_dim": 1,
            "encoder_rates": (20, 40),
            "latent_dim": 4,
            "latent_channels": 2,
            "num_attention_heads": 2,
            "decoder_dim": 4,
            "decoder_rates": (40, 20),
            "decoder_kernel_sizes": (80, 40),
            "resblock_kernel_sizes": (3,),
            "resblock_dilation_sizes": ((1,),),
            "latents_mean": (0.0, 0.0),
            "latents_std": (1.0, 1.0),
            "trainable": False,
            "save_ckpt": True,
        },
        gradient_checkpointing=True,
        conditioning_layer=2,
    )
    assert bundle._LOCAL_TYPES == {
        "text_encoder": MiniMaxH3Qwen3VLEncoder,
        "vae": AutoencoderKLMiniMaxH3,
        "audio_vae": AutoencoderKLMiniMaxH3Audio,
        "transformer": MiniMaxH3Transformer3DModel,
        "scheduler": MiniMaxH3Scheduler,
        "audio_scheduler": MiniMaxH3Scheduler,
    }
    assert isinstance(bundle.tokenizer, MiniMaxH3Tokenizer)
    assert isinstance(bundle.processor, MiniMaxH3Processor)
    assert bundle.transformer.gradient_checkpointing
    assert bundle.transformer.token_refiner.gradient_checkpointing
    with torch.device("meta"):
        meta_vae = AutoencoderKLMiniMaxH3.from_config(bundle.vae.config)
    assert meta_vae.decoder.rope.inv_freq.is_meta

    artifact = tmp_path / "bundle"
    bundle.save_pretrained(artifact)
    restored = MiniMaxH3Bundle.from_pretrained(str(artifact), load_conditioner=False)
    assert type(restored.transformer) is MiniMaxH3Transformer3DModel
    assert type(restored.scheduler) is MiniMaxH3Scheduler
    assert type(restored.audio_scheduler) is MiniMaxH3Scheduler
    assert type(restored.tokenizer) is MiniMaxH3Tokenizer
    assert type(restored.processor) is MiniMaxH3Processor
    assert restored.scheduler.shift == 12.0
    assert restored.audio_scheduler.shift == 3.0
    assert restored.conditioning_layer == 2
    assert restored.transformer._load_report["parameter_coverage"] == 1.0
    assert restored.vae._load_report["parameter_coverage"] == 1.0
    assert restored.audio_vae._load_report["parameter_coverage"] == 1.0
    assert not any(
        tensor.is_meta
        for component in (restored.vae, restored.audio_vae)
        for tensor in (*component.parameters(), *component.buffers())
    )
    torch.testing.assert_close(
        restored.vae.decoder.rope.inv_freq,
        bundle.vae.decoder.rope.inv_freq,
        rtol=0,
        atol=0,
    )
    if torch.cuda.is_available():
        restored_cuda = MiniMaxH3Bundle.from_pretrained(
            str(artifact), load_conditioner=False, device="cuda"
        )
        assert all(
            tensor.device.type == "cuda"
            for tensor in (
                *restored_cuda.vae.parameters(),
                *restored_cuda.vae.buffers(),
            )
        )
        with torch.no_grad():
            decoded = restored_cuda.vae.decode(
                torch.randn(1, 2, 7, 2, 2, device="cuda")
            ).sample
        assert decoded.device.type == "cuda"


def test_lora_bundle_export_merges_to_a_reloadable_standalone_artifact(
    tmp_path: Path,
):
    from hftrainer.models.minimax_h3 import MiniMaxH3Bundle

    bundle = MiniMaxH3Bundle(
        transformer={
            "type": "MiniMaxH3Transformer3DModel",
            **_tiny_config(),
            "trainable": "lora",
            "save_ckpt": True,
            "checkpoint_format": "lora",
            "lora_cfg": {
                "rank": 2,
                "alpha": 2,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            },
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
    ).eval()
    with torch.no_grad():
        for name, parameter in bundle.transformer.named_parameters():
            if ".lora_B." in name:
                parameter.normal_(mean=0.0, std=0.02)
        expected = bundle.transformer(**_tiny_inputs(batch_size=1))

    rejected = tmp_path / "unbound"
    with pytest.raises(ValueError, match="unbound adapter"):
        bundle.save_pretrained(rejected, merge_lora=False)
    assert not rejected.exists()

    artifact = tmp_path / "merged"
    manifest_path = Path(bundle.save_pretrained(artifact))
    assert not bundle.is_lora_module("transformer")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["merged_lora_modules"] == ["transformer"]

    restored = MiniMaxH3Bundle.from_pretrained(
        artifact,
        torch_dtype=torch.float32,
        load_conditioner=False,
        load_vaes=False,
    ).eval()
    actual = restored.transformer(**_tiny_inputs(batch_size=1))
    torch.testing.assert_close(actual.sample, expected.sample, atol=3e-7, rtol=1e-5)
    torch.testing.assert_close(
        actual.audio_sample,
        expected.audio_sample,
        atol=3e-7,
        rtol=1e-5,
    )


def test_bundle_loader_uses_explicit_component_placement_and_rejects_device_map(
    tmp_path: Path,
):
    from hftrainer.models.minimax_h3 import MiniMaxH3Bundle

    config = MiniMaxH3Bundle._bundle_config_from_pretrained(
        str(tmp_path),
        transformer_device="cuda:0",
        conditioner_device="cuda:1",
        vae_device="cuda:2",
        audio_vae_device="cpu",
    )
    assert config["variant"] == "fl2va"
    assert config["transformer"]["from_pretrained"]["device"] == "cuda:0"
    assert config["text_encoder"]["from_pretrained"]["device"] == "cuda:1"
    assert config["vae"]["from_pretrained"]["device"] == "cuda:2"
    assert config["audio_vae"]["from_pretrained"]["device"] == "cpu"
    assert config["transformer"]["from_pretrained"]["torch_dtype"] is torch.bfloat16
    assert config["vae"]["from_pretrained"]["torch_dtype"] is torch.float32
    assert config["audio_vae"]["from_pretrained"]["torch_dtype"] is torch.float32

    with pytest.raises(ValueError, match="does not silently emulate"):
        MiniMaxH3Bundle._bundle_config_from_pretrained(str(tmp_path), device_map="auto")


def test_ref2va_bundle_artifact_uses_transformer_ref_and_infers_variant(
    tmp_path: Path,
):
    from hftrainer.models.minimax_h3 import MiniMaxH3Bundle

    bundle = MiniMaxH3Bundle(
        variant="ref2va",
        transformer={
            "type": "MiniMaxH3Transformer3DModel",
            **_tiny_config(),
            "trainable": False,
            "save_ckpt": True,
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
    artifact = tmp_path / "ref-bundle"
    bundle.save_pretrained(artifact)
    assert (artifact / "transformer_ref").is_dir()
    assert not (artifact / "transformer").exists()

    restored = MiniMaxH3Bundle.from_pretrained(
        str(artifact), load_conditioner=False, load_vaes=False
    )
    assert restored.variant == "ref2va"
    assert restored.transformer._load_report["parameter_coverage"] == 1.0

    with pytest.raises(ValueError, match="saved bundle contains"):
        MiniMaxH3Bundle.from_pretrained(
            str(artifact),
            variant="fl2va",
            load_conditioner=False,
            load_vaes=False,
        )

    processor_config = artifact / "processor" / "preprocessor_config.json"
    processor_config.write_text(
        processor_config.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MiniMaxH3Bundle.from_pretrained(
            str(artifact), load_conditioner=False, load_vaes=False
        )
