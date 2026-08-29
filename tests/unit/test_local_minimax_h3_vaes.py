"""CPU contracts for the repository-local MiniMax-H3 video and audio VAEs."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from hftrainer.models.minimax_h3.network.audio_vae import (
    AutoencoderKLMiniMaxH3Audio,
)
from hftrainer.models.minimax_h3.network.video_vae import AutoencoderKLMiniMaxH3


def _tiny_video_vae(*, decoder_num_layers: int = 1) -> AutoencoderKLMiniMaxH3:
    model = AutoencoderKLMiniMaxH3(
        latent_channels=2,
        block_out_channels=(4,),
        layers_per_block=1,
        spatial_downsample_factors=(2,),
        temporal_downsample_factors=(4,),
        norm_num_groups=1,
        decoder_num_layers=decoder_num_layers,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=1,
        decoder_rope_dim_ratio=0.75,
        clip_length=17,
        token_drop=3,
        latents_mean=(0.0, 0.0),
        latents_std=(1.0, 1.0),
    )
    model.disable_tiling()
    return model


def _tiny_audio_vae() -> AutoencoderKLMiniMaxH3Audio:
    # The configurable stage rates still multiply to the released 800-sample
    # hop while keeping this CPU test small.
    return AutoencoderKLMiniMaxH3Audio(
        encoder_dim=1,
        encoder_rates=(20, 40),
        latent_dim=4,
        latent_channels=2,
        num_attention_heads=2,
        decoder_dim=4,
        decoder_rates=(40, 20),
        decoder_kernel_sizes=(80, 40),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
    )


@pytest.mark.parametrize("n", (0, 1, 2))
def test_video_vae_has_released_17n_plus_5_geometry(n: int):
    model = _tiny_video_vae().eval()
    pixels = torch.randn(1, 3, 17 * n + 5, 4, 4)

    with torch.no_grad():
        latents = model.encode(pixels).latent_dist.mode()

    assert latents.shape == (1, 2, 5 * n + 2, 2, 2)


def test_video_vae_cpu_encode_decode_checkpointed_backward():
    model = _tiny_video_vae().train()
    model.gradient_checkpointing_enable()
    pixels = torch.randn(1, 3, 22, 4, 4, requires_grad=True)

    posterior = model.encode(pixels).latent_dist
    latents = posterior.mode()
    decoded = model.decode(latents).sample

    assert latents.shape == (1, 2, 7, 2, 2)
    assert decoded.shape == pixels.shape
    assert all(
        module.gradient_checkpointing
        for module in model.modules()
        if hasattr(module, "gradient_checkpointing")
    )

    (decoded.square().mean() + latents.square().mean()).backward()
    assert pixels.grad is not None
    assert torch.isfinite(pixels.grad).all()
    assert pixels.grad.abs().sum() > 0


def test_video_decoder_checkpointing_matches_all_eager_layers_on_backward():
    torch.manual_seed(7)
    eager = _tiny_video_vae(decoder_num_layers=3).train()
    checkpointed = copy.deepcopy(eager).train()
    checkpointed.gradient_checkpointing_enable()
    eager_latents = torch.randn(1, 2, 7, 2, 2, requires_grad=True)
    checkpointed_latents = eager_latents.detach().clone().requires_grad_(True)

    eager_output = eager.decode(eager_latents).sample
    checkpointed_output = checkpointed.decode(checkpointed_latents).sample
    torch.testing.assert_close(checkpointed_output, eager_output)

    probe = torch.randn_like(eager_output)
    (eager_output * probe).sum().backward()
    (checkpointed_output * probe).sum().backward()
    torch.testing.assert_close(checkpointed_latents.grad, eager_latents.grad)
    for eager_block, checkpointed_block in zip(
        eager.decoder.transformer_blocks,
        checkpointed.decoder.transformer_blocks,
        strict=True,
    ):
        assert eager_block.attn.to_q.weight.grad is not None
        assert checkpointed_block.attn.to_q.weight.grad is not None
        torch.testing.assert_close(
            checkpointed_block.attn.to_q.weight.grad,
            eager_block.attn.to_q.weight.grad,
        )


def test_video_vae_keeps_official_checkpoint_key_layout():
    keys = set(_tiny_video_vae().state_dict())
    expected = {
        "encoder.conv_in.weight",
        "encoder.down_blocks.0.resnets.0.conv1.weight",
        "quant_conv.weight",
        "post_quant_conv.weight",
        "decoder.register_tokens",
        "decoder.transformer_blocks.0.attn.to_q.weight",
        "decoder.transformer_blocks.0.ff.net.0.proj.weight",
        "decoder.transformer_blocks.0.ff.net.2.weight",
        "decoder.proj_out.weight",
    }
    assert expected <= keys


def test_audio_vae_uses_800_hop_and_stereo_as_batch_contract():
    model = _tiny_audio_vae().train()
    model.gradient_checkpointing_enable()
    # MiniMax-H3 represents left/right channels as two mono batch rows. The
    # final sample is padded from 801 to 1600 before encoding.
    stereo_as_batch = torch.randn(2, 1, 801, requires_grad=True)

    posterior = model.encode(stereo_as_batch).latent_dist
    latents = posterior.mode()
    decoded = model.decode(latents).sample

    assert model.hop_length == 800
    assert latents.shape == (2, 2, 2)
    assert decoded.shape == (2, 1, 1600)
    assert decoded.min() >= -1.0
    assert decoded.max() <= 1.0

    (decoded.square().mean() + latents.square().mean()).backward()
    assert stereo_as_batch.grad is not None
    assert torch.isfinite(stereo_as_batch.grad).all()
    assert stereo_as_batch.grad.abs().sum() > 0

    with pytest.raises(ValueError, match="batch_size, 1, samples"):
        model.encode(torch.randn(1, 2, 800))


def test_audio_vae_keeps_official_checkpoint_key_layout():
    keys = set(_tiny_audio_vae().state_dict())
    expected = {
        "encoder.block.0.weight_g",
        "encoder.block.0.weight_v",
        "pre_block.attn.qkv.weight",
        "pre_block.attn.q_bias",
        "pre_block.attn.v_bias",
        "pre_block.attn.zero_k_bias",
        "mean_proj.weight",
        "logs_proj.weight",
        "dec_in_proj.weight",
        "decoder.ups.0.0.weight_g",
        "decoder.resblocks.0.convs1.0.weight_g",
        "decoder.conv_post.weight_g",
    }
    assert expected <= keys


@pytest.mark.parametrize(
    ("name", "factory", "model_type"),
    (
        ("video", _tiny_video_vae, AutoencoderKLMiniMaxH3),
        ("audio", _tiny_audio_vae, AutoencoderKLMiniMaxH3Audio),
    ),
)
def test_local_vae_artifact_roundtrip(
    tmp_path: Path,
    name: str,
    factory,
    model_type,
):
    model = factory().eval()
    artifact = tmp_path / name

    model.save_pretrained(artifact, safe_serialization=True)
    restored = model_type.from_pretrained(artifact).eval()

    assert (artifact / "config.json").is_file()
    assert (artifact / "diffusion_pytorch_model.safetensors").is_file()
    assert (artifact / "minimax_h3_local_manifest.json").is_file()
    assert restored.config == model.config
    assert list(restored.state_dict()) == list(model.state_dict())
    for key, expected in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], expected), key
