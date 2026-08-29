"""Repository-local StyleGAN2 execution and artifact contracts."""

from __future__ import annotations

import torch

from hftrainer.models.stylegan2 import (
    StyleGAN2Bundle,
    StyleGAN2Discriminator,
    StyleGAN2Generator,
)


def _tiny_bundle() -> StyleGAN2Bundle:
    return StyleGAN2Bundle(
        generator={
            'type': 'StyleGAN2Generator',
            'z_dim': 8,
            'w_dim': 8,
            'img_resolution': 8,
            'img_channels': 3,
            'channel_base': 64,
            'channel_max': 32,
            'mapping_layers': 2,
            'style_mixing_prob': 0.0,
            'trainable': True,
            'save_ckpt': True,
        },
        discriminator={
            'type': 'StyleGAN2Discriminator',
            'img_resolution': 8,
            'img_channels': 3,
            'channel_base': 64,
            'channel_max': 32,
            'mbstd_group_size': 2,
            'trainable': True,
            'save_ckpt': True,
        },
    )


def test_local_stylegan2_forward_and_backward():
    bundle = _tiny_bundle().train()
    latent = torch.randn(2, 8)

    images = bundle.sample(latent)
    scores = bundle.discriminate(images)
    (-scores.mean()).backward()

    assert images.shape == (2, 3, 8, 8)
    assert scores.shape == (2, 1)
    assert any(
        parameter.grad is not None
        for parameter in bundle.generator.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in bundle.discriminator.parameters()
    )


def test_local_stylegan2_artifact_round_trip(tmp_path):
    bundle = _tiny_bundle().eval()
    bundle.save_pretrained(tmp_path, safe_serialization=False)

    restored = StyleGAN2Bundle.from_pretrained(tmp_path).eval()

    assert type(restored.generator) is type(bundle.generator)
    assert type(restored.discriminator) is type(bundle.discriminator)
    for name, value in bundle.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_local_stylegan2_class_object_config_exports_portable_artifact(tmp_path):
    bundle = StyleGAN2Bundle(
        generator={
            **_tiny_bundle().get_module_build_cfg('generator'),
            'type': StyleGAN2Generator,
        },
        discriminator={
            **_tiny_bundle().get_module_build_cfg('discriminator'),
            'type': StyleGAN2Discriminator,
        },
    ).eval()

    bundle.save_pretrained(tmp_path, safe_serialization=False)
    restored = StyleGAN2Bundle.from_pretrained(tmp_path).eval()

    assert type(restored.generator) is StyleGAN2Generator
    assert type(restored.discriminator) is StyleGAN2Discriminator
    for name, value in bundle.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
