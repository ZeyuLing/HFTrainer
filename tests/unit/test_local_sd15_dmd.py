"""Executable contracts for the repository-local SD1.5 and DMD cores."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from hftrainer.models.dmd import DMDBundle
from hftrainer.models.sd15 import SD15Bundle
from hftrainer.models.sd15.network import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
)


def _text(trainable=False):
    return {
        'type': 'CLIPTextModel',
        'vocab_size': 64,
        'hidden_size': 32,
        'intermediate_size': 64,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'max_position_embeddings': 16,
        'bos_token_id': 62,
        'eos_token_id': 63,
        'trainable': trainable,
        'save_ckpt': trainable,
    }


def _vae(trainable=False):
    return {
        'type': 'AutoencoderKL',
        'block_out_channels': (16, 32),
        'layers_per_block': 1,
        'latent_channels': 4,
        'norm_num_groups': 8,
        'sample_size': 16,
        'trainable': trainable,
        'save_ckpt': trainable,
    }


def _unet(trainable=True):
    return {
        'type': 'UNet2DConditionModel',
        'sample_size': 8,
        'block_out_channels': (32, 64),
        'layers_per_block': 1,
        'down_block_types': ('CrossAttnDownBlock2D', 'DownBlock2D'),
        'up_block_types': ('UpBlock2D', 'CrossAttnUpBlock2D'),
        'cross_attention_dim': 32,
        'attention_head_dim': 4,
        'norm_num_groups': 8,
        'trainable': trainable,
        'save_ckpt': trainable,
    }


def _scheduler(name='DDPMScheduler'):
    return {
        'type': name,
        'num_train_timesteps': 20,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'beta_schedule': 'linear',
        'trainable': False,
        'save_ckpt': False,
    }


def _sd_bundle():
    return SD15Bundle(
        text_encoder=_text(),
        vae=_vae(),
        unet=_unet(),
        scheduler=_scheduler(),
        tokenizer={'vocab_size': 64, 'model_max_length': 16},
        max_token_length=16,
    )


def _dmd_bundle():
    return DMDBundle(
        text_encoder=_text(),
        vae=_vae(),
        real_score_unet=_unet(trainable=False),
        fake_score_unet=_unet(trainable=True),
        generator_unet=_unet(trainable=True),
        scheduler=_scheduler('DDIMScheduler'),
        tokenizer={'vocab_size': 64, 'model_max_length': 16},
        max_token_length=16,
        image_size=16,
        conditioning_timestep=19,
        dm_min_timestep_percent=0.1,
        dm_max_timestep_percent=0.9,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def test_local_component_key_layout_and_forward_backward():
    torch.manual_seed(7)
    bundle = _sd_bundle()
    assert type(bundle.text_encoder).__module__.startswith('hftrainer.models.sd15.')
    assert type(bundle.vae).__module__.startswith('hftrainer.models.sd15.')
    assert type(bundle.unet).__module__.startswith('hftrainer.models.sd15.')
    assert type(bundle.scheduler).__module__.startswith('hftrainer.models.sd15.')

    text_keys = bundle.text_encoder.state_dict()
    unet_keys = bundle.unet.state_dict()
    vae_keys = bundle.vae.state_dict()
    assert 'text_model.encoder.layers.0.self_attn.q_proj.weight' in text_keys
    assert 'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight' in unet_keys
    assert 'encoder.down_blocks.0.resnets.0.conv1.weight' in vae_keys

    images = torch.randn(1, 3, 16, 16)
    hidden = bundle.encode_text(['a tiny red cube'])
    latents = bundle.encode_image(images)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([7])
    noisy = bundle.add_noise(latents, noise, timesteps)
    prediction = bundle.predict_noise(noisy, timesteps, hidden)
    assert prediction.shape == latents.shape == (1, 4, 8, 8)
    loss = F.mse_loss(prediction, noise)
    loss.backward()
    assert bundle.unet.conv_out.weight.grad is not None
    assert torch.isfinite(bundle.unet.conv_out.weight.grad).all()

    sampler = DDIMScheduler(num_train_timesteps=20, beta_schedule='linear')
    sampler.set_timesteps(3)
    sample = torch.randn_like(latents)
    for timestep in sampler.timesteps:
        predicted = bundle.predict_noise(
            sample, timestep.expand(1), hidden
        )
        sample = sampler.step(predicted, timestep, sample).prev_sample
    decoded = bundle.decode_latent(sample)
    assert decoded.shape == images.shape
    assert torch.isfinite(decoded).all()


def test_schedulers_cover_training_and_sampling_api():
    original = torch.randn(2, 4, 4, 4)
    noise = torch.randn_like(original)
    timesteps = torch.tensor([2, 11])
    for scheduler_cls in (DDPMScheduler, DDIMScheduler, PNDMScheduler):
        scheduler = scheduler_cls(num_train_timesteps=20, beta_schedule='linear')
        noisy = scheduler.add_noise(original, noise, timesteps)
        velocity = scheduler.get_velocity(original, noise, timesteps)
        assert noisy.shape == velocity.shape == original.shape
        scheduler.set_timesteps(4)
        output = scheduler.step(torch.zeros_like(original), scheduler.timesteps[0], noisy)
        assert output.prev_sample.shape == original.shape


def test_component_and_bundle_save_load_round_trip(tmp_path):
    torch.manual_seed(11)
    bundle = _sd_bundle().eval()
    root = tmp_path / 'sd15'
    manifest = bundle.save_pretrained(root)
    assert manifest['format'] == 'hftrainer-local-bundle'
    assert (root / 'manifest.json').is_file()
    assert (root / 'unet' / 'manifest.json').is_file()
    disk_manifest = json.loads((root / 'unet' / 'manifest.json').read_text(encoding='utf-8'))
    assert len(disk_manifest['sha256']) == 64

    restored = SD15Bundle.from_pretrained(str(root), max_token_length=16).eval()
    assert isinstance(restored.scheduler, DDPMScheduler)
    assert restored.unet._checkpoint_load_report['parameter_coverage'] == 1.0
    assert restored.vae._checkpoint_load_report['parameter_coverage'] == 1.0
    assert restored.text_encoder._checkpoint_load_report['parameter_coverage'] == 1.0
    for key, value in bundle.unet.state_dict().items():
        assert torch.equal(value, restored.unet.state_dict()[key])


def test_sd15_bundle_rejects_tampered_config(tmp_path):
    root = tmp_path / 'sd15'
    _sd_bundle().save_pretrained(root, safe_serialization=False)
    config_path = root / 'bundle_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['max_token_length'] += 1
    config_path.write_text(json.dumps(config), encoding='utf-8')

    with pytest.raises(RuntimeError, match='config SHA-256 mismatch'):
        SD15Bundle.from_pretrained(str(root))


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('format', 'foreign-bundle', 'manifest format'),
        ('schema_version', 2, 'manifest schema'),
        ('bundle', 'DMDBundle', 'manifest type mismatch'),
        ('config', '../bundle_config.json', 'config must be the basename'),
        ('sha256', '0' * 64, 'config SHA-256 mismatch'),
    ],
)
def test_sd15_bundle_validates_root_manifest(
    tmp_path,
    field,
    value,
    message,
):
    root = tmp_path / 'sd15'
    _sd_bundle().save_pretrained(root, safe_serialization=False)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(RuntimeError, match=message):
        SD15Bundle.from_pretrained(str(root))


def test_dmd_bundle_rejects_component_path_traversal(tmp_path):
    root = tmp_path / 'dmd'
    _dmd_bundle().save_pretrained(root, safe_serialization=False)
    config_path = root / 'bundle_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['components']['generator_unet']['path'] = '../outside'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['sha256'] = _sha256(config_path)
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(RuntimeError, match='escapes the artifact root'):
        DMDBundle.from_pretrained(str(root))


def test_dmd_reuses_local_sd_core_and_backpropagates(tmp_path):
    torch.manual_seed(17)
    bundle = _dmd_bundle()
    for name in ('text_encoder', 'vae', 'real_score_unet', 'fake_score_unet', 'generator_unet'):
        assert type(getattr(bundle, name)).__module__.startswith('hftrainer.models.sd15.')

    cond = bundle.encode_text(['a toy robot'])
    uncond = bundle.get_unconditional_text_embeddings(1)
    noise = bundle.sample_latent_noise(1)
    generated = bundle.generate_latents(noise, cond, uncond_embeddings=uncond)
    dm_loss, logs = bundle.compute_distribution_matching_loss(
        generated, cond, uncond_embeddings=uncond
    )
    dm_loss.backward()
    assert bundle.generator_unet.conv_out.weight.grad is not None
    assert torch.isfinite(logs['dm_grad_norm'])

    score_loss, _ = bundle.compute_fake_score_loss(generated.detach(), cond)
    score_loss.backward()
    assert bundle.fake_score_unet.conv_out.weight.grad is not None
    teacher = bundle.sample_teacher_deterministic(
        noise, cond, uncond_embeddings=uncond, num_inference_steps=2
    )
    assert teacher.shape == generated.shape == (1, 4, 8, 8)

    root = tmp_path / 'dmd'
    bundle.save_pretrained(root)
    restored = DMDBundle.from_pretrained(str(root))
    assert isinstance(restored.scheduler, DDIMScheduler)
    assert restored.generator_unet._checkpoint_load_report['parameter_coverage'] == 1.0
    assert restored.conditioning_timestep == 19


def test_local_model_imports_work_while_forbidden_roots_are_blocked():
    project_root = Path(__file__).resolve().parents[2]
    script = r'''
import importlib.abc
import sys

blocked = {'transformers', 'diffusers', 'peft', 'ltx_core', 'ltx_pipelines', 'ltx_trainer'}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise RuntimeError('forbidden import: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())

from hftrainer.models.sd15.network import AutoencoderKL, CLIPTextModel, UNet2DConditionModel
from hftrainer.models.sd15.bundle import SD15Bundle
from hftrainer.models.dmd.bundle import DMDBundle
assert all(cls.__module__.startswith('hftrainer.models.') for cls in (
    AutoencoderKL, CLIPTextModel, UNet2DConditionModel, SD15Bundle, DMDBundle
))
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(project_root), env.get('PYTHONPATH', '')]
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
