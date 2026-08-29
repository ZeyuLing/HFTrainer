"""Stable Diffusion 1.5 bundle backed exclusively by repository-local code."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES

from .network import (
    AutoencoderKL,
    CLIPTextModel,
    CLIPTokenizer,
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)

_COMPONENT_TYPES = {
    'text_encoder': {'CLIPTextModel': CLIPTextModel},
    'vae': {'AutoencoderKL': AutoencoderKL},
    'unet': {'UNet2DConditionModel': UNet2DConditionModel},
    'scheduler': {
        'DDPMScheduler': DDPMScheduler,
        'DDIMScheduler': DDIMScheduler,
        'PNDMScheduler': PNDMScheduler,
    },
}
_LOCAL_BUNDLE_FORMAT = 'hftrainer-local-bundle'
_LOCAL_BUNDLE_SCHEMA_VERSION = 1
_LOCAL_BUNDLE_CONFIG = 'bundle_config.json'


def _local_component_config(name: str, config: Mapping[str, Any]) -> dict:
    if not isinstance(config, Mapping):
        raise TypeError(f"Component '{name}' must be configured with a mapping.")
    config = dict(config)
    requested = config.get('type')
    if isinstance(requested, str):
        local_type = _COMPONENT_TYPES[name].get(requested)
        if local_type is None:
            allowed = ', '.join(_COMPONENT_TYPES[name])
            raise ValueError(
                f"Component '{name}' requested unsupported type '{requested}'. "
                f'Local choices: {allowed}.'
            )
        config['type'] = local_type
    elif requested not in _COMPONENT_TYPES[name].values():
        raise ValueError(f"Component '{name}' must use its repository-local implementation.")
    return config


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open('r', encoding='utf-8') as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Invalid {description}: {path}') from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(
            f'{description.capitalize()} must contain a JSON object: {path}'
        )
    return dict(value)


def _load_bundle_artifact_config(
    root: Path,
    *,
    expected_bundle: str,
) -> tuple[dict[str, Any], bool]:
    """Load and authenticate a repository-owned bundle configuration."""

    manifest_path = root / 'manifest.json'
    default_config_path = root / _LOCAL_BUNDLE_CONFIG
    if not manifest_path.is_file():
        if not default_config_path.is_file():
            return {}, False
        return _read_json_object(default_config_path, 'bundle configuration'), False

    manifest = _read_json_object(manifest_path, 'bundle manifest')
    if manifest.get('format') != _LOCAL_BUNDLE_FORMAT:
        raise RuntimeError(
            'Invalid HFTrainer bundle manifest format: '
            f"{manifest.get('format')!r}."
        )
    if manifest.get('schema_version') != _LOCAL_BUNDLE_SCHEMA_VERSION:
        raise RuntimeError(
            'Unsupported HFTrainer bundle manifest schema: '
            f"{manifest.get('schema_version')!r}."
        )
    if manifest.get('bundle') != expected_bundle:
        raise RuntimeError(
            'HFTrainer bundle manifest type mismatch: '
            f"expected {expected_bundle!r}, got {manifest.get('bundle')!r}."
        )
    config_name = manifest.get('config')
    if (
        not isinstance(config_name, str)
        or Path(config_name).name != config_name
        or config_name != _LOCAL_BUNDLE_CONFIG
    ):
        raise RuntimeError(
            'HFTrainer bundle manifest config must be the basename '
            f'{_LOCAL_BUNDLE_CONFIG!r}.'
        )
    config_path = root / config_name
    expected_sha256 = manifest.get('sha256')
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError('HFTrainer bundle manifest has an invalid config SHA-256.')
    if not config_path.is_file():
        raise RuntimeError(f'HFTrainer bundle configuration is missing: {config_path}')
    actual_sha256 = _hash_file(config_path)
    if actual_sha256 != expected_sha256.lower():
        raise RuntimeError(
            'HFTrainer bundle config SHA-256 mismatch: '
            f'expected {expected_sha256}, got {actual_sha256}.'
        )
    config = _read_json_object(config_path, 'bundle configuration')
    if config.get('schema_version') != _LOCAL_BUNDLE_SCHEMA_VERSION:
        raise RuntimeError(
            'Unsupported HFTrainer bundle configuration schema: '
            f"{config.get('schema_version')!r}."
        )
    if config.get('bundle') != expected_bundle:
        raise RuntimeError(
            'HFTrainer bundle configuration type mismatch: '
            f"expected {expected_bundle!r}, got {config.get('bundle')!r}."
        )
    return config, True


@MODEL_BUNDLES.register_module(force=True)
class SD15Bundle(ModelBundle):
    """Shared training/inference core for the local SD1.5 implementation."""

    def __init__(
        self,
        text_encoder: dict,
        vae: dict,
        unet: dict,
        scheduler: dict,
        tokenizer_path: str | None = None,
        tokenizer: CLIPTokenizer | dict | None = None,
        max_token_length: int = 77,
    ):
        super().__init__()
        self.max_token_length = int(max_token_length)
        self._build_modules({
            'text_encoder': _local_component_config('text_encoder', text_encoder),
            'vae': _local_component_config('vae', vae),
            'unet': _local_component_config('unet', unet),
            'scheduler': _local_component_config('scheduler', scheduler),
        })
        if type(tokenizer) is CLIPTokenizer:
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, Mapping):
            self.tokenizer = CLIPTokenizer(**dict(tokenizer))
        elif tokenizer_path is not None:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = CLIPTokenizer(
                vocab_size=self.text_encoder.config.vocab_size,
                model_max_length=self.max_token_length,
            )
        self._extra_attributes['tokenizer'] = self.tokenizer

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        text_encoder_overrides: dict | None = None,
        vae_overrides: dict | None = None,
        unet_overrides: dict | None = None,
        scheduler_overrides: dict | None = None,
        scheduler_type: str | None = None,
        max_token_length: int = 77,
        **kwargs,
    ) -> dict:
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected local artifact options: {unknown}')

        def component(component_type, subfolder, overrides, trainable, save_ckpt):
            value = {
                'type': component_type,
                'from_pretrained': {
                    'pretrained_model_name_or_path': pretrained_model_name_or_path,
                    'subfolder': subfolder,
                },
                'trainable': trainable,
                'save_ckpt': save_ckpt,
            }
            cls._merge_nested_dict(value, overrides)
            return value

        artifact_root = Path(pretrained_model_name_or_path)
        artifact_config, _ = _load_bundle_artifact_config(
            artifact_root,
            expected_bundle=cls.__name__,
        )
        if scheduler_type is None:
            scheduler_type = (
                artifact_config.get('components', {})
                .get('scheduler', {})
                .get('class')
            )
            scheduler_type = scheduler_type or 'PNDMScheduler'
        scheduler_cls = _COMPONENT_TYPES['scheduler'].get(scheduler_type)
        if scheduler_cls is None:
            raise ValueError(f'Unknown local scheduler: {scheduler_type}')
        return {
            'text_encoder': component(CLIPTextModel, 'text_encoder', text_encoder_overrides, False, False),
            'vae': component(AutoencoderKL, 'vae', vae_overrides, False, False),
            'unet': component(UNet2DConditionModel, 'unet', unet_overrides, True, True),
            'scheduler': component(scheduler_cls, 'scheduler', scheduler_overrides, False, False),
            'tokenizer_path': pretrained_model_name_or_path,
            'max_token_length': max_token_length,
        }

    def save_pretrained(
        self,
        save_directory: str,
        *,
        safe_serialization: bool = True,
        merge_lora: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        if merge_lora:
            self.merge_lora_weights(['text_encoder', 'unet'])
        manifests = {
            'text_encoder': self.text_encoder.save_pretrained(
                root / 'text_encoder', safe_serialization=safe_serialization
            ),
            'vae': self.vae.save_pretrained(root / 'vae', safe_serialization=safe_serialization),
            'unet': self.unet.save_pretrained(root / 'unet', safe_serialization=safe_serialization),
        }
        self.scheduler.save_pretrained(root / 'scheduler')
        self.tokenizer.save_pretrained(root / 'tokenizer')
        bundle_config = {
            'schema_version': 1,
            'bundle': type(self).__name__,
            'max_token_length': self.max_token_length,
            'components': {
                name: {'class': type(getattr(self, name)).__name__, 'path': name}
                for name in ('text_encoder', 'vae', 'unet', 'scheduler')
            },
            'tokenizer': {'class': type(self.tokenizer).__name__, 'path': 'tokenizer'},
        }
        config_path = root / 'bundle_config.json'
        with config_path.open('w', encoding='utf-8') as handle:
            json.dump(bundle_config, handle, indent=2, sort_keys=True)
            handle.write('\n')
        manifest = {
            'schema_version': 1,
            'format': _LOCAL_BUNDLE_FORMAT,
            'bundle': type(self).__name__,
            'config': config_path.name,
            'sha256': _hash_file(config_path),
            'components': manifests,
        }
        with (root / 'manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write('\n')
        return manifest

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.max_token_length,
            truncation=True,
            return_tensors='pt',
        )
        input_ids = tokens.input_ids.to(next(self.text_encoder.parameters()).device)
        with torch.set_grad_enabled(self.text_encoder.training):
            outputs = self.text_encoder(input_ids=input_ids)
        return outputs.last_hidden_state

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        with torch.set_grad_enabled(self.vae.training):
            latents = self.vae.encode(images.to(dtype=vae_dtype)).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype) / self.vae.config.scaling_factor
        with torch.set_grad_enabled(self.vae.training):
            return self.vae.decode(latents).sample

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        unet_dtype = next(self.unet.parameters()).dtype
        return self.unet(
            noisy_latents.to(dtype=unet_dtype),
            timesteps,
            encoder_hidden_states=encoder_hidden_states.to(dtype=unet_dtype),
        ).sample

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.scheduler.add_noise(latents, noise, timesteps)
