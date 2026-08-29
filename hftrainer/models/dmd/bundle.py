"""Distribution Matching Distillation bundle."""

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.models.sd15.bundle import (
    _LOCAL_BUNDLE_FORMAT,
    _load_bundle_artifact_config,
)
from hftrainer.models.sd15.network import (
    AutoencoderKL,
    CLIPTextModel,
    CLIPTokenizer,
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from hftrainer.registry import MODEL_BUNDLES

_SCHEDULERS = {
    'DDPMScheduler': DDPMScheduler,
    'DDIMScheduler': DDIMScheduler,
    'PNDMScheduler': PNDMScheduler,
}


def _component(config: Mapping[str, Any], expected, name: str) -> dict:
    if not isinstance(config, Mapping):
        raise TypeError(f"Component '{name}' must be configured with a mapping.")
    config = dict(config)
    requested = config.get('type')
    choices = expected if isinstance(expected, dict) else {expected.__name__: expected}
    if isinstance(requested, str):
        if requested not in choices:
            raise ValueError(f"Component '{name}' must use a repository-local class.")
        config['type'] = choices[requested]
    elif requested not in choices.values():
        raise ValueError(f"Component '{name}' must use a repository-local class.")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_component_paths(root: Path, artifact_config: Mapping[str, Any]) -> None:
    components = artifact_config.get('components', {})
    if not isinstance(components, Mapping):
        raise RuntimeError('DMD bundle components must contain a JSON object.')
    for name, component in components.items():
        if not isinstance(component, Mapping):
            raise RuntimeError(f'DMD component {name!r} must contain a JSON object.')
        if 'path' not in component:
            continue
        subfolder = component['path']
        if not isinstance(subfolder, str) or not subfolder:
            raise RuntimeError(f'DMD component {name!r} has an invalid path.')
        resolved = (root / subfolder).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f'DMD component {name!r} path escapes the artifact root: '
                f'{subfolder!r}.'
            ) from exc


@MODEL_BUNDLES.register_module(force=True)
class DMDBundle(ModelBundle):
    """
    Bundle for DMD-style one-step diffusion distillation.

    Modules:
      - text_encoder: frozen text encoder
      - vae: frozen VAE
      - real_score_unet: frozen teacher score network
      - fake_score_unet: trainable fake score network
      - generator_unet: trainable one-step generator UNet
      - scheduler: diffusion scheduler
    """

    def __init__(
        self,
        text_encoder: dict,
        vae: dict,
        real_score_unet: dict,
        fake_score_unet: dict,
        generator_unet: dict,
        scheduler: dict,
        tokenizer_path: str | None = None,
        tokenizer: CLIPTokenizer | dict | None = None,
        max_token_length: int = 77,
        image_size: int = 512,
        conditioning_timestep: int = 999,
        dm_min_timestep_percent: float = 0.02,
        dm_max_timestep_percent: float = 0.98,
        generator_guidance_scale: float = 1.0,
        real_score_guidance_scale: float = 7.5,
        fake_score_guidance_scale: float = 1.0,
        regression_guidance_scale: float = 7.5,
    ):
        super().__init__()
        self.max_token_length = max_token_length
        self.image_size = image_size
        self.conditioning_timestep = conditioning_timestep
        self.dm_min_timestep_percent = dm_min_timestep_percent
        self.dm_max_timestep_percent = dm_max_timestep_percent
        self.generator_guidance_scale = generator_guidance_scale
        self.real_score_guidance_scale = real_score_guidance_scale
        self.fake_score_guidance_scale = fake_score_guidance_scale
        self.regression_guidance_scale = regression_guidance_scale

        self._build_modules({
            'text_encoder': _component(text_encoder, CLIPTextModel, 'text_encoder'),
            'vae': _component(vae, AutoencoderKL, 'vae'),
            'real_score_unet': _component(real_score_unet, UNet2DConditionModel, 'real_score_unet'),
            'fake_score_unet': _component(fake_score_unet, UNet2DConditionModel, 'fake_score_unet'),
            'generator_unet': _component(generator_unet, UNet2DConditionModel, 'generator_unet'),
            'scheduler': _component(scheduler, _SCHEDULERS, 'scheduler'),
        })

        pretrained_path = tokenizer_path
        if pretrained_path is None and isinstance(text_encoder, dict):
            fp = text_encoder.get('from_pretrained', {})
            pretrained_path = fp.get('pretrained_model_name_or_path') if fp else None

        if type(tokenizer) is CLIPTokenizer:
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, Mapping):
            self.tokenizer = CLIPTokenizer(**dict(tokenizer))
        elif pretrained_path is not None:
            self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_path)
        else:
            self.tokenizer = CLIPTokenizer(
                vocab_size=self.text_encoder.config.vocab_size,
                model_max_length=self.max_token_length,
            )
        self._extra_attributes['tokenizer'] = self.tokenizer

        self.latent_channels = getattr(
            self.generator_unet.config, 'in_channels', 4
        )
        vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_size = image_size // vae_scale

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        scheduler_type: str | None = None,
        image_size: int | None = None,
        max_token_length: int | None = None,
        component_overrides: dict | None = None,
        **bundle_options,
    ) -> dict:
        artifact_root = Path(pretrained_model_name_or_path).resolve()
        artifact_config, _ = _load_bundle_artifact_config(
            artifact_root,
            expected_bundle=cls.__name__,
        )
        _validate_component_paths(artifact_root, artifact_config)
        if scheduler_type is None:
            scheduler_type = (
                artifact_config.get('components', {})
                .get('scheduler', {})
                .get('class', 'DDIMScheduler')
            )
        scheduler_cls = _SCHEDULERS.get(scheduler_type)
        if scheduler_cls is None:
            raise ValueError(f'Unknown local scheduler: {scheduler_type}')
        overrides = component_overrides or {}

        def local(component_type, fallback_subfolder, name, trainable, save_ckpt):
            subfolder = (
                artifact_config.get('components', {})
                .get(name, {})
                .get('path', fallback_subfolder)
            )
            value = {
                'type': component_type,
                'from_pretrained': {
                    'pretrained_model_name_or_path': str(artifact_root),
                    'subfolder': subfolder,
                },
                'trainable': trainable,
                'save_ckpt': save_ckpt,
            }
            cls._merge_nested_dict(value, overrides.get(name))
            return value

        result = {
            'text_encoder': local(CLIPTextModel, 'text_encoder', 'text_encoder', False, False),
            'vae': local(AutoencoderKL, 'vae', 'vae', False, False),
            'real_score_unet': local(UNet2DConditionModel, 'unet', 'real_score_unet', False, False),
            'fake_score_unet': local(UNet2DConditionModel, 'unet', 'fake_score_unet', True, True),
            'generator_unet': local(UNet2DConditionModel, 'unet', 'generator_unet', True, True),
            'scheduler': local(scheduler_cls, 'scheduler', 'scheduler', False, False),
            'tokenizer_path': str(artifact_root),
            'max_token_length': max_token_length or artifact_config.get('max_token_length', 77),
            'image_size': image_size or artifact_config.get('image_size', 512),
        }
        for name in (
            'conditioning_timestep', 'dm_min_timestep_percent',
            'dm_max_timestep_percent', 'generator_guidance_scale',
            'real_score_guidance_scale', 'fake_score_guidance_scale',
            'regression_guidance_scale',
        ):
            if name in artifact_config:
                result[name] = artifact_config[name]
        result.update(bundle_options)
        return result

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
            self.merge_lora_weights(['text_encoder', 'real_score_unet', 'fake_score_unet', 'generator_unet'])
        neural_components = (
            'text_encoder', 'vae', 'real_score_unet',
            'fake_score_unet', 'generator_unet',
        )
        manifests = {}
        for name in neural_components:
            manifests[name] = getattr(self, name).save_pretrained(
                root / name, safe_serialization=safe_serialization
            )
        self.scheduler.save_pretrained(root / 'scheduler')
        self.tokenizer.save_pretrained(root / 'tokenizer')
        config = {
            'schema_version': 1,
            'bundle': type(self).__name__,
            'image_size': self.image_size,
            'max_token_length': self.max_token_length,
            'conditioning_timestep': self.conditioning_timestep,
            'dm_min_timestep_percent': self.dm_min_timestep_percent,
            'dm_max_timestep_percent': self.dm_max_timestep_percent,
            'generator_guidance_scale': self.generator_guidance_scale,
            'real_score_guidance_scale': self.real_score_guidance_scale,
            'fake_score_guidance_scale': self.fake_score_guidance_scale,
            'regression_guidance_scale': self.regression_guidance_scale,
            'components': {
                name: {'class': type(getattr(self, name)).__name__, 'path': name}
                for name in (*neural_components, 'scheduler')
            },
        }
        config_path = root / 'bundle_config.json'
        with config_path.open('w', encoding='utf-8') as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write('\n')
        manifest = {
            'schema_version': 1,
            'format': _LOCAL_BUNDLE_FORMAT,
            'bundle': type(self).__name__,
            'config': config_path.name,
            'sha256': _sha256(config_path),
            'components': manifests,
        }
        with (root / 'manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write('\n')
        return manifest

    def _module_device(self, module) -> torch.device:
        return next(module.parameters()).device

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.max_token_length,
            truncation=True,
            return_tensors='pt',
        )
        input_ids = tokens.input_ids.to(self._module_device(self.text_encoder))
        with torch.set_grad_enabled(self.text_encoder.training):
            outputs = self.text_encoder(input_ids=input_ids)
        return outputs.last_hidden_state

    def get_unconditional_text_embeddings(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        embeddings = self.encode_text([''] * batch_size)
        if device is not None:
            embeddings = embeddings.to(device)
        return embeddings

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(dtype=next(self.vae.parameters()).dtype)
        with torch.set_grad_enabled(self.vae.training):
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(dtype=next(self.vae.parameters()).dtype)
        latents = latents / self.vae.config.scaling_factor
        with torch.set_grad_enabled(self.vae.training):
            images = self.vae.decode(latents).sample
        return images

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        return noisy_latents.to(dtype=latents.dtype)

    def sample_latent_noise(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        device = device or self._module_device(self.generator_unet)
        dtype = dtype or next(self.generator_unet.parameters()).dtype
        return torch.randn(
            batch_size,
            self.latent_channels,
            self.latent_size,
            self.latent_size,
            device=device,
            dtype=dtype,
        )

    def _predict_noise(
        self,
        unet,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        model_dtype = next(unet.parameters()).dtype
        noisy_latents = noisy_latents.to(dtype=model_dtype)
        cond_embeddings = cond_embeddings.to(dtype=model_dtype)
        if uncond_embeddings is not None:
            uncond_embeddings = uncond_embeddings.to(dtype=model_dtype)

        if guidance_scale == 1.0 or uncond_embeddings is None:
            return unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=cond_embeddings,
            ).sample

        model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
        model_timesteps = torch.cat([timesteps, timesteps], dim=0)
        hidden_states = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
        noise_pred = unet(
            model_input,
            model_timesteps,
            encoder_hidden_states=hidden_states,
        ).sample
        noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
        return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    def _predict_x0(
        self,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alphas = self.scheduler.alphas_cumprod.to(noisy_latents.device)[timesteps]
        alphas = alphas.view(-1, 1, 1, 1).to(noisy_latents.dtype)
        return (
            noisy_latents - torch.sqrt(1 - alphas) * noise_pred
        ) / torch.sqrt(alphas)

    def generate_latents(
        self,
        noise_latents: torch.Tensor,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: torch.Tensor | None = None,
        guidance_scale: float | None = None,
        timestep: int | None = None,
        return_noise: bool = False,
    ):
        guidance_scale = (
            self.generator_guidance_scale if guidance_scale is None else guidance_scale
        )
        timestep = (
            self.conditioning_timestep if timestep is None else timestep
        )
        timesteps = torch.full(
            (noise_latents.shape[0],),
            int(timestep),
            device=noise_latents.device,
            dtype=torch.long,
        )
        noise_pred = self._predict_noise(
            self.generator_unet,
            noise_latents,
            timesteps,
            cond_embeddings,
            uncond_embeddings=uncond_embeddings,
            guidance_scale=guidance_scale,
        )
        latents = self._predict_x0(noise_latents, noise_pred, timesteps)
        if return_noise:
            return latents, noise_pred
        return latents

    def _sample_dm_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        total_steps = int(self.scheduler.config.num_train_timesteps)
        min_step = int(total_steps * self.dm_min_timestep_percent)
        max_step = int(total_steps * self.dm_max_timestep_percent)
        max_step = max(min_step + 1, min(total_steps, max_step + 1))
        return torch.randint(min_step, max_step, (batch_size,), device=device).long()

    def compute_distribution_matching_loss(
        self,
        fake_latents: torch.Tensor,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            latents = fake_latents.detach()
            noise = torch.randn_like(latents)
            timesteps = self._sample_dm_timesteps(latents.shape[0], latents.device)
            noisy_latents = self.add_noise(latents, noise, timesteps)

            pred_fake_noise = self._predict_noise(
                self.fake_score_unet,
                noisy_latents,
                timesteps,
                cond_embeddings,
                guidance_scale=self.fake_score_guidance_scale,
            )
            pred_fake_x0 = self._predict_x0(noisy_latents, pred_fake_noise, timesteps)

            pred_real_noise = self._predict_noise(
                self.real_score_unet,
                noisy_latents,
                timesteps,
                cond_embeddings,
                uncond_embeddings=uncond_embeddings,
                guidance_scale=self.real_score_guidance_scale,
            )
            pred_real_x0 = self._predict_x0(noisy_latents, pred_real_noise, timesteps)

            p_real = latents - pred_real_x0
            p_fake = latents - pred_fake_x0
            denom = p_real.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
            grad = torch.nan_to_num((p_real - p_fake) / denom)

        loss = 0.5 * F.mse_loss(
            fake_latents.float(),
            (fake_latents - grad).detach().float(),
        )
        log_dict = {
            'dm_timesteps': timesteps.detach(),
            'dm_grad_norm': grad.norm().detach(),
            'pred_real_x0': pred_real_x0.detach(),
            'pred_fake_x0': pred_fake_x0.detach(),
        }
        return loss, log_dict

    def compute_fake_score_loss(
        self,
        fake_latents: torch.Tensor,
        cond_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        latents = fake_latents.detach()
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()
        noisy_latents = self.add_noise(latents, noise, timesteps)
        pred_noise = self._predict_noise(
            self.fake_score_unet,
            noisy_latents,
            timesteps,
            cond_embeddings,
            guidance_scale=self.fake_score_guidance_scale,
        )
        loss = F.mse_loss(pred_noise.float(), noise.float())
        return loss, {
            'fake_score_timesteps': timesteps.detach(),
            'fake_score_pred': pred_noise.detach(),
        }

    def sample_teacher_deterministic(
        self,
        noise_latents: torch.Tensor,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: torch.Tensor | None = None,
        num_inference_steps: int = 20,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        guidance_scale = (
            self.regression_guidance_scale
            if guidance_scale is None else guidance_scale
        )
        scheduler = copy.deepcopy(self.scheduler)
        try:
            scheduler.set_timesteps(num_inference_steps, device=noise_latents.device)
        except TypeError:
            scheduler.set_timesteps(num_inference_steps)
        sample = noise_latents * getattr(scheduler, 'init_noise_sigma', 1.0)

        with torch.no_grad():
            for t in scheduler.timesteps:
                timestep_value = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                batch_timesteps = torch.full(
                    (sample.shape[0],),
                    timestep_value,
                    device=sample.device,
                    dtype=torch.long,
                )
                noise_pred = self._predict_noise(
                    self.real_score_unet,
                    sample,
                    batch_timesteps,
                    cond_embeddings,
                    uncond_embeddings=uncond_embeddings,
                    guidance_scale=guidance_scale,
                )
                sample = scheduler.step(noise_pred, t, sample).prev_sample
        return sample
