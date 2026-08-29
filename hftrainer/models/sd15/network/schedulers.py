"""Pure PyTorch noise schedules used by SD1.5 training and inference."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from hftrainer.registry import MODEL_COMPONENTS

from .configuration import ConfigDict, clean_config
from .outputs import SchedulerOutput


def _betas_for_alpha_bar(num_steps: int, max_beta: float = 0.999) -> torch.Tensor:
    def alpha_bar(time_step: float) -> float:
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for index in range(num_steps):
        start = index / num_steps
        end = (index + 1) / num_steps
        betas.append(min(1 - alpha_bar(end) / alpha_bar(start), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def _broadcast(values: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    while values.ndim < sample.ndim:
        values = values.unsqueeze(-1)
    return values.to(device=sample.device, dtype=sample.dtype)


class NoiseScheduler:
    """Shared local checkpoint/config and forward-process implementation."""

    scheduler_name = 'noise'

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = 'scaled_linear',
        trained_betas: Iterable[float] | None = None,
        prediction_type: str = 'epsilon',
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        set_alpha_to_one: bool = False,
        steps_offset: int = 0,
        timestep_spacing: str = 'leading',
        rescale_betas_zero_snr: bool = False,
        **kwargs,
    ):
        # Unknown metadata is retained to make round trips lossless.
        self.config = ConfigDict(
            num_train_timesteps=int(num_train_timesteps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            beta_schedule=beta_schedule,
            trained_betas=list(trained_betas) if trained_betas is not None else None,
            prediction_type=prediction_type,
            clip_sample=bool(clip_sample),
            clip_sample_range=float(clip_sample_range),
            set_alpha_to_one=bool(set_alpha_to_one),
            steps_offset=int(steps_offset),
            timestep_spacing=timestep_spacing,
            rescale_betas_zero_snr=bool(rescale_betas_zero_snr),
            **kwargs,
        )
        if trained_betas is not None:
            betas = torch.tensor(list(trained_betas), dtype=torch.float32)
        elif beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == 'scaled_linear':
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        elif beta_schedule in {'squaredcos_cap_v2', 'cosine'}:
            betas = _betas_for_alpha_bar(num_train_timesteps)
        else:
            raise ValueError(f'Unsupported beta schedule: {beta_schedule}')
        if rescale_betas_zero_snr:
            alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
            sqrt_alphas = alphas_cumprod.sqrt()
            initial, final = sqrt_alphas[0].clone(), sqrt_alphas[-1].clone()
            sqrt_alphas = (sqrt_alphas - final) * initial / (initial - final)
            alphas_cumprod = sqrt_alphas.square()
            alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.cat([1 - alphas_cumprod[:1], 1 - alphas]).clamp(0, 0.999)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.final_alpha_cumprod = (
            self.one if set_alpha_to_one else self.alphas_cumprod[0]
        )
        self.init_noise_sigma = 1.0
        self.num_inference_steps: int | None = None
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides):
        config = clean_config(config)
        config.update(overrides)
        return cls(**config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        subfolder: str | None = None,
        **overrides,
    ):
        root = Path(pretrained_model_name_or_path)
        if subfolder:
            root = root / subfolder
        if not root.is_dir():
            raise FileNotFoundError(f'{cls.__name__} requires a local scheduler directory: {root}')
        candidates = (root / 'scheduler_config.json', root / 'config.json')
        config_path = next((path for path in candidates if path.is_file()), None)
        if config_path is None:
            raise FileNotFoundError(f'Missing scheduler config under {root}.')
        with config_path.open('r', encoding='utf-8') as handle:
            config = clean_config(json.load(handle))
        config.update(overrides)
        return cls(**config)

    def save_pretrained(self, save_directory: str | Path):
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'scheduler_config.json'
        payload = self.config.to_dict()
        payload['_class_name'] = type(self).__name__
        with path.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            'schema_version': 1,
            'format': 'hftrainer-local-scheduler',
            'scheduler': type(self).__name__,
            'config': path.name,
            'sha256': digest,
        }
        with (root / 'manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write('\n')
        return (str(path),)

    def set_timesteps(self, num_inference_steps: int, device=None):
        num_inference_steps = int(num_inference_steps)
        if not 0 < num_inference_steps <= self.config.num_train_timesteps:
            raise ValueError('num_inference_steps must be in the training schedule range.')
        self.num_inference_steps = num_inference_steps
        spacing = self.config.timestep_spacing
        if spacing == 'linspace':
            steps = torch.linspace(
                0, self.config.num_train_timesteps - 1, num_inference_steps
            ).round().long().flip(0)
        elif spacing == 'trailing':
            ratio = self.config.num_train_timesteps / num_inference_steps
            steps = torch.arange(self.config.num_train_timesteps, 0, -ratio).round().long() - 1
        else:
            ratio = self.config.num_train_timesteps // num_inference_steps
            steps = (torch.arange(num_inference_steps).long() * ratio).flip(0)
            steps = steps + self.config.steps_offset
        self.timesteps = steps.to(device=device)
        return self.timesteps

    def _previous_timestep(self, timestep: int) -> int:
        if self.num_inference_steps is None:
            return timestep - 1
        matches = (self.timesteps.cpu() == timestep).nonzero(as_tuple=False)
        if matches.numel() and int(matches[0]) + 1 < len(self.timesteps):
            return int(self.timesteps[int(matches[0]) + 1].item())
        return -1

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = timesteps.long().cpu()
        alpha = _broadcast(self.alphas_cumprod[timesteps].sqrt(), original_samples)
        sigma = _broadcast((1 - self.alphas_cumprod[timesteps]).sqrt(), original_samples)
        return alpha * original_samples + sigma * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = timesteps.long().cpu()
        alpha = _broadcast(self.alphas_cumprod[timesteps].sqrt(), sample)
        sigma = _broadcast((1 - self.alphas_cumprod[timesteps]).sqrt(), sample)
        return alpha * noise - sigma * sample

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        return sample

    def __len__(self) -> int:
        return self.config.num_train_timesteps


@MODEL_COMPONENTS.register_module(name='DDPMScheduler', force=True)
class DDPMScheduler(NoiseScheduler):
    scheduler_name = 'ddpm'

    def __init__(self, variance_type: str = 'fixed_small', thresholding: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.config.variance_type = variance_type
        self.config.thresholding = bool(thresholding)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
        **_,
    ):
        t = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        previous = self._previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t].to(sample.device, sample.dtype)
        alpha_prod_prev = (
            self.alphas_cumprod[previous] if previous >= 0 else self.final_alpha_cumprod
        ).to(sample.device, sample.dtype)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev = 1 - alpha_prod_prev
        current_alpha = alpha_prod_t / alpha_prod_prev
        current_beta = 1 - current_alpha

        prediction_type = self.config.prediction_type
        if prediction_type == 'epsilon':
            pred_original = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif prediction_type == 'sample':
            pred_original = model_output
        elif prediction_type == 'v_prediction':
            pred_original = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(f'Unsupported prediction type: {prediction_type}')
        if self.config.thresholding:
            flat = pred_original.float().abs().reshape(pred_original.shape[0], -1)
            dynamic = torch.quantile(flat, 0.995, dim=1).clamp(min=1, max=1).to(pred_original.dtype)
            dynamic = _broadcast(dynamic, pred_original)
            pred_original = pred_original.clamp(-dynamic, dynamic) / dynamic
        elif self.config.clip_sample:
            limit = self.config.clip_sample_range
            pred_original = pred_original.clamp(-limit, limit)

        pred_original_coeff = alpha_prod_prev.sqrt() * current_beta / beta_prod_t
        current_sample_coeff = current_alpha.sqrt() * beta_prod_prev / beta_prod_t
        prev_sample = pred_original_coeff * pred_original + current_sample_coeff * sample
        if t > 0:
            variance = ((1 - alpha_prod_prev) / (1 - alpha_prod_t) * current_beta).clamp(min=1e-20)
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = prev_sample + variance.sqrt() * noise
        output = SchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original)
        return output if return_dict else (prev_sample,)


@MODEL_COMPONENTS.register_module(name='DDIMScheduler', force=True)
class DDIMScheduler(NoiseScheduler):
    scheduler_name = 'ddim'

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        variance_noise: torch.Tensor | None = None,
        return_dict: bool = True,
        **_,
    ):
        t = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        previous = self._previous_timestep(t)
        alpha_t = self.alphas_cumprod[t].to(sample.device, sample.dtype)
        alpha_prev = (
            self.alphas_cumprod[previous] if previous >= 0 else self.final_alpha_cumprod
        ).to(sample.device, sample.dtype)
        beta_t = 1 - alpha_t
        if self.config.prediction_type == 'epsilon':
            pred_original = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
            pred_epsilon = model_output
        elif self.config.prediction_type == 'sample':
            pred_original = model_output
            pred_epsilon = (sample - alpha_t.sqrt() * pred_original) / beta_t.sqrt()
        elif self.config.prediction_type == 'v_prediction':
            pred_original = alpha_t.sqrt() * sample - beta_t.sqrt() * model_output
            pred_epsilon = alpha_t.sqrt() * model_output + beta_t.sqrt() * sample
        else:
            raise ValueError(f'Unsupported prediction type: {self.config.prediction_type}')
        if self.config.clip_sample:
            pred_original = pred_original.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )
        variance = ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).clamp(min=0)
        std = float(eta) * variance.sqrt()
        direction = (1 - alpha_prev - std.square()).clamp(min=0).sqrt() * pred_epsilon
        prev_sample = alpha_prev.sqrt() * pred_original + direction
        if eta > 0:
            noise = variance_noise
            if noise is None:
                noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = prev_sample + std * noise
        output = SchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original)
        return output if return_dict else (prev_sample,)


@MODEL_COMPONENTS.register_module(name='PNDMScheduler', force=True)
class PNDMScheduler(DDIMScheduler):
    """Pseudo numerical sampler with Runge--Kutta and multistep phases."""

    scheduler_name = 'pndm'

    def __init__(self, skip_prk_steps: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.config.skip_prk_steps = bool(skip_prk_steps)
        self.pndm_order = 4
        self.ets: list[torch.Tensor] = []
        self.counter = 0
        self.cur_model_output: torch.Tensor | int = 0
        self.cur_sample: torch.Tensor | None = None
        self.prk_timesteps = torch.empty(0, dtype=torch.long)
        self.plms_timesteps = torch.empty(0, dtype=torch.long)

    def set_timesteps(self, num_inference_steps: int, device=None):
        num_inference_steps = int(num_inference_steps)
        if not 0 < num_inference_steps <= self.config.num_train_timesteps:
            raise ValueError('num_inference_steps must be in the training schedule range.')
        self.num_inference_steps = num_inference_steps
        if self.config.timestep_spacing == 'linspace':
            base = torch.linspace(
                0, self.config.num_train_timesteps - 1, num_inference_steps
            ).round().long()
        elif self.config.timestep_spacing == 'trailing':
            ratio = self.config.num_train_timesteps / num_inference_steps
            base = torch.arange(
                self.config.num_train_timesteps, 0, -ratio
            ).round().long().flip(0) - 1
        else:
            ratio = self.config.num_train_timesteps // num_inference_steps
            base = torch.arange(num_inference_steps).long() * ratio
            base = base + self.config.steps_offset

        if self.config.skip_prk_steps:
            self.prk_timesteps = torch.empty(0, dtype=torch.long)
            self.plms_timesteps = torch.cat([base[:-1], base[-2:-1], base[-1:]]).flip(0)
        else:
            half_step = self.config.num_train_timesteps // num_inference_steps // 2
            tail = base[-self.pndm_order:].repeat_interleave(2)
            offsets = torch.tensor([0, half_step] * self.pndm_order, dtype=torch.long)
            prk = tail + offsets
            self.prk_timesteps = prk[:-1].repeat_interleave(2)[1:-1].flip(0)
            self.plms_timesteps = base[:-3].flip(0)
        self.timesteps = torch.cat([self.prk_timesteps, self.plms_timesteps]).to(device=device)
        self.ets = []
        self.counter = 0
        self.cur_model_output = 0
        self.cur_sample = None
        return self.timesteps

    def step(self, model_output, timestep, sample, return_dict=True, **_):
        if self.num_inference_steps is None:
            raise ValueError('set_timesteps() must be called before step().')
        if self.counter < len(self.prk_timesteps) and not self.config.skip_prk_steps:
            return self.step_prk(model_output, timestep, sample, return_dict=return_dict)
        return self.step_plms(model_output, timestep, sample, return_dict=return_dict)

    def step_prk(self, model_output, timestep, sample, return_dict=True):
        timestep = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        half_step = self.config.num_train_timesteps // self.num_inference_steps // 2
        difference = 0 if self.counter % 2 else half_step
        previous_timestep = timestep - difference
        timestep = int(self.prk_timesteps[self.counter // 4 * 4].item())

        if self.counter % 4 == 0:
            self.cur_model_output = model_output / 6
            self.ets.append(model_output)
            self.cur_sample = sample
        elif (self.counter - 1) % 4 == 0 or (self.counter - 2) % 4 == 0:
            self.cur_model_output = self.cur_model_output + model_output / 3
        else:
            model_output = self.cur_model_output + model_output / 6
            self.cur_model_output = 0
        current_sample = self.cur_sample if self.cur_sample is not None else sample
        previous = self._get_prev_sample(
            current_sample, timestep, previous_timestep, model_output
        )
        self.counter += 1
        output = SchedulerOutput(prev_sample=previous)
        return output if return_dict else (previous,)

    def step_plms(self, model_output, timestep, sample, return_dict=True):
        timestep = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        if not self.config.skip_prk_steps and len(self.ets) < 3:
            raise RuntimeError('The Runge--Kutta warmup must finish before multistep sampling.')
        step_size = self.config.num_train_timesteps // self.num_inference_steps
        previous_timestep = timestep - step_size
        if self.counter != 1:
            self.ets = self.ets[-3:]
            self.ets.append(model_output)
        else:
            previous_timestep = timestep
            timestep = timestep + step_size

        if len(self.ets) == 1 and self.counter == 0:
            estimate = model_output
            self.cur_sample = sample
        elif len(self.ets) == 1 and self.counter == 1:
            estimate = (model_output + self.ets[-1]) / 2
            sample = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
            estimate = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            estimate = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        else:
            estimate = (
                55 * self.ets[-1] - 59 * self.ets[-2]
                + 37 * self.ets[-3] - 9 * self.ets[-4]
            ) / 24
        previous = self._get_prev_sample(sample, timestep, previous_timestep, estimate)
        self.counter += 1
        output = SchedulerOutput(prev_sample=previous)
        return output if return_dict else (previous,)

    def _get_prev_sample(self, sample, timestep, previous_timestep, model_output):
        alpha_t = self.alphas_cumprod[timestep].to(sample.device, sample.dtype)
        alpha_prev = (
            self.alphas_cumprod[previous_timestep]
            if previous_timestep >= 0 else self.final_alpha_cumprod
        ).to(sample.device, sample.dtype)
        beta_t = 1 - alpha_t
        beta_prev = 1 - alpha_prev
        if self.config.prediction_type == 'v_prediction':
            model_output = alpha_t.sqrt() * model_output + beta_t.sqrt() * sample
        elif self.config.prediction_type != 'epsilon':
            raise ValueError('PNDM supports epsilon and v_prediction outputs.')
        sample_coefficient = (alpha_prev / alpha_t).sqrt()
        output_denominator = alpha_t * beta_prev.sqrt() + (
            alpha_t * beta_t * alpha_prev
        ).sqrt()
        return (
            sample_coefficient * sample
            - (alpha_prev - alpha_t) * model_output / output_denominator
        )
