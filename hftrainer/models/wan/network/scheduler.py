"""Local Flow-Matching Euler scheduler used for Wan training and sampling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from .common import (
    FORMAT_VERSION,
    LOCAL_FORMAT,
    SchedulerOutput,
    WanConfig,
    read_json,
    resolve_pretrained_directory,
    sha256_file,
    write_json,
)


class FlowMatchEulerDiscreteScheduler:
    """Euler integration over a shifted flow-matching sigma schedule.

    This implements the scheduler surface consumed by ``WanTrainer`` and
    ``WanPipeline``: training noise scaling, inference timestep construction,
    and first-order Euler updates.
    """

    config_name = "scheduler_config.json"
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        invert_sigmas: bool = False,
        shift_terminal: float | None = None,
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        time_shift_type: str = "exponential",
        **kwargs,
    ):
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if shift <= 0:
            raise ValueError("shift must be positive")
        sigma_modes = sum(
            bool(value)
            for value in (use_karras_sigmas, use_exponential_sigmas, use_beta_sigmas)
        )
        if sigma_modes > 1:
            raise ValueError("Only one alternate sigma schedule can be enabled")
        self.config = WanConfig(
            num_train_timesteps=int(num_train_timesteps),
            shift=float(shift),
            use_dynamic_shifting=bool(use_dynamic_shifting),
            base_shift=None if base_shift is None else float(base_shift),
            max_shift=None if max_shift is None else float(max_shift),
            base_image_seq_len=int(base_image_seq_len),
            max_image_seq_len=int(max_image_seq_len),
            invert_sigmas=bool(invert_sigmas),
            shift_terminal=shift_terminal,
            use_karras_sigmas=bool(use_karras_sigmas),
            use_exponential_sigmas=bool(use_exponential_sigmas),
            use_beta_sigmas=bool(use_beta_sigmas),
            time_shift_type=time_shift_type,
            **kwargs,
        )
        training_sigmas = torch.linspace(
            1.0, 1.0 / num_train_timesteps, num_train_timesteps
        )
        training_sigmas = self._shift_sigmas(training_sigmas, shift)
        if invert_sigmas:
            training_sigmas = 1.0 - training_sigmas
        self.sigmas = training_sigmas
        self.timesteps = training_sigmas * num_train_timesteps
        self.sigma_min = float(training_sigmas[-1])
        self.sigma_max = float(training_sigmas[0])
        self._shift = float(shift)
        self.num_inference_steps: int | None = None
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @staticmethod
    def _shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)

    @property
    def shift(self) -> float:
        return self._shift

    def set_shift(self, shift: float) -> None:
        if shift <= 0:
            raise ValueError("shift must be positive")
        self._shift = float(shift)

    def _dynamic_shift(self, sigmas: torch.Tensor, mu: float) -> torch.Tensor:
        if self.config.time_shift_type == "linear":
            return mu / (mu + (1.0 / sigmas.clamp(min=1e-12) - 1.0))
        return torch.exp(torch.tensor(mu, dtype=sigmas.dtype, device=sigmas.device)) / (
            torch.exp(torch.tensor(mu, dtype=sigmas.dtype, device=sigmas.device))
            + (1.0 / sigmas.clamp(min=1e-12) - 1.0)
        )

    def _stretch_to_terminal(
        self, sigmas: torch.Tensor, terminal: float
    ) -> torch.Tensor:
        if sigmas.numel() < 2:
            return sigmas
        scale = (1.0 - terminal) / max(float(1.0 - sigmas[-1]), 1e-12)
        return 1.0 - (1.0 - sigmas) * scale

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | str | None = None,
        sigmas: Sequence[float] | torch.Tensor | None = None,
        mu: float | None = None,
        timesteps: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        if sigmas is not None and timesteps is not None:
            raise ValueError("Pass sigmas or timesteps, not both")
        if sigmas is None:
            if timesteps is not None:
                sigma_tensor = (
                    torch.as_tensor(timesteps, dtype=torch.float32)
                    / self.config.num_train_timesteps
                )
            else:
                if num_inference_steps is None or num_inference_steps <= 0:
                    raise ValueError("num_inference_steps must be positive")
                sigma_tensor = torch.linspace(
                    1.0, 1.0 / num_inference_steps, num_inference_steps
                )
        else:
            sigma_tensor = torch.as_tensor(sigmas, dtype=torch.float32).flatten()
            if sigma_tensor.numel() == 0:
                raise ValueError("sigmas must not be empty")
            if sigma_tensor[-1] == 0:
                sigma_tensor = sigma_tensor[:-1]

        if self.config.use_dynamic_shifting:
            if mu is None:
                raise ValueError("mu is required when use_dynamic_shifting=True")
            sigma_tensor = self._dynamic_shift(sigma_tensor, float(mu))
        else:
            sigma_tensor = self._shift_sigmas(sigma_tensor, self.shift)
        if self.config.shift_terminal is not None:
            sigma_tensor = self._stretch_to_terminal(
                sigma_tensor, float(self.config.shift_terminal)
            )
        if self.config.invert_sigmas:
            sigma_tensor = 1.0 - sigma_tensor

        if device is not None:
            sigma_tensor = sigma_tensor.to(device)
        self.num_inference_steps = int(sigma_tensor.numel())
        self.timesteps = sigma_tensor * self.config.num_train_timesteps
        terminal = (
            sigma_tensor.new_ones(1)
            if self.config.invert_sigmas
            else sigma_tensor.new_zeros(1)
        )
        self.sigmas = torch.cat((sigma_tensor, terminal))
        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def index_for_timestep(self, timestep: torch.Tensor | float) -> int:
        value = torch.as_tensor(timestep, device=self.timesteps.device).reshape(-1)[0]
        distances = (self.timesteps - value).abs()
        indices = (distances == distances.min()).nonzero(as_tuple=False).flatten()
        # Starting at the second duplicate avoids skipping a sigma when schedules overlap.
        return int(indices[1] if indices.numel() > 1 else indices[0])

    def _init_step_index(self, timestep: torch.Tensor | float) -> None:
        self._step_index = (
            self._begin_index
            if self._begin_index is not None
            else self.index_for_timestep(timestep)
        )

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        return sample

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(sample)
        timesteps = timestep.reshape(-1).to(self.timesteps.device)
        indices = torch.stack(
            [torch.argmin((self.timesteps - value).abs()) for value in timesteps]
        )
        sigma = self.sigmas[indices].to(device=sample.device, dtype=sample.dtype)
        sigma = sigma.view(sample.shape[0], *([1] * (sample.ndim - 1)))
        return sigma * noise + (1.0 - sigma) * sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.scale_noise(original_samples, timesteps, noise)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        del s_churn, s_tmin, s_tmax, s_noise, generator
        if self._step_index is None:
            self._init_step_index(timestep)
        if self._step_index >= len(self.sigmas) - 1:
            raise IndexError("Scheduler step called after the final sigma")
        sigma = self.sigmas[self._step_index].to(sample.device, torch.float32)
        sigma_next = self.sigmas[self._step_index + 1].to(sample.device, torch.float32)
        prev_sample = sample.float() + (sigma_next - sigma) * model_output.float()
        prev_sample = prev_sample.to(sample.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    @classmethod
    def from_config(cls, config, **kwargs):
        values = dict(config)
        values.update(kwargs)
        values.pop("_class_name", None)
        return cls(**values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        subfolder: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> FlowMatchEulerDiscreteScheduler:
        del strict
        directory = resolve_pretrained_directory(
            pretrained_model_name_or_path, subfolder
        )
        config_path = directory / cls.config_name
        if not config_path.is_file():
            fallback = directory / "config.json"
            if not fallback.is_file():
                raise FileNotFoundError(f"Missing scheduler config in {directory}")
            config_path = fallback
        config = read_json(config_path)
        for key in tuple(config):
            if key.startswith("_"):
                config.pop(key)
        for key in (
            "cache_dir",
            "local_files_only",
            "revision",
            "token",
            "torch_dtype",
            "dtype",
        ):
            kwargs.pop(key, None)
        config.update(kwargs)
        return cls(**config)

    def save_pretrained(self, save_directory: str | Path, **kwargs) -> str:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected scheduler save kwargs: {unexpected}")
        directory = Path(save_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        config = self.config.to_dict()
        config["_class_name"] = type(self).__name__
        config_path = directory / self.config_name
        write_json(config_path, config)
        manifest_path = directory / "wan_scheduler_manifest.json"
        write_json(
            manifest_path,
            {
                "format": LOCAL_FORMAT,
                "format_version": FORMAT_VERSION,
                "class_name": type(self).__name__,
                "files": [
                    {"name": config_path.name, "sha256": sha256_file(config_path)}
                ],
            },
        )
        return str(manifest_path)
