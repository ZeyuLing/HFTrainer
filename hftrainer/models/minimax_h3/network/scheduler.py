# Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MODIFIED BY HFTRAINER: configuration and serialization are repository-local
# and have no Diffusers dependency.  The numerical schedule is unchanged.

"""Rectified-flow Euler scheduler used by MiniMax-H3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import torch

from .common import (
    LOCAL_FORMAT,
    read_json,
    register_to_config,
    resolve_pretrained_directory,
    sha256_file,
    write_json,
)
from .configuration import clean_config
from .outputs import MiniMaxH3SchedulerOutput


class MiniMaxH3Scheduler:
    r"""Rectified-flow Euler scheduler with MiniMax-H3's timestep convention.

    ``t = 1 - sigma`` and ``t=1`` means clean data.  The transformer predicts a
    data-ward velocity, so the denoised estimate uses ``x0 = x_t + sigma*v``.
    """

    _compatibles: ClassVar[list[str]] = []
    order = 1
    config_name = "scheduler_config.json"

    @register_to_config
    def __init__(self, shift: float = 12.0):
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self.num_inference_steps: int | None = None
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self._shift = float(shift)
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def __len__(self) -> int:
        return 0 if self.timesteps is None else len(self.timesteps)

    def set_begin_index(self, begin_index: int = 0) -> None:
        if begin_index < 0:
            raise ValueError("begin_index cannot be negative")
        self._begin_index = int(begin_index)

    def set_shift(self, shift: float) -> None:
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self._shift = float(shift)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        """Build the shifted sigma grid, including terminal zero."""

        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(
                    "set_timesteps requires explicit sigmas or "
                    f"num_inference_steps >= 2, got {num_inference_steps}"
                )
            base = torch.linspace(
                1.0, 0.0, int(num_inference_steps), dtype=torch.float32
            )
            sigma_tensor = self._shift * base / (1.0 + (self._shift - 1.0) * base)
            sigma_tensor = torch.unique_consecutive(sigma_tensor)
        else:
            sigma_tensor = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
            valid = (
                sigma_tensor.numel() >= 2
                and bool((sigma_tensor[1:] < sigma_tensor[:-1]).all())
                and sigma_tensor[-1].item() == 0.0
            )
            if not valid:
                raise ValueError(
                    "sigmas must contain at least two strictly decreasing "
                    "values ending at 0.0"
                )

        self.sigmas = sigma_tensor.to(device=device)
        self.timesteps = (1.0 - sigma_tensor[:-1]).to(device=device)
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep: float | torch.Tensor) -> int:
        if self.timesteps is None:
            raise RuntimeError("Call set_timesteps before looking up a timestep")
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero(as_tuple=False)
        if len(indices) == 0:
            raise ValueError(
                "Passed timestep is not in scheduler.timesteps; pass a schedule value"
            )
        return int(indices[0].item())

    def scale_model_input(
        self, sample: torch.Tensor, timestep: float | torch.Tensor | None = None
    ) -> torch.Tensor:
        del timestep
        return sample

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: float | torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward process ``x_t = t*x0 + (1-t)*noise``."""

        if sample.shape != noise.shape:
            raise ValueError("sample and noise must have identical shapes")
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype, device=sample.device)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)
        return timestep * sample + (1.0 - timestep) * noise

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: float | torch.Tensor,
    ) -> torch.Tensor:
        return self.scale_noise(original_samples, timesteps, noise)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> MiniMaxH3SchedulerOutput | tuple[torch.Tensor]:
        if isinstance(timestep, int) or (
            isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()
        ):
            raise ValueError(
                "Integer step indices are not valid timesteps; pass one value "
                "from scheduler.timesteps"
            )
        if self.sigmas is None or self.timesteps is None:
            raise RuntimeError("Call set_timesteps before step")
        if model_output.shape != sample.shape:
            raise ValueError("model_output and sample must have identical shapes")
        if self._step_index is None:
            self._step_index = (
                self.index_for_timestep(timestep)
                if self._begin_index is None
                else self._begin_index
            )
        if self._step_index >= len(self.sigmas) - 1:
            raise IndexError("Scheduler step called after the terminal sigma")

        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype)
        sigma_from_timestep = 1.0 - timestep.to(
            device=sample.device, dtype=sample.dtype
        )
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
        denoised = sample + sigma_from_timestep * model_output

        compute_dtype = (
            torch.float32
            if sample.dtype in (torch.float16, torch.bfloat16)
            else sample.dtype
        )
        sigma = self.sigmas[self._step_index].to(
            device=sample.device, dtype=compute_dtype
        )
        sigma_next = self.sigmas[self._step_index + 1].to(
            device=sample.device, dtype=compute_dtype
        )
        ratio = sigma_next / sigma
        prev_sample = ratio * sample.to(compute_dtype) + (1.0 - ratio) * denoised.to(
            compute_dtype
        )
        prev_sample = prev_sample.to(sample.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)

    @classmethod
    def from_config(cls, config: Mapping[str, object], **kwargs) -> MiniMaxH3Scheduler:
        values = clean_config(config)
        values.update(kwargs)
        return cls(**values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        subfolder: str | None = None,
        **kwargs,
    ) -> MiniMaxH3Scheduler:
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
        for key in (
            "cache_dir",
            "local_files_only",
            "revision",
            "token",
            "torch_dtype",
            "dtype",
        ):
            kwargs.pop(key, None)
        return cls.from_config(config, **kwargs)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        safe_serialization: bool | None = None,
        **kwargs,
    ) -> str:
        # Accepted for bundle-wide component saving; schedulers have no tensor
        # payload, so the serialization flag does not change their artifact.
        del safe_serialization
        if kwargs:
            raise TypeError(
                "Unexpected scheduler save options: " + ", ".join(sorted(kwargs))
            )
        directory = Path(save_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        config = self.config.to_dict()
        config["_class_name"] = type(self).__name__
        config_path = directory / self.config_name
        write_json(config_path, config)
        manifest_path = directory / "minimax_h3_scheduler_manifest.json"
        write_json(
            manifest_path,
            {
                "format": LOCAL_FORMAT,
                "format_version": 1,
                "class_name": type(self).__name__,
                "files": [
                    {"name": config_path.name, "sha256": sha256_file(config_path)}
                ],
            },
        )
        return str(manifest_path)


__all__ = ["MiniMaxH3Scheduler", "MiniMaxH3SchedulerOutput"]
