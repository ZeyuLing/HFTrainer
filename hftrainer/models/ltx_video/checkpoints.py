"""LTX-2.5 split-checkpoint contracts shared by training and inference.

LTX-2.5 ships separate transformer, text encoder, video VAE, audio VAE,
duration-head and upsampler files.  Treating them as interchangeable paths is
dangerous: the distilled transformer is an inference model, the dev
transformer is the trainable base, and ComfyUI quantized files are not accepted
by the native PyTorch loader used by HFTrainer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LTX_25_OFFICIAL_REPO = 'Lightricks/LTX-2.5'
LTX_25_OFFICIAL_REVISION = 'main'
LTX_25_SOURCE_COMMIT = '400fd31054597515f47125691032c04b1c3ee24e'


def _path_text(value: str | Path) -> str:
    return str(Path(value).expanduser())


def _basename(value: str | Path) -> str:
    return Path(value).name.lower()


def _reject_comfy_only(path: str | Path, role: str) -> None:
    name = _basename(path)
    if 'int8-convrot' in name:
        raise ValueError(
            f"{role} points to a ComfyUI-only int8-convrot artifact: {path}. "
            "Use the BF16 native-PyTorch checkpoint instead."
        )


def _require_existing_file(path: str | Path, role: str) -> None:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing LTX-2.5 {role}: {resolved}")


@dataclass(frozen=True)
class LTXVideoLoraSpec:
    """One LoRA artifact and the strength used by the local loader."""

    path: str
    strength: float = 1.0

    @classmethod
    def from_value(cls, value: Any) -> LTXVideoLoraSpec:
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            return cls(path=_path_text(value))
        if isinstance(value, Mapping):
            unknown = set(value) - {'path', 'strength'}
            if unknown:
                raise ValueError(f"Unknown LoRA fields: {sorted(unknown)}")
            if 'path' not in value:
                raise ValueError("Each LTX LoRA entry requires a 'path'.")
            return cls(
                path=_path_text(value['path']),
                strength=float(value.get('strength', 1.0)),
            )
        if isinstance(value, (tuple, list)) and 1 <= len(value) <= 2:
            return cls(
                path=_path_text(value[0]),
                strength=float(value[1]) if len(value) == 2 else 1.0,
            )
        raise TypeError(
            "LTX LoRA entries must be a path, {'path', 'strength'} mapping, "
            "or (path, strength) pair."
        )


@dataclass(frozen=True)
class LTX25InferenceCheckpoints:
    """Native LTX-2.5 split pack used by an inference pipeline."""

    transformer_path: str
    text_encoder_path: str
    video_vae_path: str
    audio_vae_path: str
    spatial_upsampler_path: str
    duration_head_path: str | None = None
    distilled_lora_path: str | None = None

    def validate(
        self,
        *,
        mode: str,
        require_files: bool = True,
        strict_roles: bool = True,
    ) -> None:
        if mode not in {'distilled', 'dev_two_stage'}:
            raise ValueError(
                f"Unsupported LTX inference mode {mode!r}; expected "
                "'distilled' or 'dev_two_stage'."
            )

        roles = {
            'transformer': self.transformer_path,
            'packed Gemma 4 text encoder': self.text_encoder_path,
            'video VAE': self.video_vae_path,
            'audio VAE/vocoder': self.audio_vae_path,
            'spatial upsampler': self.spatial_upsampler_path,
        }
        if self.duration_head_path is not None:
            roles['duration head'] = self.duration_head_path
        if self.distilled_lora_path is not None:
            roles['distilled LoRA'] = self.distilled_lora_path

        for role, path in roles.items():
            if not str(path).strip():
                raise ValueError(f"LTX-2.5 {role} path cannot be empty.")
            _reject_comfy_only(path, role)
            if require_files:
                _require_existing_file(path, role)

        transformer_name = _basename(self.transformer_path)
        if strict_roles:
            if mode == 'distilled' and 'distilled-transformer' not in transformer_name:
                raise ValueError(
                    "Distilled inference requires the LTX-2.5 distilled transformer; "
                    f"got {self.transformer_path}."
                )
            if mode == 'dev_two_stage' and 'distilled-transformer' in transformer_name:
                raise ValueError(
                    "dev_two_stage inference requires the trainable dev transformer, "
                    "not the distilled transformer."
                )

        text_name = _basename(self.text_encoder_path)
        if strict_roles and not (
            'gemma4' in text_name and 'with-proj-ltx-2.5' in text_name
        ):
            raise ValueError(
                "LTX-2.5 requires its packed Gemma 4 text encoder "
                "(gemma4-12b-with-proj-ltx-2.5-*.safetensors); vanilla Gemma 4 "
                f"is not compatible. Got {self.text_encoder_path}."
            )

        if mode == 'dev_two_stage' and not self.distilled_lora_path:
            raise ValueError(
                "dev_two_stage inference requires the LTX-Video 2.5 distilled "
                "LoRA for the refinement stage."
            )


def validate_ltx25_generation_shape(
    *,
    height: int,
    width: int,
    num_frames: int | None,
    has_duration_head: bool,
) -> None:
    """Validate public two-stage generation constraints before loading 22B weights."""

    if height <= 0 or width <= 0:
        raise ValueError('height and width must be positive.')
    if height % 64 != 0 or width % 64 != 0:
        raise ValueError(
            "LTX-2.5 two-stage output height and width must be divisible by 64 "
            f"(got {height}x{width})."
        )
    if num_frames is None:
        if not has_duration_head:
            raise ValueError(
                "num_frames is required when duration_head_path is not configured."
            )
        return
    if num_frames <= 0 or num_frames % 8 != 1:
        raise ValueError(
            f"LTX-2.5 num_frames must satisfy num_frames % 8 == 1; got {num_frames}."
        )


def validate_ltx25_training_config(
    config: Mapping[str, Any],
    *,
    require_files: bool = False,
    strict_roles: bool = True,
) -> None:
    """Validate LTX-Video-specific invariants before local Pydantic parsing."""

    model = dict(config.get('model') or {})
    missing = [
        key
        for key in ('model_path', 'text_encoder_path', 'video_vae_path')
        if not model.get(key)
    ]
    if missing:
        raise ValueError(f"LTX-2.5 training model config is missing: {missing}")

    training_mode = model.get('training_mode', 'lora')
    if training_mode not in {'lora', 'full'}:
        raise ValueError("model.training_mode must be 'lora' or 'full'.")
    if training_mode == 'lora' and not config.get('lora'):
        raise ValueError("LoRA training requires a top-level lora configuration block.")
    acceleration = dict(config.get('acceleration') or {})
    if training_mode == 'full' and acceleration.get('quantization') is not None:
        raise ValueError('Full LTX-2.5 fine-tuning cannot use quantization.')

    model_path = model['model_path']
    _reject_comfy_only(model_path, 'training transformer')
    if strict_roles and 'distilled-transformer' in _basename(model_path):
        raise ValueError(
            "The LTX-2.5 distilled transformer is an 8-step inference model and "
            "is not a supported training base. Use the dev transformer."
        )
    if strict_roles and 'ltx-2.5' in _basename(model_path) and 'dev-transformer' not in _basename(model_path):
        raise ValueError(
            "An LTX-Video 2.5 training base must be the dev transformer; "
            f"got {model_path}."
        )

    text_path = model['text_encoder_path']
    if strict_roles and not (
        'gemma4' in _basename(text_path) and 'with-proj-ltx-2.5' in _basename(text_path)
    ):
        raise ValueError(
            "LTX-2.5 training requires the packed Gemma 4 text encoder with the "
            "LTX-2.5 projection."
        )

    strategy = dict(config.get('training_strategy') or {})
    audio_cfg = dict(strategy.get('audio') or {})
    needs_audio = bool(audio_cfg.get('is_generated', False))
    if needs_audio and not model.get('audio_vae_path'):
        raise ValueError(
            "Joint audio-video LTX-2.5 training requires model.audio_vae_path."
        )

    if require_files:
        file_roles = {
            'training transformer': model_path,
            'packed Gemma 4 text encoder': text_path,
            'video VAE': model['video_vae_path'],
        }
        if model.get('audio_vae_path'):
            file_roles['audio VAE/vocoder'] = model['audio_vae_path']
        for role, path in file_roles.items():
            _require_existing_file(path, role)


def normalize_lora_specs(values: Iterable[Any] | None) -> tuple[LTXVideoLoraSpec, ...]:
    return tuple(LTXVideoLoraSpec.from_value(value) for value in (values or ()))
