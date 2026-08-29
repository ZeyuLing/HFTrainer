"""Repository-owned MiniMax-H3 model bundle."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES

from .network import (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Processor,
    MiniMaxH3Qwen3VLEncoder,
    MiniMaxH3Scheduler,
    MiniMaxH3Tokenizer,
    MiniMaxH3Transformer3DModel,
)
from .network.layout import (
    MiniMaxH3PackedLayout,
    MiniMaxH3ReferenceGeometry,
    build_fl2va_layout,
    build_ref2va_layout,
)


@dataclass(frozen=True)
class MiniMaxH3PromptEncoding:
    """Conditioner output plus the H3 modality tag of every row."""

    prompt_embeds: torch.Tensor
    token_tags: torch.Tensor
    token_ids: tuple[int, ...]
    presentation: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bundle_manifest(root: Path) -> dict[str, Any]:
    """Verify an HFTrainer-exported bundle before reading any of its config."""

    manifest_path = root / "minimax_h3_bundle_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"Expected a JSON object in {manifest_path}.")
    if (
        manifest.get("format") != "hftrainer-minimax-h3-local"
        or manifest.get("format_version") != 1
        or manifest.get("class_name") != "MiniMaxH3Bundle"
    ):
        raise ValueError(f"Unsupported MiniMax-H3 bundle manifest in {manifest_path}.")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise TypeError(f"Bundle manifest {manifest_path} has no file inventory.")
    resolved_root = root.resolve()
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise TypeError(f"Invalid file inventory in {manifest_path}.")
        relative = Path(item["name"])
        path = (resolved_root / relative).resolve()
        if relative.is_absolute() or resolved_root not in path.parents:
            raise ValueError(f"Unsafe bundle-manifest path: {relative}.")
        if not path.is_file():
            raise FileNotFoundError(f"Bundle-manifest file is missing: {path}.")
        if _sha256(path) != item.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {path}.")
    return manifest


def _class_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    return getattr(value, "__name__", type(value).__name__)


@MODEL_BUNDLES.register_module()
class MiniMaxH3Bundle(ModelBundle):
    """One H3 checkpoint partition and its shared local conditioners/codecs.

    ``variant='fl2va'`` owns the official ``transformer/`` partition and
    serves both text-only and first/last-frame requests. ``variant='ref2va'``
    owns ``transformer_ref/``.  The two partitions are never silently aliased.
    """

    _LOCAL_TYPES: ClassVar[dict[str, type]] = {
        "text_encoder": MiniMaxH3Qwen3VLEncoder,
        "vae": AutoencoderKLMiniMaxH3,
        "audio_vae": AutoencoderKLMiniMaxH3Audio,
        "transformer": MiniMaxH3Transformer3DModel,
        "scheduler": MiniMaxH3Scheduler,
        "audio_scheduler": MiniMaxH3Scheduler,
    }

    def __init__(
        self,
        *,
        transformer: Mapping[str, Any],
        scheduler: Mapping[str, Any],
        audio_scheduler: Mapping[str, Any],
        text_encoder: Mapping[str, Any] | None = None,
        vae: Mapping[str, Any] | None = None,
        audio_vae: Mapping[str, Any] | None = None,
        tokenizer_path: str | None = None,
        processor_path: str | None = None,
        variant: str = "fl2va",
        conditioning_layer: int = 50,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in {"fl2va", "ref2va"}:
            raise ValueError("variant must be 'fl2va' or 'ref2va'.")
        if conditioning_layer < 1:
            raise ValueError("conditioning_layer must be positive.")
        self.variant = variant
        self.conditioning_layer = int(conditioning_layer)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        module_cfgs: dict[str, Mapping[str, Any]] = {
            "transformer": transformer,
            "scheduler": scheduler,
            "audio_scheduler": audio_scheduler,
        }
        for name, config in (
            ("text_encoder", text_encoder),
            ("vae", vae),
            ("audio_vae", audio_vae),
        ):
            if config is not None:
                module_cfgs[name] = config

        normalized = {}
        for name, raw_config in module_cfgs.items():
            if not isinstance(raw_config, Mapping):
                raise TypeError(
                    f"MiniMax-H3 component {name!r} needs a config mapping."
                )
            config = copy.deepcopy(dict(raw_config))
            expected = self._LOCAL_TYPES[name]
            component_type = config.get("type", expected.__name__)
            if component_type not in (expected, expected.__name__):
                raise ValueError(
                    f"MiniMax-H3 component {name!r} only accepts the local "
                    f"{expected.__name__}, got {_class_name(component_type)!r}."
                )
            config["type"] = expected.__name__
            if name == "transformer" and self.gradient_checkpointing:
                config.setdefault("gradient_checkpointing", True)
            normalized[name] = config
        self._build_modules(normalized)
        for name, expected in self._LOCAL_TYPES.items():
            if name not in normalized:
                continue
            actual = getattr(self, name)
            # LoRA intentionally wraps Linear children, not the root class.
            if type(actual) is not expected:
                raise TypeError(
                    f"MiniMax-H3 component {name!r} resolved to "
                    f"{type(actual).__module__}.{type(actual).__name__}, expected "
                    f"the repository-owned {expected.__module__}.{expected.__name__}."
                )

        if tokenizer_path is None:
            vocab_size = 151_936
            if hasattr(self, "text_encoder"):
                config = getattr(self.text_encoder, "config", None)
                text_config = getattr(config, "text_config", config)
                vocab_size = int(getattr(text_config, "vocab_size", vocab_size))
            self.tokenizer = MiniMaxH3Tokenizer(vocab_size=vocab_size)
        else:
            self.tokenizer = MiniMaxH3Tokenizer.from_pretrained(tokenizer_path)
        processor_root = processor_path or tokenizer_path
        if processor_root is None:
            self.processor = MiniMaxH3Processor(tokenizer=self.tokenizer)
        else:
            self.processor = MiniMaxH3Processor.from_pretrained(
                processor_root, tokenizer=self.tokenizer
            )
        self._extra_attributes.update(
            {"tokenizer": self.tokenizer, "processor": self.processor}
        )

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        variant: str | None = None,
        torch_dtype: str | torch.dtype | None = None,
        device_map: str | Mapping[str, Any] | None = None,
        device: torch.device | str = "cpu",
        transformer_device: torch.device | str | None = None,
        conditioner_device: torch.device | str | None = None,
        vae_device: torch.device | str | None = None,
        audio_vae_device: torch.device | str | None = None,
        conditioner_dtype: str | torch.dtype | None = None,
        vae_dtype: str | torch.dtype = torch.float32,
        audio_vae_dtype: str | torch.dtype = torch.float32,
        low_cpu_mem_usage: bool = True,
        trainable: bool | str = False,
        lora_cfg: Mapping[str, Any] | None = None,
        gradient_checkpointing: bool = False,
        conditioning_layer: int | None = None,
        load_conditioner: bool = True,
        load_vaes: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        if kwargs:
            raise TypeError(
                "Unexpected MiniMaxH3Bundle.from_pretrained options: "
                + ", ".join(sorted(kwargs))
            )
        if device_map is not None:
            raise ValueError(
                "MiniMaxH3Bundle does not silently emulate a layer-wise "
                "device_map. Use transformer_device, conditioner_device, "
                "vae_device, and audio_vae_device for explicit component "
                "placement."
            )
        root_path = Path(pretrained_model_name_or_path).expanduser()
        if root_path.is_dir():
            _verify_bundle_manifest(root_path)
        saved_bundle_config = root_path / "minimax_h3_bundle_config.json"
        saved_variant = None
        saved_conditioning_layer = None
        if saved_bundle_config.is_file():
            saved = json.loads(saved_bundle_config.read_text(encoding="utf-8"))
            saved_variant = saved.get("variant")
            saved_conditioning_layer = saved.get("conditioning_layer")
        if variant is None:
            variant = saved_variant or "fl2va"
        variant = str(variant).lower()
        if variant not in {"fl2va", "ref2va"}:
            raise ValueError("variant must be 'fl2va' or 'ref2va'.")
        if saved_variant is not None and variant != saved_variant:
            raise ValueError(
                f"The saved bundle contains variant={saved_variant!r}, not "
                f"variant={variant!r}."
            )
        if conditioning_layer is None:
            conditioning_layer = (
                50
                if saved_conditioning_layer is None
                else int(saved_conditioning_layer)
            )
        conditioning_layer = int(conditioning_layer)
        if conditioning_layer < 1:
            raise ValueError("conditioning_layer must be positive.")
        root = str(pretrained_model_name_or_path)

        transformer_device = transformer_device or device
        conditioner_device = conditioner_device or device
        vae_device = vae_device or device
        audio_vae_device = audio_vae_device or vae_device
        transformer_dtype = torch_dtype or torch.bfloat16
        conditioner_dtype = conditioner_dtype or transformer_dtype

        def load_config(
            type_name: str,
            subfolder: str,
            *,
            component_dtype: str | torch.dtype | None,
            component_device: torch.device | str,
            trainable_: Any = False,
        ):
            loading: dict[str, Any] = {
                "pretrained_model_name_or_path": root,
                "subfolder": subfolder,
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "device": component_device,
            }
            if component_dtype is not None:
                loading["torch_dtype"] = component_dtype
            result: dict[str, Any] = {
                "type": type_name,
                "from_pretrained": loading,
                "trainable": trainable_,
                "save_ckpt": bool(trainable_),
            }
            return result

        transformer_cfg = load_config(
            "MiniMaxH3Transformer3DModel",
            "transformer" if variant == "fl2va" else "transformer_ref",
            component_dtype=transformer_dtype,
            component_device=transformer_device,
            trainable_=trainable,
        )
        if trainable == "lora":
            transformer_cfg["lora_cfg"] = dict(lora_cfg or {})
            transformer_cfg["checkpoint_format"] = "lora"
        config: dict[str, Any] = {
            "variant": variant,
            "transformer": transformer_cfg,
            "scheduler": {
                "type": "MiniMaxH3Scheduler",
                "from_pretrained": {
                    "pretrained_model_name_or_path": root,
                    "subfolder": "scheduler",
                },
                "trainable": False,
                "save_ckpt": False,
            },
            "audio_scheduler": {
                "type": "MiniMaxH3Scheduler",
                "from_pretrained": {
                    "pretrained_model_name_or_path": root,
                    "subfolder": "audio_scheduler",
                },
                "trainable": False,
                "save_ckpt": False,
            },
            "tokenizer_path": str(Path(root) / "tokenizer"),
            "processor_path": str(Path(root) / "processor"),
            "gradient_checkpointing": gradient_checkpointing,
            "conditioning_layer": conditioning_layer,
        }
        if load_conditioner:
            config["text_encoder"] = load_config(
                "MiniMaxH3Qwen3VLEncoder",
                "text_encoder",
                component_dtype=conditioner_dtype,
                component_device=conditioner_device,
            )
        if load_vaes:
            config["vae"] = load_config(
                "AutoencoderKLMiniMaxH3",
                "vae",
                component_dtype=vae_dtype,
                component_device=vae_device,
            )
            config["audio_vae"] = load_config(
                "AutoencoderKLMiniMaxH3Audio",
                "audio_vae",
                component_dtype=audio_vae_dtype,
                component_device=audio_vae_device,
            )
        return config

    def require_components(self, *names: str) -> None:
        missing = [name for name in names if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                "This MiniMax-H3 operation needs components not loaded in the "
                "cached-feature profile: " + ", ".join(missing)
            )

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    def encode_prompt(
        self,
        prompt: str,
        *,
        mode: str = "t2va",
        first_frame=None,
        last_frame=None,
        references: Sequence[Any] = (),
    ) -> MiniMaxH3PromptEncoding:
        self.require_components("text_encoder")
        presentation = self.processor.encode_presentation(
            prompt,
            mode=mode,
            first_frame=first_frame,
            last_frame=last_frame,
            references=references,
        )
        if isinstance(presentation, Mapping):
            token_ids = presentation["token_ids"]
            token_tags = presentation["token_tags"]
            vision_inputs = presentation.get("vision_inputs")
            presentation_text = presentation.get("presentation", prompt)
        else:
            token_ids = presentation.token_ids
            token_tags = presentation.token_tags
            vision_inputs = getattr(presentation, "vision_inputs", None)
            presentation_text = getattr(presentation, "presentation", prompt)
        embeds = self.text_encoder.encode(
            token_ids,
            processor=self.processor,
            vision_inputs=vision_inputs,
            conditioning_layer=self.conditioning_layer,
            device=self.device,
            dtype=self.dtype,
        )
        return MiniMaxH3PromptEncoding(
            prompt_embeds=embeds,
            token_tags=torch.as_tensor(token_tags, dtype=torch.long),
            token_ids=tuple(int(value) for value in token_ids),
            presentation=str(presentation_text),
        )

    @staticmethod
    def _latent_stats(config, channels: int, *, device, dtype, ndim: int):
        mean = torch.as_tensor(
            getattr(config, "latents_mean", [0.0] * channels),
            device=device,
            dtype=dtype,
        )
        std = torch.as_tensor(
            getattr(config, "latents_std", [1.0] * channels),
            device=device,
            dtype=dtype,
        )
        shape = (1, channels, *([1] * (ndim - 2)))
        return mean.view(shape), std.view(shape)

    def encode_video(
        self,
        pixels: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        sample_posterior: bool = True,
        condition_rounding: bool = False,
    ) -> torch.Tensor:
        self.require_components("vae")
        if pixels.ndim != 5:
            raise ValueError("Video input must be BCTHW or BTCHW.")
        if pixels.shape[1] != 3 and pixels.shape[2] == 3:
            pixels = pixels.permute(0, 2, 1, 3, 4)
        if pixels.shape[1] != 3:
            raise ValueError("Video input needs three RGB channels.")
        parameter = next(self.vae.parameters())
        values = pixels.to(device=parameter.device)
        if values.dtype == torch.uint8:
            values = values.to(torch.float32).div(255.0)
        else:
            values = values.to(torch.float32)
            if not torch.isfinite(values).all():
                raise ValueError("Video pixels must be finite.")
            minimum, maximum = values.aminmax()
            if minimum < 0 or maximum > 1:
                raise ValueError(
                    "Floating-point video pixels must use the [0, 1] RGB range."
                )
        pixel_mean = torch.tensor(
            (0.485, 0.456, 0.406), device=values.device, dtype=values.dtype
        ).view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor(
            (0.229, 0.224, 0.225), device=values.device, dtype=values.dtype
        ).view(1, 3, 1, 1, 1)
        values = ((values - pixel_mean) / pixel_std).to(parameter.dtype)
        posterior = self.vae.encode(values).latent_dist
        latents = (
            posterior.sample(generator=generator)
            if sample_posterior
            else posterior.mode()
        )
        if condition_rounding:
            latents = latents.to(torch.float16).float()
        mean, std = self._latent_stats(
            self.vae.config,
            latents.shape[1],
            device=latents.device,
            dtype=latents.dtype,
            ndim=latents.ndim,
        )
        return (latents - mean) / std

    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized H3 latents to BCTHW RGB pixels in ``[0, 1]``."""
        self.require_components("vae")
        parameter = next(self.vae.parameters())
        latents = latents.to(parameter.device)
        mean, std = self._latent_stats(
            self.vae.config,
            latents.shape[1],
            device=latents.device,
            dtype=latents.dtype,
            ndim=latents.ndim,
        )
        values = (latents * std + mean).to(parameter.dtype)
        with torch.autocast(
            device_type=parameter.device.type,
            dtype=torch.float16,
            enabled=parameter.device.type == "cuda",
        ):
            output = self.vae.decode(values)
        decoded = output.sample if hasattr(output, "sample") else output[0]
        pixel_mean = torch.tensor(
            (0.485, 0.456, 0.406), device=decoded.device, dtype=torch.float32
        ).view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor(
            (0.229, 0.224, 0.225), device=decoded.device, dtype=torch.float32
        ).view(1, 3, 1, 1, 1)
        return (decoded.float() * pixel_std + pixel_mean).clamp(0, 1)

    def encode_audio(
        self,
        waveform: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        sample_posterior: bool = True,
    ) -> torch.Tensor:
        self.require_components("audio_vae")
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError("Audio input must be [B, channels, samples].")
        batch, channels, samples = waveform.shape
        if channels not in (1, 2):
            raise ValueError("MiniMax-H3 audio must be mono or stereo.")
        if channels == 1:
            waveform = waveform.expand(batch, 2, samples)
            channels = 2
        parameter = next(self.audio_vae.parameters())
        values = waveform.reshape(batch * channels, 1, samples).to(
            parameter.device, parameter.dtype
        )
        posterior = self.audio_vae.encode(values).latent_dist
        latents = (
            posterior.sample(generator=generator)
            if sample_posterior
            else posterior.mode()
        )
        mean, std = self._latent_stats(
            self.audio_vae.config,
            latents.shape[1],
            device=latents.device,
            dtype=latents.dtype,
            ndim=latents.ndim,
        )
        latents = (latents - mean) / std
        return latents.reshape(batch, channels, latents.shape[1], latents.shape[2])

    def decode_audio(self, latents: torch.Tensor) -> torch.Tensor:
        self.require_components("audio_vae")
        if latents.ndim != 4 or latents.shape[1] != 2:
            raise ValueError("Audio latents must be [B, 2, channels, time].")
        batch, stereo, channels, length = latents.shape
        values = latents.reshape(batch * stereo, channels, length)
        parameter = next(self.audio_vae.parameters())
        values = values.to(parameter.device)
        mean, std = self._latent_stats(
            self.audio_vae.config,
            channels,
            device=values.device,
            dtype=values.dtype,
            ndim=values.ndim,
        )
        output = self.audio_vae.decode((values * std + mean).to(parameter.dtype))
        samples = output.sample if hasattr(output, "sample") else output[0]
        return samples.reshape(batch, stereo, -1)

    def build_layout(
        self,
        text_token_tags: torch.Tensor,
        *,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        keyframe_anchors: Sequence[str] = (),
        references: Sequence[MiniMaxH3ReferenceGeometry] = (),
    ) -> MiniMaxH3PackedLayout:
        patch_size = tuple(self.transformer.config.patch_size)
        if self.variant == "ref2va":
            if keyframe_anchors:
                raise ValueError("Ref2VA does not use first/last-frame anchors.")
            return build_ref2va_layout(
                text_token_tags,
                references,
                num_latent_frames=num_latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                num_audio_latents=num_audio_latents,
                patch_size=patch_size,
            )
        if references:
            raise ValueError("Omni references require the ref2va checkpoint partition.")
        return build_fl2va_layout(
            text_token_tags,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            patch_size=patch_size,
            keyframe_anchors=keyframe_anchors,
        )

    def predict_velocity(
        self,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        prompt_embeds: torch.Tensor,
        layout: MiniMaxH3PackedLayout,
        timesteps: torch.Tensor,
        timestep_indices: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_rows.ndim == 2:
            video_rows = video_rows.unsqueeze(0)
        if audio_rows.ndim == 2:
            audio_rows = audio_rows.unsqueeze(0)
        output = self.transformer(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            timestep_indices=timestep_indices,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            attention_kwargs=attention_kwargs,
            return_dict=True,
        )
        return output.sample, output.audio_sample

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        *,
        merge_lora: bool = True,
    ) -> str:
        lora_modules = tuple(self._lora_modules)
        if lora_modules and not merge_lora:
            raise ValueError(
                "MiniMaxH3Bundle cannot export an unbound adapter as a full "
                "pretrained bundle. Keep the HFTrainer training checkpoint "
                "for resumable LoRA, or use merge_lora=True to export a "
                "standalone inference artifact."
            )
        if lora_modules:
            # A pretrained bundle is self-contained. Fold local adapters into
            # their repository-owned base layers before serializing so the
            # emitted checkpoint schema is exactly reloadable without hidden
            # adapter configuration or an external base-model reference.
            self.merge_lora_weights(list(lora_modules))

        root = Path(save_directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        for name in self._LOCAL_TYPES:
            component = getattr(self, name, None)
            if component is not None and hasattr(component, "save_pretrained"):
                subfolder = name
                if name == "transformer" and self.variant == "ref2va":
                    subfolder = "transformer_ref"
                save_kwargs = (
                    {"safe_serialization": safe_serialization}
                    if isinstance(component, nn.Module)
                    else {}
                )
                component.save_pretrained(root / subfolder, **save_kwargs)
        self.tokenizer.save_pretrained(root / "tokenizer")
        self.processor.save_pretrained(root / "processor")
        config = {
            "format": "hftrainer-minimax-h3-local",
            "format_version": 1,
            "class_name": type(self).__name__,
            "variant": self.variant,
            "conditioning_layer": self.conditioning_layer,
            "transformer_partition": (
                "transformer" if self.variant == "fl2va" else "transformer_ref"
            ),
            "merged_lora_modules": list(lora_modules),
        }
        config_path = root / "minimax_h3_bundle_config.json"
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files = [
            {
                "name": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "minimax_h3_bundle_manifest.json"
        ]
        manifest = root / "minimax_h3_bundle_manifest.json"
        manifest.write_text(
            json.dumps({**config, "files": files}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return str(manifest)


__all__ = ["MiniMaxH3Bundle", "MiniMaxH3PromptEncoding"]
