"""Repository-local Wan text-to-video model bundle."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES

from .network import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UMT5EncoderModel,
    WanTokenizer,
    WanTransformer3DModel,
)
from .network.common import (
    FORMAT_VERSION,
    LOCAL_FORMAT,
    read_json,
    sha256_file,
    write_json,
)

BUNDLE_CONFIG_NAME = "wan_bundle_config.json"
BUNDLE_MANIFEST_NAME = "wan_bundle_manifest.json"


def _component_class_name(component_type) -> str:
    if isinstance(component_type, str):
        return component_type
    return getattr(component_type, "__name__", type(component_type).__name__)


@MODEL_BUNDLES.register_module()
class WanBundle(ModelBundle):
    """Local Wan components shared by the trainer and inference pipeline.

    Component types are resolved against an explicit local class table. Dotted
    import paths and arbitrary registry fallbacks are rejected, so installing a
    separate model stack cannot change which implementation executes.
    """

    PRETRAINED_SPEC: ClassVar[dict[str, Any]] = {
        "shared_pretrained_kwargs_arg": "shared_pretrained_kwargs",
        "components": {
            "text_encoder": {
                "default_type": "UMT5EncoderModel",
                "type_arg": "text_encoder_type",
                "subfolder": "text_encoder",
                "overrides_arg": "text_encoder_overrides",
            },
            "vae": {
                "default_type": "AutoencoderKLWan",
                "type_arg": "vae_type",
                "subfolder": "vae",
                "overrides_arg": "vae_overrides",
            },
            "transformer": {
                "default_type": "WanTransformer3DModel",
                "type_arg": "transformer_type",
                "subfolder": "transformer",
                "overrides_arg": "transformer_overrides",
            },
            "scheduler": {
                "default_type": "FlowMatchEulerDiscreteScheduler",
                "type_arg": "scheduler_type",
                "subfolder": "scheduler",
                "overrides_arg": "scheduler_overrides",
            },
        },
        "init_args": {
            "tokenizer_path": {"default": ModelBundle._PRETRAINED_PATH_SENTINEL},
            "max_token_length": 512,
            "gradient_checkpointing": False,
        },
    }
    _LOCAL_TYPES: ClassVar[dict[str, type]] = {
        "text_encoder": UMT5EncoderModel,
        "vae": AutoencoderKLWan,
        "transformer": WanTransformer3DModel,
        "scheduler": FlowMatchEulerDiscreteScheduler,
    }

    def __init__(
        self,
        text_encoder: dict,
        vae: dict,
        transformer: dict,
        scheduler: dict,
        tokenizer_path: str | None = None,
        max_token_length: int = 512,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if max_token_length <= 0:
            raise ValueError("max_token_length must be positive")
        self.max_token_length = int(max_token_length)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        transformer_cfg = copy.deepcopy(dict(transformer))
        if gradient_checkpointing and "gradient_checkpointing" not in transformer_cfg:
            transformer_cfg["gradient_checkpointing"] = True
        self._build_local_components(
            {
                "text_encoder": text_encoder,
                "vae": vae,
                "transformer": transformer_cfg,
                "scheduler": scheduler,
            }
        )

        pretrained_path = tokenizer_path
        if pretrained_path is None:
            te_cfg = text_encoder if isinstance(text_encoder, Mapping) else {}
            load_cfg = te_cfg.get("from_pretrained", {})
            if isinstance(load_cfg, Mapping):
                pretrained_path = load_cfg.get("pretrained_model_name_or_path")
        if pretrained_path is None:
            self.tokenizer = WanTokenizer(
                vocab_size=int(self.text_encoder.config.vocab_size),
                pad_token_id=int(self.text_encoder.config.pad_token_id),
                eos_token_id=int(self.text_encoder.config.eos_token_id),
                model_max_length=self.max_token_length,
            )
        else:
            self.tokenizer = WanTokenizer.from_pretrained(pretrained_path)
        if len(self.tokenizer) > int(self.text_encoder.config.vocab_size):
            raise ValueError(
                f"Tokenizer vocabulary ({len(self.tokenizer)}) exceeds text encoder "
                f"vocabulary ({self.text_encoder.config.vocab_size})"
            )
        self._extra_attributes["tokenizer"] = self.tokenizer

    def _build_local_components(self, modules_cfg: Mapping[str, dict]) -> None:
        self._save_ckpt_modules = []
        self._trainable_modules = []
        self._frozen_modules = []
        self._lora_modules = []
        self._module_checkpoint_formats = {}
        self._module_build_configs = {}

        for name, raw_cfg in modules_cfg.items():
            expected_cls = self._LOCAL_TYPES[name]
            if type(raw_cfg) is expected_cls:
                cfg: dict[str, Any] = {}
                component = raw_cfg
                trainable = isinstance(component, nn.Module)
                save_ckpt = trainable
                gradient_checkpointing = False
                module_dtype_spec = None
            else:
                if not isinstance(raw_cfg, Mapping):
                    raise TypeError(f"Wan component '{name}' must be a config mapping")
                cfg = copy.deepcopy(dict(raw_cfg))
                component_type = cfg.pop("type", expected_cls.__name__)
                if component_type not in (expected_cls, expected_cls.__name__):
                    raise ValueError(
                        f"Wan component '{name}' only accepts local type "
                        f"'{expected_cls.__name__}', got {_component_class_name(component_type)!r}"
                    )
                trainable = cfg.pop("trainable", True)
                save_ckpt = cfg.pop("save_ckpt", bool(trainable))
                gradient_checkpointing = cfg.pop("gradient_checkpointing", False)
                module_dtype_spec = cfg.pop("module_dtype", None)
                checkpoint_format = cfg.pop("checkpoint_format", "full")
                lora_cfg = cfg.pop("lora_cfg", None)
                if (
                    trainable == "lora"
                    or checkpoint_format == "lora"
                    or lora_cfg is not None
                ):
                    raise ValueError(
                        "WanBundle's dependency-isolated path accepts full local weights "
                        "only; inject adapters before constructing the bundle."
                    )
                if checkpoint_format != "full":
                    raise ValueError(
                        f"Unsupported checkpoint format for '{name}': {checkpoint_format}"
                    )

                direct_dtype = cfg.pop("torch_dtype", None)
                if "dtype" in cfg:
                    direct_dtype = cfg.pop("dtype")
                load_cfg = cfg.pop("from_pretrained", None)
                init_cfg = cfg.pop("from_config", None)
                if load_cfg is not None and init_cfg is not None:
                    raise ValueError(
                        f"Wan component '{name}' cannot use two load modes"
                    )
                if load_cfg is not None:
                    load_kwargs = copy.deepcopy(dict(load_cfg))
                    load_kwargs.update(cfg)
                    if direct_dtype is not None and not (
                        {"torch_dtype", "dtype"} & load_kwargs.keys()
                    ):
                        load_kwargs["torch_dtype"] = direct_dtype
                    component = expected_cls.from_pretrained(**load_kwargs)
                elif init_cfg is not None:
                    constructor_cfg = copy.deepcopy(dict(init_cfg))
                    constructor_cfg.update(cfg)
                    component = expected_cls.from_config(constructor_cfg)
                    if direct_dtype is not None and isinstance(component, nn.Module):
                        component.to(dtype=self._resolve_module_dtype(direct_dtype))
                else:
                    component = expected_cls(**cfg)
                    if direct_dtype is not None and isinstance(component, nn.Module):
                        component.to(dtype=self._resolve_module_dtype(direct_dtype))

            normalized_cfg = (
                copy.deepcopy(dict(raw_cfg))
                if isinstance(raw_cfg, Mapping)
                else {"type": expected_cls.__name__}
            )
            normalized_cfg.setdefault("type", expected_cls.__name__)
            normalized_cfg["trainable"] = trainable
            normalized_cfg["save_ckpt"] = bool(save_ckpt)
            normalized_cfg["checkpoint_format"] = "full"
            self._module_build_configs[name] = normalized_cfg

            if isinstance(component, nn.Module):
                if gradient_checkpointing:
                    self._enable_gradient_checkpointing(
                        component, name, gradient_checkpointing
                    )
                if module_dtype_spec is not None:
                    component.to(dtype=self._resolve_module_dtype(module_dtype_spec))
                if not bool(trainable):
                    component.requires_grad_(False)
                    self._frozen_modules.append(name)
                else:
                    self._trainable_modules.append(name)
                setattr(self, name, component)
                self._module_checkpoint_formats[name] = "full"
                if save_ckpt:
                    self._save_ckpt_modules.append(name)
            else:
                if gradient_checkpointing or module_dtype_spec is not None:
                    raise ValueError(
                        f"Non-module Wan component '{name}' has module-only options"
                    )
                object.__setattr__(self, name, component)
                self._extra_attributes[name] = component

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> dict[str, Any]:
        root = Path(pretrained_model_name_or_path).expanduser()
        config_path = root / BUNDLE_CONFIG_NAME
        saved_config = read_json(config_path) if config_path.is_file() else {}
        bundle_cfg = super()._bundle_config_from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )
        if saved_config:
            bundle_cfg["max_token_length"] = int(
                saved_config.get("max_token_length", bundle_cfg["max_token_length"])
            )
            bundle_cfg["gradient_checkpointing"] = bool(
                saved_config.get(
                    "gradient_checkpointing", bundle_cfg["gradient_checkpointing"]
                )
            )
            metadata = saved_config.get("component_options", {})
            for name in cls._LOCAL_TYPES:
                options = (
                    metadata.get(name, {}) if isinstance(metadata, Mapping) else {}
                )
                for key in ("trainable", "save_ckpt", "gradient_checkpointing"):
                    if key in options:
                        bundle_cfg[name][key] = copy.deepcopy(options[key])
        return bundle_cfg

    @classmethod
    def _verify_bundle_manifest(cls, root: Path) -> dict[str, Any]:
        manifest_path = root / BUNDLE_MANIFEST_NAME
        if not manifest_path.is_file():
            return {}
        manifest = read_json(manifest_path)
        if (
            manifest.get("format") != LOCAL_FORMAT
            or manifest.get("format_version") != FORMAT_VERSION
        ):
            raise ValueError(f"Unsupported Wan bundle manifest: {manifest_path}")
        if manifest.get("class_name") != cls.__name__:
            raise ValueError(f"Manifest in {root} does not describe {cls.__name__}")
        for file_info in manifest.get("files", []):
            path = root / file_info["name"]
            if not path.is_file():
                raise FileNotFoundError(f"Bundle manifest file is missing: {path}")
            if sha256_file(path) != file_info.get("sha256"):
                raise ValueError(f"SHA-256 mismatch for bundle artifact {path}")
        return manifest

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config_overrides: dict[str, Any] | None = None,
        strict: bool | None = None,
        **kwargs,
    ):
        root = Path(pretrained_model_name_or_path).expanduser()
        if root.is_dir():
            cls._verify_bundle_manifest(root.resolve())
        if strict is not None:
            shared = copy.deepcopy(kwargs.pop("shared_pretrained_kwargs", None) or {})
            shared["strict"] = bool(strict)
            kwargs["shared_pretrained_kwargs"] = shared
        return super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config_overrides=config_overrides,
            **kwargs,
        )

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_token_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.text_encoder.device)
        attention_mask = tokens.attention_mask.to(self.text_encoder.device)
        with torch.set_grad_enabled(self.text_encoder.training):
            outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        return outputs.last_hidden_state * attention_mask.unsqueeze(-1).to(
            outputs.last_hidden_state.dtype
        )

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean_values = getattr(self.vae.config, "latents_mean", None)
        std_values = getattr(self.vae.config, "latents_std", None)
        if mean_values is not None and std_values is not None:
            mean = torch.as_tensor(
                mean_values, dtype=latents.dtype, device=latents.device
            )
            std = torch.as_tensor(
                std_values, dtype=latents.dtype, device=latents.device
            )
            shape = [1, -1] + [1] * (latents.ndim - 2)
            return (latents - mean.view(*shape)) / std.view(*shape)
        return latents * float(getattr(self.vae.config, "scaling_factor", 1.0))

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean_values = getattr(self.vae.config, "latents_mean", None)
        std_values = getattr(self.vae.config, "latents_std", None)
        if mean_values is not None and std_values is not None:
            mean = torch.as_tensor(
                mean_values, dtype=latents.dtype, device=latents.device
            )
            std = torch.as_tensor(
                std_values, dtype=latents.dtype, device=latents.device
            )
            shape = [1, -1] + [1] * (latents.ndim - 2)
            return latents * std.view(*shape) + mean.view(*shape)
        return latents / float(getattr(self.vae.config, "scaling_factor", 1.0))

    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5:
            raise ValueError(
                f"Expected a five-dimensional video batch, got {videos.shape}"
            )
        if videos.shape[1] != self.vae.config.in_channels:
            if videos.shape[2] != self.vae.config.in_channels:
                raise ValueError("Video tensor is neither BCTHW nor BTCHW")
            videos = videos.permute(0, 2, 1, 3, 4)
        vae_parameter = next(self.vae.parameters())
        videos = videos.to(device=vae_parameter.device, dtype=vae_parameter.dtype)
        with torch.set_grad_enabled(self.vae.training):
            latents = self.vae.encode(videos).latent_dist.sample()
        return self._normalize_latents(latents)

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        vae_parameter = next(self.vae.parameters())
        latents = self._denormalize_latents(latents).to(
            device=vae_parameter.device, dtype=vae_parameter.dtype
        )
        with torch.set_grad_enabled(self.vae.training):
            return self.vae.decode(latents).sample

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        **kwargs,
    ) -> str:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected WanBundle save kwargs: {unexpected}")
        root = Path(save_directory).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        for name in ("text_encoder", "vae", "transformer"):
            getattr(self, name).save_pretrained(
                root / name, safe_serialization=safe_serialization
            )
        self.scheduler.save_pretrained(root / "scheduler")
        self.tokenizer.save_pretrained(root / "tokenizer")

        component_options = {}
        for name in self._LOCAL_TYPES:
            build_cfg = self.get_module_build_cfg(name)
            component_options[name] = {
                key: copy.deepcopy(build_cfg[key])
                for key in ("trainable", "save_ckpt", "gradient_checkpointing")
                if key in build_cfg
            }
        bundle_config = {
            "_class_name": type(self).__name__,
            "format": LOCAL_FORMAT,
            "format_version": FORMAT_VERSION,
            "max_token_length": self.max_token_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "components": {
                name: {"class_name": cls.__name__, "subfolder": name}
                for name, cls in self._LOCAL_TYPES.items()
            },
            "tokenizer": {"class_name": "WanTokenizer", "subfolder": "tokenizer"},
            "component_options": component_options,
            "source_notice": "SOURCES.md in the repository package",
        }
        config_path = root / BUNDLE_CONFIG_NAME
        write_json(config_path, bundle_config)

        files = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != BUNDLE_MANIFEST_NAME:
                files.append(
                    {
                        "name": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path),
                    }
                )
        manifest_path = root / BUNDLE_MANIFEST_NAME
        write_json(
            manifest_path,
            {
                "format": LOCAL_FORMAT,
                "format_version": FORMAT_VERSION,
                "class_name": type(self).__name__,
                "files": files,
            },
        )
        return str(manifest_path)
