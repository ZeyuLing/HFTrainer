"""Shared configuration, outputs, and checkpoint I/O for local Wan models.

The classes in :mod:`hftrainer.models.wan.network` intentionally expose the
small subset of the Hugging Face-style API used by HFTrainer while keeping the
execution path dependent on PyTorch and the Python standard library only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

LOCAL_FORMAT = "hftrainer-wan-local"
FORMAT_VERSION = 1
MANIFEST_NAME = "wan_local_manifest.json"


class WanConfig(dict):
    """A JSON-serializable mapping with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(dict(self))


class WanModelOutput:
    """Tiny tuple/mapping-like output compatible with the bundle call sites."""

    _fields: tuple[str, ...] = ()

    def __iter__(self):
        for name in self._fields:
            yield getattr(self, name)

    def __getitem__(self, key):
        if isinstance(key, int):
            return tuple(self)[key]
        return getattr(self, key)

    def keys(self):
        return self._fields

    def to_tuple(self):
        return tuple(self)


@dataclass
class BaseModelOutput(WanModelOutput):
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    _fields = ("last_hidden_state", "hidden_states", "attentions")


@dataclass
class AutoencoderKLOutput(WanModelOutput):
    latent_dist: DiagonalGaussianDistribution
    _fields = ("latent_dist",)


@dataclass
class DecoderOutput(WanModelOutput):
    sample: torch.Tensor
    _fields = ("sample",)


@dataclass
class Transformer3DModelOutput(WanModelOutput):
    sample: torch.Tensor
    _fields = ("sample",)


@dataclass
class SchedulerOutput(WanModelOutput):
    prev_sample: torch.Tensor
    _fields = ("prev_sample",)


class DiagonalGaussianDistribution:
    """Diagonal Gaussian posterior used by the local KL autoencoder."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other: DiagonalGaussianDistribution | None = None) -> torch.Tensor:
        reduce_dims = tuple(range(1, self.mean.ndim))
        if other is None:
            return 0.5 * torch.sum(
                self.mean.square() + self.var - 1.0 - self.logvar,
                dim=reduce_dims,
            )
        return 0.5 * torch.sum(
            (self.mean - other.mean).square() / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=reduce_dims,
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, WanConfig):
        value = dict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_pretrained_directory(
    pretrained_model_name_or_path: str | Path,
    subfolder: str | None = None,
) -> Path:
    directory = Path(pretrained_model_name_or_path).expanduser()
    if subfolder:
        directory = directory / subfolder
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Local Wan artifacts must be directories; no directory exists at {directory}"
        )
    return directory.resolve()


def _load_tensor_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                f"Reading {path.name} requires the optional safetensors package."
            ) from exc
        return dict(load_file(str(path), device="cpu"))

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility
        value = torch.load(path, map_location="cpu")
    if isinstance(value, Mapping) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, Mapping):
        raise TypeError(f"Checkpoint {path} does not contain a state dictionary")
    return {
        str(key): tensor for key, tensor in value.items() if torch.is_tensor(tensor)
    }


def load_tensor_directory(
    directory: Path,
) -> tuple[dict[str, torch.Tensor], tuple[Path, ...]]:
    index_candidates = (
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    for name in index_candidates:
        index_path = directory / name
        if not index_path.is_file():
            continue
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise TypeError(f"Invalid shard index: {index_path}")
        shard_paths = tuple(
            directory / shard for shard in sorted(set(weight_map.values()))
        )
        state_dict: dict[str, torch.Tensor] = {}
        for shard_path in shard_paths:
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
            state_dict.update(_load_tensor_file(shard_path))
        return state_dict, (index_path, *shard_paths)

    candidates = (
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "pytorch_model.safetensors",
        "pytorch_model.bin",
        "diffusion_pytorch_model.bin",
    )
    for name in candidates:
        path = directory / name
        if path.is_file():
            return _load_tensor_file(path), (path,)
    raise FileNotFoundError(
        f"No local model weights found in {directory}; expected one of {candidates}"
    )


def save_tensor_directory(
    directory: Path,
    state_dict: Mapping[str, torch.Tensor],
    safe_serialization: bool,
) -> tuple[Path, dict[str, str]]:
    cpu_state = {
        key: tensor.detach().cpu().contiguous() for key, tensor in state_dict.items()
    }
    aliases: dict[str, str] = {}
    if safe_serialization:
        try:
            from safetensors.torch import save_file
        except ImportError:
            pass
        else:
            # Safe tensor files cannot encode shared storage. Keep one canonical
            # tensor and record exact aliases in the strict local manifest.
            storage_owners: dict[
                tuple[int, int, tuple[int, ...], tuple[int, ...]], str
            ] = {}
            safe_state: dict[str, torch.Tensor] = {}
            for key in sorted(cpu_state, key=lambda item: (len(item), item)):
                tensor = cpu_state[key]
                identity = (
                    tensor.untyped_storage().data_ptr(),
                    tensor.storage_offset(),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                )
                owner = storage_owners.get(identity)
                if owner is None:
                    storage_owners[identity] = key
                    safe_state[key] = tensor
                else:
                    aliases[key] = owner
            path = directory / "model.safetensors"
            save_file(safe_state, str(path))
            return path, aliases
    path = directory / "pytorch_model.bin"
    torch.save(cpu_state, path)
    return path, aliases


def tensor_schema(state_dict: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        for key, tensor in sorted(state_dict.items())
    }


def verify_local_manifest(
    directory: Path, expected_class: str | None = None
) -> dict[str, Any]:
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        return {}
    manifest = read_json(manifest_path)
    if (
        manifest.get("format") != LOCAL_FORMAT
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError(f"Unsupported local Wan manifest in {manifest_path}")
    if expected_class and manifest.get("class_name") != expected_class:
        raise ValueError(
            f"Manifest class is {manifest.get('class_name')!r}, expected {expected_class!r}"
        )
    for file_info in manifest.get("files", []):
        path = directory / file_info["name"]
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-declared file is missing: {path}")
        actual = sha256_file(path)
        if actual != file_info.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {path}")
    return manifest


def _strip_component_prefix(key: str, component_name: str | None) -> str:
    prefixes = ["module.", "model."]
    if component_name:
        prefixes.extend((f"{component_name}.", f"module.{component_name}."))
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def compatible_state_dict(
    expected: Mapping[str, torch.Tensor],
    incoming: Mapping[str, torch.Tensor],
    component_name: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Match exact keys first, then unambiguous suffixes with equal shapes."""

    normalized = {
        _strip_component_prefix(str(key), component_name): value
        for key, value in incoming.items()
        if torch.is_tensor(value)
    }
    matched: dict[str, torch.Tensor] = {}
    consumed = set()
    mismatched: dict[str, dict[str, Any]] = {}

    for key, target in expected.items():
        source = normalized.get(key)
        if source is not None:
            if tuple(source.shape) == tuple(target.shape):
                matched[key] = source
                consumed.add(key)
            else:
                mismatched[key] = {
                    "expected": list(target.shape),
                    "found": list(source.shape),
                }

    # Reconstruct tied parameters when a foreign safe-tensor file retained one
    # canonical key only (for example UMT5's shared token embedding).
    expected_storage: dict[
        tuple[int, int, tuple[int, ...], tuple[int, ...]], list[str]
    ] = {}
    for key, tensor in expected.items():
        identity = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        expected_storage.setdefault(identity, []).append(key)
    for aliases in expected_storage.values():
        source_key = next((key for key in aliases if key in matched), None)
        if source_key is not None:
            for alias in aliases:
                if alias not in matched and alias not in mismatched:
                    matched[alias] = matched[source_key]

    remaining = [
        key for key in expected if key not in matched and key not in mismatched
    ]
    incoming_remaining = [key for key in normalized if key not in consumed]
    for target_key in remaining:
        target = expected[target_key]
        candidates = [
            key
            for key in incoming_remaining
            if tuple(normalized[key].shape) == tuple(target.shape)
            and (key.endswith("." + target_key) or target_key.endswith("." + key))
        ]
        if len(candidates) == 1:
            source_key = candidates[0]
            matched[target_key] = normalized[source_key]
            consumed.add(source_key)
            incoming_remaining.remove(source_key)

    expected_numel = sum(tensor.numel() for tensor in expected.values())
    matched_numel = sum(expected[key].numel() for key in matched)
    report = {
        "matched_keys": sorted(matched),
        "missing_keys": sorted(set(expected) - set(matched)),
        "unexpected_keys": sorted(set(normalized) - consumed),
        "mismatched_shapes": mismatched,
        "parameter_coverage": matched_numel / max(expected_numel, 1),
    }
    return matched, report


def _resolve_dtype(torch_dtype=None, dtype=None) -> torch.dtype | None:
    value = dtype if dtype is not None else torch_dtype
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        aliases = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        value = value.removeprefix("torch.")
        if value in aliases:
            return aliases[value]
    raise ValueError(f"Unsupported dtype: {value!r}")


class LocalWanModelMixin:
    """Local ``from_pretrained``/``save_pretrained`` contract for nn.Modules."""

    config_name = "config.json"
    component_name: str | None = None

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | WanConfig, **kwargs):
        values = dict(config)
        values.update(kwargs)
        for key in tuple(values):
            if key.startswith("_") or key == "architectures":
                values.pop(key)
        return cls(**values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        subfolder: str | None = None,
        torch_dtype=None,
        dtype=None,
        strict: bool | None = None,
        allow_partial_load: bool = False,
        **config_overrides,
    ):
        directory = resolve_pretrained_directory(
            pretrained_model_name_or_path, subfolder
        )
        manifest = verify_local_manifest(directory, cls.__name__)
        config_path = directory / cls.config_name
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing model config: {config_path}")
        config = read_json(config_path)
        for key in tuple(config):
            if key.startswith("_") or key == "architectures":
                config.pop(key)

        ignored_loading_keys = {
            "cache_dir",
            "device_map",
            "force_download",
            "local_files_only",
            "low_cpu_mem_usage",
            "revision",
            "token",
            "use_auth_token",
            "use_safetensors",
            "variant",
        }
        for key in tuple(config_overrides):
            if key in ignored_loading_keys:
                config_overrides.pop(key)
        config.update(config_overrides)
        model = cls(**config)
        requested_dtype = _resolve_dtype(torch_dtype=torch_dtype, dtype=dtype)

        incoming, _ = load_tensor_directory(directory)
        for alias, canonical in manifest.get("aliases", {}).items():
            if canonical in incoming:
                incoming[alias] = incoming[canonical]
        expected = model.state_dict()
        matched, report = compatible_state_dict(
            expected, incoming, component_name=cls.component_name
        )
        report["source"] = str(directory)
        report["local_manifest"] = bool(manifest)
        model._load_report = report

        use_strict = bool(manifest) if strict is None else bool(strict)
        if use_strict and (
            report["missing_keys"]
            or report["unexpected_keys"]
            or report["mismatched_shapes"]
        ):
            raise RuntimeError(
                f"Strict {cls.__name__} checkpoint load failed: "
                f"{len(report['missing_keys'])} missing, "
                f"{len(report['unexpected_keys'])} unexpected, "
                f"{len(report['mismatched_shapes'])} shape mismatches"
            )
        if not matched and expected and not allow_partial_load:
            raise RuntimeError(
                f"Checkpoint at {directory} has no tensors compatible with "
                f"the local {cls.__name__}. Pass allow_partial_load=True only "
                "when intentionally initializing unmatched layers."
            )
        if not use_strict and report["missing_keys"] and not allow_partial_load:
            coverage = report["parameter_coverage"]
            if coverage < 0.5:
                raise RuntimeError(
                    f"Only {coverage:.1%} of local {cls.__name__} parameters are "
                    "covered by this foreign checkpoint. Refusing a mostly-random load; "
                    "set allow_partial_load=True to opt in."
                )
        model.load_state_dict(matched, strict=use_strict)
        if requested_dtype is not None:
            model.to(dtype=requested_dtype)
        return model

    def save_pretrained(
        self,
        save_directory: str | Path,
        safe_serialization: bool = True,
        **kwargs,
    ) -> str:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected save_pretrained kwargs: {unexpected}")
        directory = Path(save_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        config = self.config.to_dict()
        config["_class_name"] = type(self).__name__
        config_path = directory / self.config_name
        write_json(config_path, config)
        weight_path, aliases = save_tensor_directory(
            directory, self.state_dict(), safe_serialization=safe_serialization
        )
        manifest = {
            "format": LOCAL_FORMAT,
            "format_version": FORMAT_VERSION,
            "class_name": type(self).__name__,
            "component_name": self.component_name,
            "source_notice": "../SOURCES.md",
            "config_sha256": sha256_json(config),
            "state_dict": tensor_schema(self.state_dict()),
            "aliases": aliases,
            "files": [
                {"name": config_path.name, "sha256": sha256_file(config_path)},
                {"name": weight_path.name, "sha256": sha256_file(weight_path)},
            ],
        }
        manifest_path = directory / MANIFEST_NAME
        write_json(manifest_path, manifest)
        return str(manifest_path)

    def load_checkpoint(
        self,
        checkpoint_directory: str | Path,
        strict: bool = True,
        allow_partial_load: bool = False,
    ) -> dict[str, Any]:
        directory = resolve_pretrained_directory(checkpoint_directory)
        manifest = verify_local_manifest(directory, type(self).__name__)
        incoming, _ = load_tensor_directory(directory)
        for alias, canonical in manifest.get("aliases", {}).items():
            if canonical in incoming:
                incoming[alias] = incoming[canonical]
        matched, report = compatible_state_dict(
            self.state_dict(), incoming, component_name=self.component_name
        )
        if strict and (
            report["missing_keys"]
            or report["unexpected_keys"]
            or report["mismatched_shapes"]
        ):
            raise RuntimeError(f"Strict checkpoint load failed: {report}")
        if not matched and not allow_partial_load:
            raise RuntimeError("Checkpoint contains no compatible local tensors")
        self.load_state_dict(matched, strict=strict)
        self._load_report = report
        return report


def make_sinusoidal_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Create fp32 sinusoidal embeddings and preserve odd dimensions."""

    timesteps = timesteps.float().reshape(-1)
    half = dim // 2
    if half == 0:
        return timesteps[:, None]
    exponent = (
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps[:, None] * torch.exp(exponent)[None]
    embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


def pick_group_count(channels: int, requested: int = 32) -> int:
    groups = min(max(1, requested), channels)
    while channels % groups:
        groups -= 1
    return groups
