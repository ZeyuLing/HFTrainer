"""Shared local runtime and artifact I/O for MiniMax-H3 components.

This module intentionally implements only repository-local behavior.  It does
not resolve Hub identifiers and never imports an external model framework.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from .configuration import ConfigDict, clean_config, load_config

LOCAL_FORMAT = "hftrainer-minimax-h3-local"
FORMAT_VERSION = 1
MANIFEST_NAME = "minimax_h3_local_manifest.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def register_to_config(init: Callable[..., None]) -> Callable[..., None]:
    """Record constructor arguments in ``self.config``.

    This is the local equivalent of the serialization-oriented decorator used
    by the official implementation.  Constructor names and defaults therefore
    remain byte-for-byte compatible with the published ``config.json`` files.
    """

    signature = inspect.signature(init)

    @functools.wraps(init)
    def wrapped(self, *args, **kwargs) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = {
            name: _jsonable(value)
            for name, value in bound.arguments.items()
            if name != "self"
        }
        init(self, *args, **kwargs)
        self.config = ConfigDict(values)

    return wrapped


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
            "MiniMax-H3 components are loaded from local directories only; "
            f"no directory exists at {directory}"
        )
    return directory.resolve()


def get_parameter_dtype(module: torch.nn.Module) -> torch.dtype:
    """Return the first parameter/buffer dtype, or float32 for empty modules."""

    for parameter in module.parameters():
        return parameter.dtype
    for buffer in module.buffers():
        return buffer.dtype
    return torch.float32


def randn_tensor(
    shape: tuple[int, ...] | list[int] | torch.Size,
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
) -> torch.Tensor:
    """Create seeded noise with Diffusers-compatible generator routing.

    A CPU generator cannot be passed directly to ``torch.randn(...,
    device="cuda")``.  The official MiniMax-H3 path instead samples on CPU and
    transfers the result to the requested accelerator.  A generator list
    supplies one independently seeded row per batch item; a one-item list is
    intentionally treated like one generator for the complete tensor.
    """

    shape = tuple(int(value) for value in shape)
    target_device = torch.device(device or "cpu")
    random_device: torch.device | str = target_device
    layout = layout or torch.strided

    if isinstance(generator, list):
        if not generator:
            raise ValueError("generator lists must not be empty")
        generator_device_type = generator[0].device.type
        if any(item.device.type != generator_device_type for item in generator):
            raise ValueError("all generators in a list must use the same device type")
    elif generator is not None:
        generator_device_type = generator.device.type
    else:
        generator_device_type = None

    if generator_device_type != target_device.type:
        if generator_device_type == "cpu":
            random_device = torch.device("cpu")
        elif generator_device_type == "cuda":
            raise ValueError(
                f"Cannot generate a {target_device} tensor from a generator "
                "of type cuda"
            )

    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        if not shape:
            raise ValueError("generator lists require a batch dimension")
        batch_size = shape[0]
        if len(generator) != batch_size:
            raise ValueError(
                "A generator list must contain one item per batch row, or one "
                "item for the complete tensor"
            )
        row_shape = (1, *shape[1:])
        rows = [
            torch.randn(
                row_shape,
                generator=item,
                device=random_device,
                dtype=dtype,
                layout=layout,
            )
            for item in generator
        ]
        return torch.cat(rows, dim=0).to(target_device)

    return torch.randn(
        shape,
        generator=generator,
        device=random_device,
        dtype=dtype,
        layout=layout,
    ).to(target_device)


class DiagonalGaussianDistribution:
    """Diagonal Gaussian posterior parameterized by mean and log variance."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        return self.mean + self.std * randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other: DiagonalGaussianDistribution | None = None) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros(
                self.mean.shape[0], device=self.mean.device, dtype=self.mean.dtype
            )
        dimensions = tuple(range(1, self.mean.ndim))
        if other is None:
            return 0.5 * torch.sum(
                self.mean.square() + self.var - 1.0 - self.logvar,
                dim=dimensions,
            )
        return 0.5 * torch.sum(
            (self.mean - other.mean).square() / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=dimensions,
        )

    def nll(
        self, sample: torch.Tensor, dims: tuple[int, ...] | None = None
    ) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros(
                self.mean.shape[0], device=self.mean.device, dtype=self.mean.dtype
            )
        if dims is None:
            dims = tuple(range(1, self.mean.ndim))
        log_two_pi = math.log(2.0 * math.pi)
        return 0.5 * torch.sum(
            log_two_pi + self.logvar + (sample - self.mean).square() / self.var,
            dim=dims,
        )


def _parse_size(value: int | str | None) -> int:
    if value is None:
        return 5_000_000_000
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("max_shard_size must be positive")
        return value
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)\s*", value.lower())
    if match is None:
        raise ValueError(f"Invalid max_shard_size: {value!r}")
    number = float(match.group(1))
    suffix = match.group(2)
    powers = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    result = int(number * powers[suffix])
    if result <= 0:
        raise ValueError("max_shard_size must be positive")
    return result


def _storage_identity(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _tensor_schema(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        for key, tensor in sorted(state_dict.items())
    }


def _checkpoint_files(
    directory: Path,
    *,
    allowed_names: set[str] | None = None,
) -> tuple[Path | None, tuple[Path, ...]]:
    indices = (
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "diffusion_pytorch_model.bin.index.json",
    )
    for name in indices:
        if allowed_names is not None and name not in allowed_names:
            continue
        index_path = directory / name
        if not index_path.is_file():
            continue
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise TypeError(f"Invalid or empty shard index: {index_path}")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in weight_map.items()
        ):
            raise TypeError(f"Invalid weight map entries in {index_path}")
        shard_names = tuple(sorted(set(weight_map.values())))
        if allowed_names is not None:
            undeclared = sorted(set(shard_names) - allowed_names)
            if undeclared:
                raise RuntimeError(
                    f"Checkpoint index {index_path} references files outside "
                    "the local manifest inventory: " + ", ".join(undeclared)
                )
        paths = tuple(directory / name for name in shard_names)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Checkpoint index references missing shards: " + ", ".join(missing)
            )
        return index_path, paths

    weights = (
        "diffusion_pytorch_model.safetensors",
        "model.safetensors",
        "pytorch_model.safetensors",
        "pytorch_model.bin",
        "diffusion_pytorch_model.bin",
    )
    for name in weights:
        if allowed_names is not None and name not in allowed_names:
            continue
        path = directory / name
        if path.is_file():
            return None, (path,)
    raise FileNotFoundError(
        f"No local model weights found in {directory}; expected {weights}"
    )


def _load_tensor_file(
    path: Path, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                f"Reading {path.name} requires the safetensors package"
            ) from exc
        return dict(load_file(str(path), device=str(device)))
    try:
        value = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - PyTorch 2.0 compatibility
        value = torch.load(path, map_location=device)
    if isinstance(value, Mapping) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, Mapping):
        raise TypeError(f"Checkpoint {path} does not contain a state dictionary")
    return {
        str(key): tensor for key, tensor in value.items() if torch.is_tensor(tensor)
    }


def _verify_manifest(
    directory: Path, expected_class: str | None = None
) -> dict[str, Any]:
    path = directory / MANIFEST_NAME
    if not path.is_file():
        return {}
    manifest = read_json(path)
    if (
        manifest.get("format") != LOCAL_FORMAT
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError(f"Unsupported local MiniMax-H3 manifest in {path}")
    if expected_class is not None and manifest.get("class_name") != expected_class:
        raise ValueError(
            f"Manifest class is {manifest.get('class_name')!r}, expected {expected_class!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise TypeError(f"Manifest {path} has no file inventory")
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise TypeError(f"Invalid file inventory in {path}")
        file_path = directory / item["name"]
        if not file_path.is_file():
            raise FileNotFoundError(f"Manifest-declared file is missing: {file_path}")
        if sha256_file(file_path) != item.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {file_path}")
    return manifest


def _resolve_dtype(value: torch.dtype | str | None) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    aliases = {
        "float": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
        "half": torch.float16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    key = value.removeprefix("torch.").lower()
    if key not in aliases:
        raise ValueError(f"Unsupported dtype: {value!r}")
    return aliases[key]


def _strip_prefix(key: str, component_name: str | None) -> str:
    prefixes = ["module."]
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


def _split_state_dict(
    state_dict: Mapping[str, torch.Tensor], max_shard_size: int
) -> tuple[list[dict[str, torch.Tensor]], dict[str, str]]:
    aliases: dict[str, str] = {}
    owners: dict[tuple[Any, ...], str] = {}
    unique: list[tuple[str, torch.Tensor]] = []
    for key in sorted(state_dict):
        tensor = state_dict[key].detach()
        identity = _storage_identity(tensor)
        owner = owners.get(identity)
        if owner is not None:
            aliases[key] = owner
            continue
        owners[identity] = key
        unique.append((key, tensor))

    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_size = 0
    for key, tensor in unique:
        size = tensor.numel() * tensor.element_size()
        if current and current_size + size > max_shard_size:
            shards.append(current)
            current = {}
            current_size = 0
        current[key] = tensor
        current_size += size
    if current or not shards:
        shards.append(current)
    return shards, aliases


def _save_sharded_state_dict(
    directory: Path,
    state_dict: Mapping[str, torch.Tensor],
    *,
    weights_name: str,
    safe_serialization: bool,
    max_shard_size: int | str | None,
) -> tuple[tuple[Path, ...], dict[str, str]]:
    limit = _parse_size(max_shard_size)
    shards, aliases = _split_state_dict(state_dict, limit)
    if safe_serialization:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise RuntimeError(
                "safe_serialization=True requires the safetensors package"
            ) from exc
        suffix = ".safetensors"
        if not weights_name.endswith(suffix):
            weights_name = f"{Path(weights_name).stem}{suffix}"
        save = lambda payload, path: save_file(payload, str(path))
    else:
        suffix = ".bin"
        weights_name = f"{Path(weights_name).stem}{suffix}"
        save = lambda payload, path: torch.save(payload, path)

    def save_one(shard: Mapping[str, torch.Tensor], path: Path) -> None:
        # Materialize at most one shard on CPU.  A full H3 transformer is ~66 GB
        # in bfloat16, so cloning the complete state dict before sharding would
        # make an otherwise valid save path unusable.
        payload = {
            key: tensor.detach().cpu().contiguous() for key, tensor in shard.items()
        }
        save(payload, path)

    if len(shards) == 1:
        path = directory / weights_name
        save_one(shards[0], path)
        return (path,), aliases

    stem = weights_name[: -len(suffix)]
    shard_paths: list[Path] = []
    weight_map: dict[str, str] = {}
    count = len(shards)
    for index, shard in enumerate(shards, start=1):
        name = f"{stem}-{index:05d}-of-{count:05d}{suffix}"
        path = directory / name
        save_one(shard, path)
        shard_paths.append(path)
        weight_map.update({key: name for key in shard})
    index_path = directory / f"{weights_name}.index.json"
    write_json(
        index_path,
        {
            "metadata": {
                "total_size": sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in state_dict.values()
                )
            },
            "weight_map": weight_map,
        },
    )
    return (index_path, *shard_paths), aliases


_STANDARD_WEIGHT_FILE = re.compile(
    r"^(?:diffusion_pytorch_model|model|pytorch_model)"
    r"(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)(?:\.index\.json)?$"
)


def _remove_stale_checkpoint_files(directory: Path, *, keep_names: set[str]) -> None:
    """Remove only recognized checkpoint files superseded by the latest save."""

    for path in directory.iterdir():
        if (
            path.is_file()
            and path.name not in keep_names
            and _STANDARD_WEIGHT_FILE.fullmatch(path.name)
        ):
            path.unlink()


class LocalMiniMaxH3ModelMixin:
    """Strict local config/checkpoint contract shared by H3 ``nn.Module``s."""

    config_name = "config.json"
    weights_name = "diffusion_pytorch_model.safetensors"
    component_name: str | None = None
    _keep_in_fp32_modules: tuple[str, ...] | list[str] = ()

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return get_parameter_dtype(self)

    @classmethod
    def _convert_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        return dict(config)

    @classmethod
    def _convert_checkpoint_tensor(
        cls, key: str, tensor: torch.Tensor, config: Mapping[str, Any]
    ) -> Mapping[str, torch.Tensor]:
        del config
        return {key: tensor}

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | ConfigDict, **kwargs):
        values = cls._convert_config(clean_config(config))
        values.update(kwargs)
        return cls(**values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        subfolder: str | None = None,
        *,
        torch_dtype: torch.dtype | str | None = None,
        dtype: torch.dtype | str | None = None,
        strict: bool = True,
        allow_partial_load: bool = False,
        low_cpu_mem_usage: bool = True,
        device: torch.device | str = "cpu",
        **config_overrides,
    ):
        directory = resolve_pretrained_directory(
            pretrained_model_name_or_path, subfolder
        )
        manifest = _verify_manifest(directory, cls.__name__)
        raw_config = load_config(directory, config_name=cls.config_name)

        unsupported_loading_keys = {
            "cache_dir",
            "device_map",
            "force_download",
            "local_files_only",
            "revision",
            "token",
            "use_auth_token",
            "use_safetensors",
            "variant",
        }
        unsupported = sorted(set(config_overrides) & unsupported_loading_keys)
        if unsupported:
            raise TypeError(
                f"{cls.__name__}.from_pretrained only loads local component "
                "directories and does not support these loading options: "
                + ", ".join(unsupported)
            )
        config = cls._convert_config(clean_config(raw_config))
        config.update(config_overrides)
        requested_dtype = _resolve_dtype(dtype if dtype is not None else torch_dtype)

        if low_cpu_mem_usage:
            if allow_partial_load:
                raise ValueError(
                    "low_cpu_mem_usage cannot be combined with partial loading; "
                    "uninitialized meta parameters would be unsafe"
                )
            with torch.device("meta"):
                model = cls(**config)
        else:
            model = cls(**config)
            if requested_dtype is not None:
                model.to(dtype=requested_dtype)
                for name, module in model.named_modules():
                    if any(pattern in name for pattern in cls._keep_in_fp32_modules):
                        module.to(dtype=torch.float32)

        expected = model.state_dict()
        expected_schema = _tensor_schema(expected)
        manifest_schema = manifest.get("state_dict") if manifest else None
        if manifest_schema is not None:
            if not isinstance(manifest_schema, Mapping):
                raise TypeError("Local artifact state_dict schema must be a mapping")
            if set(manifest_schema) != set(expected_schema):
                raise RuntimeError(
                    f"Local {cls.__name__} artifact keys do not match its config"
                )
            wrong_shapes = [
                key
                for key, expected_item in expected_schema.items()
                if manifest_schema[key].get("shape") != expected_item["shape"]
            ]
            if wrong_shapes:
                raise RuntimeError(
                    f"Local {cls.__name__} artifact shapes do not match its config: "
                    + ", ".join(wrong_shapes[:8])
                )

        manifest_names = (
            {str(item["name"]) for item in manifest["files"]} if manifest else None
        )
        index_path, shard_paths = _checkpoint_files(
            directory, allowed_names=manifest_names
        )
        index_map: Mapping[str, str] | None = None
        if index_path is not None:
            index_map = read_json(index_path)["weight_map"]

        matched: set[str] = set()
        unexpected: set[str] = set()
        mismatched: dict[str, dict[str, Any]] = {}
        duplicate: set[str] = set()
        source_keys_seen: set[str] = set()
        load_device = torch.device(device)
        checkpoint_device = load_device if low_cpu_mem_usage else torch.device("cpu")

        for shard_path in shard_paths:
            source_state = _load_tensor_file(shard_path, device=checkpoint_device)
            if index_map is not None:
                declared = {
                    key
                    for key, filename in index_map.items()
                    if filename == shard_path.name
                }
                actual = set(source_state)
                if declared != actual:
                    missing_from_shard = sorted(declared - actual)
                    undeclared = sorted(actual - declared)
                    raise RuntimeError(
                        f"Shard index/content mismatch for {shard_path}: "
                        f"{len(missing_from_shard)} missing and {len(undeclared)} undeclared keys"
                    )

            load_part: dict[str, torch.Tensor] = {}
            for source_key, source_tensor in source_state.items():
                if source_key in source_keys_seen:
                    raise RuntimeError(f"Duplicate checkpoint key: {source_key}")
                source_keys_seen.add(source_key)
                normalized = _strip_prefix(source_key, cls.component_name)
                converted = cls._convert_checkpoint_tensor(
                    normalized, source_tensor, config
                )
                for target_key, tensor in converted.items():
                    if target_key in matched or target_key in load_part:
                        duplicate.add(target_key)
                        continue
                    target = expected.get(target_key)
                    if target is None:
                        unexpected.add(target_key)
                        continue
                    if tuple(tensor.shape) != tuple(target.shape):
                        mismatched[target_key] = {
                            "expected": list(target.shape),
                            "found": list(tensor.shape),
                        }
                        continue
                    if manifest_schema is not None:
                        declared_dtype = manifest_schema[target_key].get("dtype")
                        actual_dtype = str(tensor.dtype).removeprefix("torch.")
                        if declared_dtype != actual_dtype:
                            raise RuntimeError(
                                f"Manifest dtype mismatch for {target_key}: "
                                f"declared {declared_dtype}, found {actual_dtype}"
                            )
                    if requested_dtype is not None and tensor.is_floating_point():
                        keep_fp32 = any(
                            pattern in target_key
                            for pattern in cls._keep_in_fp32_modules
                        )
                        target_dtype = torch.float32 if keep_fp32 else requested_dtype
                        tensor = tensor.to(dtype=target_dtype)
                    load_part[target_key] = tensor
                    matched.add(target_key)
            if load_part:
                model.load_state_dict(
                    load_part,
                    strict=False,
                    assign=low_cpu_mem_usage,
                )
            del source_state, load_part

        aliases = manifest.get("aliases", {}) if manifest else {}
        for alias, canonical in aliases.items():
            if alias in expected and canonical in matched:
                matched.add(alias)

        missing = set(expected) - matched
        parameter_total = sum(tensor.numel() for tensor in expected.values())
        parameter_matched = sum(expected[key].numel() for key in matched)
        report = {
            "source": str(directory),
            "local_manifest": bool(manifest),
            "matched_keys": sorted(matched),
            "missing_keys": sorted(missing),
            "unexpected_keys": sorted(unexpected),
            "duplicate_keys": sorted(duplicate),
            "mismatched_shapes": mismatched,
            "parameter_coverage": parameter_matched / max(parameter_total, 1),
        }
        model._load_report = report

        has_errors = bool(missing or unexpected or duplicate or mismatched)
        if strict and has_errors:
            raise RuntimeError(
                f"Strict {cls.__name__} checkpoint load failed: "
                f"{len(missing)} missing, {len(unexpected)} unexpected, "
                f"{len(duplicate)} duplicate, {len(mismatched)} shape mismatches"
            )
        if not strict and missing and not allow_partial_load:
            coverage = report["parameter_coverage"]
            if coverage < 0.5:
                raise RuntimeError(
                    f"Only {coverage:.1%} of {cls.__name__} parameters are covered; "
                    "pass allow_partial_load=True to intentionally keep random parameters"
                )

        if low_cpu_mem_usage:
            remaining_meta = [
                name
                for name, tensor in (*model.named_parameters(), *model.named_buffers())
                if tensor.is_meta
            ]
            materialize = getattr(model, "_materialize_nonpersistent_buffers", None)
            if remaining_meta and callable(materialize):
                materialize(load_device)
                remaining_meta = [
                    name
                    for name, tensor in (
                        *model.named_parameters(),
                        *model.named_buffers(),
                    )
                    if tensor.is_meta
                ]
            if remaining_meta:
                raise RuntimeError(
                    "Low-memory load left meta tensors unmaterialized: "
                    + ", ".join(remaining_meta[:8])
                )
        else:
            model.to(device=load_device)
        return model

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        safe_serialization: bool = True,
        max_shard_size: int | str | None = "5GB",
    ) -> str:
        directory = Path(save_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if not hasattr(self, "config"):
            raise AttributeError(
                f"{type(self).__name__} has no serializable `config` attribute"
            )
        config = (
            self.config.to_dict()
            if hasattr(self.config, "to_dict")
            else dict(self.config)
        )
        config["_class_name"] = type(self).__name__
        config_path = directory / self.config_name
        write_json(config_path, config)
        weight_files, aliases = _save_sharded_state_dict(
            directory,
            self.state_dict(),
            weights_name=self.weights_name,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )
        _remove_stale_checkpoint_files(
            directory, keep_names={path.name for path in weight_files}
        )
        files = (config_path, *weight_files)
        manifest = {
            "format": LOCAL_FORMAT,
            "format_version": FORMAT_VERSION,
            "class_name": type(self).__name__,
            "component_name": self.component_name,
            "config_sha256": sha256_json(config),
            "state_dict": _tensor_schema(self.state_dict()),
            "aliases": aliases,
            "files": [
                {"name": path.name, "sha256": sha256_file(path)} for path in files
            ],
        }
        manifest_path = directory / MANIFEST_NAME
        write_json(manifest_path, manifest)
        return str(manifest_path)

    def load_checkpoint(
        self,
        checkpoint_directory: str | Path,
        *,
        strict: bool = True,
        allow_partial_load: bool = False,
    ) -> dict[str, Any]:
        loaded = type(self).from_pretrained(
            checkpoint_directory,
            strict=strict,
            allow_partial_load=allow_partial_load,
            low_cpu_mem_usage=False,
            dtype=self.dtype,
        )
        loaded_state = loaded.state_dict()
        matched_state = {
            key: loaded_state[key]
            for key in loaded._load_report["matched_keys"]
            if key in loaded_state
        }
        self.load_state_dict(matched_state, strict=strict)
        self._load_report = loaded._load_report
        return self._load_report

    def _gradient_checkpointing_func(self, function, *args):
        from torch.utils.checkpoint import checkpoint

        options = dict(getattr(self, "_gradient_checkpointing_kwargs", {}))
        options.setdefault("use_reentrant", False)
        return checkpoint(function, *args, **options)

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        options = dict(gradient_checkpointing_kwargs or {})
        options.update(kwargs)
        self._gradient_checkpointing_kwargs = options
        self.gradient_checkpointing = True
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
            if hasattr(module, "gradient_checkpointing_kwargs"):
                module.gradient_checkpointing_kwargs = dict(options)

    def gradient_checkpointing_disable(self) -> None:
        self._gradient_checkpointing_kwargs = {}
        self.gradient_checkpointing = False
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
            if hasattr(module, "gradient_checkpointing_kwargs"):
                module.gradient_checkpointing_kwargs = {}

    enable_gradient_checkpointing = gradient_checkpointing_enable
    disable_gradient_checkpointing = gradient_checkpointing_disable


__all__ = [
    "FORMAT_VERSION",
    "LOCAL_FORMAT",
    "MANIFEST_NAME",
    "ConfigDict",
    "DiagonalGaussianDistribution",
    "LocalMiniMaxH3ModelMixin",
    "get_parameter_dtype",
    "randn_tensor",
    "read_json",
    "register_to_config",
    "resolve_pretrained_directory",
    "sha256_file",
    "write_json",
]
