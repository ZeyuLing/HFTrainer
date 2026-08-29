"""Local checkpoint I/O for Stable Diffusion components.

The loader intentionally accepts only local directories.  Downloading and
cache management belong to data preparation; model execution remains fully
self-contained and deterministic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .network.configuration import ConfigDict, clean_config, load_config

WEIGHT_CANDIDATES = (
    'model.safetensors',
    'diffusion_pytorch_model.safetensors',
    'pytorch_model.bin',
    'diffusion_pytorch_model.bin',
)

# Non-strict loading exists for small, intentional architecture differences
# such as a replaced input/output projection.  It must not turn a mostly
# unrelated or truncated checkpoint into a silently random-initialized model.
_MIN_NON_STRICT_TENSOR_COVERAGE = 0.90
_MIN_NON_STRICT_PARAMETER_COVERAGE = 0.90
_LOCAL_COMPONENT_FORMAT = 'hftrainer-local-component'
_LOCAL_COMPONENT_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tensor_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file

        return dict(load_file(str(path), device='cpu'))
    try:
        value = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        value = torch.load(path, map_location='cpu')
    if isinstance(value, Mapping) and 'state_dict' in value:
        value = value['state_dict']
    if not isinstance(value, Mapping):
        raise TypeError(f'Checkpoint {path} does not contain a state dictionary.')
    return dict(value)


def _resolve_weight_files(root: Path) -> list[Path]:
    for name in WEIGHT_CANDIDATES:
        candidate = root / name
        if candidate.is_file():
            return [candidate]
        index_path = root / f'{name}.index.json'
        if index_path.is_file():
            with index_path.open('r', encoding='utf-8') as handle:
                index = json.load(handle)
            shards = sorted(set(index.get('weight_map', {}).values()))
            files = [root / shard for shard in shards]
            missing = [str(path) for path in files if not path.is_file()]
            if missing:
                raise FileNotFoundError(f'Missing checkpoint shards: {missing}')
            return files
    raise FileNotFoundError(
        f'No supported tensor file under {root}; expected one of {WEIGHT_CANDIDATES}.'
    )


def _read_local_component_manifest(root: Path) -> dict[str, Any] | None:
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open('r', encoding='utf-8') as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Invalid checkpoint manifest: {manifest_path}') from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeError(
            f'Checkpoint manifest must contain an object: {manifest_path}'
        )
    if manifest.get('format') != _LOCAL_COMPONENT_FORMAT:
        return None
    return dict(manifest)


def _validate_local_manifest_files(
    root: Path,
    files: list[Path],
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get('schema_version') != _LOCAL_COMPONENT_SCHEMA_VERSION:
        raise RuntimeError(
            'Unsupported HFTrainer component manifest schema: '
            f"{manifest.get('schema_version')!r}."
        )
    weight_name = manifest.get('weights')
    if (
        not isinstance(weight_name, str)
        or not weight_name
        or Path(weight_name).name != weight_name
    ):
        raise RuntimeError('HFTrainer component manifest has an invalid weights entry.')
    expected_weight = root / weight_name
    if files != [expected_weight]:
        resolved = [path.name for path in files]
        raise RuntimeError(
            'HFTrainer component manifest does not match the resolved weight file: '
            f'expected {weight_name!r}, resolved {resolved!r}.'
        )
    expected_sha256 = manifest.get('sha256')
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError(
            'HFTrainer component manifest has an invalid SHA-256 digest.'
        )
    actual_sha256 = _sha256(expected_weight)
    if actual_sha256 != expected_sha256.lower():
        raise RuntimeError(
            'HFTrainer component weight SHA-256 mismatch: '
            f'expected {expected_sha256}, got {actual_sha256}.'
        )


def _validate_local_manifest_state(
    state: Mapping[str, torch.Tensor],
    manifest: Mapping[str, Any],
) -> None:
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError(
            'HFTrainer component checkpoint contains a non-tensor value.'
        )
    actual_tensor_count = len(state)
    expected_tensor_count = manifest.get('tensor_count')
    if (
        type(expected_tensor_count) is not int
        or expected_tensor_count != actual_tensor_count
    ):
        raise RuntimeError(
            'HFTrainer component tensor count mismatch: '
            f'expected {expected_tensor_count!r}, got {actual_tensor_count}.'
        )
    actual_parameter_count = sum(value.numel() for value in state.values())
    expected_parameter_count = manifest.get('parameter_count')
    if (
        type(expected_parameter_count) is not int
        or expected_parameter_count != actual_parameter_count
    ):
        raise RuntimeError(
            'HFTrainer component parameter count mismatch: '
            f'expected {expected_parameter_count!r}, got {actual_parameter_count}.'
        )


def load_state_dict(root: str | Path) -> tuple[dict[str, torch.Tensor], list[Path]]:
    root = Path(root)
    manifest = _read_local_component_manifest(root)
    files = _resolve_weight_files(root)
    if manifest is not None:
        _validate_local_manifest_files(root, files, manifest)
    state: dict[str, torch.Tensor] = {}
    for path in files:
        shard = _load_tensor_file(path)
        duplicate = state.keys() & shard.keys()
        if duplicate:
            raise ValueError(
                'Duplicate keys across checkpoint shards: '
                f'{sorted(duplicate)[:5]}'
            )
        state.update(shard)
    if manifest is not None:
        _validate_local_manifest_state(state, manifest)
    return state, files


def _strip_common_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ('module.', 'model.')
    result = dict(state)
    for prefix in prefixes:
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix):]: value for key, value in result.items()}
    return result


def load_compatible_state(
    module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    strict: bool = False,
    normalize_keys: bool = True,
) -> dict[str, Any]:
    """Load a compatible checkpoint and return an auditable coverage report.

    ``strict=False`` permits a limited number of missing or shape-mismatched
    tensors for intentional SD-compatible variants.  It still requires at
    least 90% coverage by both tensor count and element count, so truncated or
    materially different checkpoints fail before mutating ``module``.
    """

    state = _strip_common_prefix(state) if normalize_keys else dict(state)
    target = module.state_dict()
    if normalize_keys and state and not (state.keys() & target.keys()):
        prefixed = {f'text_model.{key}': value for key, value in state.items()}
        if prefixed.keys() & target.keys():
            state = prefixed
    compatible = {
        key: value
        for key, value in state.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    mismatched = {
        key: {'checkpoint': tuple(value.shape), 'model': tuple(target[key].shape)}
        for key, value in state.items()
        if key in target and tuple(value.shape) != tuple(target[key].shape)
    }
    missing = sorted(target.keys() - compatible.keys())
    unexpected = sorted(state.keys() - target.keys())
    if strict and (missing or unexpected or mismatched):
        raise RuntimeError(
            'Checkpoint is not an exact match: '
            f'{len(missing)} missing, {len(unexpected)} unexpected, '
            f'{len(mismatched)} shape mismatches.'
        )
    if not compatible:
        raise RuntimeError(
            'Checkpoint contains no tensors compatible with this architecture.'
        )
    target_numel = sum(value.numel() for value in target.values())
    loaded_numel = sum(target[key].numel() for key in compatible)
    tensor_coverage = len(compatible) / max(1, len(target))
    parameter_coverage = loaded_numel / max(1, target_numel)
    if not strict and (
        tensor_coverage < _MIN_NON_STRICT_TENSOR_COVERAGE
        or parameter_coverage < _MIN_NON_STRICT_PARAMETER_COVERAGE
    ):
        raise RuntimeError(
            'Checkpoint is materially incomplete for this architecture: '
            f'{len(compatible)}/{len(target)} tensors '
            f'({tensor_coverage:.1%}) and {loaded_numel}/{target_numel} elements '
            f'({parameter_coverage:.1%}) are compatible; non-strict loading '
            f'requires at least {_MIN_NON_STRICT_TENSOR_COVERAGE:.0%} tensor '
            f'coverage and {_MIN_NON_STRICT_PARAMETER_COVERAGE:.0%} parameter '
            'coverage.'
        )
    module.load_state_dict(compatible, strict=False)
    return {
        'loaded_keys': len(compatible),
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'mismatched_shapes': mismatched,
        'tensor_coverage': tensor_coverage,
        'parameter_coverage': parameter_coverage,
    }


def save_component(
    module: torch.nn.Module,
    save_directory: str | Path,
    *,
    config: Mapping[str, Any],
    safe_serialization: bool = True,
    component_kind: str,
) -> dict[str, Any]:
    root = Path(save_directory)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / 'config.json'
    with config_path.open('w', encoding='utf-8') as handle:
        json.dump(dict(config), handle, indent=2, sort_keys=True)
        handle.write('\n')

    state = module.state_dict()
    if safe_serialization:
        from safetensors.torch import save_file

        weight_path = root / 'model.safetensors'
        values = {
            key: value.detach().contiguous().cpu()
            for key, value in state.items()
        }
        save_file(values, str(weight_path))
    else:
        weight_path = root / 'pytorch_model.bin'
        torch.save(state, weight_path)

    manifest = {
        'schema_version': 1,
        'format': _LOCAL_COMPONENT_FORMAT,
        'component': component_kind,
        'config': config_path.name,
        'weights': weight_path.name,
        'sha256': _sha256(weight_path),
        'tensor_count': len(state),
        'parameter_count': sum(value.numel() for value in state.values()),
    }
    with (root / 'manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return manifest


class LocalComponentMixin:
    """Construction and checkpoint protocol shared by local neural components."""

    config: ConfigDict
    component_kind = 'model'

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | str | Path, **overrides):
        if isinstance(config, (str, Path)):
            config = load_config(config)
        values = clean_config(dict(config))
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        subfolder: str | None = None,
        torch_dtype: torch.dtype | None = None,
        dtype: torch.dtype | None = None,
        strict: bool = True,
        **kwargs,
    ):
        root = Path(pretrained_model_name_or_path)
        if subfolder:
            root = root / subfolder
        if not root.is_dir():
            raise FileNotFoundError(
                f'{cls.__name__} requires a prepared local checkpoint directory: {root}'
            )
        config = clean_config(load_config(root))
        config.update(kwargs.pop('config_overrides', {}) or {})
        # Accept common loader-only options without leaking them into constructors.
        for key in (
            'local_files_only', 'low_cpu_mem_usage', 'use_safetensors',
            'variant', 'revision', 'cache_dir', 'force_download', 'token',
        ):
            kwargs.pop(key, None)
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise TypeError(f'Unsupported local checkpoint options: {unknown}')
        model = cls(**config)
        local_manifest = _read_local_component_manifest(root)
        state, files = load_state_dict(root)
        if local_manifest is not None:
            model._checkpoint_load_report = load_compatible_state(
                model,
                state,
                strict=True,
                normalize_keys=False,
            )
        else:
            model._checkpoint_load_report = load_compatible_state(
                model,
                state,
                strict=strict,
            )
        model._checkpoint_load_report['local_artifact'] = local_manifest is not None
        model._checkpoint_load_report['files'] = [str(path) for path in files]
        target_dtype = dtype or torch_dtype
        if target_dtype is not None:
            model.to(dtype=target_dtype)
        return model

    def save_pretrained(
        self,
        save_directory: str | Path,
        safe_serialization: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        return save_component(
            self,
            save_directory,
            config=self.config.to_dict(),
            safe_serialization=safe_serialization,
            component_kind=self.component_kind,
        )
