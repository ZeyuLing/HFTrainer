"""Small configuration primitives shared by the local SD1.5 implementation."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ConfigDict(dict):
    """A JSON-serialisable mapping with attribute access.

    Model code historically accesses configuration through both
    ``config['field']`` and ``config.field``.  Keeping that convenience local
    avoids pulling in a second configuration/model framework.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self))


def load_config(path: str | Path, subfolder: str | None = None) -> dict[str, Any]:
    root = Path(path)
    if subfolder:
        root = root / subfolder
    config_path = root / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing model config: {config_path}')
    with config_path.open('r', encoding='utf-8') as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f'Expected a JSON object in {config_path}.')
    return dict(value)


def clean_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Drop metadata fields that are not constructor arguments."""

    return {
        key: value
        for key, value in config.items()
        if not key.startswith('_')
        and key not in {'architectures', 'torch_dtype', 'transformers_version'}
    }
