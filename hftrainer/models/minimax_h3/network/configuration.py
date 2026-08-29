"""Small, dependency-free configuration primitives for MiniMax-H3.

The public MiniMax-H3 artifacts use Hugging Face-style JSON configuration
files.  HFTrainer keeps that on-disk contract while deliberately avoiding a
runtime dependency on either ``diffusers`` or ``transformers``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ConfigDict(dict):
    """A JSON-serializable mapping with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self))


def load_config(
    path: str | Path,
    subfolder: str | None = None,
    *,
    config_name: str = "config.json",
) -> dict[str, Any]:
    """Read one local JSON object without resolving remote identifiers."""

    root = Path(path).expanduser()
    if subfolder:
        root = root / subfolder
    config_path = root / config_name
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a JSON object in {config_path}.")
    return dict(value)


def clean_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Drop serialization metadata that is not a model constructor argument."""

    metadata = {
        "architectures",
        "torch_dtype",
        "transformers_version",
    }
    return {
        key: value
        for key, value in config.items()
        if not key.startswith("_") and key not in metadata
    }


__all__ = ["ConfigDict", "clean_config", "load_config"]
