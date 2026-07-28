#!/usr/bin/env python3
"""Shared, model-free helpers for ProjFlow HumanML3D evaluation adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTIUS = ROOT.parent / "Motius"
DEFAULT_PROJFLOW_REPO = DEFAULT_MOTIUS / "ref_repo/ProjFlow"
DEFAULT_PROJFLOW_ARTIFACT = (
    DEFAULT_MOTIUS / "outputs/checkpoints/projflow-official"
)
DEFAULT_DATA = ROOT / "data/eval/m2m_v2/eval_hml3d_official_control_4012.json"
DEFAULT_GT_HML263 = ROOT / "ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    raw = load_json(path)
    values = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(values, dict):
        return {str(key): dict(value) for key, value in values.items()}
    if isinstance(values, list):
        return {
            str(value.get("motion_id") or value.get("id") or index): dict(value)
            for index, value in enumerate(values)
        }
    raise TypeError(f"unsupported record container in {path}: {type(values)!r}")


def load_ids(path: Path | None, records: dict[str, dict[str, Any]]) -> list[str]:
    if path is None:
        return list(records)
    if path.suffix.lower() == ".json":
        raw = load_json(path)
        if isinstance(raw, dict):
            raw = raw.get("ids", raw.get("source_ids", raw.get("data_list", raw)))
        if isinstance(raw, dict):
            values = list(raw)
        elif isinstance(raw, list):
            values = raw
        else:
            raise TypeError(f"unsupported id container in {path}: {type(raw)!r}")
        return [str(value) for value in values]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def caption(record: dict[str, Any]) -> str:
    for key in ("caption_en", "caption", "selected_caption", "text"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def load_caption_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    raw = load_json(path)
    values = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    result: dict[str, str] = {}
    if isinstance(values, dict):
        for key, value in values.items():
            result[str(key)] = caption(value) if isinstance(value, dict) else str(value)
    elif isinstance(values, list):
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                continue
            motion_id = str(value.get("motion_id") or value.get("id") or index)
            result[motion_id] = caption(value)
    else:
        raise TypeError(f"unsupported caption container in {path}: {type(values)!r}")
    return result


def frame_indices_from_fractions(
    fractions: Sequence[float], length: int
) -> list[int]:
    if length < 1:
        raise ValueError("length must be positive")
    return sorted(
        {
            int(np.clip(np.round(float(value) * (length - 1)), 0, length - 1))
            for value in fractions
        }
    )


def chunks(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def validate_hml263(motion_id: str, value: np.ndarray, max_length: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)[:max_length]
    if result.ndim != 2 or result.shape[1] != 263:
        raise RuntimeError(f"{motion_id}: invalid HML263 prediction {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError(f"{motion_id}: HML263 prediction contains non-finite values")
    return result


def validate_joints22(motion_id: str, value: np.ndarray, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)[:length]
    if result.shape != (length, 22, 3):
        raise RuntimeError(f"{motion_id}: invalid 22-joint prediction {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError(f"{motion_id}: joint prediction contains non-finite values")
    return result


__all__ = [
    "DEFAULT_DATA",
    "DEFAULT_GT_HML263",
    "DEFAULT_MOTIUS",
    "DEFAULT_PROJFLOW_ARTIFACT",
    "DEFAULT_PROJFLOW_REPO",
    "caption",
    "chunks",
    "frame_indices_from_fractions",
    "load_ids",
    "load_caption_map",
    "load_json",
    "load_records",
    "validate_hml263",
    "validate_joints22",
]
