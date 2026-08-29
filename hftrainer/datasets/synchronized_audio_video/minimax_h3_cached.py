"""On-disk cached-feature dataset for experimental MiniMax-H3 fine-tuning."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from hftrainer.registry import DATASETS

_REQUIRED_TENSORS = (
    "video_latents",
    "audio_latents",
    "prompt_embeds",
    "text_token_tags",
)
_OPTIONAL_TENSORS = ("condition_video_rows", "condition_audio_rows")


def _load_tensor_mapping(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path), device="cpu"))
    if path.suffix in {".pt", ".pth"}:
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - PyTorch 2.0 compatibility
            value = torch.load(path, map_location="cpu")
        if not isinstance(value, Mapping):
            raise TypeError(f"Cached feature file must contain a mapping: {path}")
        return {
            str(name): tensor
            for name, tensor in value.items()
            if torch.is_tensor(tensor)
        }
    raise ValueError(
        f"Unsupported MiniMax-H3 cache suffix {path.suffix!r}; use .safetensors or .pt."
    )


def _plain_sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return tuple(value)


@DATASETS.register_module()
class MiniMaxH3CachedFeatureDataset(Dataset):
    """Load one H3 latent/conditioner bundle per JSONL record.

    Every line in ``manifest`` contains ``feature_file`` (relative to
    ``data_root`` unless absolute), plus optional structural metadata:
    ``keyframe_anchors`` and ``reference_geometries``. Tensor files contain
    the four required tensors documented by :class:`MiniMaxH3Trainer` and may
    additionally contain packed condition rows.

    The dataset intentionally requires every item in one minibatch to share
    exactly the same packed geometry. Variable-size data should be bucketed by
    the sampler or trained with ``batch_size=1``.
    """

    def __init__(
        self,
        manifest: str,
        data_root: str | None = None,
        *,
        expected_video_channels: int = 24,
        expected_audio_channels: int = 32,
        expected_prompt_dim: int = 5120,
        verify_files: bool = True,
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(
                f"MiniMax-H3 cache manifest not found: {self.manifest}"
            )
        self.data_root = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else self.manifest.parent
        )
        self.expected_video_channels = int(expected_video_channels)
        self.expected_audio_channels = int(expected_audio_channels)
        self.expected_prompt_dim = int(expected_prompt_dim)
        if (
            min(
                self.expected_video_channels,
                self.expected_audio_channels,
                self.expected_prompt_dim,
            )
            <= 0
        ):
            raise ValueError("Expected cached feature dimensions must be positive.")

        self.records: list[dict[str, Any]] = []
        with self.manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {self.manifest}:{line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise TypeError(
                        f"Manifest row {line_number} must be a JSON object."
                    )
                relative = record.get("feature_file", record.get("features"))
                if not isinstance(relative, str) or not relative:
                    raise KeyError(
                        f"Manifest row {line_number} needs a non-empty feature_file."
                    )
                path = Path(relative).expanduser()
                if not path.is_absolute():
                    path = self.data_root / path
                path = path.resolve()
                if verify_files and not path.is_file():
                    raise FileNotFoundError(
                        f"Manifest row {line_number} points to missing cache: {path}"
                    )
                normalized = dict(record)
                normalized["feature_file"] = str(path)
                normalized["_line_number"] = line_number
                self.records.append(normalized)
        if not self.records:
            raise ValueError(f"MiniMax-H3 cache manifest is empty: {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def _validate(self, values: dict[str, torch.Tensor], path: Path) -> None:
        missing = [name for name in _REQUIRED_TENSORS if name not in values]
        if missing:
            raise KeyError(f"Cache {path} is missing tensors: {', '.join(missing)}")
        video = values["video_latents"]
        audio = values["audio_latents"]
        prompt = values["prompt_embeds"]
        tags = values["text_token_tags"]
        if video.ndim not in {4, 5}:
            raise ValueError(f"video_latents in {path} must be CTHW or BCTHW.")
        if audio.ndim not in {3, 4}:
            raise ValueError(f"audio_latents in {path} must be 2CL or B2CL.")
        if prompt.ndim not in {2, 3}:
            raise ValueError(f"prompt_embeds in {path} must be ND or BND.")
        if tags.ndim != 1:
            raise ValueError(f"text_token_tags in {path} must be one-dimensional.")
        if video.shape[-4] != self.expected_video_channels:
            raise ValueError(
                f"video_latents in {path} have {video.shape[-4]} channels; "
                f"expected {self.expected_video_channels}."
            )
        if audio.shape[-3:-1] != (2, self.expected_audio_channels):
            raise ValueError(
                f"audio_latents in {path} must have stereo/{self.expected_audio_channels} "
                f"channels, got {tuple(audio.shape)}."
            )
        if prompt.shape[-1] != self.expected_prompt_dim:
            raise ValueError(
                f"prompt_embeds in {path} have width {prompt.shape[-1]}; "
                f"expected {self.expected_prompt_dim}."
            )
        if prompt.shape[-2] != tags.numel():
            raise ValueError(
                f"prompt rows and text_token_tags disagree in {path}: "
                f"{prompt.shape[-2]} vs {tags.numel()}."
            )

    @staticmethod
    def _remove_singleton_batch(
        tensor: torch.Tensor, expected_ndim: int
    ) -> torch.Tensor:
        if tensor.ndim == expected_ndim + 1:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "A cache file may contain at most one sample; DataLoader owns batching."
                )
            return tensor[0]
        return tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = Path(record["feature_file"])
        values = _load_tensor_mapping(path)
        self._validate(values, path)
        item: dict[str, Any] = {
            "video_latents": self._remove_singleton_batch(values["video_latents"], 4),
            "audio_latents": self._remove_singleton_batch(values["audio_latents"], 3),
            "prompt_embeds": self._remove_singleton_batch(values["prompt_embeds"], 2),
            "text_token_tags": values["text_token_tags"].long(),
            "keyframe_anchors": _plain_sequence(
                record.get("keyframe_anchors"), name="keyframe_anchors"
            ),
            "reference_geometries": _plain_sequence(
                record.get("reference_geometries"), name="reference_geometries"
            ),
        }
        for name in _OPTIONAL_TENSORS:
            if name in values:
                item[name] = self._remove_singleton_batch(values[name], 2)
        return item

    @staticmethod
    def collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise ValueError("Cannot collate an empty MiniMax-H3 batch.")
        structural = (
            "text_token_tags",
            "keyframe_anchors",
            "reference_geometries",
        )
        first = items[0]
        for item in items[1:]:
            if not torch.equal(item["text_token_tags"], first["text_token_tags"]):
                raise ValueError(
                    "MiniMax-H3 prompt row geometry differs inside one batch; "
                    "bucket by prompt/reference layout or use batch_size=1."
                )
            for name in structural[1:]:
                if item[name] != first[name]:
                    raise ValueError(
                        f"MiniMax-H3 {name} differs inside one batch; bucket the data."
                    )
        tensor_names = [*_REQUIRED_TENSORS[:-1], *_OPTIONAL_TENSORS]
        result: dict[str, Any] = {
            "text_token_tags": first["text_token_tags"],
            "keyframe_anchors": first["keyframe_anchors"],
            "reference_geometries": first["reference_geometries"],
        }
        for name in tensor_names:
            present = [name in item for item in items]
            if any(present) and not all(present):
                raise ValueError(
                    f"Optional cached tensor {name} is missing from part of a batch."
                )
            if all(present):
                try:
                    result[name] = torch.stack([item[name] for item in items])
                except RuntimeError as exc:
                    raise ValueError(
                        f"Cached tensor {name} has variable shapes; bucket the data."
                    ) from exc
        return result


__all__ = ["MiniMaxH3CachedFeatureDataset"]
