import json

import pytest
import torch
from safetensors.torch import save_file

from hftrainer.datasets.synchronized_audio_video import (
    MiniMaxH3CachedFeatureDataset,
)


def _write_cache(root, name: str, *, tags=(1, 1, 1)):
    path = root / f"{name}.safetensors"
    save_file(
        {
            "video_latents": torch.randn(2, 3, 4, 4),
            "audio_latents": torch.randn(2, 3, 5),
            "prompt_embeds": torch.randn(len(tags), 7),
            "text_token_tags": torch.tensor(tags, dtype=torch.long),
        },
        str(path),
    )
    return path


def test_cached_feature_dataset_loads_and_collates(tmp_path):
    _write_cache(tmp_path, "one")
    _write_cache(tmp_path, "two")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"feature_file": f"{name}.safetensors"})
            for name in ("one", "two")
        ),
        encoding="utf-8",
    )
    dataset = MiniMaxH3CachedFeatureDataset(
        str(manifest),
        expected_video_channels=2,
        expected_audio_channels=3,
        expected_prompt_dim=7,
    )
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["video_latents"].shape == (2, 2, 3, 4, 4)
    assert batch["audio_latents"].shape == (2, 2, 3, 5)
    assert batch["prompt_embeds"].shape == (2, 3, 7)
    assert batch["text_token_tags"].tolist() == [1, 1, 1]


def test_cached_feature_dataset_rejects_mixed_packed_layouts(tmp_path):
    _write_cache(tmp_path, "one", tags=(1, 1, 1))
    _write_cache(tmp_path, "two", tags=(1, 0, 1))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"feature_file": f"{name}.safetensors"})
            for name in ("one", "two")
        ),
        encoding="utf-8",
    )
    dataset = MiniMaxH3CachedFeatureDataset(
        str(manifest),
        expected_video_channels=2,
        expected_audio_channels=3,
        expected_prompt_dim=7,
    )
    with pytest.raises(ValueError, match="prompt row geometry differs"):
        dataset.collate_fn([dataset[0], dataset[1]])
