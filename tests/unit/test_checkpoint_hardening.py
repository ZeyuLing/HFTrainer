"""Regression tests for local checkpoint compatibility boundaries."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from hftrainer.models.sd15.checkpoint import load_compatible_state
from hftrainer.models.sd15.network import CLIPTextModel
from hftrainer.models.vit import LocalViTForImageClassification, ViTConfig


def _tiny_vit_config(*, num_labels: int = 3) -> ViTConfig:
    return ViTConfig(
        image_size=16,
        patch_size=4,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=num_labels,
    )


def _write_vit_checkpoint(directory, state: dict[str, torch.Tensor]) -> None:
    directory.mkdir()
    (directory / 'config.json').write_text(
        json.dumps(_tiny_vit_config().to_dict()), encoding='utf-8'
    )
    torch.save(state, directory / 'pytorch_model.bin')


def _tiny_clip_text_model() -> CLIPTextModel:
    return CLIPTextModel(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        bos_token_id=62,
        eos_token_id=63,
    )


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def test_sd15_non_strict_load_rejects_one_tensor_checkpoint():
    module = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8),
    )
    state = module.state_dict()
    one_tensor_checkpoint = {'0.bias': state['0.bias'].clone()}

    with pytest.raises(RuntimeError, match='materially incomplete'):
        load_compatible_state(module, one_tensor_checkpoint, strict=False)


def test_sd15_non_strict_load_preserves_small_compatible_variants():
    module = torch.nn.Sequential(*(torch.nn.Linear(8, 8) for _ in range(12)))
    state = dict(module.state_dict())
    state['11.bias'] = torch.zeros(9)

    report = load_compatible_state(module, state, strict=False)

    assert report['tensor_coverage'] > 0.90
    assert report['parameter_coverage'] > 0.90
    assert report['mismatched_shapes'] == {
        '11.bias': {'checkpoint': (9,), 'model': (8,)}
    }


def test_sd15_local_artifact_rejects_missing_bias_even_if_non_strict_requested(
    tmp_path,
):
    model = _tiny_clip_text_model()
    model.save_pretrained(tmp_path, safe_serialization=False)
    weight_path = tmp_path / 'pytorch_model.bin'
    state = torch.load(weight_path, map_location='cpu', weights_only=True)
    state.pop('text_model.final_layer_norm.bias')
    torch.save(state, weight_path)
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['sha256'] = _file_sha256(weight_path)
    manifest['tensor_count'] = len(state)
    manifest['parameter_count'] = sum(value.numel() for value in state.values())
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(RuntimeError, match='not an exact match'):
        CLIPTextModel.from_pretrained(tmp_path, strict=False)


def test_sd15_local_artifact_rejects_tampered_weight_file(tmp_path):
    model = _tiny_clip_text_model()
    model.save_pretrained(tmp_path, safe_serialization=False)
    with (tmp_path / 'pytorch_model.bin').open('ab') as handle:
        handle.write(b'tampered')

    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        CLIPTextModel.from_pretrained(tmp_path)


def test_sd15_local_artifact_validates_manifest_tensor_count(tmp_path):
    model = _tiny_clip_text_model()
    model.save_pretrained(tmp_path, safe_serialization=False)
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['tensor_count'] += 1
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(RuntimeError, match='tensor count mismatch'):
        CLIPTextModel.from_pretrained(tmp_path)


def test_sd15_upstream_compatibility_requires_explicit_non_strict(tmp_path):
    model = _tiny_clip_text_model()
    model.save_pretrained(tmp_path, safe_serialization=False)
    (tmp_path / 'manifest.json').unlink()
    weight_path = tmp_path / 'pytorch_model.bin'
    state = torch.load(weight_path, map_location='cpu', weights_only=True)
    state['text_model.final_layer_norm.bias'] = torch.zeros(33)
    torch.save(state, weight_path)

    with pytest.raises(RuntimeError, match='not an exact match'):
        CLIPTextModel.from_pretrained(tmp_path)

    restored = CLIPTextModel.from_pretrained(tmp_path, strict=False)
    assert restored._checkpoint_load_report['local_artifact'] is False
    assert restored._checkpoint_load_report['tensor_coverage'] > 0.90


def test_vit_ignore_mismatched_sizes_allows_only_classifier_replacement(tmp_path):
    source = LocalViTForImageClassification(_tiny_vit_config(num_labels=3))
    source.save_pretrained(str(tmp_path), safe_serialization=False)

    _, info = LocalViTForImageClassification.from_pretrained(
        str(tmp_path),
        num_labels=5,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
    )

    assert {item[0] for item in info['mismatched_keys']} == {
        'classifier.weight',
        'classifier.bias',
    }
    assert set(info['missing_keys']) == {'classifier.weight', 'classifier.bias'}
    assert info['unexpected_keys'] == []


@pytest.mark.parametrize('corruption', ['missing', 'unexpected', 'mismatched'])
def test_vit_ignore_mismatched_sizes_rejects_backbone_corruption(
    tmp_path,
    corruption,
):
    model = LocalViTForImageClassification(_tiny_vit_config())
    state = dict(model.state_dict())
    backbone_key = 'vit.embeddings.patch_embeddings.projection.weight'
    if corruption == 'missing':
        state.pop(backbone_key)
    elif corruption == 'unexpected':
        state['vit.unexpected.weight'] = torch.zeros(1)
    else:
        state[backbone_key] = state[backbone_key][:1]
    checkpoint = tmp_path / corruption
    _write_vit_checkpoint(checkpoint, state)

    with pytest.raises(RuntimeError, match='classifier|non-classifier'):
        LocalViTForImageClassification.from_pretrained(
            str(checkpoint), ignore_mismatched_sizes=True
        )
