"""Regression gates for LTX model-component ownership and dependency injection."""

from __future__ import annotations

import ast
from pathlib import Path

from hftrainer.models.ltx_video.component_loader import LTXComponentStore


ROOT = Path(__file__).resolve().parents[2]


class _TrackingRegistry:
    def __init__(self):
        self.clear_calls = 0

    def clear(self):
        self.clear_calls += 1


def _calls(tree: ast.AST, names: set[str]) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
    ]


def _has_keyword(call: ast.Call, name: str) -> bool:
    return any(keyword.arg == name for keyword in call.keywords)


def test_component_store_separates_inference_and_training_cache_policy():
    store = LTXComponentStore()

    assert store.inference_registry is not store.training_registry
    assert store.inference_registry.cache_models is True
    assert store.inference_registry.cache_weights is False
    assert store.training_registry.cache_models is False
    assert store.training_registry.cache_weights is False


def test_component_store_clears_both_owned_registries():
    inference = _TrackingRegistry()
    training = _TrackingRegistry()
    store = LTXComponentStore(
        inference_registry=inference,
        training_registry=training,
    )

    store.clear()

    assert inference.clear_calls == 1
    assert training.clear_calls == 1


def test_every_component_loader_injects_its_registry_into_the_builder():
    path = ROOT / 'hftrainer' / 'models' / 'ltx_video' / 'component_loader.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    loader_names = {
        'load_transformer',
        'load_video_vae_encoder',
        'load_video_vae_decoder',
        'load_audio_vae_encoder',
        'load_audio_vae_decoder',
        'load_vocoder',
        'load_text_encoder',
        'load_embeddings_processor',
    }
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in loader_names
    }

    assert set(functions) == loader_names
    for name, function in functions.items():
        builder_calls = _calls(function, {'SingleGPUModelBuilder'})
        assert len(builder_calls) == 1, name
        assert _has_keyword(builder_calls[0], 'registry'), name


def test_native_training_and_validation_forward_the_owned_registry():
    trainer_path = (
        ROOT / 'hftrainer' / 'trainers' / 'ltx_video' / 'native' / 'trainer.py'
    )
    trainer_tree = ast.parse(
        trainer_path.read_text(encoding='utf-8'), filename=str(trainer_path)
    )
    trainer_calls = _calls(
        trainer_tree,
        {'ValidationRunner', 'load_transformer', 'load_embeddings_processor'},
    )
    assert {call.func.id for call in trainer_calls} == {
        'ValidationRunner',
        'load_transformer',
        'load_embeddings_processor',
    }
    for call in trainer_calls:
        keyword = (
            'component_registry'
            if call.func.id == 'ValidationRunner'
            else 'registry'
        )
        assert _has_keyword(call, keyword), call.func.id

    validation_path = trainer_path.with_name('validation_runner.py')
    validation_tree = ast.parse(
        validation_path.read_text(encoding='utf-8'), filename=str(validation_path)
    )
    loader_calls = _calls(
        validation_tree,
        {
            'load_text_encoder',
            'load_embeddings_processor',
            'load_video_vae_encoder',
            'load_audio_vae_encoder',
            'load_video_vae_decoder',
            'load_audio_vae_decoder',
            'load_vocoder',
        },
    )
    assert len(loader_calls) == 7
    assert all(_has_keyword(call, 'registry') for call in loader_calls)
