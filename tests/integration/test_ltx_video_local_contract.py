"""Contracts for the repository-owned LTX-Video 2.5 execution stack.

These tests intentionally import only ``hftrainer.*`` model implementations.
Media/runtime libraries may be supplied by an LTX extra, but no separately
installed model framework or LTX source checkout participates in execution.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from mmengine.config import Config

from hftrainer.models.ltx_video.checkpoints import LTX_25_SOURCE_COMMIT


def _dummy_file(root: Path, name: str) -> str:
    path = root / name
    path.touch()
    return str(path)


def _native_config(repo_root: Path, tmp_path: Path) -> dict:
    data_root = tmp_path / '.precomputed'
    for name in ('latents', 'audio_latents', 'conditions'):
        (data_root / name).mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    config = Config.fromfile(
        repo_root / 'configs' / 'ltx_video' / 'train_ltx_video_2_5_lora.py',
        import_custom_modules=False,
    ).trainer.native_config.to_dict()
    config['model'].update(
        model_path=_dummy_file(
            tmp_path,
            'ltx-2.5-22b-dev-transformer-bf16.safetensors',
        ),
        text_encoder_path=_dummy_file(
            tmp_path,
            'gemma4-12b-with-proj-ltx-2.5-bf16.safetensors',
        ),
        video_vae_path=_dummy_file(
            tmp_path,
            'ltx-2.5-video-vae-bf16.safetensors',
        ),
        audio_vae_path=_dummy_file(
            tmp_path,
            'ltx-2.5-audio-vae-bf16.safetensors',
        ),
    )
    config['data']['preprocessed_data_root'] = str(data_root)
    config['output_dir'] = str(output_dir)
    return config


def test_local_ltx_training_config_contract(repo_root, tmp_path):
    from hftrainer.trainers.ltx_video.native.config import LtxTrainerConfig

    assert LtxTrainerConfig.__module__.startswith('hftrainer.')
    native = LtxTrainerConfig(**_native_config(repo_root, tmp_path))
    assert native.model.training_mode == 'lora'
    assert native.optimization.steps == 2000


def test_local_ltx_backend_public_signatures():
    pytest.importorskip('av', reason='install hftrainer[ltx-video] for media runtime')

    from hftrainer.pipelines.ltx_video.backend.distilled import DistilledPipeline
    from hftrainer.pipelines.ltx_video.backend.ti2vid_two_stages import (
        TI2VidTwoStagesPipeline,
    )
    from hftrainer.pipelines.ltx_video.backend.utils.model_paths import ModelPaths
    from hftrainer.trainers.ltx_video.native.trainer import LtxvTrainer

    for local_type in (
        DistilledPipeline,
        TI2VidTwoStagesPipeline,
        ModelPaths,
        LtxvTrainer,
    ):
        assert local_type.__module__.startswith('hftrainer.')

    distilled_init = inspect.signature(DistilledPipeline.__init__).parameters
    dev_init = inspect.signature(TI2VidTwoStagesPipeline.__init__).parameters
    model_paths = inspect.signature(ModelPaths.from_split).parameters
    train_call = inspect.signature(LtxvTrainer.train).parameters

    assert {'model_paths', 'spatial_upsampler_path', 'loras'} <= set(distilled_init)
    assert {'model_paths', 'distilled_lora', 'loras'} <= set(dev_init)
    assert {
        'transformer_path',
        'text_encoder_path',
        'video_vae_path',
        'audio_vae_path',
        'duration_head_path',
    } <= set(model_paths)
    assert {'disable_progress_bars', 'step_callback'} <= set(train_call)


def test_ltx_source_revision_is_packaged_as_provenance(repo_root):
    record = (
        repo_root / 'hftrainer' / 'models' / 'ltx_video' / 'UPSTREAM.md'
    ).read_text(encoding='utf-8')
    assert LTX_25_SOURCE_COMMIT in record
    assert 'hftrainer.models.ltx_video.network' in record
    assert 'hftrainer.pipelines.ltx_video.backend' in record
    assert 'hftrainer.trainers.ltx_video.native' in record
