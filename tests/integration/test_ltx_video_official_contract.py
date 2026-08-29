"""Optional contract test against the real pinned Lightricks source checkout.

Set ``HFTRAINER_LTX_SOURCE_ROOT`` to the root of Lightricks/LTX-2 at the
revision pinned by HFTrainer. The normal offline suite intentionally skips this
test so core CI does not clone a rapidly evolving 22B model stack.
"""

from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path

import pytest
from mmengine.config import Config

from hftrainer.models.ltx_video.checkpoints import LTX_25_SOURCE_COMMIT

SOURCE_ROOT = os.environ.get('HFTRAINER_LTX_SOURCE_ROOT')
pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(
        not SOURCE_ROOT,
        reason='set HFTRAINER_LTX_SOURCE_ROOT to run the pinned-source contract',
    ),
]


def _dummy_file(root: Path, name: str) -> str:
    path = root / name
    path.touch()
    return str(path)


def test_real_ltx_config_and_public_signatures(repo_root, tmp_path, monkeypatch):
    source_root = Path(SOURCE_ROOT).expanduser().resolve()
    package_roots = [
        source_root / 'packages' / 'ltx-core' / 'src',
        source_root / 'packages' / 'ltx-pipelines' / 'src',
        source_root / 'packages' / 'ltx-trainer' / 'src',
    ]
    missing = [path for path in package_roots if not path.is_dir()]
    assert not missing, f'Invalid LTX source root; missing package sources: {missing}'
    for path in package_roots:
        monkeypatch.syspath_prepend(str(path))

    if (source_root / '.git').exists():
        revision = subprocess.run(
            ['git', '-C', str(source_root), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert revision == LTX_25_SOURCE_COMMIT

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

    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_trainer.config import LtxTrainerConfig
    from ltx_trainer.trainer import LtxvTrainer

    native = LtxTrainerConfig(**config)
    assert native.model.training_mode == 'lora'
    assert native.optimization.steps == 2000

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
