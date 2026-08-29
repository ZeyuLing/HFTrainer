"""Offline tests for managed LTX-2.5 training and preprocessing adapters."""

import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from hftrainer.trainers.ltx_video.preprocess import build_ltx_preprocess_command
from hftrainer.trainers.ltx_video.trainer import LTXVideoTrainer

DEV_TRANSFORMER = (
    'weights/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors'
)
TEXT_ENCODER = (
    'weights/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'
)
VIDEO_VAE = 'weights/vae/ltx-2.5-video-vae-bf16.safetensors'
AUDIO_VAE = 'weights/vae/ltx-2.5-audio-vae-bf16.safetensors'


def native_training_config(output_dir='outputs/native-ltx-test'):
    return {
        'model': {
            'model_path': DEV_TRANSFORMER,
            'text_encoder_path': TEXT_ENCODER,
            'video_vae_path': VIDEO_VAE,
            'audio_vae_path': AUDIO_VAE,
            'training_mode': 'lora',
            'load_checkpoint': None,
        },
        'lora': {'rank': 8, 'alpha': 8},
        'training_strategy': {
            'name': 'flexible',
            'video': {'is_generated': True, 'latents_dir': 'latents'},
            'audio': {'is_generated': True, 'latents_dir': 'audio_latents'},
        },
        'acceleration': {'quantization': None},
        'checkpoints': {'no_resume': False},
        'output_dir': str(output_dir),
    }


@pytest.mark.parametrize(
    ('load_scope', 'expected_no_resume'),
    [('model', True), ('full', False)],
)
def test_framework_load_scope_maps_to_official_checkpoint_semantics(
    tmp_path, load_scope, expected_no_resume
):
    checkpoint = tmp_path / 'lora_weights_step_00100.safetensors'
    trainer = LTXVideoTrainer(
        native_config=native_training_config(),
        output_dir=tmp_path / 'run',
        load_from={'path': str(checkpoint), 'load_scope': load_scope},
        require_files=False,
        require_linux=False,
    )

    resolved = trainer.resolve_config()

    assert resolved['model']['load_checkpoint'] == str(checkpoint)
    assert resolved['checkpoints']['no_resume'] is expected_no_resume
    assert resolved['output_dir'] == str((tmp_path / 'run').resolve())


def test_string_load_from_defaults_to_weights_only(tmp_path):
    checkpoint = tmp_path / 'lora_weights_step_00100.safetensors'
    trainer = LTXVideoTrainer(
        native_config=native_training_config(),
        load_from=str(checkpoint),
        require_files=False,
        require_linux=False,
    )

    resolved = trainer.resolve_config()

    assert resolved['model']['load_checkpoint'] == str(checkpoint)
    assert resolved['checkpoints']['no_resume'] is True


def test_auto_resume_maps_checkpoint_directory_and_enables_training_state(tmp_path):
    output_dir = tmp_path / 'run'
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / 'lora_weights_step_00025.safetensors').touch()
    trainer = LTXVideoTrainer(
        native_config=native_training_config(),
        output_dir=output_dir,
        auto_resume=True,
        require_files=False,
        require_linux=False,
    )

    resolved = trainer.resolve_config()

    assert resolved['model']['load_checkpoint'] == str(checkpoint_dir)
    assert resolved['checkpoints']['no_resume'] is False


def test_explicit_load_from_takes_precedence_over_auto_resume(tmp_path):
    output_dir = tmp_path / 'run'
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / 'lora_weights_step_00025.safetensors').touch()
    explicit = tmp_path / 'external' / 'lora_weights_step_00050.safetensors'
    trainer = LTXVideoTrainer(
        native_config=native_training_config(),
        output_dir=output_dir,
        load_from={'path': str(explicit), 'load_scope': 'model'},
        auto_resume=True,
        require_files=False,
        require_linux=False,
    )

    resolved = trainer.resolve_config()

    assert resolved['model']['load_checkpoint'] == str(explicit)
    assert resolved['checkpoints']['no_resume'] is True


def test_from_framework_config_maps_cli_level_fields(tmp_path):
    work_dir = tmp_path / 'framework-run'
    checkpoint = tmp_path / 'resume' / 'lora_weights_step_00075.safetensors'
    framework_cfg = SimpleNamespace(
        trainer={
            'type': 'LTXVideoTrainer',
            'native_config': native_training_config(),
            'require_files': False,
            'require_linux': False,
        },
        work_dir=str(work_dir),
        load_from={'path': str(checkpoint), 'load_scope': 'full'},
        auto_resume=True,
    )

    trainer = LTXVideoTrainer.from_framework_config(framework_cfg)
    resolved = trainer.resolve_config()

    assert isinstance(trainer, LTXVideoTrainer)
    assert resolved['output_dir'] == str(work_dir.resolve())
    assert resolved['model']['load_checkpoint'] == str(checkpoint)
    assert resolved['checkpoints']['no_resume'] is False


def test_managed_trainer_builds_official_schema_and_forwards_step_callback(
    tmp_path, monkeypatch
):
    events = {}
    callback_events = []

    class FakeOfficialConfig:
        def __init__(self, **kwargs):
            self.values = kwargs
            events['config'] = self

    class FakeOfficialTrainer:
        def __init__(self, config, *, component_registry):
            self.config = config
            self.component_registry = component_registry
            events['trainer'] = self

        def train(self, *, disable_progress_bars, step_callback):
            events['train_kwargs'] = {
                'disable_progress_bars': disable_progress_bars,
                'step_callback': step_callback,
            }
            samples = [Path('sample-step-1.mp4')]
            if step_callback is not None:
                step_callback(1, 10, samples)
            return Path('adapter.safetensors'), {'steps': 10}

    monkeypatch.setattr(
        LTXVideoTrainer,
        '_import_training_api',
        staticmethod(lambda: (FakeOfficialConfig, FakeOfficialTrainer)),
    )
    def callback(step, total, samples):
        callback_events.append((step, total, samples))

    trainer = LTXVideoTrainer(
        native_config=native_training_config(),
        output_dir=tmp_path / 'official-run',
        disable_progress_bars=True,
        require_files=False,
        require_linux=False,
        require_cuda=False,
        write_resolved_config=False,
        step_callback=callback,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = trainer.train()

    assert events['config'].values['output_dir'] == str(
        (tmp_path / 'official-run').resolve()
    )
    assert events['config'].values['model']['model_path'] == DEV_TRANSFORMER
    assert events['trainer'].config is events['config']
    assert events['trainer'].component_registry is trainer.components.training_registry
    assert (
        events['trainer'].component_registry
        is not trainer.components.inference_registry
    )
    assert events['train_kwargs'] == {
        'disable_progress_bars': True,
        'step_callback': callback,
    }
    assert callback_events == [(1, 10, [Path('sample-step-1.mp4')])]
    assert result == (Path('adapter.safetensors'), {'steps': 10})


def test_training_runtime_rejects_a_cpu_only_environment(monkeypatch):
    import torch

    trainer = LTXVideoTrainer(
        native_config=native_training_config(),
        require_files=False,
        require_linux=False,
        require_cuda=True,
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    with pytest.raises(RuntimeError, match='requires an NVIDIA CUDA runtime'):
        trainer._validate_runtime()


def test_global_rank_takes_precedence_over_local_rank(monkeypatch):
    monkeypatch.setenv('RANK', '1')
    monkeypatch.setenv('LOCAL_RANK', '0')
    assert LTXVideoTrainer._is_global_main_process() is False

    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '1')
    assert LTXVideoTrainer._is_global_main_process() is True


def test_managed_runner_builder_selects_ltx_without_accelerate(tmp_path):
    from hftrainer.runner.builder import build_runner_from_cfg

    cfg = SimpleNamespace(
        trainer={
            'type': 'LTXVideoTrainer',
            'native_config': native_training_config(),
            'require_files': False,
            'require_linux': False,
        },
        work_dir=str(tmp_path / 'managed'),
        load_from=None,
        auto_resume=False,
    )

    runner = build_runner_from_cfg(cfg)

    assert isinstance(runner, LTXVideoTrainer)


def test_framework_config_preserves_an_injected_component_store_identity(tmp_path):
    from hftrainer.models.ltx_video.component_loader import LTXComponentStore

    components = LTXComponentStore()
    cfg = SimpleNamespace(
        trainer={
            'type': 'LTXVideoTrainer',
            'native_config': native_training_config(),
            'require_files': False,
            'require_linux': False,
            'components': components,
        },
        work_dir=str(tmp_path / 'managed'),
        load_from=None,
        auto_resume=False,
    )

    trainer = LTXVideoTrainer.from_framework_config(cfg)

    assert trainer.components is components


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_preprocess_argv_contains_all_split_components_and_boolean_flags(tmp_path):
    import hftrainer.trainers.ltx_video.preprocess as preprocess_module

    script = (
        Path(preprocess_module.__file__).with_name('preprocess_scripts')
        / 'process_dataset.py'
    )
    model_path = _touch(tmp_path / DEV_TRANSFORMER)
    text_encoder_path = _touch(tmp_path / TEXT_ENCODER)
    video_vae_path = _touch(tmp_path / VIDEO_VAE)
    audio_vae_path = _touch(tmp_path / AUDIO_VAE)
    output_dir = tmp_path / 'preprocessed'

    command = build_ltx_preprocess_command(
        dataset_path=tmp_path / 'dataset.jsonl',
        resolution_buckets='960x544x49',
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        video_vae_path=video_vae_path,
        audio_vae_path=audio_vae_path,
        output_dir=output_dir,
        python_executable='python-test',
        device='cuda:1',
        batch_size=2,
        skip_audio=True,
        overwrite=True,
        vae_tiling=True,
        extra_args=['--caption-column', 'text'],
    )

    assert command[:3] == [
        'python-test',
        str(script),
        str(tmp_path / 'dataset.jsonl'),
    ]
    expected_pairs = {
        '--resolution-buckets': '960x544x49',
        '--model-path': str(model_path),
        '--text-encoder-path': str(text_encoder_path),
        '--video-vae-path': str(video_vae_path),
        '--audio-vae-path': str(audio_vae_path),
        '--output-dir': str(output_dir),
        '--device': 'cuda:1',
        '--batch-size': '2',
        '--caption-column': 'text',
    }
    for flag, value in expected_pairs.items():
        index = command.index(flag)
        assert command[index + 1] == value
    assert '--skip-audio' in command
    assert '--overwrite' in command
    assert '--vae-tiling' in command


def test_preprocess_requires_audio_vae_unless_audio_is_skipped(tmp_path):
    model_path = _touch(tmp_path / DEV_TRANSFORMER)
    text_encoder_path = _touch(tmp_path / TEXT_ENCODER)
    video_vae_path = _touch(tmp_path / VIDEO_VAE)

    kwargs = {
        'dataset_path': tmp_path / 'dataset.jsonl',
        'resolution_buckets': '960x544x49',
        'model_path': model_path,
        'text_encoder_path': text_encoder_path,
        'video_vae_path': video_vae_path,
        'audio_vae_path': None,
    }
    with pytest.raises(ValueError, match='requires model.audio_vae_path'):
        build_ltx_preprocess_command(**kwargs, skip_audio=False)

    command = build_ltx_preprocess_command(**kwargs, skip_audio=True)
    assert '--skip-audio' in command
    assert '--audio-vae-path' not in command


def test_ltx_configs_register_adapters_without_importing_heavy_backend(
    repo_root, smoke_env
):
    config_paths = [
        'configs/ltx_video/infer_ltx_video_2_5_distilled.py',
        'configs/ltx_video/infer_ltx_video_2_5_dev_lora.py',
        'configs/ltx_video/train_ltx_video_2_5_lora.py',
    ]
    script = f"""
import sys
from mmengine.config import Config
from hftrainer.registry import MODEL_BUNDLES, PIPELINES, TRAINERS
from hftrainer.utils.setup_env import import_custom_modules

for config_path in {config_paths!r}:
    cfg = Config.fromfile(config_path)
    import_custom_modules(cfg)

assert MODEL_BUNDLES.get('LTXVideoBundle') is not None
assert PIPELINES.get('LTXVideoPipeline') is not None
assert TRAINERS.get('LTXVideoTrainer') is not None
assert not any(
    name == root or name.startswith(root + '.')
    for name in sys.modules
    for root in ('ltx_core', 'ltx_pipelines', 'ltx_trainer')
), sorted(name for name in sys.modules if name.startswith('ltx'))
"""
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=repo_root,
        env=smoke_env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
