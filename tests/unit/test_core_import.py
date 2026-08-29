"""Tests for the lightweight package import and explicit registration API."""

import subprocess
import sys
from types import SimpleNamespace


def test_core_import_does_not_import_accelerate_runner(repo_root, smoke_env):
    script = """
import sys
before = set(sys.modules)
import hftrainer

assert 'hftrainer.runner.accelerate_runner' not in sys.modules
assert 'accelerate' not in set(sys.modules) - before
assert hftrainer.ModelBundle.__name__ == 'ModelBundle'
assert hftrainer.BaseTrainer.__name__ == 'BaseTrainer'
assert 'hftrainer.runner.accelerate_runner' not in sys.modules
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


def test_compatibility_runner_export_when_accelerate_is_installed():
    import pytest

    pytest.importorskip('accelerate')

    from hftrainer import AccelerateRunner
    from hftrainer.runner.accelerate_runner import AccelerateRunner as DirectRunner

    assert AccelerateRunner is DirectRunner


def test_register_all_modules_is_idempotent():
    import hftrainer
    from hftrainer.registry import DATASETS, MODEL_BUNDLES, PIPELINES, TRAINERS

    hftrainer.register_all_modules()
    hftrainer.register_all_modules()

    assert MODEL_BUNDLES.get('ViTBundle') is not None
    assert MODEL_BUNDLES.get('SD15Bundle') is not None
    assert MODEL_BUNDLES.get('CausalLMBundle') is not None
    assert MODEL_BUNDLES.get('WanBundle') is not None
    assert MODEL_BUNDLES.get('StyleGAN2Bundle') is not None
    assert MODEL_BUNDLES.get('DMDBundle') is not None
    assert TRAINERS.get('ClassificationTrainer') is not None
    assert PIPELINES.get('WanPipeline') is not None
    assert DATASETS.get('AlpacaDataset') is not None


def test_custom_imports_loads_config_extension(tmp_path, monkeypatch):
    from mmengine.config import Config

    from hftrainer.utils.setup_env import import_custom_modules

    module_name = 'hftrainer_test_custom_extension'
    module_path = tmp_path / f'{module_name}.py'
    module_path.write_text('CUSTOM_IMPORT_WAS_EXECUTED = True\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))

    cfg = Config(
        {
            'custom_imports': {
                'imports': [module_name],
                'allow_failed_imports': False,
            }
        }
    )
    import_custom_modules(cfg)

    imported = sys.modules[module_name]
    assert imported.CUSTOM_IMPORT_WAS_EXECUTED is True


def test_train_merges_cfg_options_before_custom_imports(tmp_path):
    from tools.train import _load_and_register_config

    config_path = tmp_path / 'override_imports.py'
    config_path.write_text(
        "custom_imports = dict(imports=['module_that_must_not_import'], "
        "allow_failed_imports=False)\n"
        "trainer = dict(type='TrainerThatDoesNotExist')\n",
        encoding='utf-8',
    )

    cfg = _load_and_register_config(
        config_path,
        [
            "custom_imports.imports=['hftrainer.trainers.ltx_video']",
            "trainer.type='LTXVideoTrainer'",
        ],
    )

    assert cfg.trainer.type == 'LTXVideoTrainer'


def test_train_binds_local_rank_to_visible_cuda_device(monkeypatch):
    from tools import train

    selected_devices = []
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        set_device=selected_devices.append,
    )
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setenv('LOCAL_RANK', '5')

    train._bind_local_cuda_device()

    assert selected_devices == [1]
