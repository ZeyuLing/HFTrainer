"""Tests for declarative loading of repository-owned components."""

from pathlib import Path

import torch
import torch.nn as nn

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_COMPONENTS


class DummyLocalModel(nn.Module):
    last_from_pretrained_kwargs = None

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.last_from_pretrained_kwargs = dict(kwargs)
        return cls()


MODEL_COMPONENTS.register_module(
    name='DummyLocalModel', module=DummyLocalModel, force=True
)


class DummyArtifactBundle(ModelBundle):
    PRETRAINED_SPEC = {
        'components': {
            'model': {
                'default_type': 'DummyLocalModel',
                'type_arg': 'model_type',
                'pretrained_kwargs_arg': 'model_kwargs',
                'overrides_arg': 'model_overrides',
            },
        },
        'init_args': {
            'tokenizer_path': {'default': ModelBundle._PRETRAINED_PATH_SENTINEL},
        },
    }

    def __init__(self, model: dict, tokenizer_path: str):
        super().__init__()
        self._build_modules({'model': model})
        self.tokenizer_path = tokenizer_path

    def save_pretrained(self, save_directory: str, **kwargs):
        if kwargs:
            raise TypeError(f'Unexpected options: {sorted(kwargs)}')
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / 'model.pt')
        (path / 'artifact.version').write_text('dummy-local-v1\n', encoding='utf-8')


def test_pretrained_spec_builds_only_registered_local_component():
    bundle = DummyArtifactBundle.from_pretrained(
        pretrained_model_name_or_path='local/artifact',
        model_kwargs={'revision': 'frozen-copy'},
    )

    assert isinstance(bundle, DummyArtifactBundle)
    assert bundle.tokenizer_path == 'local/artifact'
    assert DummyLocalModel.last_from_pretrained_kwargs == {
        'pretrained_model_name_or_path': 'local/artifact',
        'revision': 'frozen-copy',
    }


def test_bundle_owns_its_artifact_export(tmp_path):
    bundle = DummyArtifactBundle.from_pretrained('local/artifact')
    bundle.save_pretrained(str(tmp_path))

    assert (tmp_path / 'model.pt').is_file()
    assert (tmp_path / 'artifact.version').read_text(encoding='utf-8') == 'dummy-local-v1\n'


def test_model_registry_rejects_dynamic_or_unknown_class_paths():
    for name in ('external_package.Model', 'DefinitelyNotRegistered'):
        try:
            MODEL_COMPONENTS.build({'type': name})
        except KeyError as exc:
            assert 'not registered' in str(exc)
        else:
            raise AssertionError(f'{name!r} unexpectedly bypassed the local registry')
