from pathlib import Path

import torch.nn as nn

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import HF_MODELS


class DummyHFModel(nn.Module):
    last_from_pretrained_kwargs = None
    last_save_pretrained_kwargs = None

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.last_from_pretrained_kwargs = dict(kwargs)
        return cls()

    def save_pretrained(self, save_directory: str, **kwargs):
        type(self).last_save_pretrained_kwargs = dict(kwargs)
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'dummy-model.txt').write_text('ok', encoding='utf-8')


class DummyTokenizer:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, save_directory: str):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'dummy-tokenizer.txt').write_text('ok', encoding='utf-8')
        self.saved_to = str(path)


class DummyPipeline:
    last_components = None
    last_save_kwargs = None

    def __init__(self, model, tokenizer):
        type(self).last_components = {'model': model, 'tokenizer': tokenizer}

    def save_pretrained(self, save_directory: str, **kwargs):
        type(self).last_save_kwargs = dict(kwargs)
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'dummy-pipeline.txt').write_text('ok', encoding='utf-8')


HF_MODELS.register_module(name='DummyHFModel', module=DummyHFModel, force=True)


class DummySingleModelBundle(ModelBundle):
    HF_PRETRAINED_SPEC = {
        'components': {
            'model': {
                'default_type': 'DummyHFModel',
                'type_arg': 'model_type',
                'pretrained_kwargs_arg': 'model_kwargs',
                'overrides_arg': 'model_overrides',
            },
        },
        'init_args': {
            'tokenizer_path': {'default': ModelBundle._PRETRAINED_PATH_SENTINEL},
        },
    }
    HF_SAVE_PRETRAINED_SPEC = {
        'kind': 'module',
        'module': 'model',
        'extra_artifacts': ['tokenizer'],
    }

    def __init__(self, model: dict, tokenizer_path: str):
        super().__init__()
        self._build_modules({'model': model})
        self.tokenizer_path = tokenizer_path
        self.tokenizer = DummyTokenizer()


class DummyPipelineBundle(ModelBundle):
    HF_SAVE_PRETRAINED_SPEC = {
        'kind': 'pipeline',
        'pipeline_class': DummyPipeline,
        'components': {
            'model': 'model',
            'tokenizer': 'tokenizer',
        },
    }

    def __init__(self):
        super().__init__()
        self.model = DummyHFModel()
        self.tokenizer = DummyTokenizer()


def test_hf_pretrained_spec_builds_bundle_from_common_definition():
    bundle = DummySingleModelBundle.from_pretrained(
        pretrained_model_name_or_path='dummy/model',
        model_kwargs={'revision': 'main'},
    )

    assert isinstance(bundle, DummySingleModelBundle)
    assert bundle.tokenizer_path == 'dummy/model'
    assert DummyHFModel.last_from_pretrained_kwargs == {
        'pretrained_model_name_or_path': 'dummy/model',
        'revision': 'main',
    }


def test_hf_save_pretrained_spec_exports_module_and_extra_artifacts(tmp_path):
    bundle = DummySingleModelBundle.from_pretrained('dummy/model')
    bundle.save_pretrained(str(tmp_path))

    assert (tmp_path / 'dummy-model.txt').is_file()
    assert (tmp_path / 'dummy-tokenizer.txt').is_file()
    assert DummyHFModel.last_save_pretrained_kwargs['safe_serialization'] is True


def test_hf_save_pretrained_spec_supports_pipeline_exports(tmp_path):
    bundle = DummyPipelineBundle()
    bundle.save_pretrained(str(tmp_path))

    assert (tmp_path / 'dummy-pipeline.txt').is_file()
    assert DummyPipeline.last_components['model'] is bundle.model
    assert DummyPipeline.last_components['tokenizer'] is bundle.tokenizer
