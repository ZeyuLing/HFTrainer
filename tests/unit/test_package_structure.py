"""Structural contracts for HFTrainer's implementation-first taxonomy."""

from importlib import import_module
from pathlib import Path


IMPLEMENTATIONS = {
    'llama',
    'dmd',
    'ltx_video',
    'sd15',
    'stylegan2',
    'vit',
    'wan',
}

BUNDLE_OWNERS = {
    'llama': ('LlamaBundle', 'hftrainer.models.llama.bundle'),
    'dmd': ('DMDBundle', 'hftrainer.models.dmd.bundle'),
    'ltx_video': ('LTXVideoBundle', 'hftrainer.models.ltx_video.bundle'),
    'sd15': ('SD15Bundle', 'hftrainer.models.sd15.bundle'),
    'stylegan2': ('StyleGAN2Bundle', 'hftrainer.models.stylegan2.bundle'),
    'vit': ('ViTBundle', 'hftrainer.models.vit.bundle'),
    'wan': ('WanBundle', 'hftrainer.models.wan.bundle'),
}

COMPONENT_OWNERS = {
    'AutoencoderKL': 'hftrainer.models.sd15.network.vae',
    'AutoencoderKLWan': 'hftrainer.models.wan.network.vae',
    'CLIPTextModel': 'hftrainer.models.sd15.network.clip',
    'DDIMScheduler': 'hftrainer.models.sd15.network.schedulers',
    'DDPMScheduler': 'hftrainer.models.sd15.network.schedulers',
    'FlowMatchEulerDiscreteScheduler': 'hftrainer.models.wan.network.scheduler',
    'LocalLlamaForCausalLM': 'hftrainer.models.llama.network.modeling_llama',
    'LocalViTForImageClassification': 'hftrainer.models.vit.network.modeling_vit',
    'PNDMScheduler': 'hftrainer.models.sd15.network.schedulers',
    'StyleGAN2Discriminator': 'hftrainer.models.stylegan2.network.model',
    'StyleGAN2Generator': 'hftrainer.models.stylegan2.network.model',
    'UMT5EncoderModel': 'hftrainer.models.wan.network.text_encoder',
    'UNet2DConditionModel': 'hftrainer.models.sd15.network.unet',
    'WanTokenizer': 'hftrainer.models.wan.network.tokenizer',
    'WanTransformer3DModel': 'hftrainer.models.wan.network.transformer',
}


def test_models_namespace_contains_only_implementation_directories(repo_root: Path):
    models_root = repo_root / 'hftrainer' / 'models'
    actual = {
        path.name
        for path in models_root.iterdir()
        if path.is_dir() and any(path.rglob('*.py'))
    }
    assert actual == IMPLEMENTATIONS


def test_models_namespace_contains_only_framework_root_modules(repo_root: Path):
    models_root = repo_root / 'hftrainer' / 'models'
    assert {path.name for path in models_root.glob('*.py')} == {
        '__init__.py',
        'base_model_bundle.py',
        'lora.py',
    }


def test_each_implementation_exports_one_canonical_bundle():
    for implementation, (name, owner) in BUNDLE_OWNERS.items():
        package = import_module(f'hftrainer.models.{implementation}')
        assert getattr(package, name).__module__ == owner


def test_registered_components_are_owned_by_local_network_modules():
    import hftrainer
    from hftrainer.registry import MODEL_COMPONENTS

    hftrainer.register_all_modules()
    for name, owner in COMPONENT_OWNERS.items():
        component = MODEL_COMPONENTS.get(name)
        assert component is not None, name
        assert component.__module__ == owner
        assert component.__module__.startswith('hftrainer.models.')


def test_register_all_modules_includes_ltx_vertical_slice():
    import hftrainer
    from hftrainer.registry import MODEL_BUNDLES, PIPELINES, TRAINERS

    hftrainer.register_all_modules()
    assert MODEL_BUNDLES.get('LTXVideoBundle') is not None
    assert PIPELINES.get('LTXVideoPipeline') is not None
    assert TRAINERS.get('LTXVideoTrainer') is not None


def test_trainers_and_pipelines_are_not_nested_under_models(repo_root: Path):
    models_root = repo_root / 'hftrainer' / 'models'
    forbidden = {'trainer.py', 'pipeline.py', 'dataset.py'}
    offenders = [
        path.relative_to(repo_root)
        for path in models_root.rglob('*.py')
        if path.name in forbidden
    ]
    assert not offenders
