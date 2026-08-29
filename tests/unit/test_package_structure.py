"""Structural contracts for HFTrainer's package taxonomy."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

CANONICAL_MODEL_EXPORTS = {
    'causal_lm': {
        'CausalLMBundle': 'hftrainer.models.causal_lm.bundle',
    },
    'dmd': {
        'DMDBundle': 'hftrainer.models.dmd.bundle',
    },
    'ltx_video': {
        'LTXVideoBundle': 'hftrainer.models.ltx_video.bundle',
    },
    'sd15': {
        'SD15Bundle': 'hftrainer.models.sd15.bundle',
    },
    'stylegan2': {
        'StyleGAN2Bundle': 'hftrainer.models.stylegan2.bundle',
        'StyleGAN2Discriminator': 'hftrainer.models.stylegan2.model',
        'StyleGAN2Generator': 'hftrainer.models.stylegan2.model',
    },
    'vit': {
        'ViTBundle': 'hftrainer.models.vit.bundle',
    },
    'wan': {
        'WanBundle': 'hftrainer.models.wan.bundle',
    },
}

CANONICAL_ROOT_MODULES = {
    '__init__.py',
    'base_model_bundle.py',
    'peft_utils.py',
}


def test_models_namespace_contains_only_implementation_directories(repo_root: Path):
    """Task aliases must not create regular or namespace model packages."""
    models_root = repo_root / 'hftrainer' / 'models'
    implementation_directories = {
        path.name
        for path in models_root.iterdir()
        if path.is_dir() and any(path.rglob('*.py'))
    }

    assert implementation_directories == set(CANONICAL_MODEL_EXPORTS)


def test_models_namespace_contains_only_framework_root_modules(repo_root: Path):
    """Task aliases must not return as root-level modules."""
    models_root = repo_root / 'hftrainer' / 'models'
    root_modules = {path.name for path in models_root.glob('*.py')}

    assert root_modules == CANONICAL_ROOT_MODULES


def test_model_package_exports_are_owned_by_the_canonical_module():
    """Package exports must point at their one real implementation."""
    for package_name, exports in CANONICAL_MODEL_EXPORTS.items():
        package = import_module(f'hftrainer.models.{package_name}')
        for export_name, owner_module in exports.items():
            exported = getattr(package, export_name)
            assert exported.__module__ == owner_module


def test_registered_model_classes_have_one_canonical_owner(repo_root: Path):
    """Every model registry decorator must live on one canonical export."""
    expected = sorted(
        (export_name, owner_module)
        for exports in CANONICAL_MODEL_EXPORTS.values()
        for export_name, owner_module in exports.items()
    )
    registered = []
    models_root = repo_root / 'hftrainer' / 'models'

    for source_path in models_root.rglob('*.py'):
        module_path = source_path.relative_to(repo_root).with_suffix('')
        module_name = '.'.join(module_path.parts)
        tree = ast.parse(
            source_path.read_text(encoding='utf-8'),
            filename=str(source_path),
        )
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(target, ast.Attribute) and target.attr == 'register_module':
                    registered.append((node.name, module_name))

    assert sorted(registered) == expected
