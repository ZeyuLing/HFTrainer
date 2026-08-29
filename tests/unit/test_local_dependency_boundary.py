"""Hard contracts for repository-owned model execution."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOTS = (ROOT / 'hftrainer', ROOT / 'tools')
FORBIDDEN_MODEL_PACKAGES = {
    'transformers',
    'diffusers',
    'peft',
    'tokenizers',
    'ltx_core',
    'ltx_pipelines',
    'ltx_trainer',
    'ltx_kernels',
    'timm',
    'mmagic',
    'mmpretrain',
    'comfy',
    'comfy_kitchen',
}


def _python_files():
    for root in RUNTIME_ROOTS:
        yield from root.rglob('*.py')


def _root(name: str | None) -> str:
    return (name or '').split('.', 1)[0]


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f'{parent}.{node.attr}' if parent else None
    return None


def test_runtime_has_no_forbidden_model_imports():
    violations = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            else:
                continue
            for name in names:
                if _root(name) in FORBIDDEN_MODEL_PACKAGES:
                    violations.append(f'{path.relative_to(ROOT)}:{node.lineno}: {name}')
    assert not violations, 'Forbidden model-runtime imports:\n' + '\n'.join(violations)


def test_model_execution_chain_has_no_dynamic_import_escape_hatch():
    violations = []
    for layer in ('models', 'pipelines', 'trainers'):
        for path in (ROOT / 'hftrainer' / layer).rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = (
                        [alias.name for alias in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module or '']
                    )
                    if any(_root(name) == 'importlib' for name in names):
                        violations.append(f'{path.relative_to(ROOT)}:{node.lineno}')
                elif isinstance(node, ast.Call):
                    callee = _dotted_name(node.func)
                    if callee in {
                        '__import__',
                        'eval',
                        'exec',
                        'importlib.import_module',
                        'torch.hub.load',
                    }:
                        violations.append(
                            f'{path.relative_to(ROOT)}:{node.lineno}: {callee}'
                        )
    assert not violations, (
        'Model execution code may not dynamically resolve implementation classes:\n'
        + '\n'.join(violations)
    )


def test_model_execution_chain_does_not_mutate_python_path():
    violations = []
    for layer in ('models', 'pipelines', 'trainers'):
        for path in (ROOT / 'hftrainer' / layer).rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                owner = node.func.value
                if (
                    isinstance(owner, ast.Attribute)
                    and isinstance(owner.value, ast.Name)
                    and owner.value.id == 'sys'
                    and owner.attr == 'path'
                    and node.func.attr in {'append', 'insert', 'extend'}
                ):
                    violations.append(f'{path.relative_to(ROOT)}:{node.lineno}')
    assert not violations, (
        'Model execution code may not mutate sys.path to locate implementations:\n'
        + '\n'.join(violations)
    )


def test_model_layer_does_not_depend_on_pipeline_or_trainer_layers():
    violations = []
    for path in (ROOT / 'hftrainer' / 'models').rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            else:
                continue
            for name in names:
                if name.startswith(('hftrainer.pipelines', 'hftrainer.trainers')):
                    violations.append(
                        f'{path.relative_to(ROOT)}:{node.lineno}: {name}'
                    )
    assert not violations, (
        'The model layer must not import orchestration layers:\n'
        + '\n'.join(violations)
    )


def test_builtin_models_import_when_external_model_packages_are_blocked():
    script = r'''
import importlib.abc
import sys

blocked = {
    "transformers", "diffusers", "peft", "tokenizers",
    "ltx_core", "ltx_pipelines", "ltx_trainer", "ltx_kernels",
    "timm", "mmagic", "mmpretrain", "comfy", "comfy_kitchen",
}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"forbidden import attempted: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import hftrainer
hftrainer.register_all_modules()
import hftrainer.models.ltx_video
import hftrainer.pipelines.ltx_video
import hftrainer.trainers.ltx_video
print("local-import-ok")
'''
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT)
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'local-import-ok' in result.stdout


def test_declared_dependencies_do_not_install_external_model_implementations():
    import tomllib

    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    dependencies = list(project['project']['dependencies'])
    for optional in project['project'].get('optional-dependencies', {}).values():
        dependencies.extend(optional)
    roots = {
        value.split('[', 1)[0].split('=', 1)[0].split('<', 1)[0].split('>', 1)[0].lower()
        for value in dependencies
    }
    assert roots.isdisjoint(FORBIDDEN_MODEL_PACKAGES)


def test_complete_builtin_component_registry_is_repository_owned():
    """Every built-in component, not only the canonical names, stays local."""

    script = r'''
import hftrainer
from hftrainer.registry import MODEL_COMPONENTS

hftrainer.register_all_modules()
foreign = {
    name: component.__module__
    for name, component in MODEL_COMPONENTS.module_dict.items()
    if not component.__module__.startswith("hftrainer.models.")
}
assert not foreign, foreign
print("all-components-local")
'''
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT)
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'all-components-local' in result.stdout


def test_repository_registries_reject_dotted_import_paths():
    """Unknown config strings may not become implicit Python imports."""

    import pytest

    import hftrainer
    from hftrainer.registry import (
        DATASETS,
        EVALUATORS,
        HOOKS,
        MODEL_BUNDLES,
        MODEL_COMPONENTS,
        PIPELINES,
        TRAINERS,
        TRANSFORMS,
        VISUALIZERS,
    )

    hftrainer.register_all_modules()
    registries = (
        MODEL_COMPONENTS,
        MODEL_BUNDLES,
        TRAINERS,
        PIPELINES,
        DATASETS,
        TRANSFORMS,
        HOOKS,
        EVALUATORS,
        VISUALIZERS,
    )
    for registry in registries:
        dotted_path = 'torch.nn.Identity'
        assert registry.get(dotted_path) is None
        with pytest.raises(KeyError):
            registry.build({'type': dotted_path})


def test_repository_registries_reject_unregistered_class_objects():
    """A Python class object is not a back door around explicit registration."""

    from types import SimpleNamespace

    import pytest
    import torch.nn as nn

    from hftrainer.registry import (
        DATASETS,
        EVALUATORS,
        HOOKS,
        MODEL_BUNDLES,
        MODEL_COMPONENTS,
        PIPELINES,
        TRAINERS,
        TRANSFORMS,
        VISUALIZERS,
    )

    for registry, cls in (
        (MODEL_COMPONENTS, nn.Identity),
        (MODEL_BUNDLES, SimpleNamespace),
        (TRAINERS, SimpleNamespace),
        (PIPELINES, SimpleNamespace),
        (DATASETS, SimpleNamespace),
        (TRANSFORMS, SimpleNamespace),
        (HOOKS, SimpleNamespace),
        (EVALUATORS, SimpleNamespace),
        (VISUALIZERS, SimpleNamespace),
    ):
        with pytest.raises(KeyError, match='not explicitly registered'):
            registry.build({'type': cls})


def test_stylegan_bundle_rejects_registered_foreign_components():
    """Registration alone cannot change the implementation family owner."""

    import pytest
    import torch.nn as nn

    import hftrainer.models.stylegan2  # noqa: F401
    from hftrainer.models.stylegan2.bundle import StyleGAN2Bundle
    from hftrainer.registry import MODEL_COMPONENTS

    @MODEL_COMPONENTS.register_module(name='BoundaryForeignGenerator', force=True)
    class ForeignGenerator(nn.Module):
        pass

    @MODEL_COMPONENTS.register_module(name='BoundaryForeignDiscriminator', force=True)
    class ForeignDiscriminator(nn.Module):
        pass

    with pytest.raises(TypeError, match='repository-owned'):
        StyleGAN2Bundle(
            generator={'type': 'BoundaryForeignGenerator'},
            discriminator={'type': 'BoundaryForeignDiscriminator'},
        )


def test_managed_runner_rejects_an_unregistered_class_object():
    """The managed-runner shortcut must enforce the trainer registry too."""

    from types import SimpleNamespace

    import pytest

    from hftrainer.runner.builder import build_runner_from_cfg

    class ForeignManagedTrainer:
        manages_training_loop = True

        @classmethod
        def from_framework_config(cls, cfg):
            raise AssertionError('unregistered trainer executed')

    cfg = SimpleNamespace(trainer={'type': ForeignManagedTrainer})
    with pytest.raises(KeyError, match='not explicitly registered'):
        build_runner_from_cfg(cfg)
