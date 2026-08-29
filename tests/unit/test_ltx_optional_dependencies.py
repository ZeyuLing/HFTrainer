"""Import and error contracts for opt-in LTX support integrations."""

from __future__ import annotations

import ast
import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hftrainer.trainers.ltx_video.native.optional_dependencies import (
    OptionalLTXDependencyError,
    require_huggingface_hub,
    require_imageio,
    require_wandb,
)

ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = ROOT / "hftrainer" / "trainers" / "ltx_video" / "native"
OPTIONAL_INTEGRATIONS = {"wandb", "huggingface_hub", "imageio"}


def _root(name: str | None) -> str:
    return (name or "").split(".", 1)[0]


def _module_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(_root(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(_root(node.module))
    return imports


def test_ltx_integration_packages_are_not_imported_at_module_scope():
    paths = [NATIVE_ROOT / "trainer.py", NATIVE_ROOT / "hf_hub_utils.py"]
    violations = {
        str(path.relative_to(ROOT)): sorted(
            _module_level_imports(path) & OPTIONAL_INTEGRATIONS
        )
        for path in paths
        if _module_level_imports(path) & OPTIONAL_INTEGRATIONS
    }
    assert not violations


def test_ltx_public_packages_import_with_integrations_blocked():
    script = r'''
import importlib.abc
import sys

blocked = {"wandb", "huggingface_hub", "imageio"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in blocked:
            error = ModuleNotFoundError(f"blocked optional integration: {fullname}")
            error.name = root
            raise error
        return None

sys.meta_path.insert(0, Blocker())
import hftrainer.models.ltx_video
import hftrainer.pipelines.ltx_video
import hftrainer.trainers.ltx_video
import hftrainer.trainers.ltx_video.native.hf_hub_utils
print("ltx-public-import-ok")
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ltx-public-import-ok" in result.stdout


@pytest.mark.parametrize(
    ("blocked", "loader", "feature", "package"),
    [
        ("wandb", require_wandb, "test W&B feature", "wandb"),
        (
            "huggingface_hub",
            require_huggingface_hub,
            "test Hub feature",
            "huggingface-hub",
        ),
        ("imageio", require_imageio, "test GIF feature", "imageio"),
    ],
)
def test_enabled_optional_integration_reports_actionable_error(
    monkeypatch, blocked, loader, feature, package
):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] == blocked:
            error = ModuleNotFoundError(f"blocked optional integration: {name}")
            error.name = blocked
            raise error
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(OptionalLTXDependencyError) as error:
        loader(feature=feature)

    message = str(error.value)
    assert feature in message
    assert package in message
    assert "hftrainer[ltx-video-integrations]" in message
