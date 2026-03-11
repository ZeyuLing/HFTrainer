import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope='session')
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def python_executable() -> str:
    return sys.executable


@pytest.fixture(scope='session')
def has_cuda() -> bool:
    return torch.cuda.is_available()


@pytest.fixture(scope='session')
def smoke_env(repo_root: Path):
    env = os.environ.copy()
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    env['PYTHONPATH'] = str(repo_root) + os.pathsep + env.get('PYTHONPATH', '')
    return env


def run_cli(
    args,
    repo_root: Path,
    env: dict,
    timeout: int = 600,
):
    result = subprocess.run(
        args,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Command failed.\n"
            f"Args: {' '.join(str(a) for a in args)}\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


@pytest.fixture
def cli_runner(repo_root: Path, smoke_env: dict):
    def _run(args, timeout: int = 600):
        return run_cli(args=args, repo_root=repo_root, env=smoke_env, timeout=timeout)

    return _run
