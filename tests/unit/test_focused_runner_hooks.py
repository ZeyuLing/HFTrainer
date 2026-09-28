"""A clean focused training process must be able to build shared runner hooks."""
import subprocess
import sys


def test_hooks_registered_without_importing_full_catalogue():
    subprocess.run([sys.executable, '-c', '''
from hftrainer.runner.accelerate_runner import AccelerateRunner
hooks = AccelerateRunner._build_hooks({
    'checkpoint': {'type': 'CheckpointHook', 'interval': 1},
    'logger': {'type': 'LoggerHook', 'interval': 1},
})
assert {type(h).__name__ for h in hooks} == {'CheckpointHook', 'LoggerHook'}
'''], check=True, timeout=60)
