"""Smoke test for HyMotion-V2M (pre-extracted feature -> motion).

Stage 1 is inference-only (no trainer), so this exercises the full wiring via
``tools/infer.py``: registry import -> build tiny vendored ``MotionGenerationV2M``
-> ``HyMotionV2MPipeline.infer_from_feature`` on a random feature stream ->
sliding-window flow-matching ODE -> 349-dim decode -> SMPL FK + floor fit ->
saved npz.  Runs on CPU in seconds (tiny network, 2 ODE steps).
"""

from pathlib import Path

import numpy as np
import pytest


CONFIG_PATH = 'configs/hymotion_v2m/hymotion_v2m_smoke.py'


@pytest.mark.smoke
def test_hymotion_v2m_feature_to_motion(
    tmp_path: Path, repo_root: Path, python_executable: str, cli_runner
):
    config = repo_root / CONFIG_PATH
    assert config.exists(), f"missing config: {config}"

    output_path = tmp_path / 'hymotion_v2m_infer.npz'
    args = [
        python_executable,
        'tools/infer.py',
        '--config', str(config),
        '--checkpoint', 'none',  # bundle has no ckpt; warns & uses random init
        '--num-frames', '48',    # > train_frames(40) to exercise sliding window
        '--output', str(output_path),
        '--device', 'cpu',
    ]
    cli_runner(args, timeout=600)

    assert output_path.exists(), 'inference did not produce an output npz'
    data = np.load(output_path)
    for key in ('rot6d', 'transl', 'shapes', 'keypoints3d'):
        assert key in data, f'missing key {key} in output'

    rot6d = data['rot6d']
    transl = data['transl']
    k3d = data['keypoints3d']
    # (B, L, J, 6) / (B, L, 3) / (B, L, J, 3)
    assert rot6d.ndim == 4 and rot6d.shape[-1] == 6
    assert transl.ndim == 3 and transl.shape[-1] == 3
    assert k3d.ndim == 4 and k3d.shape[-1] == 3
    L = transl.shape[1]
    assert L >= 48, f'expected >=48 frames after sliding window, got {L}'
    assert rot6d.shape[1] == L and k3d.shape[1] == L
    assert np.isfinite(rot6d).all() and np.isfinite(k3d).all()
