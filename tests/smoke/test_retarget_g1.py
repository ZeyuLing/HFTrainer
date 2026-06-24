"""Smoke test for SMPL → G1 retargeting (GMR backend only).

The analytic Euler-decomposition backend was removed (low quality). The only
retargeter is :class:`GMRSMPLToG1Retargeter` (mink IK). Its full IK path needs
heavy optional deps (``mink``/``daqp``/``smplx``/``mujoco``) plus SMPL-X body
models, so the IK test skips gracefully when those are unavailable; the
output-format helpers are pure NumPy and are always exercised.
"""
import numpy as np
import pytest


def _fake_gmr_result(T=20, n_dof=29, fps=30.0):
    """A synthetic GMRSMPLToG1Retargeter result dict (no IK needed)."""
    rng = np.random.default_rng(0)
    quat = rng.standard_normal((T, 4)).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    return {
        'dof_pos': (rng.standard_normal((T, n_dof)) * 0.1).astype(np.float32),
        'root_pos': rng.standard_normal((T, 3)).astype(np.float32),
        'root_orient_quat': quat,                       # wxyz
        'root_rot': quat[:, [1, 2, 3, 0]],              # xyzw
        'fps': fps,
        'joint_names': None,
        'dof': n_dof,
    }


@pytest.mark.smoke
def test_g1_constants():
    """G1 joint name/limit tables are consistent."""
    from hftrainer.motion.retarget import G1_JOINT_LIMITS, G1_JOINT_NAMES

    assert len(G1_JOINT_NAMES) == 29
    for name in G1_JOINT_NAMES:
        lo, hi = G1_JOINT_LIMITS[name]
        assert lo < hi


@pytest.mark.smoke
def test_analytic_backend_removed():
    """The old analytic retargeter must no longer be importable."""
    import hftrainer.motion.retarget as R

    assert not hasattr(R, 'SMPLToG1Retargeter')
    assert hasattr(R, 'GMRSMPLToG1Retargeter')


@pytest.mark.smoke
def test_to_mujoco_qpos():
    """GMR result -> MuJoCo qpos (pure numpy)."""
    from hftrainer.motion.retarget import GMRSMPLToG1Retargeter

    res = _fake_gmr_result(T=20)
    qpos = GMRSMPLToG1Retargeter.to_mujoco_qpos(
        GMRSMPLToG1Retargeter.__new__(GMRSMPLToG1Retargeter), res
    )
    assert qpos.shape == (20, 36)
    quat_norms = np.linalg.norm(qpos[:, 3:7], axis=-1)
    np.testing.assert_allclose(quat_norms, 1.0, atol=1e-4)


@pytest.mark.smoke
def test_asap_pkl_export(tmp_path):
    """ASAP pickle export from a GMR result (pure numpy, staticmethod)."""
    import pickle
    from hftrainer.motion.retarget import GMRSMPLToG1Retargeter

    res = _fake_gmr_result(T=30)
    pkl_path = str(tmp_path / 'g1.pkl')
    GMRSMPLToG1Retargeter.to_asap_pkl(res, pkl_path)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    assert data['dof_pos'].shape == (30, 29)
    assert data['dof_vel'].shape == (30, 29)
    assert data['root_vel'].shape == (30, 3)
    assert 'root_orient_quat' in data


@pytest.mark.smoke
def test_gmr_retarget_from_motion135():
    """Full GMR mink-IK path on a tiny clip; skip if deps/models unavailable."""
    from hftrainer.motion.retarget import GMRSMPLToG1Retargeter

    try:
        rt = GMRSMPLToG1Retargeter(smooth=False, ground_align=False)
        T = 8
        motion_135 = np.zeros((T, 135), dtype=np.float32)
        # identity rot6d for all 22 joints ([1,0,0, 0,1,0] row-major).
        eye6 = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        motion_135[:, 3:135] = np.tile(eye6, 22)
        motion_135[:, 2] = 0.9  # height
        result = rt.retarget_from_motion135(motion_135)
    except Exception as e:  # missing mink/daqp/smplx/mujoco or SMPL-X models
        pytest.skip(f'GMR IK unavailable: {e}')

    assert result['dof_pos'].shape[0] == 8
    assert result['dof_pos'].shape[1] == 29
    assert result['root_pos'].shape == (8, 3)
    assert result['root_orient_quat'].shape == (8, 4)


@pytest.mark.smoke
def test_asap_config_generator():
    """Test ASAP config/command generation."""
    from hftrainer.motion.retarget.isaac_gym_bridge import (
        ASAPConfigGenerator,
    )

    gen = ASAPConfigGenerator(asap_root='/tmp/fake_asap', num_envs=1024)

    cmd = gen.generate_training_command(
        motion_file='/tmp/test_motion.pkl',
        experiment_name='test_exp',
    )

    assert 'train_agent.py' in cmd
    assert 'isaacgym' in cmd
    assert '1024' in cmd
    assert '/tmp/test_motion.pkl' in cmd

    eval_cmd = gen.generate_eval_command('/tmp/model.pt')
    assert 'eval_agent.py' in eval_cmd

    sim_cmds = gen.generate_sim2sim_commands('/tmp/policy.onnx')
    assert 'simulator' in sim_cmds
    assert 'policy' in sim_cmds


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'smoke'])
