"""Smoke test for SMPL → G1 retargeting pipeline."""
import numpy as np
import pytest


@pytest.mark.smoke
def test_smpl_to_g1_retarget_basic():
    """Test basic retargeting from SMPL 135-dim to G1 29-DOF."""
    from hftrainer.models.motion.components.retarget import SMPLToG1Retargeter

    retargeter = SMPLToG1Retargeter(apply_limits=True, g1_dof=29)

    # Create synthetic SMPL motion: 60 frames, 135-dim
    T = 60
    motion_135 = np.random.randn(T, 135).astype(np.float32) * 0.1
    # Set translation to something reasonable
    motion_135[:, 0:3] = np.stack([
        np.linspace(0, 1, T),   # forward
        np.zeros(T),            # lateral
        np.ones(T) * 0.9,       # height
    ], axis=-1)

    result = retargeter.retarget_from_hymotion(motion_135, fps=30.0)

    # Check output shapes
    assert result['joint_angles'].shape == (T, 29)
    assert result['root_pos'].shape == (T, 3)
    assert result['root_orient_quat'].shape == (T, 4)
    assert result['root_orient_euler'].shape == (T, 3)
    assert result['fps'] == 30.0
    assert len(result['joint_names']) == 29

    # Check joint limits are respected
    from hftrainer.models.motion.components.retarget import G1_JOINT_LIMITS, G1_JOINT_NAMES
    for i, name in enumerate(G1_JOINT_NAMES):
        lo, hi = G1_JOINT_LIMITS[name]
        assert np.all(result['joint_angles'][:, i] >= lo - 1e-6), \
            f'{name}: min={result["joint_angles"][:, i].min():.4f} < {lo:.4f}'
        assert np.all(result['joint_angles'][:, i] <= hi + 1e-6), \
            f'{name}: max={result["joint_angles"][:, i].max():.4f} > {hi:.4f}'


@pytest.mark.smoke
def test_smpl_to_g1_retarget_201dim():
    """Test retargeting from 201-dim format."""
    from hftrainer.models.motion.components.retarget import SMPLToG1Retargeter

    retargeter = SMPLToG1Retargeter(g1_dof=29)

    T = 30
    motion_201 = np.random.randn(T, 201).astype(np.float32) * 0.1
    motion_201[:, 0:3] = 0.0

    result = retargeter.retarget_from_hymotion_201(motion_201)
    assert result['joint_angles'].shape == (T, 29)


@pytest.mark.smoke
def test_asap_pkl_export():
    """Test ASAP pickle export."""
    import tempfile
    import pickle
    from hftrainer.models.motion.components.retarget import SMPLToG1Retargeter

    retargeter = SMPLToG1Retargeter(g1_dof=29)

    T = 30
    motion_135 = np.random.randn(T, 135).astype(np.float32) * 0.1
    motion_135[:, 0:3] = 0.0

    result = retargeter.retarget_from_hymotion(motion_135, fps=30.0)

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pkl_path = f.name

    retargeter.to_asap_pkl(result, pkl_path)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    assert 'dof_pos' in data
    assert 'root_pos' in data
    assert 'dof_vel' in data
    assert data['dof_pos'].shape == (T, 29)
    assert data['dof_vel'].shape == (T, 29)

    import os
    os.unlink(pkl_path)


@pytest.mark.smoke
def test_mujoco_qpos_export():
    """Test MuJoCo qpos conversion."""
    from hftrainer.models.motion.components.retarget import SMPLToG1Retargeter

    retargeter = SMPLToG1Retargeter(g1_dof=29)

    T = 20
    motion_135 = np.random.randn(T, 135).astype(np.float32) * 0.1
    motion_135[:, 0:3] = 0.0

    result = retargeter.retarget_from_hymotion(motion_135)
    qpos = retargeter.to_mujoco_qpos(result)

    # MuJoCo qpos: [root_pos(3), root_quat(4), joints(29)] = 36
    assert qpos.shape == (T, 36)
    # Root quaternion should be unit norm
    quat_norms = np.linalg.norm(qpos[:, 3:7], axis=-1)
    np.testing.assert_allclose(quat_norms, 1.0, atol=1e-4)


@pytest.mark.smoke
def test_g1_23dof():
    """Test 23-DOF basic G1 variant."""
    from hftrainer.models.motion.components.retarget import SMPLToG1Retargeter

    retargeter = SMPLToG1Retargeter(g1_dof=23)

    T = 20
    motion_135 = np.random.randn(T, 135).astype(np.float32) * 0.1
    motion_135[:, 0:3] = 0.0

    result = retargeter.retarget_from_hymotion(motion_135)
    assert result['joint_angles'].shape == (T, 23)
    assert result['dof'] == 23


@pytest.mark.smoke
def test_asap_config_generator():
    """Test ASAP config/command generation."""
    from hftrainer.models.motion.components.retarget.isaac_gym_bridge import (
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
