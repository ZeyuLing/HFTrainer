import numpy as np

from hftrainer.evaluation.motion.mbench_body_penetration import body_axis_angle_for_file


def test_body_penetration_smpl_npz_uses_body_pose_only(tmp_path):
    path = tmp_path / "smpl_params.npz"
    global_orient = np.array([[100.0, 101.0, 102.0]], dtype=np.float32)
    body_pose = np.arange(63, dtype=np.float32).reshape(1, 63)
    np.savez(path, global_orient=global_orient, body_pose=body_pose)

    aa = body_axis_angle_for_file(path, "m135")

    assert aa.shape == (1, 21, 3)
    np.testing.assert_allclose(aa.reshape(1, 63), body_pose)


def test_body_penetration_poses_npz_skips_root(tmp_path):
    path = tmp_path / "poses.npz"
    poses = np.arange(72, dtype=np.float32).reshape(1, 72)
    np.savez(path, poses=poses)

    aa = body_axis_angle_for_file(path, "m135")

    assert aa.shape == (1, 21, 3)
    np.testing.assert_allclose(aa.reshape(1, 63), poses[:, 3:66])
