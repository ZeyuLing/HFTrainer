import numpy as np

from hftrainer.evaluation.motion.mbench_poseq import poseq_axis_angle_for_file


def test_poseq_smpl_npz_uses_global_orient_and_first_20_body_joints(tmp_path):
    path = tmp_path / "smpl_params.npz"
    global_orient = np.array([[100.0, 101.0, 102.0]], dtype=np.float32)
    body_pose = np.arange(63, dtype=np.float32).reshape(1, 63)
    np.savez(path, global_orient=global_orient, body_pose=body_pose)

    aa = poseq_axis_angle_for_file(path, "m135")

    assert aa.shape == (1, 21, 3)
    np.testing.assert_allclose(aa[:, 0], global_orient)
    np.testing.assert_allclose(aa[:, 1], body_pose.reshape(1, 21, 3)[:, 0])
    np.testing.assert_allclose(aa[:, -1], body_pose.reshape(1, 21, 3)[:, 19])


def test_poseq_poses_npz_uses_first_63_dims(tmp_path):
    path = tmp_path / "poses.npz"
    poses = np.arange(72, dtype=np.float32).reshape(1, 72)
    np.savez(path, poses=poses)

    aa = poseq_axis_angle_for_file(path, "m135")

    assert aa.shape == (1, 21, 3)
    np.testing.assert_allclose(aa.reshape(1, 63), poses[:, :63])
