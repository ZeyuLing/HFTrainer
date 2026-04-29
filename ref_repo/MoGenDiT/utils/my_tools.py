import numpy as np


def auto_loop_interp_args(motion_joint22, fps=20):
    ani_frames = motion_joint22.shape[0]  # [n, 22, 3]
    trans = motion_joint22[:, [0]]
    hands_move = motion_joint22[:, [20, 21]] - trans
    loop_k = min(10, ani_frames // 4)  # 取动画长度的1/4作为上限
    trans_A = motion_joint22[0, 0, [0, 2]]
    trans_B = motion_joint22[-1, 0, [0, 2]]
    distance_gap = np.linalg.norm(trans_A - trans_B)  # 计算起始和结束位置的距离
    n_interpolation_return = int(
        fps * distance_gap / 0.8
    )  # 允许平均0.8m/s的速度回到起始点

    avg_hands_move_speed = np.linalg.norm(hands_move[1:] - hands_move[:-1]).mean()
    hands_return_distance = np.linalg.norm(hands_move[-1] - hands_move[0]).mean()
    n_interpolation_motion = int(hands_return_distance / avg_hands_move_speed) * 1.5
    n_interpolation_motion = int(n_interpolation_motion)

    n_interpolation = max(n_interpolation_motion, n_interpolation_return)
    n_interpolation = min(
        n_interpolation, 196 - loop_k * 2
    )  # 限制总输入长度不超过196帧
    return max(n_interpolation, 20), loop_k


def auto_seam_interp_args(motion_joint22_1, motion_joint22_2, fps=20):
    ani_frames_1 = motion_joint22_1.shape[0]  # [n, 22, 3]
    trans_1 = motion_joint22_1[:, [0]]
    hands_move_1 = motion_joint22_1[:, [20, 21]] - trans_1

    ani_frames_2 = motion_joint22_2.shape[0]  # [n, 22, 3]
    trans_2 = motion_joint22_2[:, [0]]
    hands_move_2 = motion_joint22_2[:, [20, 21]] - trans_2

    loop_k = min(10, ani_frames_1 // 4)  # 取动画长度的1/4作为上限

    trans_A = motion_joint22_1[-1, 0, [0, 2]]
    trans_B = motion_joint22_2[0, 0, [0, 2]]
    distance_gap = np.linalg.norm(trans_A - trans_B)  # 计算起始和结束位置的距离
    n_interpolation_return = int(
        fps * distance_gap / 0.5
    )  # 允许平均1.0m/s的速度执行A-B移动

    avg_hands_move_vel = np.linalg.norm(hands_move_1[1:] - hands_move_1[:-1]).mean()
    hands_return_distance = np.linalg.norm(hands_move_1[-1] - hands_move_2[0]).mean()
    n_interpolation_motion = int(hands_return_distance / avg_hands_move_vel) + 20

    n_interpolation = max(n_interpolation_motion, n_interpolation_return)
    n_interpolation = min(
        n_interpolation, 196 - loop_k * 2
    )  # 限制总输入长度不超过196帧
    return max(n_interpolation, 20), loop_k
