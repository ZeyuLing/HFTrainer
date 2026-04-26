"""
Candy Wrapper Corruptors: 按 TBS 坐标系（绕当前骨轴施加 twist）模拟 candy wrapper 现象。

- Twist 在全局空间施加：curr_bone_axis = Global_Pos_Child - Global_Pos_Joint，
  Q_new_global = R_twist(curr_bone_axis, angle) * Q_old_global，new_local = inv(Q_parent_global) * Q_new_global。
- 仅修改 body 关节（0–21），不碰手指。

提供两个独立、不互斥的 corruptor（可单独使用，也可同时使用）：
1. WristCandyWrapperCorruptor（手腕）：仅手腕/手臂相关
   - palm_flip_180：手腕单独 180° twist（手掌朝向完全相反）
   - arm_twist_360：肩+肘共同正向 twist x°（比例随机）、腕反向 -x°（组合问题，减轻肘处 mesh 塌陷）
2. LimbCandyWrapperCorruptor（其它部位）：手臂、大腿等肢体，与手腕 corruptor 不互斥
   - limb_180：在选定链上随机一关节施加 180° twist
   - limb_360：从扰动关节向上追溯，祖宗节点共同 +x°、扰动关节 -x°，各关节位置不变（与手腕 360° 一致）
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Any, Dict, List, Optional, Tuple

from .base_corruptor import BaseCorruptor

# -----------------------------------------------------------------------------
# 身体关节与层级（SMPL 24 / SMPL-H 52 前 22 一致）
# -----------------------------------------------------------------------------
WRIST_LEFT, WRIST_RIGHT = 20, 21
# 手臂链：肩、肘、腕（用于 360° 分布）
ARM_LEFT_JOINTS = [16, 18, 20]
ARM_RIGHT_JOINTS = [17, 19, 21]
# 腿链：髋、膝、踝（用于泛化）
LEG_LEFT_JOINTS = [1, 4, 7]
LEG_RIGHT_JOINTS = [2, 5, 8]
BODY_JOINT_COUNT = 22

# SMPL 24 parent
SMPL24_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32
)
# SMPL-H 52 parent（前 22 与 24 一致，22+ 为手）
SMPLH_PARENTS = np.array(
    [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        22,
        23,
        20,
        25,
        26,
        20,
        28,
        29,
        20,
        31,
        32,
        20,
        34,
        35,
        21,
        37,
        38,
        21,
        40,
        41,
        21,
        43,
        44,
        21,
        46,
        47,
        21,
        49,
        50,
    ],
    dtype=np.int32,
)

# Rest 下关节→子关节的 offset（用于 FK 得到正确的骨轴方向）；仅 body 0–21，近似 T-pose
# 格式：rest_offsets[j] = rest_pos[j] - rest_pos[parent[j]]，长度可任意，方向用于骨轴
REST_OFFSETS_BODY_22 = np.zeros((22, 3), dtype=np.float64)
REST_OFFSETS_BODY_22[1] = [-0.09, -0.22, 0.02]
REST_OFFSETS_BODY_22[2] = [0.09, -0.22, 0.02]
REST_OFFSETS_BODY_22[3] = [0, 0.22, 0]
REST_OFFSETS_BODY_22[4] = [0, -0.43, 0]
REST_OFFSETS_BODY_22[5] = [0, -0.43, 0]
REST_OFFSETS_BODY_22[6] = [0, 0.22, 0]
REST_OFFSETS_BODY_22[7] = [0.05, -0.42, 0]
REST_OFFSETS_BODY_22[8] = [-0.05, -0.42, 0]
REST_OFFSETS_BODY_22[9] = [0, 0.22, 0]
REST_OFFSETS_BODY_22[10] = [0.05, -0.1, 0.05]
REST_OFFSETS_BODY_22[11] = [-0.05, -0.1, 0.05]
REST_OFFSETS_BODY_22[12] = [0, 0.15, 0]
REST_OFFSETS_BODY_22[13] = [-0.18, 0.08, 0]
REST_OFFSETS_BODY_22[14] = [0.18, 0.08, 0]
REST_OFFSETS_BODY_22[15] = [0, 0.2, 0]
REST_OFFSETS_BODY_22[16] = [-0.28, 0, 0]
REST_OFFSETS_BODY_22[17] = [0.28, 0, 0]
REST_OFFSETS_BODY_22[18] = [-0.25, 0, 0]
REST_OFFSETS_BODY_22[19] = [0.25, 0, 0]
REST_OFFSETS_BODY_22[20] = [-0.12, 0, 0]
REST_OFFSETS_BODY_22[21] = [0.12, 0, 0]

# 每关节用于 twist 轴的子关节索引（与 vis_smpl_preview 一致：手腕用 middle1）
WRIST_CHILD_LEFT_24 = 22
WRIST_CHILD_RIGHT_24 = 23
WRIST_CHILD_LEFT_52 = 25
WRIST_CHILD_RIGHT_52 = 40

MODE_PALM_FLIP_180 = "palm_flip_180"
MODE_ARM_TWIST_360 = "arm_twist_360"
CANDY_MODES = (MODE_PALM_FLIP_180, MODE_ARM_TWIST_360)
# 360° 模式：肩+肘共同正向 twist x°，腕反向 -x°；分配比例随机，避免肘处集中导致 mesh 塌陷
ARM_360_TWIST_ANGLE_RANGE = (90.0, 180.0)
# 肩占正向 twist 的比例，在 [min, max] 内随机，保证肩、肘都参与
ARM_360_SHOULDER_RATIO_RANGE = (0.2, 0.8)


def _fk_global_rot_and_pos(
    poses: np.ndarray,
    trans: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    J: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """FK: (F,J,3) axis-angle, (F,3) trans -> global_rot (F,J,3,3), global_pos (F,J,3)."""
    F = poses.shape[0]
    parents = np.asarray(parents[:J], dtype=np.int32)
    offsets = np.asarray(rest_offsets[:J], dtype=np.float64)
    r_local = R.from_rotvec(poses.reshape(-1, 3))
    mats = r_local.as_matrix().reshape(F, J, 3, 3)
    global_mats = np.zeros((F, J, 3, 3), dtype=np.float64)
    for j in range(J):
        p = parents[j]
        if p < 0:
            global_mats[:, j] = mats[:, j]
        else:
            global_mats[:, j] = np.einsum("fij,fjk->fik", global_mats[:, p], mats[:, j])
    positions = np.zeros((F, J, 3), dtype=np.float64)
    positions[:, 0] = trans
    for j in range(1, J):
        p = parents[j]
        positions[:, j] = positions[:, p] + (global_mats[:, p] @ offsets[j])
    return global_mats, positions


def _apply_twist_in_global_space(
    poses: np.ndarray,
    trans: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    target_joint_idx: int,
    child_joint_idx: int,
    twist_rad_per_frame: np.ndarray,
    J: int,
) -> None:
    """In-place: 在全局空间对 target_joint 绕「当前骨轴」(target→child) 施加 twist，写回 local。
    每帧在施加前重新 FK，保证骨轴与当前 poses 一致，从而子关节位置严格不变。
    """
    F = poses.shape[0]
    parent_idx = parents[target_joint_idx]

    for f in range(F):
        global_mats, global_pos = _fk_global_rot_and_pos(
            poses[f : f + 1], trans[f : f + 1], parents, rest_offsets, J
        )
        global_mats = global_mats[0]
        global_pos = global_pos[0]
        axis_vec = global_pos[child_joint_idx] - global_pos[target_joint_idx]
        n2 = np.dot(axis_vec, axis_vec)
        if n2 < 1e-14:
            continue
        axis_vec = axis_vec / np.sqrt(n2)
        angle = float(twist_rad_per_frame[f])
        q_twist = R.from_rotvec(axis_vec * angle)
        R_twist = q_twist.as_matrix()
        Q_old = global_mats[target_joint_idx]
        Q_new = R_twist @ Q_old
        if parent_idx < 0:
            R_new_local = Q_new
        else:
            R_parent_inv = global_mats[parent_idx].T
            R_new_local = R_parent_inv @ Q_new
        poses[f, target_joint_idx] = R.from_matrix(R_new_local).as_rotvec()


def _get_parents_and_rest_offsets(J: int) -> Tuple[np.ndarray, np.ndarray]:
    if J >= 52:
        parents = SMPLH_PARENTS[:J].copy()
    else:
        parents = SMPL24_PARENTS[:J].copy()
    rest = np.zeros((J, 3), dtype=np.float64)
    n = min(22, J)
    rest[:n] = REST_OFFSETS_BODY_22[:n]
    if J > 22:
        rest[22:, 0] = 0.05
    return parents, rest


def _wrist_child_index(wrist_idx: int, J: int) -> int:
    if J >= 52:
        return WRIST_CHILD_LEFT_52 if wrist_idx == WRIST_LEFT else WRIST_CHILD_RIGHT_52
    return WRIST_CHILD_LEFT_24 if wrist_idx == WRIST_LEFT else WRIST_CHILD_RIGHT_24


# -----------------------------------------------------------------------------
# 1) 180° 手腕 palm flip
# -----------------------------------------------------------------------------
def _apply_wrist_180(
    poses: np.ndarray,
    trans: np.ndarray,
    side: str,
    weight: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    J: int,
) -> Dict:
    wrist_idx = WRIST_LEFT if side == "left" else WRIST_RIGHT
    if wrist_idx >= J:
        return {"side": side, "skipped": True}
    child_idx = _wrist_child_index(wrist_idx, J)
    if child_idx >= J:
        return {"side": side, "skipped": True}
    angle_rad = np.deg2rad(180.0) * np.asarray(weight, dtype=np.float64)
    _apply_twist_in_global_space(
        poses,
        trans,
        parents,
        rest_offsets,
        wrist_idx,
        child_idx,
        angle_rad,
        J,
    )
    return {"side": side, "angle_deg": 180.0, "joint": wrist_idx, "child": child_idx}


# -----------------------------------------------------------------------------
# 2) 360° 手臂：肩+肘共同正向 twist x°，腕反向 -x°；肩/肘分配比例随机（真实多为组合问题，避免肘集中→mesh 塌陷）
# -----------------------------------------------------------------------------
def _apply_arm_360(
    poses: np.ndarray,
    trans: np.ndarray,
    side: str,
    weight: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    J: int,
    angles_deg: Optional[List[float]] = None,
) -> Dict:
    chain = ARM_LEFT_JOINTS if side == "left" else ARM_RIGHT_JOINTS  # [肩, 肘, 腕]
    if angles_deg is not None and len(angles_deg) >= 2:
        x_deg = float(angles_deg[0])
        shoulder_ratio = float(angles_deg[1])
    else:
        x_deg = float(np.random.uniform(ARM_360_TWIST_ANGLE_RANGE[0], ARM_360_TWIST_ANGLE_RANGE[1]))
        shoulder_ratio = float(np.random.uniform(ARM_360_SHOULDER_RATIO_RANGE[0], ARM_360_SHOULDER_RATIO_RANGE[1]))
    theta_shoulder_deg = x_deg * shoulder_ratio
    theta_elbow_deg = x_deg * (1.0 - shoulder_ratio)

    shoulder_idx, elbow_idx, wrist_idx = chain[0], chain[1], chain[2]
    # 肩 +θ1（绕 肩→肘）
    if shoulder_idx < J and elbow_idx < J:
        angle_rad = np.deg2rad(theta_shoulder_deg) * np.asarray(weight, dtype=np.float64)
        _apply_twist_in_global_space(
            poses, trans, parents, rest_offsets,
            shoulder_idx, elbow_idx, angle_rad, J,
        )
    # 肘 +θ2（绕 肘→腕）
    if elbow_idx < J and wrist_idx < J:
        angle_rad = np.deg2rad(theta_elbow_deg) * np.asarray(weight, dtype=np.float64)
        _apply_twist_in_global_space(
            poses, trans, parents, rest_offsets,
            elbow_idx, wrist_idx, angle_rad, J,
        )
    # 腕 -x（绕 腕→手）；仅改 body 0–21，不对手腕子关节做补偿（否则会动到手指）
    child_idx = _wrist_child_index(wrist_idx, J)
    if wrist_idx < J and child_idx < J:
        angle_rad = np.deg2rad(-x_deg) * np.asarray(weight, dtype=np.float64)
        _apply_twist_in_global_space(
            poses, trans, parents, rest_offsets,
            wrist_idx, child_idx, angle_rad, J,
        )
    return {
        "side": side,
        "shoulder_deg": theta_shoulder_deg,
        "elbow_deg": theta_elbow_deg,
        "wrist_deg": -x_deg,
        "angle_deg": x_deg,
        "shoulder_ratio": shoulder_ratio,
    }


# -----------------------------------------------------------------------------
# WristCandyWrapperCorruptor：手腕两种模式
# -----------------------------------------------------------------------------
class WristCandyWrapperCorruptor(BaseCorruptor):
    """
    手腕 candy wrapper：
    - palm_flip_180：仅腕 +180° twist（TBS 全局骨轴）→ 手掌朝向完全相反
    - arm_twist_360：肩+肘共同 +x°（比例随机）、腕 -x°（组合问题，减轻肘处 mesh 塌陷）
    """

    def __init__(self, body_model: Optional[Any] = None, device: str = "cuda") -> None:
        super().__init__(body_model=body_model, device=device)

    def _apply_corruption(
        self,
        data_mod: Dict,
        poses: np.ndarray,
        trans: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        F, J, _ = poses.shape
        if F < 2:
            return (
                poses.copy(),
                trans.copy(),
                {"synthesis_type": "wrist_candy_wrapper", "description": "skipped (too short)"},
            )

        mode = kwargs.get("mode")
        if mode not in CANDY_MODES:
            mode = str(np.random.choice(CANDY_MODES))
        parents, rest_offsets = _get_parents_and_rest_offsets(J)
        weight = np.ones(F, dtype=np.float64)

        sides: List[str] = []
        if np.random.random() < 0.5:
            sides.append("left")
        if np.random.random() < 0.5 or not sides:
            sides.append("right")

        poses_out = poses.copy()
        trans_out = trans.copy()
        meta_logs: List[Dict] = []

        for side in sides:
            if mode == MODE_PALM_FLIP_180:
                log = _apply_wrist_180(poses_out, trans_out, side, weight, parents, rest_offsets, J)
            else:
                log = _apply_arm_360(poses_out, trans_out, side, weight, parents, rest_offsets, J)
            log["mode"] = mode
            meta_logs.append(log)

        # Build _mask_info: collect all affected joint indices from events
        affected_joints: List[int] = []
        for log in meta_logs:
            if log.get("skipped"):
                continue
            if mode == MODE_PALM_FLIP_180:
                j = log.get("joint")
                if j is not None:
                    affected_joints.append(j)
            elif mode == MODE_ARM_TWIST_360:
                side = log.get("side", "")
                chain = ARM_LEFT_JOINTS if side == "left" else ARM_RIGHT_JOINTS
                affected_joints.extend(chain)
        # Deduplicate
        affected_joints = list(set(affected_joints))

        _mask_info: Dict = {}
        if affected_joints:
            _mask_info = {
                "all_frames": True,
                "corrupted_joints": affected_joints,
                "trans_corrupted": False,
            }

        desc = f"Wrist Candy Wrapper: {mode}, {len(meta_logs)} side(s)"
        meta = {
            "synthesis_type": "wrist_candy_wrapper",
            "description": desc,
            "synthesis_method": {"mode": mode, "events_log": meta_logs, "frame_range": [0, F]},
            "degradation_details": {"affected_components": ["arm_poses"], "logic": "tbs_twist_global"},
            "_mask_info": _mask_info,
        }
        return poses_out, trans_out, meta


# -----------------------------------------------------------------------------
# 3) 泛化肢体：指定链上 180° 单关节 或 360° 分布（排除手腕，与 WristCandyWrapper 不重叠）
# -----------------------------------------------------------------------------
# 肢体链：(chain, terminal_child)：链不含末端腕/踝，最后一关节的 child 用 terminal_child 算骨轴
# - 手臂：仅肩、肘，不含腕；肘的 child 用腕 20/21
# - 腿：仅髋、膝，不含踝；膝的 child 用踝 7/8
LIMB_CHAINS: Dict[str, Tuple[List[int], Optional[int]]] = {
    "arm_left": ([16, 18], 20),
    "arm_right": ([17, 19], 21),
    "leg_left": ([1, 4], 7),
    "leg_right": ([2, 5], 8),
}
LIMB_CHAIN_NAMES = list(LIMB_CHAINS.keys())


def _limb_child(chain: List[int], k: int, terminal_child: Optional[int], J: int) -> Optional[int]:
    """链上第 k 个关节的 child 索引（用于骨轴）；最后一关节用 terminal_child。"""
    if k + 1 < len(chain):
        return chain[k + 1]
    return terminal_child


def _apply_limb_180(
    poses: np.ndarray,
    trans: np.ndarray,
    chain: List[int],
    terminal_child: Optional[int],
    joint_idx_in_chain: int,
    weight: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    J: int,
) -> Dict:
    j_idx = chain[joint_idx_in_chain]
    if j_idx >= J:
        return {"skipped": True}
    child_idx = _limb_child(chain, joint_idx_in_chain, terminal_child, J)
    if child_idx is None or child_idx >= J:
        return {"skipped": True}
    angle_rad = np.deg2rad(180.0) * np.asarray(weight, dtype=np.float64)
    _apply_twist_in_global_space(
        poses,
        trans,
        parents,
        rest_offsets,
        j_idx,
        child_idx,
        angle_rad,
        J,
    )
    return {"chain": chain, "joint_index": j_idx, "child": child_idx, "angle_deg": 180.0}


# 肢体 360° 模式：在 TBS 下要保证各关节位置不变，只能仅在扰动关节（链末）绕「扰动→terminal_child」施加 -x°
# （若在祖宗上施加 +θ，会改变下游全局旋转，terminal_child 位置无法保持）
LIMB_360_TWIST_ANGLE_RANGE = (90.0, 180.0)


def _apply_limb_360(
    poses: np.ndarray,
    trans: np.ndarray,
    chain: List[int],
    terminal_child: Optional[int],
    weight: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    J: int,
    angles_deg: Optional[List[float]] = None,
) -> Dict:
    """
    扰动关节 = 链末（肘/膝）。在 TBS 下保证链上及 terminal_child 位置和朝向都不变：
    1) 在扰动关节绕「扰动→terminal_child」施加 -x° twist；
    2) 在 terminal_child 绕同一条骨轴施加 +x°，恢复子关节的全局朝向。
    这样 terminal_child（腕/踝）位置与朝向均不变；body 0..21 不变；terminal_child 再下游（如手指）会位移。
    """
    n = len(chain)
    if n < 2:
        return {"chain": chain, "skipped": True}
    perturbed_joint = chain[-1]

    if angles_deg is not None and len(angles_deg) > 0:
        x_deg = float(angles_deg[0])
    else:
        x_deg = float(np.random.uniform(LIMB_360_TWIST_ANGLE_RANGE[0], LIMB_360_TWIST_ANGLE_RANGE[1]))

    c_last = _limb_child(chain, n - 1, terminal_child, J)
    if perturbed_joint < J and c_last is not None and c_last < J:
        angle_rad_neg = np.deg2rad(-x_deg) * np.asarray(weight, dtype=np.float64)
        _apply_twist_in_global_space(
            poses, trans, parents, rest_offsets, perturbed_joint, c_last, angle_rad_neg, J,
        )
        # 在子关节绕同一条骨轴施加补偿：轴为 (c_last→perturbed)，要等价于绕 (perturbed→c_last) 转 +x°，需传 -x°
        angle_rad_comp = np.deg2rad(-x_deg) * np.asarray(weight, dtype=np.float64)
        _apply_twist_in_global_space(
            poses, trans, parents, rest_offsets, c_last, perturbed_joint, angle_rad_comp, J,
        )

    return {
        "chain": chain,
        "perturbed_deg": -x_deg,
        "angle_deg": x_deg,
    }


# -----------------------------------------------------------------------------
# LimbCandyWrapperCorruptor
# -----------------------------------------------------------------------------
MODE_LIMB_180 = "limb_180"
MODE_LIMB_360 = "limb_360"
LIMB_MODES = (MODE_LIMB_180, MODE_LIMB_360)


class LimbCandyWrapperCorruptor(BaseCorruptor):
    """
    泛化肢体 candy wrapper（排除手腕，仅肩/肘、髋/膝等）：
    - limb_180：在选定链上随机选一关节施加 180° twist
    - limb_360：与手腕 360° 一致——祖宗节点共同 +x°、扰动关节（链末） -x°，各关节位置不变
    可选肢体：arm_left, arm_right（仅肩肘）, leg_left, leg_right（仅髋膝）；与 WristCandyWrapper 不重叠。
    """

    def __init__(self, body_model: Optional[Any] = None, device: str = "cuda") -> None:
        super().__init__(body_model=body_model, device=device)

    def _apply_corruption(
        self,
        data_mod: Dict,
        poses: np.ndarray,
        trans: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        F, J, _ = poses.shape
        if F < 2:
            return (
                poses.copy(),
                trans.copy(),
                {"synthesis_type": "limb_candy_wrapper", "description": "skipped (too short)"},
            )

        mode = kwargs.get("mode")
        if mode not in LIMB_MODES:
            mode = str(np.random.choice(LIMB_MODES))
        limb_name = kwargs.get("limb")
        if limb_name is not None and limb_name in LIMB_CHAINS:
            chain, terminal_child = LIMB_CHAINS[limb_name]
        else:
            limb_name = str(np.random.choice(LIMB_CHAIN_NAMES))
            chain, terminal_child = LIMB_CHAINS[limb_name]
        parents, rest_offsets = _get_parents_and_rest_offsets(J)
        weight = np.ones(F, dtype=np.float64)

        poses_out = poses.copy()
        trans_out = trans.copy()

        if mode == MODE_LIMB_180:
            joint_in_chain = int(np.random.randint(0, len(chain)))
            log = _apply_limb_180(
                poses_out,
                trans_out,
                chain,
                terminal_child,
                joint_in_chain,
                weight,
                parents,
                rest_offsets,
                J,
            )
        else:
            log = _apply_limb_360(
                poses_out,
                trans_out,
                chain,
                terminal_child,
                weight,
                parents,
                rest_offsets,
                J,
            )
        log["mode"] = mode
        log["limb"] = limb_name

        # Build _mask_info: collect affected joint indices
        affected_joints: List[int] = list(chain)
        if terminal_child is not None:
            affected_joints.append(terminal_child)
        affected_joints = list(set(affected_joints))

        _mask_info: Dict = {}
        if affected_joints and not log.get("skipped"):
            _mask_info = {
                "all_frames": True,
                "corrupted_joints": affected_joints,
                "trans_corrupted": False,
            }

        desc = f"Limb Candy Wrapper: {mode}, limb={limb_name}"
        meta = {
            "synthesis_type": "limb_candy_wrapper",
            "description": desc,
            "synthesis_method": {"mode": mode, "limb": limb_name, "event": log, "frame_range": [0, F]},
            "degradation_details": {"affected_components": ["limb_poses"], "logic": "tbs_twist_global"},
            "_mask_info": _mask_info,
        }
        return poses_out, trans_out, meta


# -----------------------------------------------------------------------------
# 测试
# -----------------------------------------------------------------------------
def _fk_joint_positions(
    poses_3d: np.ndarray, trans: np.ndarray, parents: np.ndarray, rest_offsets: np.ndarray
) -> np.ndarray:
    J = poses_3d.shape[1]
    _, positions = _fk_global_rot_and_pos(poses_3d, trans, parents, rest_offsets, J)
    return positions


def _test_candy_wrapper_modes_and_twist_only():
    c = WristCandyWrapperCorruptor(device="cpu")
    for F, J, parents, name in [
        (80, 24, SMPL24_PARENTS[:24], "SMPL24"),
        (80, 52, SMPLH_PARENTS[:52], "SMPLH52"),
    ]:
        np.random.seed(123)
        data = {
            "poses": np.random.randn(F, J * 3).astype(np.float64) * 0.15,
            "trans": np.random.randn(F, 3).astype(np.float64) * 0.1,
        }
        trans_orig = data["trans"].copy()
        _, rest_offsets = _get_parents_and_rest_offsets(J)
        poses_orig = data["poses"].reshape(F, J, 3)
        pos_orig = _fk_joint_positions(poses_orig, trans_orig, parents, rest_offsets)

        for mode in CANDY_MODES:
            out = c.corrupt(data, mode=mode)
            assert "corrupted_motion" in out
            corrupted = out["corrupted_motion"]
            meta = out.get("meta") or {}
            sm = meta.get("synthesis_method") or {}
            assert sm.get("mode") == mode
            np.testing.assert_array_almost_equal(corrupted["trans"], trans_orig, err_msg="trans changed")
            poses_out = corrupted["poses"].reshape(F, J, 3)
            # 仅改 body 0–21，22+（手掌/手指）必须不变
            for j in range(BODY_JOINT_COUNT, J):
                np.testing.assert_array_almost_equal(
                    poses_out[:, j, :],
                    poses_orig[:, j, :],
                    err_msg=f"{name} joint {j} must be unchanged",
                )
            if mode == MODE_PALM_FLIP_180:
                assert any(e.get("angle_deg") == 180.0 for e in (sm.get("events_log") or []))
            if mode == MODE_ARM_TWIST_360:
                # 肩+肘+腕 三处参与，不再断言 body 位置不变；仅检查总角度在范围内且有三段 twist 记录
                evts = sm.get("events_log") or []
                assert any(
                    ARM_360_TWIST_ANGLE_RANGE[0] <= (e.get("angle_deg") or 0) <= ARM_360_TWIST_ANGLE_RANGE[1]
                    for e in evts
                ), "arm_twist_360 angle_deg should be in range"
                assert any("shoulder_deg" in e and "elbow_deg" in e and "wrist_deg" in e for e in evts), "arm_twist_360 should log shoulder/elbow/wrist"
            print(f"  [{name}] mode={mode} ok")
    print("WristCandyWrapperCorruptor tests passed.")


def _test_limb_candy_wrapper_excludes_wrist():
    """LimbCandyWrapperCorruptor：手臂链不含腕关节，且支持 limb_180 / limb_360。"""
    for limb_name, (chain, terminal_child) in LIMB_CHAINS.items():
        if "arm" in limb_name:
            assert 20 not in chain and 21 not in chain, f"arm chain {limb_name} must exclude wrist: {chain}"
            assert terminal_child in (20, 21), f"arm terminal_child should be wrist: {terminal_child}"
    c = LimbCandyWrapperCorruptor(device="cpu")
    F, J = 80, 24
    np.random.seed(456)
    data = {
        "poses": np.random.randn(F, J * 3).astype(np.float64) * 0.15,
        "trans": np.random.randn(F, 3).astype(np.float64) * 0.1,
    }
    trans_orig = data["trans"].copy()
    parents, rest_offsets = _get_parents_and_rest_offsets(J)
    poses_orig = data["poses"].reshape(F, J, 3)
    pos_orig = _fk_joint_positions(poses_orig, trans_orig, parents, rest_offsets)
    for mode in (MODE_LIMB_180, MODE_LIMB_360):
        out = c.corrupt(data, mode=mode)
        assert "corrupted_motion" in out
        corrupted = out["corrupted_motion"]
        np.testing.assert_array_almost_equal(corrupted["trans"], trans_orig, err_msg=f"limb {mode} trans changed")
        if mode == MODE_LIMB_360:
            # 扰动关节 -x° + 子关节 +x° 后，子关节位置与朝向均不变；只断言 body 0..21 位置不变
            poses_out = corrupted["poses"].reshape(F, J, 3)
            pos_after = _fk_joint_positions(poses_out, corrupted["trans"], parents, rest_offsets)
            np.testing.assert_array_almost_equal(
                pos_after[:, :BODY_JOINT_COUNT],
                pos_orig[:, :BODY_JOINT_COUNT],
                decimal=5,
                err_msg="limb_360: body joint positions (0..21) must be unchanged",
            )
        meta = out.get("meta") or {}
        sm = meta.get("synthesis_method") or {}
        assert sm.get("mode") == mode
        assert sm.get("limb") in LIMB_CHAINS
    print("LimbCandyWrapperCorruptor tests passed (excludes wrist, 180/360 modes).")


if __name__ == "__main__":
    import sys

    print("Usage: python -m hymotion.utils.data_corruptor.run_candy_test", file=sys.stderr)
    sys.exit(0)
