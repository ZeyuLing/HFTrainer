"""
物理误差指标在线计算模块

支持格式：
  - NPZ（SMPL/SMPLH/SMPLX poses + betas + trans）
  - H5（自定义 HDF 格式，含预计算 joints3d/rot_mats）

计算分为两级：
  第一级（仅需关节数据）：jerk, joint_pop, wrist_twist, velocity, acceleration, bone_length 等
    → 可直接使用 HDF 中预计算的 joints3d，跳过 BodyModel
  第二级（需要顶点数据）：penetration, floating, skating 等
    → 统一走 BodyModel 重新计算 verts + joints，确保数据一致性

单位系统：
  - 输入单位通过骨骼长度自动推断（mm/cm/dm/m），内部归一化到米进行计算
  - 输出单位通过 output_unit 参数指定，默认 "cm"
  - 返回结果中包含 length_unit 字段标明当前单位

物理指标清单（字段名不含单位后缀，实际单位由 length_unit 决定）：
  基础统计：
    - motion_duration_sec: 动画时长 (秒)
    - total_distance: 根节点总移动距离
  时间平滑性（关节级）：
    - jerk_with_rot: pelvis-local 空间（含旋转）j=1..21 关节三阶导数 (m/s^3)
    - local_pose_jerk: pelvis 坐标系（不含旋转）j=1..21 关节三阶导数 (m/s^3)
    - pelvis_rot_jerk: pelvis 旋转角 jerk (deg/s^3)
    - pelvis_trans_jerk: pelvis 平移轨迹三阶导数 (m/s^3)
    - avg_velocity / max_velocity: 关节速度 (length_unit/s)
    - avg_acceleration / max_acceleration: 关节加速度 (length_unit/s^2)
  骨骼一致性：
    - bone_length_cv_mean / bone_length_cv_max: 骨骼长度变异系数 (%)
  关节跳变：
    - joint_pop_ratio / arms/legs/wrists/ankles_pop_ratio: 关节跳变率 (%)
    - wrist_twist_ratio: 手腕扭曲率 (%)
  孤立跳变（与 tremor 共用 reversal 底层函数，不同阈值组）：
    - snap_ratio / snap_ratio_pos / snap_ratio_rot: 孤立大幅跳变帧占比 (%)
  地面交互（顶点级，分前后脚掌 support 检测）：
    - avg_penetrate: 平均穿透深度
    - avg_float: 平均浮空高度
    - avg_skate: 平均滑动距离（仅超阈值帧）
    - frame_avg_skate: 帧平均滑动距离
    - skate_ratio: 滑动帧占比 (%)
    - phys_err: 综合物理误差 = penetrate + float + skate
"""

import os
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 尝试把 t2m_database 目录加到 sys.path，以便 body_model 子包可用
# 兼容从 hftrainer 仓库和独立 t2m_database 两种位置使用
_T2M_DB_CANDIDATES = [
    os.path.join(os.path.dirname(_SCRIPT_DIR), '..', '..', '..', 'motion_annot_web', 'm2m_database'),
    os.path.join(os.path.dirname(_SCRIPT_DIR), '..', '..', '..', 't2m_database'),
]
for _cand in _T2M_DB_CANDIDATES:
    _cand = os.path.abspath(_cand)
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
from typing import Any, Dict, List, Tuple, Optional
from loguru import logger

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 默认计算设备：有 CUDA 时用 GPU，否则回退到 CPU
_DEFAULT_DEVICE: str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# BodyModel 动态导入（延迟，避免启动时报错）
_BODY_MODEL_CLASS = None

# 结果缓存：{file_path: {"metrics": dict, "mtime": float}}
PHYS_METRICS_CACHE: Dict[str, Dict] = {}

# ==============================================================================
# [SMPL 耦合] 从 body_model.smpl_skeleton 导入骨骼常量（失败时降级为内联常量）
# 关节索引、骨骼拓扑、脚底顶点等均为 SMPL 特有定义。
# ==============================================================================

try:
    from body_model.smpl_skeleton import (
        PELVIS_IDX,
        L_HIP_IDX, R_HIP_IDX, L_KNEE_IDX, R_KNEE_IDX,
        L_ANKLE_IDX, R_ANKLE_IDX, L_FOOT_IDX, R_FOOT_IDX,
        L_SHOULDER_IDX, R_SHOULDER_IDX, L_ELBOW_IDX, R_ELBOW_IDX,
        L_WRIST_IDX, R_WRIST_IDX,
        BODY_BONE_PAIRS,
        ARMS_JTR_IDS, LEGS_JTR_IDS, WRISTS_JTR_IDS, ANKLES_JTR_IDS,
        infer_smpl_type,
        build_joint_reorder_map, reorder_joints,
        SMPLH_JOINTS,
    )
    from body_model.smpl_skeleton import (
        BODY_JOINT_COUNT, SMPLX_PARENT, SMPLX_JOINTS,
        SMPLH_PARENT,
    )
except Exception as _skel_err:
    logger.warning(f"body_model 导入失败，使用内联降级常量: "
                   f"{type(_skel_err).__name__}: {_skel_err}")
    PELVIS_IDX = 0
    L_HIP_IDX, R_HIP_IDX = 1, 2
    L_KNEE_IDX, R_KNEE_IDX = 4, 5
    L_ANKLE_IDX, R_ANKLE_IDX = 7, 8
    L_FOOT_IDX, R_FOOT_IDX = 10, 11
    L_SHOULDER_IDX, R_SHOULDER_IDX = 16, 17
    L_ELBOW_IDX, R_ELBOW_IDX = 18, 19
    L_WRIST_IDX, R_WRIST_IDX = 20, 21
    BODY_BONE_PAIRS = [
        (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (4, 7), (5, 8),
        (7, 10), (8, 11), (3, 6), (6, 9), (9, 12), (9, 13), (9, 14),
        (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
    ]
    ARMS_JTR_IDS = [16, 17, 18, 19]
    LEGS_JTR_IDS = [1, 2, 4, 5]
    WRISTS_JTR_IDS = [20, 21]
    ANKLES_JTR_IDS = [7, 8]
    SMPLH_JOINTS = None
    BODY_JOINT_COUNT = 22
    SMPLX_PARENT = None
    SMPLX_JOINTS = None
    SMPLH_PARENT = None

    def _fallback_infer_smpl_type(poses):
        """内联降级 infer_smpl_type：根据 poses shape 推断 SMPL 类型。"""
        if poses.ndim == 2:
            total_dim = poses.shape[1]
            if total_dim == 72:    # 24*3 = SMPL
                return poses.reshape(-1, 24, 3), "smpl"
            elif total_dim == 156:  # 52*3 = SMPLH
                return poses.reshape(-1, 52, 3), "smplh"
            elif total_dim == 165:  # 55*3 = SMPLX
                return poses.reshape(-1, 55, 3), "smplx"
            else:
                n_joints = total_dim // 3
                return poses.reshape(-1, n_joints, 3), "smplh"
        elif poses.ndim == 3:
            n_joints = poses.shape[1]
            if n_joints <= 24:
                return poses, "smpl"
            elif n_joints <= 52:
                return poses, "smplh"
            else:
                return poses, "smplx"
        return poses, "smplh"

    infer_smpl_type = _fallback_infer_smpl_type
    build_joint_reorder_map = None
    reorder_joints = None

# foot_regions 独立导入（不依赖 smpl_skeleton）
try:
    from body_model.foot_regions import select_foot_regions
except Exception:
    select_foot_regions = None

# ==============================================================================
# 从 body_model.tbs_axes 导入 TBS 分解工具（失败时降级禁用）
# ==============================================================================

try:
    from body_model.tbs_axes import BodyTBSLayer
    from body_model.rotation_utils import rvecs_to_mats_torch

    HAS_TBS = True
except Exception as _tbs_err:
    logger.warning(f"body_model.tbs_axes 导入失败，TBS 异常关节检测将不可用: "
                   f"{type(_tbs_err).__name__}: {_tbs_err}")
    HAS_TBS = False

# ==============================================================================
# 从 body_model.mesh_distortion 导入 mesh 变形指标（失败时降级禁用）
# ==============================================================================

try:
    from body_model.mesh_distortion import (
        mesh_distort_arap,
        mesh_distort_edge_stretch,
        mesh_distort_volume,
        mesh_distort_symmetric_dirichlet,
        mesh_distort_all,
        precompute_rest_basis,
        _faces_to_unique_edges,
        get_face_joint_assignment,
    )

    HAS_MESH_DISTORTION = True
except Exception as _md_err:
    logger.warning(f"body_model.mesh_distortion 导入失败，mesh distortion 检测将不可用: "
                   f"{type(_md_err).__name__}: {_md_err}")
    HAS_MESH_DISTORTION = False

# ==============================================================================
# [SMPL 耦合] TBS 关节 ROM (Range of Motion) 阈值表
# ==============================================================================
# 每关节 6 元组：(twist_min, twist_max, bend_min, bend_max, spread_min, spread_max)
# 单位：度。基于解剖学极限 + 容差，用于检测明显不合理的关节角度。
# TBS 各轴的正方向含义参见 body_model/tbs_axes.py 文件头的含义表格。
# TBS 三轴构成右手系 (det(R_p_a) = +1)，+bend/+twist 输出与物理方向一致。
#
# ⚠️ SMPL 耦合说明：
#   这些阈值针对 SMPL T-pose（双臂水平伸展、手掌朝下）设计。
#   - 膝/肘的 twist/spread 在解剖学上极小，但 SMPL 骨骼非纯铰链，
#     实际动作数据中常出现非零值，因此 ROM 留了 buffer（如 knee twist ±60°）。
#   - 肩关节 spread 允许到 135°，是为了覆盖 SMPL 中"双手过头顶"等极限动作。
#   - A-pose 或其他 rest pose 的模型：各轴的 0° 参考点不同，ROM 需重新标定。
#
# 分解约定：
#   大多数关节: XYZ  R = Rx(twist)·Ry(bend)·Rz(spread), bend 为中间轴
#   膝/肘关节: XZY  R = Rx(twist)·Rz(spread)·Ry(bend), spread 为中间轴
# 膝/肘改用 XZY 后，spread（≈0°）为中间轴，彻底消除 bend 接近 ±90° 时的万向节锁。
# yapf: disable
DEFAULT_TBS_ROM_LIMITS = {
    # --- 中线关节（+twist=左旋, +bend=前屈, +spread=右侧屈）XZY分解,spread为中间轴 --- [通用解剖学]
    "Pelvis":     (-80,  80,  -70,  70,  -50,  50),
    "Spine1":     (-60,  60,  -70,  70,  -50,  50),       # bend放宽至±70°(弯腰/举重); spread放宽至±50°
    "Spine2":     (-60,  60,  -70,  70,  -50,  50),
    "Spine3":     (-60,  60,  -70,  70,  -50,  50),
    "Neck":       (-75,  75,  -70,  75,  -55,  55),       # +bend=低头
    "Head":       (-65,  65,  -55,  65,  -50,  50),       # +bend=点头
    # --- 髋关节（+twist=内旋, +bend=伸髋/后摆, +spread=内收）--- [通用解剖学]
    "L_Hip":      (-75,  75, -135,  90,  -90,  90),       # YXZ分解; bend=屈髋/竖叉; spread=横叉
    "R_Hip":      (-75,  75, -135,  90,  -90,  90),
    # --- 膝关节（+twist=内旋, +bend=屈膝, twist/spread 极小）--- [SMPL 耦合: buffer 放大]
    "L_Knee":     (-60,  60,  -25, 165,  -40,  40),       # SMPL非纯铰链→twist/spread放宽; XZY分解
    "R_Knee":     (-60,  60,  -25, 165,  -40,  40),
    # --- 踝关节（+twist=外翻, +bend=跖屈/踮脚, +spread=外旋）---
    "L_Ankle":    (-60,  60,  -55,  75,  -50,  50),
    "R_Ankle":    (-60,  60,  -55,  75,  -50,  50),
    # --- 足部（twist=Pelvis spread轴: +twist=内翻, bend=Ankle bend轴: +bend=趾屈, spread=cross）---
    "L_Foot":     (-80,  80,  -75,  55,  -80,  80),       # twist轴改为Pelvis spread后角度分布更宽
    "R_Foot":     (-80,  80,  -75,  55,  -80,  80),
    # --- 锁骨（+twist=内旋, +bend=上提, +spread=后缩）---
    "L_Collar":   (-70,  70,  -55,  55,  -60,  65),
    "R_Collar":   (-70,  70,  -55,  55,  -60,  65),
    # --- 肩关节（+twist=内旋, +bend=外展/抬臂, +spread=伸肩/后摆）YXZ分解,twist为中间轴 --- [SMPL 耦合: spread 放大]
    "L_Shoulder": (-75,   75, -135, 135, -100, 135),      # YXZ分解; spread放宽覆盖SMPL双手过头
    "R_Shoulder": (-75,   75, -135, 135, -100, 135),
    # --- 肘关节（+twist=内旋, +bend=屈肘, SMPL非纯铰链）--- [SMPL 耦合: buffer 放大]
    "L_Elbow":    (-110, 110,  -25, 175,  -45,  45),      # SMPL非纯铰链→spread放宽; XZY分解
    "R_Elbow":    (-110, 110,  -25, 175,  -45,  45),
    # --- 手腕（+twist=旋前, +bend=桡偏, +spread=伸腕）---
    "L_Wrist":    (-110, 110,  -55,  55,  -95,  95),
    "R_Wrist":    (-110, 110,  -55,  55,  -95,  95),
}
# yapf: enable

# ==============================================================================
# 单位系统
# ==============================================================================

UNIT_TO_METERS = {"mm": 0.001, "cm": 0.01, "dm": 0.1, "m": 1.0}
METERS_TO_UNIT = {v: k for k, v in UNIT_TO_METERS.items()}
HUMAN_FEMUR_RANGE_M = (0.25, 0.65)


def infer_input_unit(joints_first_frame: np.ndarray) -> str:
    """
    从关节位置推断输入数据的长度单位。

    使用左右股骨长度取均值，与人体参考值比对，判断 mm/cm/dm/m。
    """
    l_femur = float(np.linalg.norm(
        joints_first_frame[L_KNEE_IDX] - joints_first_frame[L_HIP_IDX]
    ))
    r_femur = float(np.linalg.norm(
        joints_first_frame[R_KNEE_IDX] - joints_first_frame[R_HIP_IDX]
    ))
    femur_len = (l_femur + r_femur) / 2.0

    for unit in ("m", "dm", "cm", "mm"):
        scale = UNIT_TO_METERS[unit]
        length_m = femur_len * scale
        if HUMAN_FEMUR_RANGE_M[0] <= length_m <= HUMAN_FEMUR_RANGE_M[1]:
            return unit
    raise ValueError(
        f"无法推断输入单位: 股骨长度={femur_len:.4f}，"
        f"对应米制范围不在 {HUMAN_FEMUR_RANGE_M} 内（任何已知单位）"
    )


def _unit_scale(from_unit: str, to_unit: str) -> float:
    """返回从 from_unit 到 to_unit 的乘法因子。"""
    return UNIT_TO_METERS[from_unit] / UNIT_TO_METERS[to_unit]


# 所有关节 ID 统一使用 Jtr 空间索引（BodyModel 输出顺序）。
# 在内部需要索引 joint_rot = poses[:, 1:22, :] 时，代码自动做 -1 偏移。
# 阈值统一使用米制（m, m/s），内部自动处理帧率和单位转换。
DEFAULT_PHYS_PARAMS = {
    "up_axis": 1,                          # [通用] Y 轴朝上
    "floor_mode": "first_n_seconds",       # [通用] 地面高度估算策略
    "floor_height_value": None,
    "floor_first_seconds": 2.0,
    "floor_n_lowest_per_frame": 10,
    "floor_cluster_gap": 0.05,
    "floor_cluster_min_seconds": 0.5,
    "floor_first_n_frames": 5,
    # [SMPL 耦合] 脚部区域参数：关节 ID 为 SMPL 骨骼拓扑特有
    "left_heel_joint_ids": [L_ANKLE_IDX],
    "left_forefoot_joint_ids": [L_FOOT_IDX],
    "right_heel_joint_ids": [R_ANKLE_IDX],
    "right_forefoot_joint_ids": [R_FOOT_IDX],
    "left_heel_vertex_ids": None,
    "left_forefoot_vertex_ids": None,
    "right_heel_vertex_ids": None,
    "right_forefoot_vertex_ids": None,
    "k_nearest": 200,
    "sole_keep_percentile": 55.0,
    # [通用] 地面交互阈值（米制），不依赖特定骨骼模型
    "heel_contact_thresh": 0.03,
    "forefoot_contact_thresh": 0.01,
    "below_tol": 0.01,
    "vertical_vel_thresh": 0.3,
    "skate_threshold": 0.3,
    "support_min_duration_sec": 0.1,
    # [通用] 关节跳变阈值（度/帧）
    "ang_pop_thresh": 20.0,
    "ang_pop_thresh_per_joint": {},
    # [SMPL 耦合] 关节分组 ID 列表：SMPL 骨骼拓扑特有
    "arms_joint_ids": ARMS_JTR_IDS,
    "legs_joint_ids": LEGS_JTR_IDS,
    "wrists_joint_ids": WRISTS_JTR_IDS,
    "ankles_joint_ids": ANKLES_JTR_IDS,
    "wrist_twist_threshold": 120.0,
    "chunk_size": 256,
    # 震颤 (tremor) 检测参数
    "tremor_window_sec": 0.5,
    "tremor_min_reversals": 3,
    "tremor_max_half_cycle_sec": 0.1,
    "tremor_max_path_efficiency": 0.5,
    "tremor_min_vel_component": 0.05,
    "tremor_min_angular_vel_component": 0.5,
    # swing 振幅上限：区分真正的小幅震颤与大幅合理运动（翻滚、快跑）的方向变化。
    # 真实 tremor swing 振幅约 1~5mm / 0.01~0.03 rad；
    # 翻滚等快速运动 swing 振幅 100~1000mm / 0.5+ rad。
    "tremor_max_swing_amplitude_m": 0.02,       # 位置空间 20mm
    "tremor_max_swing_amplitude_rad": 0.02,     # 旋转空间 ~1.1°
    # 孤立跳变 (snap) 检测参数
    # snap 与 tremor 共用 _detect_axis_reversals 底层函数，但使用不同阈值组：
    #   tremor = 小振幅 + 多反转 + 低路径效率（持续微小抖动）
    #   snap   = 大振幅 + 少反转（孤立的大幅跳变，跳出去+回来）
    # snap 的 swing 时长用秒定义，自动适配不同帧率（30fps下1-3帧，60fps下2-6帧）。
    #
    # Grid search 结论（2026-03-24, 13 个 mocap 测试数据）：
    #   - min_amp_rad 从 0.05 提高到 0.20（~11.5°）大幅降低快速运动假阳性
    #   - max_reversals 从 2 降到 1 强化孤立性条件
    #   - max_path_efficiency 从 0.9 降到 0.7 排除路径效率高的合法运动
    #   - min_amp_m 从 0.01 提高到 0.04 减少位置空间假阳性
    #   调参后 normal_avg 从 ~65% 降至 ~46%；仍偏高是因为 mocap 数据中缺乏
    #   典型 snap 样本（V2M 逐帧方法产生的孤立帧异常），对生成数据应更有区分度。
    "snap_window_sec": 0.5,                     # 滑动窗口长度（与 tremor 相同）
    "snap_max_half_cycle_sec": 0.15,            # 每个 swing ≤ 150ms
    "snap_min_swing_amplitude_m": 0.04,         # 位置空间 ≥ 40mm（排除小幅合法运动）
    "snap_max_swing_amplitude_m": 0.5,          # ≤ 500mm（超过的是合法大动作）
    "snap_min_swing_amplitude_rad": 0.20,       # 旋转空间 ≥ ~11.5°（排除常规关节运动）
    "snap_max_swing_amplitude_rad": 1.0,        # ≤ ~57°
    "snap_min_vel_component": 0.1,              # deadzone 比 tremor 更大（m/s）
    "snap_min_angular_vel_component": 1.0,      # 同上（rad/s）
    "snap_max_reversals_in_window": 1,          # 窗口内仅允许 1 次反转（严格孤立性）
    "snap_max_path_efficiency": 0.7,            # 排除路径效率高的持续快速运动
    # 头部稳定性 (head stability) 检测参数
    # 头部是视觉最敏感的关节，使用比 jerk 更严格的差异化阈值。
    # 分运动/静止阶段设置不同上限。
    #
    # Grid search 结论（2026-03-24）：
    #   - moving 从 120 提高到 300 避免翻滚/快跑等合法高角速度动作误报
    #   - static 从 15 提高到 40 减少站立微动假阳性
    #   - pelvis_vel_thresh 从 0.1 降到 0.05 更精确区分静止/运动阶段
    "head_joint_idx": 15,                       # HEAD_IDX (Jtr 空间)
    "head_ang_vel_warn_moving": 300.0,          # deg/s，运动时头部角速度警告阈值
    "head_ang_vel_warn_static": 40.0,           # deg/s，静止时头部角速度警告阈值
    "head_static_pelvis_vel_thresh": 0.05,      # m/s，判断静止的 pelvis 速度阈值
    # 自穿透 (self-penetration) 检测参数
    #
    # Grid search 结论（2026-03-24）：
    #   - tolerance 从 1cm 提高到 2cm，减少双腿/双臂自然靠近的假阳性
    #   - hands_torso_only 策略在 mocap 数据上假阳性极低（normal 2.5%），
    #     但 None (all) + 3cm 容差也可接受（normal 40%，但 known/normal 有正向 separation）
    #   - 对生成数据（常见手穿躯干），hands_torso_only 应该最有效
    "self_penetration_mode": "capsule",          # "capsule" (快速) | "mesh" (精确，未来实现)
    "self_penetration_tolerance": 0.02,          # 2cm 容差，减少合法近接触假阳性
    "self_penetration_parts_of_interest": None,  # None=检测所有预定义骨骼对
    # 接触闪烁 (contact flicker) 检测参数
    # 检测 debounce 前 raw support 序列中短于 min_run_sec 的段。
    # Grid search 结论：0.067s（2帧@30fps）只标记真正的单帧闪烁，假阳性极低。
    "contact_flicker_min_run_sec": 0.067,        # 低于此时长的接触/非接触段视为闪烁
}

# SMPL 家族模型统一存放路径：
#   /apdcephfs_cq11/share_1467498/home/dkang/assets/smpl_models/smpl_family_models/
#     smpl/SMPL_NEUTRAL.npz          (6890 verts, 24 joints)
#     smplh/neutral/model.npz        (6890 verts, 52 joints)
#     smplx/SMPLX_NEUTRAL.npz        (10475 verts, 55 joints)
DEFAULT_SMPLH_MODEL_PATH = (
    "/apdcephfs_cq11/share_1467498/home/dkang/codes/"
    "MoreDiff-Data/motion_process/body_model/smplh/neutral/model.npz"
)


# ==============================================================================
# Part A：旋转数学工具函数（内联自 articulate/math/）
# [通用算法] 纯数学工具，不依赖任何骨骼模型。
# ==============================================================================


def _normalize_tensor(x: "torch.Tensor", dim: int = -1, return_norm: bool = False):
    norm = x.norm(dim=dim, keepdim=True).clip(min=1e-8)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)


def _vector_cross_matrix(x: "torch.Tensor") -> "torch.Tensor":
    x = x.view(-1, 3)
    zeros = torch.zeros(x.shape[0], device=x.device)
    return torch.stack(
        (zeros, -x[:, 2], x[:, 1], x[:, 2], zeros, -x[:, 0], -x[:, 1], x[:, 0], zeros),
        dim=1,
    ).view(-1, 3, 3)


def _axis_angle_to_rotation_matrix(a: "torch.Tensor") -> "torch.Tensor":
    axis, angle = _normalize_tensor(a.reshape(-1, 3), return_norm=True)
    axis[torch.isnan(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c = angle.cos().view(-1, 1, 1)
    s = angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * _vector_cross_matrix(axis)
    return r


def _rotation_matrix_2_angle(rot_mat: "torch.Tensor") -> "torch.Tensor":
    batch_size = rot_mat.size(0)
    rot_mat = rot_mat.view(batch_size, 3, 3)
    cos_theta = 0.5 * (rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2] - 1)
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    return theta.unsqueeze(1)


def _angle_between(rot1: "torch.Tensor", rot2: "torch.Tensor") -> "torch.Tensor":
    rot1 = rot1.reshape(-1, 3, 3)
    rot2 = rot2.reshape(-1, 3, 3)
    offsets = rot1.transpose(1, 2).bmm(rot2)
    return _rotation_matrix_2_angle(offsets)


# ==============================================================================
# [SMPL 耦合] Part B：加载动画数据 + BodyModel 前向计算
# 本节所有函数均针对 SMPL 系列模型（SMPL/SMPLH/SMPLX），包括模型加载、
# 姿态参数解析、前向运动学计算等。
# ==============================================================================


def _get_body_model_class():
    """延迟加载 BodyModel 类（兼容包内相对导入和直接运行）"""
    global _BODY_MODEL_CLASS
    if _BODY_MODEL_CLASS is not None:
        return _BODY_MODEL_CLASS

    try:
        from .body_model.body_model import BodyModel

        _BODY_MODEL_CLASS = BodyModel
        return _BODY_MODEL_CLASS
    except ImportError:
        pass

    import importlib.util
    import types

    this_dir = os.path.dirname(os.path.abspath(__file__))
    bm_dir = os.path.join(this_dir, "body_model")

    # 1) Load lbs.py first
    lbs_path = os.path.join(bm_dir, "lbs.py")
    lbs_spec = importlib.util.spec_from_file_location("body_model.lbs", lbs_path)
    lbs_mod = importlib.util.module_from_spec(lbs_spec)
    lbs_spec.loader.exec_module(lbs_mod)

    # 2) Create package stubs so `from body_model.lbs import ...` and
    #    `from motion_process.human_body_prior.body_model.lbs import ...` both work.
    bm_pkg = types.ModuleType("body_model")
    bm_pkg.__path__ = [bm_dir]
    bm_pkg.lbs = lbs_mod
    sys.modules["body_model"] = bm_pkg
    sys.modules["body_model.lbs"] = lbs_mod

    # Alias for: from motion_process.human_body_prior.body_model.lbs import lbs
    mp_pkg = types.ModuleType("motion_process")
    mp_pkg.__path__ = []
    hbp_pkg = types.ModuleType("motion_process.human_body_prior")
    hbp_pkg.__path__ = []
    hbp_bm_pkg = types.ModuleType("motion_process.human_body_prior.body_model")
    hbp_bm_pkg.__path__ = [bm_dir]
    hbp_bm_pkg.lbs = lbs_mod
    sys.modules.setdefault("motion_process", mp_pkg)
    sys.modules.setdefault("motion_process.human_body_prior", hbp_pkg)
    sys.modules.setdefault("motion_process.human_body_prior.body_model", hbp_bm_pkg)
    sys.modules.setdefault("motion_process.human_body_prior.body_model.lbs", lbs_mod)

    # 3) Load body_model.py
    bm_path = os.path.join(bm_dir, "body_model.py")
    spec = importlib.util.spec_from_file_location("body_model.body_model", bm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _BODY_MODEL_CLASS = mod.BodyModel
    return _BODY_MODEL_CLASS


def _handle_nan_inf(arr: np.ndarray, name: str = "array") -> np.ndarray:
    if np.any(~np.isfinite(arr)):
        count = np.sum(~np.isfinite(arr))
        logger.warning(f"[phys_metrics] {name} 包含 {count} 个 NaN/Inf，已替换为 0")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def load_motion_data(file_path: str) -> Dict:
    """
    统一加载 NPZ/H5 动画文件。

    返回:
        {
          "poses": np.ndarray (T, J, 3),
          "betas": np.ndarray (num_betas,),
          "trans": np.ndarray (T, 3),
          "fps": float,
          "gender": str,
          "smpl_type": str,
          # HDF 预计算字段（可选，仅 H5 格式且文件包含时存在）
          "joints3d": np.ndarray (T, J, 3) | None,
          "rot_mats": np.ndarray (T, J, 3, 3) | None,
        }
    """
    file_path = str(file_path)
    precomputed_joints3d = None
    precomputed_rot_mats = None
    hdf_joint_names = None

    if file_path.lower().endswith(".h5"):
        if not HAS_H5PY:
            raise ImportError("h5py 未安装，请运行 pip install h5py")
        raw = {}
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                ds = f[key]
                raw[key] = ds[()] if ds.shape == () else ds[:]
        if "global_translation" in raw and "trans" not in raw:
            raw["trans"] = raw["global_translation"]
        if "frame_rate" in raw and "mocap_framerate" not in raw:
            raw["mocap_framerate"] = float(raw["frame_rate"])
        if "gender" not in raw:
            raw["gender"] = "neutral"
        if "joints3d" in raw:
            precomputed_joints3d = np.array(raw["joints3d"], dtype=np.float32)
        if "rot_mats" in raw:
            precomputed_rot_mats = np.array(raw["rot_mats"], dtype=np.float32)
        if "joint_names" in raw:
            jn = raw["joint_names"]
            if isinstance(jn, np.ndarray):
                hdf_joint_names = [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in jn
                ]
            elif isinstance(jn, (list, tuple)):
                hdf_joint_names = [str(s) for s in jn]
    elif file_path.lower().endswith(".npz"):
        raw = dict(np.load(file_path, allow_pickle=True))
    else:
        raise ValueError(f"不支持的文件格式: {file_path}（仅支持 .npz 和 .h5）")

    gender = raw.get("gender", "neutral")
    if isinstance(gender, np.ndarray):
        gender = str(gender.item())
    else:
        gender = str(gender)
    if gender not in ["male", "female", "neutral"]:
        gender = "neutral"

    fps = float(raw.get("mocap_framerate", raw.get("mocap_frame_rate", 30)))

    betas_raw = raw.get("betas", np.zeros(16))
    betas_raw = np.array(betas_raw, dtype=np.float32)
    num_betas = min(16, int(betas_raw.shape[0]))
    betas = betas_raw.flatten()[:num_betas]
    betas = _handle_nan_inf(betas, "betas")

    poses = raw.get("poses")
    if poses is None:
        raise ValueError(f"文件 {file_path} 缺少 poses 字段")
    poses = np.array(poses, dtype=np.float32)
    poses = _handle_nan_inf(poses, "poses")
    poses, smpl_type = infer_smpl_type(poses)

    T = poses.shape[0]
    trans = raw.get("trans", np.zeros((T, 3), dtype=np.float32))
    trans = np.array(trans, dtype=np.float32)
    trans = _handle_nan_inf(trans, "trans")
    if trans.ndim == 1:
        trans = np.tile(trans, (T, 1))
    elif trans.shape[0] != T:
        logger.warning(f"trans 长度 {trans.shape[0]} 与 poses 帧数 {T} 不一致，重置为零")
        trans = np.zeros((T, 3), dtype=np.float32)

    # HDF joint_names 可能与标准 SMPL 顺序不同（如 FBX DFS 顺序），
    # 需要构建重排映射来确保 joints3d/rot_mats 的关节顺序与标准一致
    joint_reorder_map = None
    if hdf_joint_names is not None:
        try:
            joint_reorder_map = build_joint_reorder_map(
                hdf_joint_names, smpl_type=smpl_type,
            )
            if joint_reorder_map is not None:
                logger.info(
                    f"HDF joint_names 顺序与标准 {smpl_type} 不一致，已构建重排映射"
                )
                if precomputed_joints3d is not None:
                    precomputed_joints3d = reorder_joints(
                        precomputed_joints3d, joint_reorder_map, joint_axis=1,
                    )
                if precomputed_rot_mats is not None:
                    precomputed_rot_mats = reorder_joints(
                        precomputed_rot_mats, joint_reorder_map, joint_axis=1,
                    )
        except ValueError as e:
            logger.warning(f"HDF joint_names 重排失败: {e}，将忽略预计算数据")
            precomputed_joints3d = None
            precomputed_rot_mats = None

    return {
        "poses": poses,
        "betas": betas,
        "trans": trans,
        "fps": fps,
        "gender": gender,
        "smpl_type": smpl_type,
        "joints3d": precomputed_joints3d,
        "rot_mats": precomputed_rot_mats,
    }


def _poses_to_smplh_components(poses: np.ndarray, smpl_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = poses.shape[0]
    root_orient = poses[:, 0, :]

    if smpl_type == "smplh":
        pose_body = poses[:, 1:22, :].reshape(T, 63)
        pose_hand = poses[:, 22:52, :].reshape(T, 90)
    elif smpl_type == "smplx":
        pose_body = poses[:, 1:22, :].reshape(T, 63)
        pose_hand = poses[:, 25:55, :].reshape(T, 90)
    else:
        J = poses.shape[1]
        pose_body = poses[:, 1 : min(22, J), :].reshape(T, -1)
        if pose_body.shape[1] < 63:
            pose_body = np.concatenate([pose_body, np.zeros((T, 63 - pose_body.shape[1]), dtype=np.float32)], axis=1)
        pose_hand = np.zeros((T, 90), dtype=np.float32)

    return root_orient, pose_body, pose_hand


def compute_verts_joints(
    poses: np.ndarray,
    betas: np.ndarray,
    trans: np.ndarray,
    smpl_type: str,
    smpl_model_path: str,
    device: str = _DEFAULT_DEVICE,
    chunk_size: int = 256,
    fps: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], float]:
    """
    通过 BodyModel forward 计算 verts 和 joints，以及 rest pose（用于 KNN 脚底选取）。

    返回:
        joint_rot: (T', 21, 3) body joints 轴角（不含 root）
        verts: (T', V, 3)
        joints: (T', J, 3)
        rest_verts: (V, 3) T-pose 顶点
        rest_joints: (J, 3) T-pose 关节
        faces: (F, 3) 面片索引（用于脚部法线计算），如模型无面片则为 None
        fps_out: float 下采样后的帧率
    """
    BodyModel = _get_body_model_class()

    T = poses.shape[0]
    num_betas = len(betas)

    root_orient, pose_body, pose_hand = _poses_to_smplh_components(poses, smpl_type)
    joint_rot = poses[:, 1:22, :].copy() if poses.shape[1] >= 22 else np.zeros((T, 21, 3), dtype=np.float32)

    bm = BodyModel(bm_fname=smpl_model_path, num_betas=num_betas).to(device)
    bm.eval()
    faces = bm.f.cpu().numpy().astype(np.int32) if hasattr(bm, 'f') and bm.f is not None else None

    root_t = torch.from_numpy(root_orient).float().to(device)
    body_t = torch.from_numpy(pose_body).float().to(device)
    hand_t = torch.from_numpy(pose_hand).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device).unsqueeze(0)
    trans_t = torch.from_numpy(trans).float().to(device)

    verts_list = []
    joints_list = []

    with torch.no_grad():
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            cs = end - start
            body = bm(
                root_orient=root_t[start:end],
                pose_body=body_t[start:end],
                pose_hand=hand_t[start:end],
                betas=betas_t.repeat(cs, 1),
                trans=trans_t[start:end],
            )
            verts_list.append(body.v.cpu().numpy().astype(np.float32))
            joints_list.append(body.Jtr.cpu().numpy().astype(np.float32))

        # Rest pose (T-pose): zero rotations, same betas, zero translation
        rest_body = bm(
            root_orient=torch.zeros(1, 3, device=device),
            pose_body=torch.zeros(1, 63, device=device),
            pose_hand=torch.zeros(1, 90, device=device),
            betas=betas_t,
            trans=torch.zeros(1, 3, device=device),
        )
        rest_verts = rest_body.v.cpu().numpy().astype(np.float32)[0]
        rest_joints = rest_body.Jtr.cpu().numpy().astype(np.float32)[0]

    verts = np.concatenate(verts_list, axis=0)
    joints = np.concatenate(joints_list, axis=0)

    down = max(1, int(fps // 30))
    if down > 1:
        joint_rot = joint_rot[::down]
        verts = verts[::down]
        joints = joints[::down]
    fps_out = fps / down

    return joint_rot, verts, joints, rest_verts, rest_joints, faces, fps_out


def _downsample_to_target_fps(fps: float, *arrays, target_fps: float = 30.0):
    """对多个数组按帧率比例下采样，返回 (下采样后的数组元组, 新帧率)"""
    down = max(1, int(fps // target_fps))
    if down > 1:
        return tuple(a[::down] for a in arrays), fps / down
    return arrays, fps


# ==============================================================================
# Part C：顶点级物理指标（穿透/浮空/滑动，需要 BodyModel verts）
# [通用算法] 这些函数的核心算法不依赖特定骨骼模型（仅需顶点/法线数据），
# 但脚部区域的顶点 ID 选取（foot_regions.py）是 SMPL 耦合的。
# ==============================================================================


def _compute_floor_height(verts: np.ndarray, up_axis: int, first_n_frames: int) -> float:
    """基于前 N 帧所有顶点最低 10 个点的平均值估计地面高度"""
    T, V, _ = verts.shape
    N = int(min(max(first_n_frames, 1), T))
    per_frame = []
    for t in range(N):
        heights = verts[t, :, up_axis]
        n_lowest = min(10, V)
        lowest_idx = np.argpartition(heights, n_lowest)[:n_lowest]
        per_frame.append(float(np.mean(heights[lowest_idx])))
    return float(np.mean(per_frame))


def _compute_floor_height_global_min(
    verts: np.ndarray, sole_vids: np.ndarray, up_axis: int,
) -> float:
    """基于所有帧脚底区域最低点的均值估计地面高度（比 first_n_frames 更稳健）"""
    if sole_vids.size == 0:
        return 0.0
    sole_h = verts[:, sole_vids, up_axis]
    per_frame_min = sole_h.min(axis=1)
    n_lowest = min(10, len(per_frame_min))
    lowest_idx = np.argpartition(per_frame_min, n_lowest)[:n_lowest]
    return float(np.mean(per_frame_min[lowest_idx]))


def estimate_floor_height(
    points: np.ndarray,
    up_axis: int,
    fps: float,
    first_seconds: float = 2.0,
    n_lowest_per_frame: int = 10,
    cluster_gap: float = 0.05,
    cluster_min_seconds: float = 0.5,
) -> float:
    """
    鲁棒地面高度估算：前 N 秒数据，每帧取最低点均值，1D 聚类识别地面层级，
    取最低有效簇的 median。

    算法流程:
      1. 每帧取最低 k 个点的均值（骨骼模式 k=1）
      2. 对帧高度排序后按 gap 阈值切分为不同高度簇
      3. 过滤帧数不足 cluster_min_seconds 的噪声簇
      4. 取最低有效簇的 median

    Args:
        points: (T, V, 3) 点云，mesh 顶点或骨骼关节均可
        up_axis: 垂直轴索引
        fps: 帧率
        first_seconds: 使用前多少秒的数据
        n_lowest_per_frame: 每帧取最低几个点的均值（骨骼模式强制为 1）
        cluster_gap: 类间最小距离（米），相邻帧高度差超过此值则分为不同簇
        cluster_min_seconds: 有效簇的最小持续时间（秒），低于此值视为噪声
    """
    T, V, _ = points.shape
    n_frames = max(1, min(int(round(first_seconds * fps)), T))
    is_skeleton = V < 100
    k = 1 if is_skeleton else min(n_lowest_per_frame, V)

    per_frame_h = np.empty(n_frames)
    for t in range(n_frames):
        heights = points[t, :, up_axis]
        if k >= V:
            per_frame_h[t] = np.min(heights)
        else:
            idx = np.argpartition(heights, k)[:k]
            per_frame_h[t] = np.mean(heights[idx])

    sorted_h = np.sort(per_frame_h)

    if len(sorted_h) <= 1:
        return float(sorted_h[0])

    gaps = np.diff(sorted_h)
    split_indices = np.where(gaps > cluster_gap)[0] + 1
    clusters = np.split(sorted_h, split_indices)

    min_samples = max(1, int(round(cluster_min_seconds * fps)))
    valid_clusters = [c for c in clusters if len(c) >= min_samples]
    if not valid_clusters:
        valid_clusters = clusters

    return float(np.median(valid_clusters[0]))


def _resolve_floor_height(
    verts: Optional[np.ndarray], sole_vids: np.ndarray, up_axis: int,
    params: Dict, fps: float = 30.0,
    joints: Optional[np.ndarray] = None,
) -> float:
    """
    根据 floor_mode 参数选择地面高度估算策略。

    模式:
      first_n_seconds — 前 N 秒，每帧取最低点均值，聚类识别地面，median（默认）
      fixed_zero      — 固定 y=0，与前端棋盘格地面一致
      first_n_frames  — 前 N 帧全局最低 10 顶点均值（旧模式）
      global_min      — 所有帧脚底区域最低帧的均值
      fixed_value     — 用户通过 floor_height_value 直接指定（米）

    当 verts 为 None 时（无 mesh），first_n_seconds 模式回退到使用 joints。
    """
    mode = params.get("floor_mode", "first_n_seconds")
    if mode == "first_n_seconds":
        points = verts if verts is not None else joints
        if points is None:
            return 0.0
        return estimate_floor_height(
            points, up_axis, fps,
            first_seconds=params.get("floor_first_seconds", 2.0),
            n_lowest_per_frame=params.get("floor_n_lowest_per_frame", 10),
            cluster_gap=params.get("floor_cluster_gap", 0.05),
            cluster_min_seconds=params.get("floor_cluster_min_seconds", 0.5),
        )
    elif mode == "fixed_zero":
        return 0.0
    elif mode == "first_n_frames":
        assert verts is not None, "first_n_frames 模式需要 mesh 顶点"
        T = verts.shape[0]
        n = min(params.get("floor_first_n_frames", 5), T)
        return _compute_floor_height(verts, up_axis, n)
    elif mode == "global_min":
        assert verts is not None, "global_min 模式需要 mesh 顶点"
        return _compute_floor_height_global_min(verts, sole_vids, up_axis)
    elif mode == "fixed_value":
        val = params.get("floor_height_value")
        if val is None:
            raise ValueError("floor_mode='fixed_value' 需要提供 floor_height_value 参数")
        return float(val)
    else:
        raise ValueError(f"未知的 floor_mode: {mode}")


def _is_region_in_contact(
    h_min: np.ndarray,
    v: np.ndarray,
    floor_height: float,
    contact_thresh: float,
    below_tol: float,
    vel_thresh_per_frame: float,
) -> np.ndarray:
    """
    判断区域是否处于接触状态：最低顶点在接触窗口内且垂直速度低。

    Returns: (T,) bool array
    """
    low = floor_height - below_tol
    high = floor_height + contact_thresh
    return (h_min >= low) & (h_min <= high) & (np.abs(v) < vel_thresh_per_frame)


def _is_region_penetrating(
    h_min: np.ndarray,
    floor_height: float,
    below_tol: float,
) -> np.ndarray:
    """
    判断区域是否处于穿透状态：最低顶点穿透到地面以下。
    穿透意味着地面一定在提供反力，无需速度判断。

    Returns: (T,) bool array
    """
    return h_min < (floor_height - below_tol)


def _compute_region_h_min_and_vel(
    verts: np.ndarray,
    region_vids: np.ndarray,
    up_axis: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算区域最低顶点高度和帧间垂直速度。"""
    T = verts.shape[0]
    h_min = np.min(verts[:, region_vids, up_axis], axis=1)
    v = np.zeros(T, dtype=np.float32)
    v[:-1] = h_min[1:] - h_min[:-1]
    v[-1] = v[-2] if T > 1 else 0.0
    return h_min, v


def _debounce_support(support: np.ndarray, min_frames: int) -> np.ndarray:
    """
    去毛刺：移除短于 min_frames 帧的 support 连续段。

    步行摆动弧线最低点可能短暂穿过接触检测窗口（通常 < 0.2s），
    而真正的支撑期至少持续 0.3s 以上（步行）或 0.1s 以上（跑步/跳跃）。
    """
    if min_frames <= 1:
        return support
    result = support.copy()
    T = len(support)
    i = 0
    while i < T:
        if support[i]:
            j = i
            while j < T and support[j]:
                j += 1
            if (j - i) < min_frames:
                result[i:j] = False
            i = j
        else:
            i += 1
    return result


def _compute_contact_flicker(raw_support: np.ndarray, fps: float, min_run_sec: float = 0.1) -> Dict:
    """统计接触/非接触状态的时序稳定性（闪烁检测）。

    在 debounce 之前的原始 support 序列上检测短暂的状态翻转。
    自然动作中接触状态转换必然持续多帧（≥3-5帧 at 30fps），
    而生成动作中可能出现逐帧闪烁（接地→浮空→接地），虽然帧级穿透/浮空值很小，
    但视觉上非常明显（脚在抖）。

    Args:
        raw_support: (T,) 布尔数组，debounce 之前的逐帧接触状态
        fps: 帧率
        min_run_sec: 低于此时长的接触/非接触段视为闪烁（默认 0.1s）
    Returns:
        dict with:
            flicker_ratio: 处于闪烁段内的帧占比 (%)
            avg_run_length: 所有段的平均持续帧数（越大越稳定）
    """
    T = len(raw_support)
    if T < 2:
        return {"flicker_ratio": 0.0, "avg_run_length": float(T)}

    min_run_frames = max(1, int(round(min_run_sec * fps)))

    # 计算 run-length：连续相同值的段长度
    runs = []
    i = 0
    while i < T:
        j = i + 1
        while j < T and raw_support[j] == raw_support[i]:
            j += 1
        runs.append(j - i)
        i = j

    if not runs:
        return {"flicker_ratio": 0.0, "avg_run_length": float(T)}

    avg_run_length = float(np.mean(runs))

    # 闪烁帧数 = 处于短段（< min_run_frames）中的帧总数
    flicker_frames = sum(r for r in runs if r < min_run_frames)
    flicker_ratio = float(flicker_frames / T * 100) if T > 0 else 0.0

    return {"flicker_ratio": flicker_ratio, "avg_run_length": avg_run_length}


def _support_and_metrics_one_foot(
    verts: np.ndarray,
    heel_vids: np.ndarray,
    forefoot_vids: np.ndarray,
    up_axis: int,
    floor_height: float,
    heel_contact_thresh: float,
    forefoot_contact_thresh: float,
    below_tol: float,
    vertical_vel_thresh: float,
    skate_threshold: float,
    fps: float,
    support_min_frames: int = 3,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    单脚物理指标计算（分区域 support 检测 + 去毛刺）。

    将脚底分为脚跟区和前脚掌区，各自独立判断 support 状态，
    使用固定顶点集计算质心位移，避免帧间顶点选取不一致。
    脚跟和前脚掌使用不同的接触窗口阈值：脚跟离地弧线高，可用较大窗口；
    前脚掌离地弧线低，需用较小窗口避免摆动穿越误判。

    阈值 vertical_vel_thresh 和 skate_threshold 单位为 m/s。

    Returns: (pen_m, flo_m, ska_m, skate_ratio, raw_any_support)
        raw_any_support: (T,) debounce 之前的原始接触状态布尔数组，供闪烁检测使用
    """
    T = verts.shape[0]
    all_sole_vids = np.union1d(heel_vids, forefoot_vids)
    if all_sole_vids.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    dt = 1.0 / fps
    vel_thresh_pf = vertical_vel_thresh * dt
    skate_thresh_pf = skate_threshold * dt
    horiz_axes = [ax for ax in range(3) if ax != up_axis]

    # --- 分区域 support 检测 + 去毛刺 ---
    region_thresholds = (heel_contact_thresh, forefoot_contact_thresh)
    regions = []
    raw_regions = []  # debounce 前的原始 support，供闪烁检测使用
    for rvids, ct in zip((heel_vids, forefoot_vids), region_thresholds):
        if rvids.size == 0:
            regions.append(np.zeros(T, dtype=bool))
            raw_regions.append(np.zeros(T, dtype=bool))
            continue
        h_min, v = _compute_region_h_min_and_vel(verts, rvids, up_axis)
        in_contact = _is_region_in_contact(
            h_min, v, floor_height, ct, below_tol, vel_thresh_pf,
        )
        in_penetration = _is_region_penetrating(h_min, floor_height, below_tol)
        raw_support = in_contact | in_penetration
        raw_regions.append(raw_support.copy())
        regions.append(_debounce_support(raw_support, support_min_frames))
    heel_support, forefoot_support = regions
    raw_heel_support, raw_forefoot_support = raw_regions

    any_support = heel_support | forefoot_support
    raw_any_support = raw_heel_support | raw_forefoot_support

    # --- 穿透/浮空（使用所有脚底顶点） ---
    low = floor_height - below_tol
    all_h_min = np.min(verts[:, all_sole_vids, up_axis], axis=1)
    pen_depth = np.maximum(low - all_h_min, 0.0)
    flo_gap = np.maximum(all_h_min - floor_height, 0.0)

    pen_frames = any_support & (all_h_min < low)
    contact_frames = any_support & (all_h_min >= low)

    pen_m = float(np.mean(pen_depth[pen_frames])) if np.any(pen_frames) else 0.0
    flo_m = float(np.mean(flo_gap[contact_frames])) if np.any(contact_frames) else 0.0

    # --- 滑步计算（分区域，固定顶点集，双帧 support 检查） ---
    skate_distances = []
    for t in range(T - 1):
        supported_vids = []
        if heel_support[t] and heel_support[t + 1] and heel_vids.size > 0:
            supported_vids.append(heel_vids)
        if forefoot_support[t] and forefoot_support[t + 1] and forefoot_vids.size > 0:
            supported_vids.append(forefoot_vids)
        if not supported_vids:
            continue

        max_disp = 0.0
        for rvids in supported_vids:
            c_t = np.mean(verts[t, rvids][:, horiz_axes], axis=0)
            c_t1 = np.mean(verts[t + 1, rvids][:, horiz_axes], axis=0)
            disp = float(np.linalg.norm(c_t1 - c_t))
            max_disp = max(max_disp, disp)
        skate_distances.append(max_disp)

    n_skating = sum(1 for d in skate_distances if d > skate_thresh_pf)
    skate_ratio = float(n_skating / max(T - 1, 1) * 100.0)

    skating_only = [d for d in skate_distances if d > skate_thresh_pf]
    ska_m = float(np.mean(skating_only)) if skating_only else 0.0

    return pen_m, flo_m, ska_m, skate_ratio, raw_any_support


def _compute_verts_based_metrics(
    verts: np.ndarray,
    joints: np.ndarray,
    fps: float,
    params: Dict,
    rest_verts: np.ndarray,
    rest_joints: np.ndarray,
    faces: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算需要顶点数据的物理指标（穿透/浮空/滑动）。
    所有长度类指标以米为单位返回（由调用方做 output_unit 转换）。
    """
    assert verts.ndim == 3 and verts.shape[2] == 3, (
        f"verts 应为 (T, V, 3)，实际 shape={verts.shape}"
    )
    assert rest_verts.ndim == 2 and rest_verts.shape[1] == 3, (
        f"rest_verts 应为 (V, 3)，实际 shape={rest_verts.shape}"
    )
    up_axis = params["up_axis"]
    assert up_axis in (0, 1, 2), (
        f"up_axis 必须为 0/1/2（对应 X/Y/Z），实际值={up_axis}"
    )
    T = verts.shape[0]

    foot_params = {
        "up_axis": up_axis,
        "k_nearest": params["k_nearest"],
        "sole_keep_percentile": params["sole_keep_percentile"],
        "left_heel_joint_ids": params.get("left_heel_joint_ids", [L_ANKLE_IDX]),
        "left_forefoot_joint_ids": params.get("left_forefoot_joint_ids", [L_FOOT_IDX]),
        "right_heel_joint_ids": params.get("right_heel_joint_ids", [R_ANKLE_IDX]),
        "right_forefoot_joint_ids": params.get("right_forefoot_joint_ids", [R_FOOT_IDX]),
        "left_heel_vertex_ids": params.get("left_heel_vertex_ids"),
        "left_forefoot_vertex_ids": params.get("left_forefoot_vertex_ids"),
        "right_heel_vertex_ids": params.get("right_heel_vertex_ids"),
        "right_forefoot_vertex_ids": params.get("right_forefoot_vertex_ids"),
    }
    regions = select_foot_regions(rest_verts, rest_joints, foot_params, faces=faces)

    all_sole_vids = np.unique(np.concatenate([
        regions["left_heel"], regions["left_forefoot"],
        regions["right_heel"], regions["right_forefoot"],
    ]))
    floor_height = _resolve_floor_height(verts, all_sole_vids, up_axis, params, fps, joints=joints)

    min_dur = params.get("support_min_duration_sec", 0.2)
    support_min_frames = max(1, int(round(min_dur * fps)))

    heel_ct = params["heel_contact_thresh"]
    forefoot_ct = params["forefoot_contact_thresh"]

    L_pen, L_flo, L_ska, L_skr, L_raw_support = _support_and_metrics_one_foot(
        verts, regions["left_heel"], regions["left_forefoot"],
        up_axis, floor_height,
        heel_ct, forefoot_ct, params["below_tol"],
        params["vertical_vel_thresh"], params["skate_threshold"], fps,
        support_min_frames=support_min_frames,
    )
    R_pen, R_flo, R_ska, R_skr, R_raw_support = _support_and_metrics_one_foot(
        verts, regions["right_heel"], regions["right_forefoot"],
        up_axis, floor_height,
        heel_ct, forefoot_ct, params["below_tol"],
        params["vertical_vel_thresh"], params["skate_threshold"], fps,
        support_min_frames=support_min_frames,
    )

    pen_avg = 0.5 * (L_pen + R_pen)
    flo_avg = 0.5 * (L_flo + R_flo)
    ska_avg = 0.5 * (L_ska + R_ska)
    skate_ratio_avg = 0.5 * (L_skr + R_skr)

    # --- Per-frame penetration/float for frame_stats ---
    all_h_min = np.min(verts[:, all_sole_vids, up_axis], axis=1)  # (T,)
    low = floor_height - params["below_tol"]
    per_frame_pen = np.maximum(low - all_h_min, 0.0)  # (T,) 穿透深度
    per_frame_flo = np.maximum(all_h_min - floor_height, 0.0)  # (T,) 浮空高度

    frame_stats = {}
    frame_stats["avg_penetrate"] = _compute_frame_stats(per_frame_pen)
    frame_stats["avg_float"] = _compute_frame_stats(per_frame_flo)

    # --- Contact Flicker（接触闪烁）---
    flicker_min_run = params.get("contact_flicker_min_run_sec", 0.067)
    L_flicker = _compute_contact_flicker(L_raw_support, fps, flicker_min_run)
    R_flicker = _compute_contact_flicker(R_raw_support, fps, flicker_min_run)
    contact_flicker_ratio = 0.5 * (L_flicker["flicker_ratio"] + R_flicker["flicker_ratio"])
    contact_avg_run_length = 0.5 * (L_flicker["avg_run_length"] + R_flicker["avg_run_length"])

    return {
        "avg_penetrate": pen_avg,
        "avg_float": flo_avg,
        "avg_skate": ska_avg,
        "frame_avg_skate": ska_avg * skate_ratio_avg / 100.0,
        "skate_ratio": skate_ratio_avg,
        # TODO: phys_err 综合物理误差的计算方式需要更新——当前仅为 penetrate + float + skate
        # 的简单求和，应考虑纳入新增的指标（如 self_penetration、contact_flicker 等），
        # 并重新设计加权/归一化策略。
        "phys_err": pen_avg + flo_avg + ska_avg,
        "contact_flicker_ratio": contact_flicker_ratio,
        "contact_avg_run_length": contact_avg_run_length,
        "frame_stats": frame_stats,
    }


# ==============================================================================
# Part C1b：Mesh Distortion 指标（需要 BodyModel verts + faces）
# [通用算法] ARAP/体积/边拉伸计算不依赖特定骨骼模型，但 face_joint_assignment
# （将面片分配到关节）使用了 SMPL 的 LBS weights（SMPL 耦合）。
# ==============================================================================


_MESH_DISTORT_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]

_ARAP_THRESH_GOOD = 0.10
_ARAP_THRESH_MODERATE = 0.40


def _compute_mesh_distortion_metrics(
    verts: np.ndarray,
    rest_verts: np.ndarray,
    faces: np.ndarray,
    device: str = _DEFAULT_DEVICE,
    chunk_size: int = 256,
    arap_severe_thresh: float = 0.1,
) -> Dict[str, Any]:
    """
    计算 posed mesh 相对于 rest pose 的 mesh distortion 指标。

    全局指标 + per-joint ARAP 统计（p95、质量分级）。

    Args:
        verts:       (T, V, 3) 所有帧的 posed 顶点，米制
        rest_verts:  (V, 3) rest pose 顶点，米制
        faces:       (F, 3) 面片索引
        device:      torch 计算设备
        chunk_size:  分块大小（帧数），避免 GPU OOM
        arap_severe_thresh: ARAP severe 判定阈值
    Returns:
        dict: mesh_distort_* 前缀的各项指标（无量纲）+ per_joint_arap 字典
    """
    if not HAS_TORCH or not HAS_MESH_DISTORTION:
        return {}
    if faces is None:
        logger.warning("[mesh_distortion] faces 为 None，跳过 mesh distortion 计算")
        return {}

    T = verts.shape[0]
    faces_t = torch.from_numpy(faces).long().to(device)
    rest_t = torch.from_numpy(rest_verts).float().to(device)

    # 预计算一次 rest 基底逆矩阵和唯一边，跨所有 chunk 复用
    D_rest_inv = precompute_rest_basis(rest_t, faces_t)   # (F, 3, 3)
    unique_edges = _faces_to_unique_edges(faces_t)         # (E, 2)

    arap_all = []
    vol_all = []
    sd_all = []
    es_all = []

    with torch.no_grad():
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            posed_t = torch.from_numpy(verts[start:end]).float().to(device)

            arap_b, vol_b, sd_b, es_b = mesh_distort_all(
                posed_t, rest_t, faces_t,
                D_rest_inv=D_rest_inv,
                unique_edges=unique_edges,
            )
            arap_all.append(arap_b.cpu())
            vol_all.append(vol_b.cpu())
            sd_all.append(sd_b.cpu())
            es_all.append(es_b.cpu())

    arap_cat = torch.cat(arap_all, dim=0)   # (T, F)
    vol_cat = torch.cat(vol_all, dim=0)     # (T, F)
    sd_cat = torch.cat(sd_all, dim=0)       # (T, F)
    es_cat = torch.cat(es_all, dim=0)       # (T, E)

    # --- ARAP 全局摘要 ---
    arap_flat = arap_cat.flatten()
    arap_mean = float(arap_flat.mean())
    arap_max = float(arap_flat.max())
    arap_p99 = float(torch.quantile(arap_flat.float(), 0.99))

    severe_count = int((arap_cat > arap_severe_thresh).sum())
    total_face_frames = arap_cat.numel()
    severe_ratio = severe_count / max(total_face_frames, 1) * 100.0

    # --- Edge Stretch 摘要 ---
    es_flat = es_cat.flatten()
    es_std = float(es_flat.std())
    es_max = float(es_flat.max())
    es_min = float(es_flat.min())

    # --- Volume 摘要 ---
    vol_flat = vol_cat.flatten()
    vol_mean = float(vol_flat.mean())
    vol_max = float(vol_flat.max())

    # --- Symmetric Dirichlet 摘要 ---
    sd_flat = sd_cat.flatten()
    sd_mean = float(sd_flat.mean())
    sd_max = float(sd_flat.max())

    # --- Per-joint ARAP ---
    # 双聚合策略：
    #   arap_p95     = 所有帧×所有面片 flatten 后的 p95（原方案，对持续性形变敏感）
    #   arap_fmax_p95 = 每帧取关节面片 max，再跨帧取 p95（对少量极端帧敏感，
    #                   避免大量正常帧稀释 candy-wrapper 等局部高 distortion 信号）
    # quality 取两者中更严格的判定。
    face_assignment = get_face_joint_assignment(num_faces=faces.shape[0])
    face_assign_t = torch.from_numpy(face_assignment).long()

    per_joint_arap = []
    num_body_joints = 22
    for jid in range(num_body_joints):
        mask = (face_assign_t == jid)
        face_count = int(mask.sum())
        if face_count == 0:
            per_joint_arap.append({
                "joint_idx": jid,
                "joint_name": _MESH_DISTORT_JOINT_NAMES[jid],
                "face_count": 0,
                "arap_mean": 0.0,
                "arap_p95": 0.0,
                "arap_fmax_p95": 0.0,
                "arap_max": 0.0,
                "quality": "good",
            })
            continue

        joint_arap_2d = arap_cat[:, mask]            # (T, face_count)
        joint_arap = joint_arap_2d.flatten()          # (T * face_count,)
        p95 = float(torch.quantile(joint_arap.float(), 0.95))

        frame_max = joint_arap_2d.max(dim=1).values   # (T,)
        fmax_p95 = float(torch.quantile(frame_max.float(), 0.95))

        j_mean = float(joint_arap.mean())
        j_max = float(joint_arap.max())

        effective_p95 = max(p95, fmax_p95)
        if effective_p95 < _ARAP_THRESH_GOOD:
            quality = "good"
        elif effective_p95 < _ARAP_THRESH_MODERATE:
            quality = "moderate"
        else:
            quality = "poor"

        per_joint_arap.append({
            "joint_idx": jid,
            "joint_name": _MESH_DISTORT_JOINT_NAMES[jid],
            "face_count": face_count,
            "arap_mean": round(j_mean, 6),
            "arap_p95": round(p95, 6),
            "arap_fmax_p95": round(fmax_p95, 6),
            "arap_max": round(j_max, 6),
            "quality": quality,
        })

    return {
        "mesh_distort_arap_mean": arap_mean,
        "mesh_distort_arap_max": arap_max,
        "mesh_distort_arap_p99": arap_p99,
        "mesh_distort_severe_ratio": severe_ratio,
        "mesh_distort_edge_stretch_std": es_std,
        "mesh_distort_edge_stretch_max": es_max,
        "mesh_distort_edge_stretch_min": es_min,
        "mesh_distort_volume_mean": vol_mean,
        "mesh_distort_volume_max": vol_max,
        "mesh_distort_symdirichlet_mean": sd_mean,
        "mesh_distort_symdirichlet_max": sd_max,
        "per_joint_arap": per_joint_arap,
    }


# ==============================================================================
# Part C2：TBS (Twist-Bend-Spread) 异常关节检测（三层函数）
# ==============================================================================
#
#   Layer 1: compute_tbs_euler_angles  — 从局部旋转计算 TBS 欧拉角
#   Layer 2: tbs_angle_abnormal        — 对 TBS 角度做 ROM 阈值判断
#   Layer 3: has_distorted_joints      — 整合 L1+L2，输出是否存在异常关节
#


def _offsets_to_template_joints(offsets: np.ndarray, parents: List[int]) -> np.ndarray:
    """从相对骨骼偏移量累加得到绝对 T-pose 关节位置。

    Args:
        offsets: (J, 3) 相对偏移。offsets[0] 为 root 绝对位置，
                 offsets[j] = joints[j] - joints[parent[j]] (j>0)
        parents: length J，parents[0] = -1
    Returns:
        template_joints: (J, 3) 绝对位置
    """
    J = len(parents)
    joints = np.zeros((J, 3), dtype=np.float64)
    joints[0] = offsets[0]
    for j in range(1, J):
        joints[j] = joints[parents[j]] + offsets[j]
    return joints.astype(np.float32)


def _infer_joint_names(num_joints: int) -> List[str]:
    """[SMPL 耦合] 根据关节数量推断关节名列表（硬编码了 SMPL/SMPLH/SMPLX 的关节数映射）。"""
    if SMPLX_JOINTS is not None:
        if num_joints <= BODY_JOINT_COUNT:
            return list(SMPLX_JOINTS[:num_joints])
        elif num_joints == 52 and SMPLH_JOINTS is not None:
            return list(SMPLH_JOINTS)
        elif num_joints <= len(SMPLX_JOINTS):
            return list(SMPLX_JOINTS[:num_joints])
    raise ValueError(
        f"无法推断 {num_joints} 个关节的名称列表，请显式传入 joint_names"
    )


def _build_rom_limits_array(
    joint_names: List[str],
    rom_dict: Optional[Dict[str, Tuple]] = None,
) -> np.ndarray:
    """[通用算法] 从 ROM 字典构建 (J, 6) ndarray。未在字典中定义的关节使用 [-180, 180] 不限制。
    函数本身通用，但默认 rom_dict 为 DEFAULT_TBS_ROM_LIMITS（SMPL 耦合）。"""
    if rom_dict is None:
        rom_dict = DEFAULT_TBS_ROM_LIMITS
    J = len(joint_names)
    limits = np.full((J, 6), [-180, 180, -180, 180, -180, 180], dtype=np.float32)
    for j, name in enumerate(joint_names):
        if name in rom_dict:
            limits[j] = rom_dict[name]
    return limits


def compute_tbs_euler_angles(
    local_rot: np.ndarray,
    parents: List[int],
    offsets: np.ndarray,
    joint_names: Optional[List[str]] = None,
    chunk_size: int = 256,
) -> np.ndarray:
    """[通用算法+SMPL 耦合入口] 从局部旋转计算每个关节在 TBS 坐标系下的欧拉角。

    算法本身通用，但 joint_names=None 时自动推断为 SMPL 关节名（走 _infer_joint_names），
    进而从 _UP_REF_TABLE（SMPL 耦合）查找 up_ref。显式传入 joint_names 可用于其他骨骼模型。

    Args:
        local_rot: (T, J, 3) 各关节局部旋转 axis-angle
        parents: length J 的父节点索引数组，parents[0] = -1
        offsets: (J, 3) 相对骨骼偏移量（beta=0 T-pose）。
                 offsets[0] 为 root 绝对位置，offsets[j>0] = joints[j] - joints[parent[j]]
        joint_names: 关节名列表，用于查找 TBS up_ref 表。
                     None 时根据 J 自动推断（22→body, 52→SMPLH）
        chunk_size: 分批处理的帧数，避免 GPU OOM

    Returns:
        euler_tbs_deg: (T, J, 3) TBS 欧拉角（度），列顺序 [twist, bend, spread]
    """
    if not HAS_TBS:
        raise RuntimeError("TBS 模块不可用（body_model.tbs_axes 导入失败）")
    if not HAS_TORCH:
        raise RuntimeError("torch 未安装，无法计算 TBS 欧拉角")

    local_rot = np.asarray(local_rot, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    T, J = local_rot.shape[0], local_rot.shape[1]
    assert local_rot.shape == (T, J, 3), f"local_rot shape 应为 (T, J, 3)，实际 {local_rot.shape}"
    assert offsets.shape == (J, 3), f"offsets shape 应为 (J, 3)，实际 {offsets.shape}"
    assert len(parents) == J, f"parents 长度 {len(parents)} 与 J={J} 不一致"

    if joint_names is None:
        joint_names = _infer_joint_names(J)

    template_joints = _offsets_to_template_joints(offsets, parents)
    template_joints_t = torch.tensor(template_joints, dtype=torch.float32)

    tbs_layer = BodyTBSLayer(template_joints_t, parents, joint_names)
    tbs_layer.eval()

    all_eulers = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        batch_aa = torch.tensor(local_rot[start:end], dtype=torch.float32)

        with torch.no_grad():
            local_rotmats = rvecs_to_mats_torch(batch_aa.reshape(-1, 3))
            local_rotmats = local_rotmats.reshape(end - start, J, 3, 3)

            euler_tbs_rad, _ = tbs_layer(local_rotmats)
            euler_deg = torch.rad2deg(euler_tbs_rad)

        all_eulers.append(euler_deg.cpu().numpy())

    return np.concatenate(all_eulers, axis=0)  # (T, J, 3)


def tbs_angle_abnormal(
    euler_tbs_deg: np.ndarray,
    rom_limits: np.ndarray,
) -> np.ndarray:
    """[通用算法] 对 TBS 角度做 ROM 阈值判断，返回异常指示矩阵。
    函数本身纯数值比较，不依赖特定骨骼模型；SMPL 耦合在调用方传入的 rom_limits。

    Args:
        euler_tbs_deg: (T, J, 3) TBS 欧拉角（度），列顺序 [twist, bend, spread]
        rom_limits: (J, 6) 每关节 ROM 范围
                    [twist_min, twist_max, bend_min, bend_max, spread_min, spread_max]

    Returns:
        indicator: (T, J) int32 矩阵，0=正常，1=异常（任一轴超限）
    """
    euler_tbs_deg = np.asarray(euler_tbs_deg, dtype=np.float32)
    rom_limits = np.asarray(rom_limits, dtype=np.float32)
    T, J = euler_tbs_deg.shape[0], euler_tbs_deg.shape[1]
    assert euler_tbs_deg.shape == (T, J, 3), f"euler_tbs_deg shape 应为 (T,J,3)，实际 {euler_tbs_deg.shape}"
    assert rom_limits.shape == (J, 6), f"rom_limits shape 应为 (J,6)，实际 {rom_limits.shape}"

    twist = euler_tbs_deg[:, :, 0]
    bend = euler_tbs_deg[:, :, 1]
    spread = euler_tbs_deg[:, :, 2]

    # rom_limits 行展开为 (1, J) 便于广播
    lim = rom_limits[np.newaxis, :, :]  # (1, J, 6)

    abnormal = (
        (twist < lim[:, :, 0]) | (twist > lim[:, :, 1]) |
        (bend < lim[:, :, 2]) | (bend > lim[:, :, 3]) |
        (spread < lim[:, :, 4]) | (spread > lim[:, :, 5])
    )

    return abnormal.astype(np.int32)


def has_distorted_joints(
    local_rot: np.ndarray,
    parents: List[int],
    offsets: np.ndarray,
    joints_of_interest: Optional[List[int]] = None,
    rom_limits: Optional[np.ndarray] = None,
    abnormal_thresh_ratio: float = 0.05,
    verbose: bool = False,
    joint_names: Optional[List[str]] = None,
    chunk_size: int = 256,
) -> Dict:
    """[通用算法+SMPL 耦合默认值] 检测动画中是否存在异常关节（TBS 角度超出 ROM 范围）。
    rom_limits=None 时默认使用 DEFAULT_TBS_ROM_LIMITS（SMPL 耦合阈值）。

    Args:
        local_rot: (T, J, 3) 各关节局部旋转 axis-angle
        parents: length J 的父节点索引数组
        offsets: (J, 3) beta=0 的相对骨骼偏移量
        joints_of_interest: 要检查的关节索引列表。None 表示检查全部关节
        rom_limits: (J, 6) ROM 范围数组。None 时从 DEFAULT_TBS_ROM_LIMITS 构建
        abnormal_thresh_ratio: 帧超限占比阈值，超过此比例则判定该关节异常（默认 5%）
        verbose: True 时在返回值中包含逐关节详细报告
        joint_names: 关节名列表，None 自动推断
        chunk_size: TBS 计算分批帧数

    Returns:
        dict 包含:
            has_distortion: bool — True 表示存在异常关节
            abnormal_joint_indices: List[int] — 异常关节索引
            abnormal_joint_names: List[str] — 异常关节名
            total_abnormal_ratio: float — 有任一关节超限的帧占比 (%)
        verbose=True 时额外包含:
            per_joint_detail: Dict[int, Dict] — 每个 JoI 的详细信息
            report: str — 人类可读的文本报告
    """
    local_rot = np.asarray(local_rot, dtype=np.float32)
    T, J = local_rot.shape[0], local_rot.shape[1]

    if joint_names is None:
        joint_names = _infer_joint_names(J)

    euler_tbs_deg = compute_tbs_euler_angles(
        local_rot, parents, offsets,
        joint_names=joint_names, chunk_size=chunk_size,
    )

    if rom_limits is None:
        rom_limits = _build_rom_limits_array(joint_names)

    indicator = tbs_angle_abnormal(euler_tbs_deg, rom_limits)  # (T, J)

    if joints_of_interest is None:
        joints_of_interest = list(range(J))

    abnormal_indices = []
    abnormal_names = []
    per_joint_detail = {}

    for j in joints_of_interest:
        j_indicator = indicator[:, j]
        ratio = float(np.mean(j_indicator))

        if verbose:
            twist = euler_tbs_deg[:, j, 0]
            bend = euler_tbs_deg[:, j, 1]
            spread = euler_tbs_deg[:, j, 2]
            lim = rom_limits[j]

            twist_exceed = float(np.mean(
                (twist < lim[0]) | (twist > lim[1])
            ))
            bend_exceed = float(np.mean(
                (bend < lim[2]) | (bend > lim[3])
            ))
            spread_exceed = float(np.mean(
                (spread < lim[4]) | (spread > lim[5])
            ))

            detail = {
                "name": joint_names[j],
                "abnormal_ratio": ratio,
                "twist_exceed_ratio": twist_exceed,
                "bend_exceed_ratio": bend_exceed,
                "spread_exceed_ratio": spread_exceed,
                "twist_range": (float(twist.min()), float(twist.max())),
                "bend_range": (float(bend.min()), float(bend.max())),
                "spread_range": (float(spread.min()), float(spread.max())),
                "twist_limits": (float(lim[0]), float(lim[1])),
                "bend_limits": (float(lim[2]), float(lim[3])),
                "spread_limits": (float(lim[4]), float(lim[5])),
            }
            per_joint_detail[j] = detail

        if ratio > abnormal_thresh_ratio:
            abnormal_indices.append(j)
            abnormal_names.append(joint_names[j])

    # 有任一 JoI 超限的帧占比
    joi_indicator = indicator[:, joints_of_interest]
    any_abnormal_per_frame = np.any(joi_indicator > 0, axis=1)
    total_abnormal_ratio = float(np.mean(any_abnormal_per_frame) * 100.0)

    result: Dict = {
        "has_distortion": len(abnormal_indices) > 0,
        "abnormal_joint_indices": abnormal_indices,
        "abnormal_joint_names": abnormal_names,
        "total_abnormal_ratio": total_abnormal_ratio,
    }

    if verbose:
        result["per_joint_detail"] = per_joint_detail

        lines = []
        lines.append(f"TBS 异常关节检测报告  |  帧数: {T}  |  检查关节数: {len(joints_of_interest)}")
        lines.append(f"异常帧占比: {total_abnormal_ratio:.1f}%  |  "
                      f"异常关节: {len(abnormal_indices)}/{len(joints_of_interest)}")
        lines.append("-" * 90)
        lines.append(
            f"{'关节':<14} {'状态':<6} {'异常帧%':>7}  "
            f"{'twist超限%':>9} {'bend超限%':>9} {'spread超限%':>10}  "
            f"{'twist范围':>16} {'bend范围':>16} {'spread范围':>16}"
        )
        lines.append("-" * 90)
        for j in joints_of_interest:
            d = per_joint_detail[j]
            status = "异常" if d["abnormal_ratio"] > abnormal_thresh_ratio else "正常"
            tr = d["twist_range"]
            br = d["bend_range"]
            sr = d["spread_range"]
            lines.append(
                f"{d['name']:<14} {status:<6} {d['abnormal_ratio']*100:6.1f}%  "
                f"{d['twist_exceed_ratio']*100:8.1f}% {d['bend_exceed_ratio']*100:8.1f}% "
                f"{d['spread_exceed_ratio']*100:9.1f}%  "
                f"[{tr[0]:6.1f},{tr[1]:6.1f}] [{br[0]:6.1f},{br[1]:6.1f}] [{sr[0]:6.1f},{sr[1]:6.1f}]"
            )
        result["report"] = "\n".join(lines)

    return result


    return result


# ==============================================================================
# Part C2：自穿透检测 (Self-Penetration)
# 使用胶囊体近似（capsule mode）或 mesh 精确碰撞检测（mesh mode，未来实现）。
# [SMPL 耦合] 胶囊半径和身体部位定义针对 SMPL 骨骼拓扑。
# ==============================================================================

# 身体部位 → 骨骼段（关节对）映射
# 每个部位包含一组骨骼段，每段由 (parent_jtr_idx, child_jtr_idx) 表示。
# 单元素元组如 (20,) 表示端点关节（退化为球体，无段长度）。
BODY_PART_SEGMENTS = {
    "Torso":      [(0, 3), (3, 6), (6, 9), (9, 12)],  # Pelvis→Spine1→Spine2→Spine3→Neck
    "Head":       [(12, 15)],                            # Neck→Head
    "L_UpperArm": [(13, 16), (16, 18)],                 # L_Collar→L_Shoulder→L_Elbow
    "R_UpperArm": [(14, 17), (17, 19)],
    "L_Forearm":  [(18, 20)],                            # L_Elbow→L_Wrist
    "R_Forearm":  [(19, 21)],
    "L_Hand":     [(20,)],                               # L_Wrist 端点球
    "R_Hand":     [(21,)],
    "L_Thigh":    [(1, 4)],                              # L_Hip→L_Knee
    "R_Thigh":    [(2, 5)],
    "L_Shin":     [(4, 7)],                              # L_Knee→L_Ankle
    "R_Shin":     [(5, 8)],
    "L_Foot":     [(7, 10)],                             # L_Ankle→L_Foot
    "R_Foot":     [(8, 11)],
}

# 预定义的非相邻骨骼对检测列表
# 只检测远端部位之间的穿透，跳过同一运动链上距离 ≤2 的骨骼对（天然近接触不算穿透）
SELF_PENETRATION_PAIRS = [
    # 手/前臂 ↔ 躯干（最常见的穿透）
    ("L_Forearm", "Torso"), ("R_Forearm", "Torso"),
    ("L_Hand", "Torso"),    ("R_Hand", "Torso"),
    # 腿 ↔ 腿（双腿交叉）
    ("L_Thigh", "R_Thigh"), ("L_Shin", "R_Shin"),
    ("L_Thigh", "R_Shin"),  ("R_Thigh", "L_Shin"),
    # 手 ↔ 腿（坐姿等）
    ("L_Forearm", "L_Thigh"), ("R_Forearm", "R_Thigh"),
    ("L_Forearm", "R_Thigh"), ("R_Forearm", "L_Thigh"),
    # 头 ↔ 手
    ("Head", "L_Hand"), ("Head", "R_Hand"),
    # 手臂 ↔ 手臂（交叉）
    ("L_Forearm", "R_Forearm"), ("L_UpperArm", "R_UpperArm"),
]

# [SMPL 耦合] SMPLH neutral rest pose 的胶囊半径（米）
# 每个骨骼段 (parent_jtr, child_jtr) → 胶囊半径
# 端点关节退化为球体，使用固定半径。
# 标定方法：argmax(skinning_weights) 确定顶点主关节，
#   只保留骨骼中段 70% 的顶点（去掉两端关节附近膨大），取 p90 到轴线距离。
# 标定脚本：tests/debug/test_for_capsule_radii.py
SMPLH_CAPSULE_RADII = {
    # 躯干
    ( 0,  1): 0.1774,  # Pelvis→L_Hip
    ( 0,  2): 0.1774,  # Pelvis→R_Hip
    ( 0,  3): 0.1713,  # Pelvis→Spine1 (腰部)
    ( 3,  6): 0.1573,  # Spine1→Spine2
    ( 6,  9): 0.1854,  # Spine2→Spine3
    ( 9, 12): 0.1898,  # Spine3→Neck (含胸腔)
    ( 9, 13): 0.1630,  # Spine3→L_Collar
    ( 9, 14): 0.1668,  # Spine3→R_Collar
    # 头
    (12, 15): 0.0983,  # Neck→Head
    # 上肢
    (13, 16): 0.0961,  # L_Collar→L_Shoulder
    (14, 17): 0.0917,  # R_Collar→R_Shoulder
    (16, 18): 0.0650,  # L_Shoulder→L_Elbow (上臂)
    (17, 19): 0.0646,  # R_Shoulder→R_Elbow
    (18, 20): 0.0432,  # L_Elbow→L_Wrist (前臂)
    (19, 21): 0.0432,  # R_Elbow→R_Wrist
    (20,): 0.06,       # L_Wrist 端点球（手部简化为拳头大小）
    (21,): 0.06,       # R_Wrist 端点球
    # 下肢
    ( 1,  4): 0.1059,  # L_Hip→L_Knee (大腿)
    ( 2,  5): 0.1048,  # R_Hip→R_Knee
    ( 4,  7): 0.0854,  # L_Knee→L_Ankle (小腿)
    ( 5,  8): 0.0867,  # R_Knee→R_Ankle
    ( 7, 10): 0.0586,  # L_Ankle→L_Foot
    ( 8, 11): 0.0533,  # R_Ankle→R_Foot
}


def _capsule_capsule_distance(
    p1: np.ndarray, q1: np.ndarray, r1: float,
    p2: np.ndarray, q2: np.ndarray, r2: float,
) -> float:
    """计算两个胶囊体之间的穿透深度（标量）。

    胶囊体 1: 端点 p1, q1, 半径 r1
    胶囊体 2: 端点 p2, q2, 半径 r2

    如果退化为球体（p==q），退化为球-胶囊距离或球-球距离。

    Returns:
        穿透深度 (>0 表示穿透), 0 表示无穿透
    """
    # 两条线段之间的最短距离
    d1 = q1 - p1  # 方向向量
    d2 = q2 - p2
    r = p1 - p2

    a = float(np.dot(d1, d1))  # |d1|²
    e = float(np.dot(d2, d2))  # |d2|²
    f = float(np.dot(d2, r))

    EPSILON = 1e-10

    if a <= EPSILON and e <= EPSILON:
        # 两个退化为点/球
        dist = float(np.linalg.norm(r))
    elif a <= EPSILON:
        # 胶囊 1 退化为球
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    elif e <= EPSILON:
        # 胶囊 2 退化为球
        t = 0.0
        b = float(np.dot(d1, r))
        s = max(0.0, min(1.0, -b / a))
    else:
        b = float(np.dot(d1, r))
        c = float(np.dot(d1, d2))
        denom = a * e - c * c

        if denom > EPSILON:
            s = max(0.0, min(1.0, (c * f - b * e) / denom))
        else:
            s = 0.0

        t = (c * s + f) / e
        if t < 0.0:
            t = 0.0
            s = max(0.0, min(1.0, -b / a))
        elif t > 1.0:
            t = 1.0
            s = max(0.0, min(1.0, (c - b) / a))

    if not (a <= EPSILON and e <= EPSILON):
        closest1 = p1 + s * d1
        closest2 = p2 + t * d2
        dist = float(np.linalg.norm(closest1 - closest2))

    penetration = (r1 + r2) - dist
    return max(0.0, penetration)


def _compute_self_penetration(
    joints: np.ndarray,
    params: Dict,
    rest_verts: np.ndarray = None,
    posed_verts: np.ndarray = None,
    faces: np.ndarray = None,
    parts_of_interest: List[str] = None,
    mode: str = "capsule",
) -> Dict:
    """自穿透检测。

    检测身体部位之间的穿透（手穿过躯干、双腿交叉穿透等）。

    Args:
        joints: (T, J, 3) 关节位置
        params: 包含 self_penetration_tolerance 等参数
        rest_verts: (V, 3) rest pose 顶点，用于动态标定胶囊半径（可选）
        posed_verts: (T, V, 3) 各帧顶点，mesh 模式需要（未来实现）
        faces: (F, 3) 面片索引，mesh 模式需要（未来实现）
        parts_of_interest: 只检测涉及这些 part 的骨骼对，None=全部。
            使用 BODY_PART_SEGMENTS 中的 key，如 ["L_Hand", "Torso", "L_Thigh"]。
            只检测**至少一端**属于 parts_of_interest 的骨骼对。
        mode: "capsule" (快速胶囊近似) | "mesh" (精确碰撞检测，未来实现)
    Returns:
        dict with:
            self_penetration_ratio: 有穿透的帧占比 (%)
            self_penetration_max_depth: 最大穿透深度 (m)
            self_penetration_pairs: 按严重程度排序的 part 对详细列表
            self_penetration_mode: 实际使用的检测模式
    """
    if mode == "mesh":
        if posed_verts is None or faces is None:
            raise ValueError("mesh 模式需要 posed_verts 和 faces 参数")
        raise NotImplementedError(
            "mesh 精确自穿透检测尚未实现。请使用 mode='capsule' 或等待后续更新。"
        )

    T = joints.shape[0]
    J = joints.shape[1]
    tolerance = params.get("self_penetration_tolerance", 0.01)  # 1cm 容差
    capsule_radii = SMPLH_CAPSULE_RADII  # TODO: 支持从 rest_verts 动态标定

    # 筛选需要检测的骨骼对
    pairs_to_check = SELF_PENETRATION_PAIRS
    if parts_of_interest is not None:
        poi_set = set(parts_of_interest)
        pairs_to_check = [
            (a, b) for a, b in SELF_PENETRATION_PAIRS
            if a in poi_set or b in poi_set
        ]

    if not pairs_to_check:
        return {
            "self_penetration_ratio": 0.0,
            "self_penetration_max_depth": 0.0,
            "self_penetration_pairs": [],
            "self_penetration_mode": mode,
        }

    # 预收集每个 part 的所有骨骼段信息
    def _get_segments_for_part(part_name: str):
        """返回 [(p_idx, q_idx, radius), ...] 列表"""
        segs = BODY_PART_SEGMENTS.get(part_name, [])
        result = []
        for seg in segs:
            if len(seg) == 1:
                # 端点球体
                jidx = seg[0]
                if jidx < J:
                    r = capsule_radii.get(seg, 0.04)
                    result.append((jidx, jidx, r))  # p==q → 球
            else:
                p_idx, q_idx = seg
                if p_idx < J and q_idx < J:
                    r = capsule_radii.get(seg, 0.05)
                    result.append((p_idx, q_idx, r))
        return result

    # 计算每对在 rest pose (frame 0) 下的穿透深度作为基线
    rest_joints_f0 = joints[0]

    # 逐对检测
    pair_results = []
    any_penetration_per_frame = np.zeros(T, dtype=bool)

    for part_a, part_b in pairs_to_check:
        segs_a = _get_segments_for_part(part_a)
        segs_b = _get_segments_for_part(part_b)
        if not segs_a or not segs_b:
            continue

        # 计算 rest pose 下该对的穿透深度作为基线
        # 如果 rest pose 下已经"穿透"（如大腿在 T-pose 下几乎接触），
        # 则只有超过 rest pose 穿透深度的部分才算真正穿透
        rest_max_depth = 0.0
        for pa, qa, ra in segs_a:
            for pb, qb, rb in segs_b:
                d = _capsule_capsule_distance(
                    rest_joints_f0[pa], rest_joints_f0[qa], ra - tolerance,
                    rest_joints_f0[pb], rest_joints_f0[qb], rb - tolerance,
                )
                rest_max_depth = max(rest_max_depth, d)

        # 逐帧逐段对计算穿透
        pair_depth = np.zeros(T, dtype=np.float64)
        for t in range(T):
            max_depth_t = 0.0
            for pa_idx, qa_idx, ra in segs_a:
                for pb_idx, qb_idx, rb in segs_b:
                    depth = _capsule_capsule_distance(
                        joints[t, pa_idx], joints[t, qa_idx], ra - tolerance,
                        joints[t, pb_idx], joints[t, qb_idx], rb - tolerance,
                    )
                    max_depth_t = max(max_depth_t, depth)
            # 只有超过 rest pose 基线的部分才算穿透
            pair_depth[t] = max(0.0, max_depth_t - rest_max_depth)

        pen_frames = pair_depth > 0
        pen_ratio = float(pen_frames.sum() / max(T, 1) * 100)

        if pen_ratio > 0:
            pair_results.append({
                "part_a": part_a,
                "part_b": part_b,
                "ratio": pen_ratio,
                "max_depth": float(pair_depth.max()),
                "avg_depth": float(pair_depth[pen_frames].mean()) if pen_frames.any() else 0.0,
            })
            any_penetration_per_frame |= pen_frames

    # 按 ratio 降序排列
    pair_results.sort(key=lambda x: x["ratio"], reverse=True)

    total_ratio = float(any_penetration_per_frame.sum() / max(T, 1) * 100)
    max_depth = max((p["max_depth"] for p in pair_results), default=0.0)

    return {
        "self_penetration_ratio": total_ratio,
        "self_penetration_max_depth": max_depth,
        "self_penetration_pairs": pair_results,
        "self_penetration_mode": mode,
    }


# ==============================================================================
# Part D-0：震颤 (tremor) 检测
# ==============================================================================


def _detect_axis_reversals(
    signal_1d: np.ndarray,
    fps: float,
    min_component: float,
    max_half_cycle_sec: float,
    max_swing_amplitude: float = float('inf'),
    min_swing_amplitude: float = 0.0,
) -> np.ndarray:
    """对一维速度/角速度信号检测符合振幅范围的反转事件。

    将连续同号帧分为 swing 段，swing 边界即为 reversal。
    reversal 需同时满足：
      1. 该 swing 时长 < max_half_cycle_sec（高频条件）
      2. min_swing_amplitude ≤ 该 swing 的位移振幅 < max_swing_amplitude（振幅范围条件）

    振幅范围的物理含义：
    - tremor 模式：min=0, max=20mm → 检测小幅高频抖动
    - snap 模式：  min=10mm, max=500mm → 检测大幅孤立跳变
    - 翻滚/快跑等合理运动 swing 振幅 100~1000mm，通过 max 上限排除
    - 大幅 popping（跳变）：joint_pop_ratio 和 jerk 指标负责检测

    Args:
        signal_1d: (N,) 一维速度分量序列（如 vel[:,j,axis]），单位 m/s 或 rad/s
        fps: 帧率
        min_component: deadzone 阈值，|signal| < 此值的帧视为静止
        max_half_cycle_sec: 单个 swing 最大时长（秒）
        max_swing_amplitude: swing 位移振幅上限（米 或 弧度），超过则不计为 reversal
        min_swing_amplitude: swing 位移振幅下限（米 或 弧度），低于则不计为 reversal（默认 0.0 = 无下限）
    Returns:
        reversals: (N,) bool 数组，True 表示该位置发生了 reversal
    """
    N = len(signal_1d)
    reversals = np.zeros(N, dtype=bool)
    if N < 2:
        return reversals

    max_half_cycle_frames = int(max_half_cycle_sec * fps)
    dt = 1.0 / fps

    active = np.abs(signal_1d) >= min_component
    pos = signal_1d > min_component

    swing_start = -1
    swing_sign = 0  # +1 / -1

    for i in range(N):
        if not active[i]:
            continue
        cur_sign = 1 if pos[i] else -1
        if swing_sign == 0:
            swing_start = i
            swing_sign = cur_sign
        elif cur_sign != swing_sign:
            swing_len = i - swing_start
            # swing 位移振幅 = |∫v dt| ≈ |Σ v[k]| / fps
            swing_amp = abs(float(np.sum(signal_1d[swing_start:i]))) * dt
            if swing_len <= max_half_cycle_frames and min_swing_amplitude <= swing_amp < max_swing_amplitude:
                reversals[i] = True
            swing_start = i
            swing_sign = cur_sign

    return reversals


def _sliding_window_count(flags: np.ndarray, window: int) -> np.ndarray:
    """用 cumsum 计算滑动窗口内 True 的数量。

    返回长度与 flags 相同的数组，result[t] = sum(flags[t-window+1 : t+1])。
    """
    N = len(flags)
    cum = np.cumsum(flags.astype(np.int32))
    cum_pad = np.concatenate([[0], cum])
    starts = np.maximum(np.arange(1, N + 1) - window, 0)
    return cum_pad[1: N + 1] - cum_pad[starts]


def _axis_path_efficiency(positions_1d: np.ndarray, window: int) -> np.ndarray:
    """计算一维位置序列在滑动窗口内的路径效率。

    path_eff[t] = |pos[t] - pos[t-W+1]| / sum(|diff(pos)|) 在窗口内。
    """
    N = len(positions_1d)
    result = np.ones(N, dtype=np.float64)
    if N < 2:
        return result

    step_abs = np.abs(np.diff(positions_1d))
    cum_step = np.concatenate([[0.0], np.cumsum(step_abs)])

    for t in range(N):
        s = max(0, t - window + 1)
        path_len = cum_step[t] - cum_step[s]
        if path_len < 1e-12:
            result[t] = 1.0
        else:
            disp = abs(positions_1d[t] - positions_1d[s])
            result[t] = disp / path_len
    return result


def _compute_tremor_position(
    joints: np.ndarray,
    fps: float,
    params: Dict,
) -> Dict:
    """Position 空间的震颤 + 孤立跳变检测。

    同时检测两种异常模式（共用 _detect_axis_reversals 底层函数，不同阈值组）：
      - tremor: 小振幅 + 多反转 + 低路径效率（持续微小抖动）
      - snap:   大振幅 + 少反转（孤立的大幅跳变）

    Args:
        joints: (T, J, 3) 全局关节位置（米制）
        fps: 帧率
        params: tremor + snap 参数字典
    Returns:
        dict with tremor_ratio_pos (%), tremor_frames_pos (int),
                   snap_ratio_pos (%), snap_frames_pos (int)
    """
    T, J_all = joints.shape[0], joints.shape[1]
    J = min(J_all, BODY_JOINT_COUNT)
    if T < 4:
        return {"tremor_ratio_pos": 0.0, "tremor_frames_pos": 0,
                "snap_ratio_pos": 0.0, "snap_frames_pos": 0}

    local_joints = joints[:, :J, :] - joints[:, 0:1, :]
    vel = np.diff(local_joints, axis=0) * fps  # (T-1, J, 3)
    T_vel = vel.shape[0]

    # --- Tremor 参数 ---
    min_vel_c = params.get("tremor_min_vel_component", 0.01)
    max_hc = params.get("tremor_max_half_cycle_sec", 0.15)
    max_swing_amp = params.get("tremor_max_swing_amplitude_m", float('inf'))
    window_frames = max(3, int(params.get("tremor_window_sec", 0.5) * fps))
    min_rev = params.get("tremor_min_reversals", 3)
    max_pe = params.get("tremor_max_path_efficiency", 0.5)

    # --- Snap 参数 ---
    snap_min_vel_c = params.get("snap_min_vel_component", 0.1)
    snap_max_hc = params.get("snap_max_half_cycle_sec", 0.15)
    snap_min_amp = params.get("snap_min_swing_amplitude_m", 0.01)
    snap_max_amp = params.get("snap_max_swing_amplitude_m", 0.5)
    snap_window_frames = max(3, int(params.get("snap_window_sec", 0.5) * fps))
    snap_max_rev = params.get("snap_max_reversals_in_window", 2)
    snap_max_pe = params.get("snap_max_path_efficiency", 0.9)

    tremor_flag = np.zeros((T_vel, J), dtype=bool)
    snap_flag = np.zeros((T_vel, J), dtype=bool)

    for j in range(1, J):  # skip pelvis (index 0)
        for axis in range(3):
            # Tremor: 小振幅反转（原有逻辑，不变）
            rev = _detect_axis_reversals(
                vel[:, j, axis], fps, min_vel_c, max_hc, max_swing_amp)
            rev_count = _sliding_window_count(rev, window_frames)
            pe = _axis_path_efficiency(local_joints[:T_vel, j, axis], window_frames)
            tremor_flag[:, j] |= (rev_count >= min_rev) & (pe < max_pe)

            # Snap: 大振幅反转（新增，复用底层函数）
            snap_rev = _detect_axis_reversals(
                vel[:, j, axis], fps, snap_min_vel_c, snap_max_hc,
                max_swing_amplitude=snap_max_amp,
                min_swing_amplitude=snap_min_amp)
            snap_rev_count = _sliding_window_count(snap_rev, snap_window_frames)
            snap_pe = _axis_path_efficiency(local_joints[:T_vel, j, axis], snap_window_frames)
            # snap = 有反转（≥1）+ 反转少（≤snap_max_rev，孤立性）+ 路径效率不太高
            snap_flag[:, j] |= ((snap_rev_count >= 1)
                                & (snap_rev_count <= snap_max_rev)
                                & (snap_pe < snap_max_pe))

    frame_has_tremor = tremor_flag[:, 1:J].any(axis=1)
    tremor_frames = int(frame_has_tremor.sum())
    tremor_ratio = float(tremor_frames / T_vel * 100) if T_vel > 0 else 0.0

    frame_has_snap = snap_flag[:, 1:J].any(axis=1)
    snap_frames = int(frame_has_snap.sum())
    snap_ratio = float(snap_frames / T_vel * 100) if T_vel > 0 else 0.0

    return {"tremor_ratio_pos": tremor_ratio, "tremor_frames_pos": tremor_frames,
            "snap_ratio_pos": snap_ratio, "snap_frames_pos": snap_frames}


def _compute_tremor_rotation(
    joint_rot: np.ndarray,
    fps: float,
    params: Dict,
) -> Dict:
    """Rotation 空间的震颤 + 孤立跳变检测。

    同时检测两种异常模式（共用 _detect_axis_reversals 底层函数，不同阈值组）：
      - tremor: 小振幅 + 多反转 + 低路径效率（持续微小抖动）
      - snap:   大振幅 + 少反转（孤立的大幅跳变）

    Args:
        joint_rot: (T, 21, 3) body joints 轴角（不含 root）
        fps: 帧率
        params: tremor + snap 参数字典
    Returns:
        dict with tremor_ratio_rot (%), tremor_frames_rot (int),
                   snap_ratio_rot (%), snap_frames_rot (int)
    """
    T_rot, J_rot = joint_rot.shape[0], joint_rot.shape[1]
    if T_rot < 4:
        return {"tremor_ratio_rot": 0.0, "tremor_frames_rot": 0,
                "snap_ratio_rot": 0.0, "snap_frames_rot": 0}

    joint_rot_t = torch.from_numpy(joint_rot).float()
    rot_mat = _axis_angle_to_rotation_matrix(
        joint_rot_t.reshape(-1, 3)
    ).reshape(T_rot, J_rot, 3, 3)

    # 帧间相对旋转 → axis-angle 向量（表示角速度方向和大小）
    # R_rel = R[t+1] @ R[t]^T
    R_curr = rot_mat[:-1].reshape(-1, 3, 3)
    R_next = rot_mat[1:].reshape(-1, 3, 3)
    R_rel = torch.bmm(R_next, R_curr.transpose(1, 2))

    # rotation matrix → axis-angle: 使用 trace 和反对称部分
    T_diff = T_rot - 1
    traces = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = ((traces - 1.0) / 2.0).clamp(-1.0, 1.0)
    angles = torch.acos(cos_angle)  # (T_diff*J_rot,)

    # 反对称部分提取轴
    ax = torch.stack([
        R_rel[:, 2, 1] - R_rel[:, 1, 2],
        R_rel[:, 0, 2] - R_rel[:, 2, 0],
        R_rel[:, 1, 0] - R_rel[:, 0, 1],
    ], dim=1)  # (N, 3)
    ax_norm = ax.norm(dim=1, keepdim=True).clamp(min=1e-8)
    ax_unit = ax / ax_norm
    # axis-angle 向量 = angle * axis
    ang_vel_vec = (angles.unsqueeze(1) * ax_unit).reshape(T_diff, J_rot, 3)
    ang_vel_np = (ang_vel_vec * fps).numpy()  # rad/s

    # --- Tremor 参数 ---
    min_ang_c = params.get("tremor_min_angular_vel_component", 0.05)
    max_hc = params.get("tremor_max_half_cycle_sec", 0.15)
    max_swing_amp = params.get("tremor_max_swing_amplitude_rad", float('inf'))
    window_frames = max(3, int(params.get("tremor_window_sec", 0.5) * fps))
    min_rev = params.get("tremor_min_reversals", 3)
    max_pe = params.get("tremor_max_path_efficiency", 0.5)

    # --- Snap 参数 ---
    snap_min_ang_c = params.get("snap_min_angular_vel_component", 1.0)
    snap_max_hc = params.get("snap_max_half_cycle_sec", 0.15)
    snap_min_amp = params.get("snap_min_swing_amplitude_rad", 0.05)
    snap_max_amp = params.get("snap_max_swing_amplitude_rad", 1.0)
    snap_window_frames = max(3, int(params.get("snap_window_sec", 0.5) * fps))
    snap_max_rev = params.get("snap_max_reversals_in_window", 2)
    snap_max_pe = params.get("snap_max_path_efficiency", 0.9)

    # 近似角位置: theta[t] = sum(ang_vel_vec[0:t]) / fps
    ang_pos = np.cumsum(ang_vel_np / fps, axis=0)  # (T_diff, J_rot, 3)

    tremor_flag = np.zeros((T_diff, J_rot), dtype=bool)
    snap_flag = np.zeros((T_diff, J_rot), dtype=bool)

    for j in range(J_rot):
        for axis in range(3):
            # Tremor: 小振幅反转（原有逻辑，不变）
            rev = _detect_axis_reversals(
                ang_vel_np[:, j, axis], fps, min_ang_c, max_hc, max_swing_amp)
            rev_count = _sliding_window_count(rev, window_frames)
            pe = _axis_path_efficiency(ang_pos[:, j, axis], window_frames)
            tremor_flag[:, j] |= (rev_count >= min_rev) & (pe < max_pe)

            # Snap: 大振幅反转（新增）
            snap_rev = _detect_axis_reversals(
                ang_vel_np[:, j, axis], fps, snap_min_ang_c, snap_max_hc,
                max_swing_amplitude=snap_max_amp,
                min_swing_amplitude=snap_min_amp)
            snap_rev_count = _sliding_window_count(snap_rev, snap_window_frames)
            snap_pe = _axis_path_efficiency(ang_pos[:, j, axis], snap_window_frames)
            snap_flag[:, j] |= ((snap_rev_count >= 1)
                                & (snap_rev_count <= snap_max_rev)
                                & (snap_pe < snap_max_pe))

    frame_has_tremor = tremor_flag.any(axis=1)
    tremor_frames = int(frame_has_tremor.sum())
    tremor_ratio = float(tremor_frames / T_diff * 100) if T_diff > 0 else 0.0

    frame_has_snap = snap_flag.any(axis=1)
    snap_frames = int(frame_has_snap.sum())
    snap_ratio = float(snap_frames / T_diff * 100) if T_diff > 0 else 0.0

    return {"tremor_ratio_rot": tremor_ratio, "tremor_frames_rot": tremor_frames,
            "snap_ratio_rot": snap_ratio, "snap_frames_rot": snap_frames}


# ==============================================================================
# Part D：关节级物理指标（jerk / pop / twist / velocity / accel / bone_length）
# [通用算法] 这些函数仅需关节旋转/位置数据，不依赖特定骨骼模型。
# 但 wrist_twist_threshold 等默认阈值是基于 SMPL T-pose 下的实测经验设定的。
# ==============================================================================


def _compute_head_stability(
    joints: np.ndarray,
    joint_rot: np.ndarray,
    fps: float,
    params: Dict,
) -> Dict:
    """头部稳定性检测。

    头部是视觉上最敏感的关节。人类的前庭-眼反射在运动中主动稳定头部，
    因此头部应比其他关节更加平稳。本函数用更严格的阈值单独评估头部角速度，
    区分运动阶段（pelvis 有位移）和静止阶段（pelvis 基本不动）。

    Args:
        joints: (T, J, 3) 全局关节位置（米制）
        joint_rot: (T, 21, 3) body joints 轴角（不含 root）
        fps: 帧率
        params: 包含 head_joint_idx, head_ang_vel_warn_moving/static,
                head_static_pelvis_vel_thresh 等参数
    Returns:
        dict with:
            head_angular_vel_p95: 头部角速度 p95 (deg/s)
            head_jitter_ratio: 超阈值帧占比 (%)
    """
    T = joints.shape[0]
    if T < 2:
        return {"head_angular_vel_p95": 0.0, "head_jitter_ratio": 0.0}

    # Head 在 joint_rot 中的索引（joint_rot 不含 root，所以 HEAD_IDX=15 对应 rot index 14）
    head_jtr_idx = params.get("head_joint_idx", 15)
    head_rot_idx = head_jtr_idx - 1  # joint_rot = poses[:, 1:22, :]
    if head_rot_idx < 0 or head_rot_idx >= joint_rot.shape[1]:
        return {"head_angular_vel_p95": 0.0, "head_jitter_ratio": 0.0}

    dt = 1.0 / fps

    # 头部帧间旋转角度（标量，度/帧 → 度/秒）
    head_rot_t = torch.from_numpy(joint_rot[:, head_rot_idx:head_rot_idx+1, :]).float()
    T_rot = head_rot_t.shape[0]
    if T_rot < 2:
        return {"head_angular_vel_p95": 0.0, "head_jitter_ratio": 0.0}

    head_mats = _axis_angle_to_rotation_matrix(head_rot_t.reshape(-1, 3)).reshape(T_rot, 3, 3)
    head_ang_diff = (
        _angle_between(head_mats[1:].reshape(-1, 3, 3), head_mats[:-1].reshape(-1, 3, 3))
        * 180.0 / np.pi
    ).reshape(-1).numpy()  # (T-1,) 度/帧

    head_ang_vel = head_ang_diff * fps  # 度/秒

    # Pelvis 水平速度（判断运动/静止阶段）
    pelvis_vel_thresh = params.get("head_static_pelvis_vel_thresh", 0.1)  # m/s
    pelvis_pos = joints[:, 0, :]  # (T, 3)
    pelvis_vel_mag = np.linalg.norm(np.diff(pelvis_pos, axis=0), axis=1) * fps  # (T-1,) m/s

    is_static = pelvis_vel_mag < pelvis_vel_thresh

    # 阈值
    warn_moving = params.get("head_ang_vel_warn_moving", 120.0)  # deg/s
    warn_static = params.get("head_ang_vel_warn_static", 15.0)   # deg/s

    # Per-frame threshold：静止时更严格
    per_frame_thresh = np.where(is_static, warn_static, warn_moving)

    # p95 of head angular velocity
    head_angular_vel_p95 = float(np.percentile(head_ang_vel, 95)) if len(head_ang_vel) > 0 else 0.0

    # Jitter ratio: 超过对应阈值的帧占比
    jitter_frames = head_ang_vel > per_frame_thresh
    head_jitter_ratio = float(jitter_frames.sum() / max(len(head_ang_vel), 1) * 100)

    return {
        "head_angular_vel_p95": head_angular_vel_p95,
        "head_jitter_ratio": head_jitter_ratio,
    }



def _compute_joint_based_metrics(
    joints: np.ndarray,
    joint_rot: np.ndarray,
    fps: float,
    params: Dict,
    root_orient: np.ndarray = None,
) -> Dict[str, float]:
    """
    计算仅需关节位置和旋转的物理指标。

    Args:
        joints: (T, J, 3) 关节位置
        joint_rot: (T, 21, 3) body joints 轴角（不含 root）
        fps: 帧率
        params: 参数字典
        root_orient: (T, 3) pelvis 轴角，用于 jerk 分解
    """
    assert joints.ndim == 3 and joints.shape[2] == 3, (
        f"joints 应为 (T, J, 3)，实际 shape={joints.shape}"
    )
    assert joint_rot.ndim == 3 and joint_rot.shape[2] == 3, (
        f"joint_rot 应为 (T, 21, 3)，实际 shape={joint_rot.shape}"
    )
    T, J_total = joints.shape[0], joints.shape[1]
    assert J_total >= 22, (
        f"joints 至少需要 22 个 body joints，实际 shape={joints.shape}"
    )
    assert joint_rot.shape[0] == T, (
        f"joints 帧数 ({T}) 与 joint_rot 帧数 ({joint_rot.shape[0]}) 不一致"
    )
    assert joint_rot.shape[1] == 21, (
        f"joint_rot 应有 21 个关节旋转（不含 root），实际 shape={joint_rot.shape}"
    )
    dt = 1.0 / fps

    # --- 基础统计 ---
    motion_duration_sec = float(T * dt)
    root_pos = joints[:, PELVIS_IDX, :]
    if T >= 2:
        total_distance_m = float(np.sum(np.linalg.norm(np.diff(root_pos, axis=0), axis=1)))
    else:
        total_distance_m = 0.0

    # --- 速度/加速度（pelvis-local 空间）---
    # 去掉全局平移分量，保留全局旋转的影响：
    #   body_local = joints - pelvis_translation
    # 这样 jerk/velocity/acceleration 反映的是肢体关节运动质量，
    # 不受角色整体位移（跑/走/滚）的干扰。
    body_joints_np = joints[:, :22, :]
    body_local = body_joints_np - body_joints_np[:, 0:1, :]
    if T >= 2:
        vel = np.diff(body_local, axis=0) / dt
        vel_mag = np.linalg.norm(vel, axis=2)
        avg_velocity = float(np.mean(vel_mag))
        max_velocity = float(np.max(vel_mag))
    else:
        avg_velocity = max_velocity = 0.0

    if T >= 3:
        accel = np.diff(body_local, n=2, axis=0) / (dt ** 2)
        accel_mag = np.linalg.norm(accel, axis=2)
        avg_acceleration = float(np.mean(accel_mag))
        max_acceleration = float(np.max(accel_mag))
    else:
        avg_acceleration = max_acceleration = 0.0

    # --- Jerk 四分量分解 ---
    # 四个 jerk 指标从不同维度刻画运动平滑度：
    #   jerk_with_rot:     pelvis-local 空间（减去平移、保留旋转）j=1..21 的 jerk [m/s³]
    #                      包含 root rotation 对子关节的影响，反映全局可感知的肢体 jerk
    #   local_pose_jerk:   pelvis 坐标系（减去平移和旋转）j=1..21 的 jerk [m/s³]
    #                      锁定 pelvis 6DoF，纯粹反映肢体关节的运动平滑度
    #   pelvis_rot_jerk:   pelvis 旋转的角 jerk [deg/s³]
    #   pelvis_trans_jerk: pelvis 全局位置的 jerk [m/s³]
    if T >= 4:
        # (1) pelvis_trans_jerk: pelvis 全局位置的三阶导数
        root_t = torch.from_numpy(body_joints_np[:, 0:1, :]).float()
        pelvis_trans_jerk = float(
            ((root_t[3:] - 3 * root_t[2:-1] + 3 * root_t[1:-2] - root_t[:-3]) * (fps ** 3))
            .norm(dim=2)
            .mean()
            .item()
        )

        # (2) jerk_with_rot: pelvis-local（仅去掉平移），j=1..21
        local_j1_t = torch.from_numpy(body_local[:, 1:, :]).float()
        jerk_with_rot = float(
            ((local_j1_t[3:] - 3 * local_j1_t[2:-1] + 3 * local_j1_t[1:-2] - local_j1_t[:-3]) * (fps ** 3))
            .norm(dim=2)
            .mean()
            .item()
        )

        # (3)(4) 需要原始 root orientation（来自 poses[:, 0]，非 joint_rot）
        # joint_rot 经过 body model FK 变换，其 j=0 可能与原始 root 不同
        if root_orient is not None:
            from scipy.spatial.transform import Rotation as ScipyR
            root_rot_mats = ScipyR.from_rotvec(root_orient).as_matrix()  # (T, 3, 3)

            # (3) local_pose_jerk: 变换到 pelvis 坐标系，仅 j=1..21
            body_in_pelvis = np.einsum(
                'tki,tjk->tji', root_rot_mats, body_local[:, 1:, :]
            )  # (T, 21, 3)
            pf_t = torch.from_numpy(body_in_pelvis).float()
            local_pose_jerk = float(
                ((pf_t[3:] - 3 * pf_t[2:-1] + 3 * pf_t[1:-2] - pf_t[:-3]) * (fps ** 3))
                .norm(dim=2)
                .mean()
                .item()
            )

            # (4) pelvis_rot_jerk: 角 jerk (deg/s³)
            R_rel = np.einsum(
                'tji,tjk->tik', root_rot_mats[:-1], root_rot_mats[1:]
            )  # (T-1, 3, 3)
            omega = ScipyR.from_matrix(R_rel).as_rotvec() * fps  # (T-1, 3) rad/s
            ang_jerk = np.diff(omega, n=2, axis=0) * (fps ** 2)  # (T-3, 3) rad/s³
            pelvis_rot_jerk = float(
                np.mean(np.linalg.norm(ang_jerk, axis=1)) * (180.0 / np.pi)
            )
        else:
            local_pose_jerk = 0.0
            pelvis_rot_jerk = 0.0
    else:
        jerk_with_rot = 0.0
        local_pose_jerk = 0.0
        pelvis_rot_jerk = 0.0
        pelvis_trans_jerk = 0.0

    # --- 骨骼长度一致性（变异系数 CV = std/mean * 100%）---
    bone_lengths_list = []
    for p_idx, c_idx in BODY_BONE_PAIRS:
        if p_idx < joints.shape[1] and c_idx < joints.shape[1]:
            bl = np.linalg.norm(joints[:, c_idx, :] - joints[:, p_idx, :], axis=1)
            bone_lengths_list.append(bl)

    if bone_lengths_list and T >= 2:
        bone_lengths_arr = np.stack(bone_lengths_list, axis=1)
        bone_mean = np.mean(bone_lengths_arr, axis=0)
        bone_std = np.std(bone_lengths_arr, axis=0)
        bone_cv = bone_std / np.maximum(bone_mean, 1e-8) * 100.0
        bone_length_cv_mean = float(np.mean(bone_cv))
        bone_length_cv_max = float(np.max(bone_cv))
    else:
        bone_length_cv_mean = bone_length_cv_max = 0.0

    # --- Joint Pop Ratio ---
    ang_pop_thresh = params["ang_pop_thresh"]
    ang_pop_per_joint = params.get("ang_pop_thresh_per_joint", {})
    wrist_twist_threshold = params["wrist_twist_threshold"]

    # params 中的关节 ID 统一使用 Jtr 索引，转为 joint_rot 索引需 -1
    # （joint_rot = poses[:, 1:22, :] 去掉了 root）
    def _jtr_to_rot(jtr_ids):
        return [i - 1 for i in jtr_ids if i >= 1]

    arms_ids = _jtr_to_rot(params["arms_joint_ids"])
    legs_ids = _jtr_to_rot(params["legs_joint_ids"])
    wrists_ids = _jtr_to_rot(params["wrists_joint_ids"])
    ankles_ids = _jtr_to_rot(params["ankles_joint_ids"])

    joint_rot_t = torch.from_numpy(joint_rot).float()
    T_rot, J_rot = joint_rot_t.shape[0], joint_rot_t.shape[1]

    if T_rot >= 2:
        rot_mat = _axis_angle_to_rotation_matrix(joint_rot_t.reshape(-1, 3)).reshape(T_rot, J_rot, 3, 3)
        ang_diff_deg = (
            _angle_between(rot_mat[1:].reshape(-1, 3, 3), rot_mat[:-1].reshape(-1, 3, 3))
            .reshape(T_rot - 1, J_rot, 1)
            * 180.0
            / np.pi
        )

        thresh_array = torch.full((J_rot,), ang_pop_thresh)
        for jtr_idx, j_thresh in ang_pop_per_joint.items():
            rot_idx = jtr_idx - 1
            if 0 <= rot_idx < J_rot:
                thresh_array[rot_idx] = j_thresh
        joint_pop_flag = ang_diff_deg[:, :, 0] > thresh_array.unsqueeze(0)
        total_frames = T_rot - 1

        def _pop_rate(ids):
            if not ids:
                return 0.0
            valid_ids = [i for i in ids if i < J_rot]
            if not valid_ids:
                return 0.0
            return float(joint_pop_flag[:, valid_ids].sum() / (total_frames * len(valid_ids)) * 100)

        overall_pop_ratio = float(joint_pop_flag.sum() / (total_frames * J_rot) * 100)
        arms_pop_ratio = _pop_rate(arms_ids)
        legs_pop_ratio = _pop_rate(legs_ids)
        wrists_pop_ratio = _pop_rate(wrists_ids)
        ankles_pop_ratio = _pop_rate(ankles_ids)
    else:
        overall_pop_ratio = arms_pop_ratio = legs_pop_ratio = 0.0
        wrists_pop_ratio = ankles_pop_ratio = 0.0

    # --- Wrist Twist Ratio ---
    if T_rot >= 1 and wrists_ids:
        valid_wrist_ids = [i for i in wrists_ids if i < J_rot]
        if valid_wrist_ids:
            wrist_angles = torch.norm(joint_rot_t[:, valid_wrist_ids], dim=2) * 180.0 / np.pi
            wrist_twist_flag = wrist_angles > wrist_twist_threshold
            wrist_twist_ratio = float(wrist_twist_flag.any(dim=1).float().mean() * 100)
        else:
            wrist_twist_ratio = 0.0
    else:
        wrist_twist_ratio = 0.0

    # --- Tremor (震颤) + Snap (孤立跳变) ---
    pos_tremor = _compute_tremor_position(joints, fps, params)
    rot_tremor = _compute_tremor_rotation(joint_rot, fps, params)
    tremor_ratio = max(pos_tremor["tremor_ratio_pos"], rot_tremor["tremor_ratio_rot"])
    snap_ratio = max(pos_tremor["snap_ratio_pos"], rot_tremor["snap_ratio_rot"])

    # --- Head Stability (头部稳定性) ---
    head_stability = _compute_head_stability(joints, joint_rot, fps, params)

    # --- Per-frame summary statistics (for frontend detail view) ---
    frame_stats = {}
    if T >= 2:
        # Velocity: per-frame mean/max across joints
        per_frame_vel_mean = np.mean(vel_mag, axis=1)  # (T-1,)
        frame_stats["avg_velocity"] = _compute_frame_stats(per_frame_vel_mean)
        per_frame_vel_max = np.max(vel_mag, axis=1)
        frame_stats["max_velocity"] = _compute_frame_stats(per_frame_vel_max)
    if T >= 3:
        per_frame_accel_mean = np.mean(accel_mag, axis=1)  # (T-2,)
        frame_stats["avg_acceleration"] = _compute_frame_stats(per_frame_accel_mean)
    if T >= 4:
        # Jerk with rot: per-frame mean jerk across joints
        local_j1_np = body_local[:, 1:, :]
        jerk_np = (local_j1_np[3:] - 3 * local_j1_np[2:-1] + 3 * local_j1_np[1:-2] - local_j1_np[:-3]) * (fps ** 3)
        per_frame_jerk = np.mean(np.linalg.norm(jerk_np, axis=2), axis=1)  # (T-3,)
        frame_stats["jerk_with_rot"] = _compute_frame_stats(per_frame_jerk)

        # Pelvis trans jerk: per-frame
        root_np = body_joints_np[:, 0:1, :]
        pelvis_jerk_np = (root_np[3:] - 3 * root_np[2:-1] + 3 * root_np[1:-2] - root_np[:-3]) * (fps ** 3)
        per_frame_pelvis_jerk = np.linalg.norm(pelvis_jerk_np[:, 0, :], axis=1)  # (T-3,)
        frame_stats["pelvis_trans_jerk"] = _compute_frame_stats(per_frame_pelvis_jerk)

    return {
        "motion_duration_sec": motion_duration_sec,
        "total_distance": total_distance_m,
        "avg_velocity": avg_velocity,
        "max_velocity": max_velocity,
        "avg_acceleration": avg_acceleration,
        "max_acceleration": max_acceleration,
        "jerk_with_rot": jerk_with_rot,
        "local_pose_jerk": local_pose_jerk,
        "pelvis_rot_jerk": pelvis_rot_jerk,
        "pelvis_trans_jerk": pelvis_trans_jerk,
        "tremor_ratio": tremor_ratio,
        "tremor_ratio_pos": pos_tremor["tremor_ratio_pos"],
        "tremor_ratio_rot": rot_tremor["tremor_ratio_rot"],
        "snap_ratio": snap_ratio,
        "snap_ratio_pos": pos_tremor["snap_ratio_pos"],
        "snap_ratio_rot": rot_tremor["snap_ratio_rot"],
        "head_angular_vel_p95": head_stability["head_angular_vel_p95"],
        "head_jitter_ratio": head_stability["head_jitter_ratio"],
        "bone_length_cv_mean": bone_length_cv_mean,
        "bone_length_cv_max": bone_length_cv_max,
        "joint_pop_ratio": overall_pop_ratio,
        "arms_pop_ratio": arms_pop_ratio,
        "legs_pop_ratio": legs_pop_ratio,
        "wrists_pop_ratio": wrists_pop_ratio,
        "ankles_pop_ratio": ankles_pop_ratio,
        "wrist_twist_ratio": wrist_twist_ratio,
        "frame_stats": frame_stats,
    }


# ==============================================================================
# Part E：主入口
# ==============================================================================


def _compute_frame_stats(arr) -> Optional[Dict]:
    """Compute summary statistics from a per-frame 1D array.

    Args:
        arr: 1D numpy array of per-frame values

    Returns:
        Dict with mean, std, min, max, median, p5, p25, p75, p95, n_frames;
        None if arr is empty or all-NaN.
    """
    if arr is None or len(arr) == 0:
        return None
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "n_frames": int(len(arr)),
    }


def _apply_unit_conversion(metrics: Dict, scale: float) -> Dict:
    """将内部米制结果按 scale 转换到目标 output_unit。"""
    SCALE_KEYS = {
        "total_distance", "avg_penetrate", "avg_float",
        "avg_skate", "frame_avg_skate", "phys_err",
        "avg_velocity", "max_velocity",
        "avg_acceleration", "max_acceleration",
    }
    # jerk 系列指标固定使用 m/s³ 或 rad/s³，不参与单位缩放

    result = {}
    for k, v in metrics.items():
        if k == "frame_stats" and isinstance(v, dict):
            # Scale frame_stats for length-based metrics
            scaled_fs = {}
            for fs_key, fs_val in v.items():
                if fs_val and isinstance(fs_val, dict) and fs_key in SCALE_KEYS:
                    scaled_fs[fs_key] = {
                        sk: sv * scale if isinstance(sv, (int, float)) and sk != "n_frames" else sv
                        for sk, sv in fs_val.items()
                    }
                else:
                    scaled_fs[fs_key] = fs_val
            result[k] = scaled_fs
        elif not isinstance(v, (int, float)):
            result[k] = v
        elif k in SCALE_KEYS:
            result[k] = v * scale
        else:
            result[k] = v
    return result


def compute_phys_metrics(
    file_path: str,
    smpl_model_path: Optional[str] = None,
    device: str = _DEFAULT_DEVICE,
    use_cache: bool = True,
    output_unit: str = "cm",
    **override_params,
) -> Dict:
    """
    计算单个动画文件的物理误差指标。

    两级计算策略：
      - 第一级（关节级）：jerk (with_rot/local_pose/rot/trans), pop, twist, velocity, acceleration, bone_length 等。
        当 HDF 文件包含预计算的 joints3d 时，无需 smpl_model_path 即可计算。
      - 第二级（顶点级）：penetrate, float, skate。
        需要 smpl_model_path，统一走 BodyModel 重新计算 verts+joints。

    输入单位通过骨骼长度自动推断，内部统一归一化到米计算，
    最终按 output_unit 转换输出。

    Args:
        file_path: 动画文件路径（.npz 或 .h5）
        smpl_model_path: SMPLH 模型路径（None 则只计算第一级指标）
        device: torch 计算设备
        use_cache: 是否使用内存缓存
        output_unit: 输出长度单位 ("mm"/"cm"/"dm"/"m")，默认 "cm"
        **override_params: 覆盖默认参数
    """
    if output_unit not in UNIT_TO_METERS:
        return {"error": f"不支持的 output_unit: {output_unit}，可选: {list(UNIT_TO_METERS.keys())}"}

    if not HAS_TORCH:
        return {"error": "torch 未安装，无法计算物理指标"}

    if not os.path.exists(file_path):
        return {"error": f"文件不存在: {file_path}"}

    _floor_mode = override_params.get("floor_mode", DEFAULT_PHYS_PARAMS["floor_mode"])
    cache_key = f"{file_path}::{output_unit}::{_floor_mode}"
    if use_cache and cache_key in PHYS_METRICS_CACHE:
        cached = PHYS_METRICS_CACHE[cache_key]
        current_mtime = os.path.getmtime(file_path)
        if abs(cached["mtime"] - current_mtime) < 1.0:
            logger.debug(f"[phys_metrics] 使用缓存结果: {file_path}")
            return cached["metrics"]

    params = {**DEFAULT_PHYS_PARAMS, **override_params}

    try:
        logger.info(f"[phys_metrics] 开始计算: {file_path}")

        motion = load_motion_data(file_path)
        fps = motion["fps"]
        logger.debug(f"[phys_metrics] poses={motion['poses'].shape}, fps={fps}, type={motion['smpl_type']}")

        has_bodymodel = smpl_model_path is not None and os.path.exists(smpl_model_path)

        if has_bodymodel:
            joint_rot, verts, joints, rest_verts, rest_joints, faces, fps_eff = compute_verts_joints(
                poses=motion["poses"],
                betas=motion["betas"],
                trans=motion["trans"],
                smpl_type=motion["smpl_type"],
                smpl_model_path=smpl_model_path,
                device=device,
                chunk_size=params["chunk_size"],
                fps=fps,
            )
            logger.debug(f"[phys_metrics] verts={verts.shape}, joints={joints.shape}, fps_eff={fps_eff}")

            # 推断输入单位并归一化到米
            input_unit = infer_input_unit(joints[0])
            if input_unit != "m":
                s = UNIT_TO_METERS[input_unit]
                logger.info(f"[phys_metrics] 检测到输入单位={input_unit}，归一化到米 (×{s})")
                verts = verts * s
                joints = joints * s
                rest_verts = rest_verts * s
                rest_joints = rest_joints * s

            root_orient = motion["poses"][:, 0, :].copy()
            metrics = _compute_joint_based_metrics(joints, joint_rot, fps_eff, params,
                                                   root_orient=root_orient)

            verts_metrics = _compute_verts_based_metrics(
                verts, joints, fps_eff, params, rest_verts, rest_joints, faces=faces,
            ) if select_foot_regions is not None else {}
            # Merge verts metrics, handling frame_stats specially
            verts_frame_stats = verts_metrics.pop("frame_stats", {})
            metrics.update(verts_metrics)
            if "frame_stats" in metrics:
                metrics["frame_stats"].update(verts_frame_stats)
            else:
                metrics["frame_stats"] = verts_frame_stats

            # --- Mesh Distortion 指标 ---
            if HAS_MESH_DISTORTION and faces is not None:
                try:
                    md_metrics = _compute_mesh_distortion_metrics(
                        verts, rest_verts, faces,
                        device=device,
                        chunk_size=params["chunk_size"],
                    )
                    metrics.update(md_metrics)
                except Exception as _md_e:
                    logger.warning(f"[phys_metrics] mesh distortion 计算失败: {_md_e}")

            # --- TBS 异常关节检测 ---
            if HAS_TBS:
                try:
                    body_J = min(BODY_JOINT_COUNT, rest_joints.shape[0])
                    body_parents = list(SMPLX_PARENT[:body_J])
                    body_offsets = np.zeros((body_J, 3), dtype=np.float32)
                    body_offsets[0] = rest_joints[0]
                    for _j in range(1, body_J):
                        body_offsets[_j] = rest_joints[_j] - rest_joints[body_parents[_j]]

                    tbs_result = has_distorted_joints(
                        local_rot=motion["poses"][:, :body_J, :],
                        parents=body_parents,
                        offsets=body_offsets,
                        abnormal_thresh_ratio=0.05,
                        chunk_size=params["chunk_size"],
                        verbose=True,
                    )
                    metrics["tbs_has_distortion"] = tbs_result["has_distortion"]
                    metrics["tbs_abnormal_ratio"] = tbs_result["total_abnormal_ratio"]
                    metrics["tbs_abnormal_joint_names"] = ",".join(tbs_result["abnormal_joint_names"])
                    if "per_joint_detail" in tbs_result:
                        details = []
                        for j_idx, d in tbs_result["per_joint_detail"].items():
                            details.append({
                                "name": d["name"],
                                "ratio": round(d["abnormal_ratio"] * 100, 1),
                                "twist_range": [round(d["twist_range"][0], 1), round(d["twist_range"][1], 1)],
                                "bend_range": [round(d["bend_range"][0], 1), round(d["bend_range"][1], 1)],
                                "spread_range": [round(d["spread_range"][0], 1), round(d["spread_range"][1], 1)],
                                "twist_limits": [round(d["twist_limits"][0], 1), round(d["twist_limits"][1], 1)],
                                "bend_limits": [round(d["bend_limits"][0], 1), round(d["bend_limits"][1], 1)],
                                "spread_limits": [round(d["spread_limits"][0], 1), round(d["spread_limits"][1], 1)],
                                "twist_exceed": round(d["twist_exceed_ratio"] * 100, 1),
                                "bend_exceed": round(d["bend_exceed_ratio"] * 100, 1),
                                "spread_exceed": round(d["spread_exceed_ratio"] * 100, 1),
                            })
                        metrics["tbs_per_joint"] = details
                except Exception as _tbs_e:
                    logger.warning(f"[phys_metrics] TBS 异常检测失败: {_tbs_e}")

            # --- Self-Penetration 自穿透检测 ---
            try:
                sp_mode = params.get("self_penetration_mode", "capsule")
                sp_poi = params.get("self_penetration_parts_of_interest", None)
                sp_metrics = _compute_self_penetration(
                    joints, params,
                    rest_verts=rest_verts,
                    posed_verts=verts if sp_mode == "mesh" else None,
                    faces=faces if sp_mode == "mesh" else None,
                    parts_of_interest=sp_poi,
                    mode=sp_mode,
                )
                metrics.update(sp_metrics)
            except NotImplementedError as _sp_ni:
                logger.warning(f"[phys_metrics] 自穿透检测: {_sp_ni}")
            except Exception as _sp_e:
                logger.warning(f"[phys_metrics] 自穿透检测失败: {_sp_e}")
        else:
            has_precomputed = motion.get("joints3d") is not None

            if has_precomputed:
                joints = motion["joints3d"]
                logger.debug(f"[phys_metrics] 使用 HDF 预计算 joints3d: {joints.shape}")
            else:
                if smpl_model_path is None:
                    return {"error": "需要 smpl_model_path 或 HDF 文件中包含预计算的 joints3d"}
                return {"error": f"SMPL 模型文件不存在: {smpl_model_path}"}

            poses = motion["poses"]
            T_poses = poses.shape[0]
            T_joints = joints.shape[0]
            if T_poses != T_joints:
                T_min = min(T_poses, T_joints)
                logger.warning(
                    f"[phys_metrics] poses 帧数 ({T_poses}) 与 joints3d 帧数 ({T_joints}) "
                    f"不一致，截断为 {T_min}"
                )
                poses = poses[:T_min]
                joints = joints[:T_min]

            # 推断输入单位并归一化到米
            input_unit = infer_input_unit(joints[0])
            if input_unit != "m":
                s = UNIT_TO_METERS[input_unit]
                logger.info(f"[phys_metrics] 检测到输入单位={input_unit}，归一化到米 (×{s})")
                joints = joints * s

            if poses.shape[1] >= 22:
                joint_rot = poses[:, 1:22, :].copy()
            else:
                joint_rot = np.zeros((poses.shape[0], 21, 3), dtype=np.float32)

            root_orient = poses[:, 0, :].copy()
            (joints, joint_rot), fps_eff = _downsample_to_target_fps(fps, joints, joint_rot)
            # downsample root_orient to match
            if joints.shape[0] < root_orient.shape[0]:
                step = root_orient.shape[0] / joints.shape[0]
                indices = np.round(np.arange(joints.shape[0]) * step).astype(int)
                indices = np.clip(indices, 0, root_orient.shape[0] - 1)
                root_orient = root_orient[indices]

            metrics = _compute_joint_based_metrics(joints, joint_rot, fps_eff, params,
                                                   root_orient=root_orient)
            logger.info("[phys_metrics] 仅计算关节级指标（无 BodyModel）")

        # 从米转换到 output_unit
        out_scale = _unit_scale("m", output_unit)
        metrics = _apply_unit_conversion(metrics, out_scale)
        metrics["length_unit"] = output_unit

        logger.info(f"[phys_metrics] 完成: {len(metrics)} 项指标, output_unit={output_unit}")

        if use_cache:
            PHYS_METRICS_CACHE[cache_key] = {
                "metrics": metrics,
                "mtime": os.path.getmtime(file_path),
            }

        return metrics

    except Exception as e:
        logger.exception(f"[phys_metrics] 计算失败: {file_path}")
        return {"error": str(e)}


# ==============================================================================
# CLI 入口
# ==============================================================================


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="计算动画文件的物理误差指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
        "  python phys_metrics.py motion.npz\n"
        "  python phys_metrics.py motion.h5 --smpl-model /path/to/model.npz\n"
        "  python phys_metrics.py test_data/*.npz -m /path/to/model.npz -v\n"
        "  python phys_metrics.py motion.npz --output-unit mm\n",
    )
    parser.add_argument("files", nargs="+", help="NPZ/H5 动画文件路径")
    parser.add_argument(
        "--smpl-model", "-m", type=str, default=None,
        help=f"SMPLH 模型路径（不提供则只计算关节级指标）。默认尝试: {DEFAULT_SMPLH_MODEL_PATH}",
    )
    parser.add_argument("--device", "-d", type=str, default=_DEFAULT_DEVICE,
                        help="计算设备 (cpu/cuda)，默认自动选择（有 GPU 用 GPU，否则 CPU）")
    parser.add_argument("--output-unit", "-u", type=str, default="cm",
                        choices=["mm", "cm", "dm", "m"],
                        help="输出长度单位 (默认: cm)")
    parser.add_argument("--save", "-s", type=str, default=None,
                        help="将结果保存为 JSONL 文件（每行一条记录）")
    parser.add_argument("--floor-mode", type=str, default="first_n_seconds",
                        choices=["first_n_seconds", "fixed_zero", "first_n_frames", "global_min", "fixed_value"],
                        help="地面高度估算模式 (默认: first_n_seconds, 取前2秒所有顶点最低点)")
    parser.add_argument("--floor-height", type=float, default=None,
                        help="floor_mode=fixed_value 时指定地面高度（米）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示 DEBUG 日志")
    return parser.parse_args()


if __name__ == "__main__":
    import json as _json

    args = parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    smpl_model = args.smpl_model
    if smpl_model is None and os.path.exists(DEFAULT_SMPLH_MODEL_PATH):
        smpl_model = DEFAULT_SMPLH_MODEL_PATH
        logger.info(f"[phys_metrics] 自动使用默认 SMPL 模型: {smpl_model}")

    all_results = []

    for f in args.files:
        if not os.path.exists(f):
            logger.error(f"文件不存在: {f}")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"文件: {f}")
        logger.info(f"{'=' * 60}")

        floor_params = {"floor_mode": args.floor_mode}
        if args.floor_height is not None:
            floor_params["floor_height_value"] = args.floor_height

        metrics = compute_phys_metrics(
            f, smpl_model_path=smpl_model, device=args.device,
            use_cache=False, output_unit=args.output_unit,
            **floor_params,
        )

        if "error" in metrics:
            logger.error(f"  计算失败: {metrics['error']}")
            all_results.append({"file": f, "error": metrics["error"]})
            continue

        unit = metrics.get("length_unit", args.output_unit)
        logger.info(f"计算结果 (长度单位: {unit}):")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k:30s} = {v:.6f}")
            else:
                logger.info(f"  {k:30s} = {v}")

        all_results.append({"file": os.path.basename(f), **metrics})

    if args.save and all_results:
        with open(args.save, "w", encoding="utf-8") as fout:
            for rec in all_results:
                fout.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"\n结果已保存到: {args.save} ({len(all_results)} 条记录)")
