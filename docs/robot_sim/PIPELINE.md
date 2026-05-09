# Robot Simulation Pipeline: HyMotion → Unitree G1

## 全流程概述

从文本描述到机器人仿真执行的完整链路：

```
Text Prompt ("a person walks forward")
    │
    ▼
┌──────────────────────────┐
│  HyMotion T2M-Lite       │  本仓库 (hftrainer)
│  Flow Matching, 0.46B    │  SMPL 22-joint 135/201-dim
│  MMDiT Transformer       │
└──────────┬───────────────┘
           │
    ▼
┌──────────────────────────┐
│  SMPL → G1 Retargeting   │  本仓库 (hftrainer)
│  Joint decomposition     │  G1 29-DOF joint angles
│  + Euler → DOF mapping   │
└──────────┬───────────────┘
           │
    ▼
┌──────────────────────────┐
│  Isaac Gym / ASAP        │  外部 (LeCAR-Lab/ASAP)
│  PPO Motion Tracking     │  4096 并行环境
│  Reward: tracking error  │
└──────────┬───────────────┘
           │
    ▼
┌──────────────────────────┐
│  Sim2Sim (MuJoCo)        │  外部 (ASAP sim2real/)
│  Cross-sim validation    │
└──────────┬───────────────┘
           │
    ▼
┌──────────────────────────┐
│  Sim2Real (Unitree SDK)  │  外部 (Unitree SDK + ROS2)
│  Real G1 deployment      │
└──────────────────────────┘
```

## 已实现模块（本仓库）

### 1. SMPL → G1 Retargeting

**位置**: `hftrainer/models/motion/components/retarget/`

**核心类**: `SMPLToG1Retargeter`

将 HyMotion 生成的 SMPL 22-joint rotation_6d 动作转换为 Unitree G1 29-DOF 关节角度。

#### 关节对应关系

| SMPL 关节        | G1 关节                              | 分解方式       |
|-----------------|--------------------------------------|---------------|
| L_Hip (1)       | left_hip_{pitch,roll,yaw}           | ZXY Euler     |
| R_Hip (2)       | right_hip_{pitch,roll,yaw}          | ZXY Euler     |
| L_Knee (4)      | left_knee                            | Y (pitch)     |
| R_Knee (5)      | right_knee                           | Y (pitch)     |
| L_Ankle (7)     | left_ankle_{pitch,roll}             | YX Euler      |
| R_Ankle (8)     | right_ankle_{pitch,roll}            | YX Euler      |
| Spine1 (3)      | waist_{yaw,roll,pitch}              | ZXY Euler     |
| L_Shoulder (16) | left_shoulder_{pitch,roll,yaw}      | YXZ Euler     |
| R_Shoulder (17) | right_shoulder_{pitch,roll,yaw}     | YXZ Euler     |
| L_Elbow (18)    | left_elbow                           | Y (pitch)     |
| R_Elbow (19)    | right_elbow                          | Y (pitch)     |
| L_Wrist (20)    | left_wrist_{roll,pitch,yaw}         | XYZ Euler     |
| R_Wrist (21)    | right_wrist_{roll,pitch,yaw}        | XYZ Euler     |

#### 关键技术点

1. **Rot6D 约定转换**: HyMotion 使用 row-major rot6d，需要 reorder `[0,2,4,1,3,5]` 转为 column-major 后才能调用 `rotation_6d_to_matrix`
2. **Rest-Pose 校准**: SMPL T-pose 手臂水平，G1 rest-pose 手臂下垂，肩部减去 90° roll 偏移
3. **关节限位**: 所有输出角度 clamp 到 G1 URDF 定义的硬件限位范围内
4. **速度计算**: 通过有限差分得到关节角速度，供 RL 训练使用

#### 使用方式

```python
from hftrainer.models.motion.components.retarget import SMPLToG1Retargeter

retargeter = SMPLToG1Retargeter(g1_dof=29)

# 从 HyMotion 135-dim 输出直接 retarget
result = retargeter.retarget_from_hymotion(motion_135, fps=30.0)
# result['joint_angles']: (T, 29) G1 关节角度 (rad)
# result['root_pos']: (T, 3) 根节点位置 (m)
# result['root_orient_quat']: (T, 4) 根节点朝向 (wxyz)

# 导出为 ASAP 训练格式
retargeter.to_asap_pkl(result, 'output/g1_motion.pkl')

# 导出为 MuJoCo qpos 格式
qpos = retargeter.to_mujoco_qpos(result)  # (T, 36)
```

### 2. Isaac Gym / ASAP 集成

**位置**: `hftrainer/models/motion/components/retarget/isaac_gym_bridge.py`

**核心类**: `ASAPConfigGenerator`, `ReferenceMotionManager`

自动生成 ASAP 训练所需的配置和命令：

```python
from hftrainer.models.motion.components.retarget.isaac_gym_bridge import ASAPConfigGenerator

gen = ASAPConfigGenerator(asap_root='~/ASAP', num_envs=4096)
cmd = gen.generate_training_command('output/g1_motion.pkl')
print(cmd)
# cd ~/ASAP && python humanoidverse/train_agent.py ...
```

### 3. 端到端 Pipeline 脚本

**位置**: `tools/robot_sim/text_to_g1.py`

```bash
# 一键生成 + retarget + 输出 ASAP 训练命令
python tools/robot_sim/text_to_g1.py \
    --prompt "a person walks forward slowly" \
    --config configs/robot_sim/g1_motion_tracking.py \
    --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --output output/g1_walk/ \
    --generate-train-cmd --asap-root ~/ASAP

# 从已有动作文件 retarget
python tools/robot_sim/text_to_g1.py \
    --input-npz output/smpl_motion.npz \
    --output output/g1_retarget/

# 批量处理
python tools/robot_sim/text_to_g1.py \
    --prompt-file prompts.txt \
    --output output/g1_batch/ \
    --config configs/robot_sim/g1_motion_tracking.py \
    --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt
```

## 外部依赖安装（ASAP + Isaac Gym）

```bash
# 使用安装助手
python tools/robot_sim/setup_asap.py --install-dir ~/ASAP

# 或手动安装
# Step 1: Isaac Gym Preview 4
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf IsaacGym_Preview_4_Package.tar.gz
pip install -e isaacgym/python

# Step 2: ASAP
git clone https://github.com/LeCAR-Lab/ASAP.git ~/ASAP
cd ~/ASAP && pip install -e . && pip install -e isaac_utils

# Step 3: SMPL 数据
# 下载 SMPL v1.1.0 放到 ~/ASAP/humanoidverse/data/smpl/

# Step 4 (可选): MuJoCo (用于 sim2sim)
pip install mujoco
```

## ASAP 训练流程

完成 retarget 后，使用 ASAP 的 motion tracking 训练机器人模仿生成的动作：

```bash
# Step 1: 训练 motion tracking policy
cd ~/ASAP
python humanoidverse/train_agent.py \
    +simulator=isaacgym \
    +exp=motion_tracking \
    +robot=g1/g1_29dof_anneal_23dof \
    +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
    +rewards=motion_tracking/reward_motion_tracking_dm_2real \
    num_envs=4096 \
    robot.motion.motion_file="/path/to/g1_motion_asap.pkl" \
    rewards.reward_penalty_curriculum=True

# Step 2: 评估
python humanoidverse/eval_agent.py \
    +checkpoint=logs/MotionTracking/model_5800.pt

# Step 3: Sim2Sim (MuJoCo验证)
# Terminal 1:
cd sim2real && python sim_env/base_sim.py --config=config/g1_29dof_hist.yaml
# Terminal 2:
python rl_policy/deepmimic_dec_loco_height.py \
    --config=config/g1_29dof_hist.yaml \
    --mimic_model_paths=./models/mimic/model.onnx
```

## 文件结构

```
hftrainer/
├── hftrainer/models/motion/components/retarget/
│   ├── __init__.py                  # Public API
│   ├── smpl_to_g1.py               # SMPL→G1 retargeting core
│   └── isaac_gym_bridge.py         # ASAP/Isaac Gym integration
├── tools/robot_sim/
│   ├── text_to_g1.py               # End-to-end pipeline
│   └── setup_asap.py               # Environment setup helper
├── configs/robot_sim/
│   └── g1_motion_tracking.py       # Pipeline configuration
├── tests/smoke/
│   └── test_retarget_g1.py         # Smoke tests
└── docs/robot_sim/
    └── PIPELINE.md                  # This document
```

## 已知局限与未来工作

### 当前局限

1. **关节映射为静态分解**: 当前使用 Euler 角分解，在 gimbal lock 附近可能有数值问题。
   ASAP 的原始 retarget pipeline 使用基于优化的方法 (`fit_smpl_motion.py`) 效果更好。

2. **肢体比例差异**: SMPL 人体和 G1 机器人的肢体比例不同，纯关节角度 retarget
   会导致末端执行器位置偏差。RL policy 的 tracking reward 会部分补偿这个问题。

3. **无脚部接触约束**: 生成的动作没有考虑脚部接触力学，可能出现滑步。
   ASAP 的 reward 设计（foot contact reward）在 RL 训练阶段处理这个问题。

### 后续改进方向

1. **接入 ASAP 的优化 retarget**: 使用 `fit_smpl_motion.py` 的梯度优化方法替代
   静态 Euler 分解，减少 FK 误差。

2. **Delta Action Model**: ASAP 的 Real2Sim2Real pipeline 使用 delta action model
   来弥合仿真与真实世界的物理差异。

3. **T2M 模型微调**: 针对 G1 可执行的动作范围微调 T2M 模型，避免生成超出
   机器人运动能力的动作。

4. **在线 retarget + RL**: 将 retarget 集成到 RL reward 中，让 policy
   直接从 SMPL 参考动作学习，不需要离线 retarget。

## 参考项目

| 项目 | 用途 | 链接 |
|------|------|------|
| ASAP | Sim2Real motion tracking | [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP) |
| HumanoidVerse | Multi-sim RL framework | [LeCAR-Lab/HumanoidVerse](https://github.com/LeCAR-Lab/ASAP) |
| Isaac Gym | GPU-accelerated simulation | [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) |
| InterMimic | Full-body interaction imitation | [Sirui-Xu/InterMimic](https://github.com/Sirui-Xu/InterMimic) |
| Unitree SDK | Real robot control | [unitreerobotics/unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2_python) |
