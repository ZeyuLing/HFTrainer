# Physics-Aware Motion Generation: RL + Simulation Research

> **状态**: 调研完成，待实施
> **日期**: 2026-03-23
> **目标**: 使用强化学习和物理仿真提升动作生成模型（HyMotion M2M 等）的物理真实性

---

## 1. 问题定义

当前 flow matching / diffusion 动作生成模型输出的是 **kinematic motion**——关节角度序列。
这些输出在运动学上合理，但物理上可能存在：

| 缺陷 | 描述 |
|------|------|
| Foot skating | 脚接触地面时仍有滑移 |
| Ground penetration | 脚/身体穿透地面 |
| Floating | 应接触地面时浮空 |
| Jitter | 高频抖动 |
| 物理不可行 | 空中转身、无支撑跳跃等 |
| 无场景交互 | 不考虑椅子、障碍物等环境 |

---

## 2. 三条技术路径

### 路径 A: 推理时物理修正（Inference-time Physics Correction）

**核心思路**: 生成模型不改，在推理 pipeline 末端加物理仿真器 + RL 追踪 policy 修正输出。

```
HyMotion M2M 生成 motion (135-dim)
    → FK 到关节位置
    → Isaac Gym / MuJoCo 中用 RL policy 追踪
    → 物理修正后的 motion
```

**代表工作**:

- **PhysDiff** (NVIDIA, arXiv 2212.02500):
  - 在 diffusion 每个 denoising step 插入物理投影
  - 预训练 RL motion imitation policy（一次性成本）
  - 不改训练代码，只改推理 pipeline
  - foot sliding 降低 78-94%，FID 提升 >20%
  - 代码未完全开源，但方法描述详细

- **PHC** (Perpetual Humanoid Control, ICCV 2023, github.com/ZhengyiLuo/PHC):
  - 在 AMASS 11k+ motion 上训练的通用 motion tracking policy
  - 100% tracking 成功率，支持失败自恢复
  - Isaac Gym 原生，GPU 并行
  - 完全开源，有预训练 checkpoint
  - 后续: PHC+, PULSE (语言控制), UniHSI (跨骨骼泛化)

**优势**: 不改训练、可插拔、开源可用
**劣势**: 增加推理延迟、需要 Isaac Gym 环境、需要 135-dim ↔ SMPL torque 的桥接
**落地难度**: ⭐⭐⭐ (中等)

---

### 路径 B: 训练时可微物理 Loss（Differentiable Physics Loss）

**核心思路**: 在 flow matching loss 基础上增加可微的物理约束 loss，无需仿真器。

```python
loss = flow_matching_loss
     + λ_fs  * foot_skating_loss      # 接地时脚部速度惩罚
     + λ_pen * penetration_loss        # 地面穿透惩罚
     + λ_jit * jitter_loss             # 加速度平滑
     + λ_fk  * fk_consistency_loss     # FK 关节位置一致性
```

**各 loss 定义**:

| Loss | 公式 | 说明 |
|------|------|------|
| Foot skating | `‖v_foot‖² * σ(−y_foot / τ)` | y_foot 接近 0 时 contact probability 高，惩罚脚速度 |
| Penetration | `ReLU(−y_foot)²` | 脚部 y < 0 时产生惩罚 |
| Jitter | `‖p[t+1] − 2p[t] + p[t−1]‖²` | 二阶差分，惩罚高频抖动 |
| FK consistency | `‖FK(pred_rot6d) − FK(gt_rot6d)‖²` | 需要可微 FK (已有 SmplxLiteJ24) |

**代表工作**:
- **StableMoFusion** (arXiv 2405.05691): contact-aware loss 改善 foot skating
- **ReinDiffuse** (arXiv 2410.07296): RL 优化 diffusion 的物理合理性

**实现位置**: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py`
**依赖**: SmplxLiteJ24 (已有)，foot contact 检测 (阈值法)
**优势**: 纯 PyTorch 可微，不需要额外仿真器，改动最小
**劣势**: 只是软约束，不能保证物理可行性
**落地难度**: ⭐⭐ (最容易)

---

### 路径 C: 端到端 RL 微调（End-to-End RL Fine-tuning）

**核心思路**: 用 DDPO/PPO 风格的 RL 微调 flow matching 模型，reward 来自物理仿真器。

```
Flow Matching Model (frozen or fine-tunable)
    → 生成 motion
    → 物理仿真器执行 → reward (physical plausibility + text alignment)
    → PPO/DDPO 更新 model 参数
```

**代表工作**:
- **"No MoCap Needed"** (arXiv 2510.06988): DDPO 微调 motion diffusion，用 TMR 作 reward
- **MaskedMimic** (NVIDIA): 统一 masked inpainting + physics tracking controller
- **CLoSD**: Diffusion planner + RL tracker 联合训练

**优势**: 模型本身学会生成物理合理的 motion
**劣势**: 训练成本极高、需要仿真器、RL 训练不稳定
**落地难度**: ⭐⭐⭐⭐⭐ (最难)

---

## 3. 场景交互（Scene Interaction）

场景交互是独立但相关的课题。需要模型感知环境几何并生成与之交互的动作。

| 工作 | 方法 | 场景表示 | 开源 |
|------|------|---------|------|
| **TRUMANS** (PKU) | 自回归 diffusion + 场景 conditioning | 3D scene mesh | 计划开源 |
| **TeSMo** (ECCV 2024) | Diffusion + collision avoidance + text | Object mesh + SDF | paper |
| **LaserHuman** (ShanghaiTech) | 语言描述场景 + diffusion | Language description | 有代码 |
| **CLoSD** | Diffusion planner + RL tracker | Scene occupancy | project page |
| **DIP** (arXiv 2412.02261) | Implicit policy during diffusion inference | PROX/Replica scenes | arXiv |

**在 HyMotion M2M 上实现场景交互需要**:
1. 场景表示 (point cloud / SDF / mesh) 作为 MMDiT 额外 conditioning
2. 碰撞检测 loss（可微 SDF query）
3. 场景交互训练数据（TRUMANS / PROX 等）

---

## 4. 推荐实施路线

```
Phase 1 (1-2周): 路径 B — 可微物理 loss [最小改动，立即可做]
  ├─ 在 M2MLoss 中增加 foot_skating + penetration + jitter loss
  ├─ 用 SmplxLiteJ24 做可微 FK
  ├─ foot contact 用 y 坐标 + 速度阈值判断
  ├─ 新增 config: hymotion_m2m_completion_*_fkloss.py
  └─ 验证: foot skating / jitter 指标是否下降

Phase 2 (2-4周): 路径 A — PHC 推理时物理修正 [需搭 Isaac Gym 环境]
  ├─ 搭建 Isaac Gym 环境 + 加载 PHC 预训练 policy
  ├─ 实现 135-dim (rot6d) → SMPL joint angles → PHC input
  ├─ 实现 PHC output → 135-dim 的逆映射
  ├─ 集成到 HyMotionM2MPipeline 作为可选后处理
  └─ 验证: 对比修正前后的 MPJPE / foot skating / penetration

Phase 3 (远期): 路径 C + 场景交互
  ├─ 基于 Phase 2 仿真器作为 RL reward
  ├─ DDPO/PPO 微调 flow matching model
  ├─ 加入场景 conditioning (SDF / point cloud)
  └─ 端到端优化
```

---

## 5. 关键依赖和风险

| 依赖 | 状态 | 风险 |
|------|------|------|
| SmplxLiteJ24 (可微 FK) | ✅ 已有 (`hftrainer/models/motion/hymotion_m2m/network/smpl_lite.py`) | 低 |
| SMPL body model | ✅ 已有 (`checkpoints/smpl_models/`) | 低 |
| Isaac Gym | ❌ 需安装 (Preview 4, 不再维护；或迁移到 Isaac Lab) | 中 |
| PHC checkpoint | ✅ GitHub 开源 | 低 |
| MuJoCo (替代方案) | ❌ 需安装，但更稳定、持续维护 | 低 |
| 场景交互数据 | ❌ 需下载 TRUMANS/PROX | 高 (数据量大) |

---

## 6. 参考文献

- PhysDiff: Physics-Guided Human Motion Diffusion (arXiv 2212.02500, NVIDIA)
- PHC: Perpetual Humanoid Control (ICCV 2023, github.com/ZhengyiLuo/PHC)
- CLoSD: Closing the Loop Between Simulation and Diffusion
- ReinDiffuse: RL + Diffusion for Motion Plausibility (arXiv 2410.07296)
- MaskedMimic: Unified Physics-Based Character Control (NVIDIA)
- StableMoFusion: Contact-Aware Motion Generation (arXiv 2405.05691)
- TRUMANS: Large-Scale Human-Scene Interaction (PKU)
- TeSMo: Text-Controlled Scene-Aware Motion (ECCV 2024)
- LaserHuman: Language-guided Scene-aware Motion (ShanghaiTech)
- DIP: Diffusion Implicit Policy for Scene Motion (arXiv 2412.02261)
- AMP: Adversarial Motion Priors (arXiv 2104.02180)
- "No MoCap Needed": DDPO for Motion Diffusion (arXiv 2510.06988)
