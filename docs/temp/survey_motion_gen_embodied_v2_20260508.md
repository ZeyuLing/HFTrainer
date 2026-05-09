# 调研报告 v2：动作生成模型 ↔ 机器人/具身智能 — 双向融合与实践方案

> **日期**: 2026-05-08（基于 2026-04-16 v1 版本全面更新）
> **课题**: (1) 如何用动作生成模型指导机器人、具身智能训练；(2) 机器人/具身智能训练方式如何提升动作生成模型的物理真实性
> **存放位置**: `docs/temp/survey_motion_gen_embodied_v2_20260508.md`

---

## 目录

1. [核心问题与研究逻辑](#1-核心问题与研究逻辑)
2. [方向 A：动作生成模型 → 指导机器人训练](#2-方向-a动作生成模型--指导机器人训练)
3. [方向 B：机器人/具身智能训练 → 提升动作生成物理真实性](#3-方向-b机器人具身智能训练--提升动作生成物理真实性)
4. [统一闭环框架](#4-统一闭环框架)
5. [已下载/待下载代码仓库索引](#5-已下载待下载代码仓库索引)
6. [可行实践方案（按优先级排序）](#6-可行实践方案按优先级排序)
7. [参考文献](#7-参考文献)

---

## 1. 核心问题与研究逻辑

```
┌───────────────────────────────────┐
│    动作生成大模型                    │
│  (HyMotion M2M / Flow Matching)    │
│                                   │
│  ✅ 多样性、语义控制、大规模数据      │
│  ❌ 不保证物理可行性（浮空/滑动/穿地）│
└────────────┬──────────────────────┘
             │                    ▲
    方向A：运动先验              方向B：物理反馈
    指导机器人训练               提升生成质量
             │                    │
             ▼                    │
┌───────────────────────────────────┐
│    物理仿真 / 强化学习 / 机器人      │
│  (IsaacGym, MuJoCo, Unitree G1)   │
│                                   │
│  ✅ 物理真实性、接触力学、Sim2Real   │
│  ❌ 动作多样性受限、数据稀缺         │
└───────────────────────────────────┘
```

**两个方向的互惠关系**：
- 方向 A：动作生成模型输出丰富的运动先验，帮助机器人快速掌握通用动作（降低 RL 探索成本、扩大动作库）
- 方向 B：物理仿真提供可行性反馈，反向微调生成模型（RLPF/PhysDiff/SimDiff/PARC 等），使生成结果物理可行

---

## 2. 方向 A：动作生成模型 → 指导机器人训练

### 2.1 技术路线分类

| 路线 | 思路 | 代表工作 | 成熟度 |
|------|------|---------|--------|
| **A1: Reference Motion Tracking** | 生成运动序列 → RL 策略跟踪执行 | PHC, BeyondMimic, ASAP | ⭐⭐⭐ 最成熟 |
| **A2: Motion Latent as Action Space** | 用生成模型隐空间作为 RL 动作空间 | PULSE, RoboGhost | ⭐⭐ |
| **A3: Diffusion Policy** | 生成模型直接作为控制策略 | BeyondMimic (guided), DreamControl | ⭐⭐ |
| **A4: Data Augmentation** | 合成大规模运动数据扩充训练集 | PARC, GR00T-Mimic, UH-1 | ⭐⭐⭐ |
| **A5: Retargeting-Free Latent Control** | 跳过解码/retargeting，潜表示直接控制 | RoboGhost | ⭐ 最新 |

### 2.2 核心工作详解（新增/更新）

#### 2.2.1 PARC — Physics-based Augmentation with RL for Character Controllers (SIGGRAPH 2025)

| 项目 | 内容 |
|------|------|
| **机构** | Simon Fraser University + NVIDIA |
| **论文** | arXiv:2505.04002 |
| **代码** | https://github.com/mshoe/PARC ✅ 已开源 |
| **核心贡献** | 迭代式「生成 → 物理修正 → 数据增强」闭环，从小规模种子数据生成大规模高质量训练数据 |

**技术要点**：
```
循环迭代：
  1. 起始数据集（core terrain traversal motions）
  2. Motion Generator 生成新地形的合成运动
  3. Physics-based Tracking Controller（RL）跟踪并修正物理伪影
  4. 修正后的运动加回数据集
  5. 重复 → 能力逐步扩展
```

**对 HyMotion 的直接价值**：
- HyMotion 可以充当 Step 2 的 Motion Generator
- PHC/ASAP 可以充当 Step 3 的 Physics-based Tracker
- 修正后的运动可以回流训练 HyMotion（形成数据飞轮）

---

#### 2.2.2 ASAP — Aligning Simulation and Real-World Physics (RSS 2025)

| 项目 | 内容 |
|------|------|
| **机构** | CMU + NVIDIA (LeCAR Lab) |
| **论文** | arXiv:2502.01143 |
| **代码** | https://github.com/LeCAR-Lab/ASAP ✅ 已开源 |
| **核心贡献** | Delta Action Model 弥合 Sim2Real 物理差异，实现高难度全身技能（篮球运球、高踢腿） |

**技术要点**：
- 基于 HumanoidVerse 多仿真器框架（IsaacGym/IsaacSim/Genesis）
- Motion Retargeting: AMASS → 机器人关节轨迹
- Delta Action Model: 学习残差动作补偿真实世界动力学差异
- 在 Unitree G1 上验证

**对 HyMotion 的意义**：
- ASAP 的 retargeting 直接接受 SMPL 格式输入 → HyMotion 输出可直接对接
- Delta Action Model 概念可反向应用：学习「生成运动 → 物理可行运动」的残差修正

---

#### 2.2.3 VideoMimic — Visual Imitation for Contextual Humanoid Control (CoRL 2025 Best Student Paper)

| 项目 | 内容 |
|------|------|
| **机构** | 多机构 |
| **论文** | CoRL 2025 |
| **代码** | https://github.com/hongsukchoi/VideoMimic ✅ 已开源 |
| **核心贡献** | 从单目 RGB 视频自动提取 4D 人体运动 + 3D 场景 → 训练机器人在真实场景中执行 |

**三阶段 Pipeline**：
```
Stage 1: Real-to-Sim
  视频 → 4D人体运动重建（VIMO/ViTPose）
       → 3D场景重建（MegaSaM/MonST3R）
       → Retarget to robot

Stage 2: Simulation Training
  PPO in IsaacGym: MoCap预训练 → 场景条件跟踪 → 蒸馏 → RL微调

Stage 3: Sim-to-Real
  仅需 local height map + root commands → 部署到 Unitree G1
```

**对 HyMotion 的意义**：
- VideoMimic 证明了「运动学序列 + 场景信息 → 机器人控制」的完整链路
- HyMotion 可替代 Stage 1 中的视频重建步骤：文本/条件 → 运动生成 → 直接进入 Stage 2
- 场景条件控制是 HyMotion 缺乏但机器人需要的能力

---

#### 2.2.4 ExBody2 — Expressive Humanoid Whole-Body Control (2024)

| 项目 | 内容 |
|------|------|
| **机构** | UC Berkeley + UCSD + MIT + NVIDIA |
| **代码** | https://github.com/jimazeyu/exbody2 ✅ 已开源 |
| **核心贡献** | PPO + Teacher-Student 框架，AMASS/CMU 数据 → Unitree G1/H1 全身表现力控制 |

**技术要点**：
- CVAE 进行 Sim2Real 迁移
- 复杂动作：舞蹈、拳击、蹲下同时保持平衡
- 两阶段 Teacher(privileged state)-Student(sparse sensor) 蒸馏

---

#### 2.2.5 GMR — General Motion Retargeting (ICRA 2026)

| 项目 | 内容 |
|------|------|
| **机构** | Yanjie Ze et al. |
| **代码** | https://github.com/YanjieZe/GMR ✅ 已开源 (MIT License) |
| **核心贡献** | CPU 实时运行的通用 Retargeting，支持 17+ 机器人 × 多种输入格式 |

**支持的输入格式**：
- BVH (Xsens, Nokov, LAFAN1)
- FBX (OptiTrack)
- **SMPLX (AMASS, OMOMO)** ← 直接兼容 HyMotion 输出
- PICO (XRoboToolkit)
- 单目视频 (GVHMR)

**支持的机器人**：Unitree H1/H1_2, Fourier GR3, PAL Talos, Booster T1 等 17+

**对 HyMotion 的直接价值**：
- **关键桥梁**：HyMotion 输出 SMPL 格式 → GMR 实时 retarget → 任意机器人
- CPU 实时运行 → 可嵌入 RL 训练循环
- 已被 BeyondMimic、TWIST 等项目使用

---

#### 2.2.6 UH-1 — Universal Humanoid from Internet Videos (Humanoids 2025 Oral)

| 项目 | 内容 |
|------|------|
| **机构** | UC Berkeley |
| **代码** | https://github.com/sihengz02/UH-1 ✅ 已开源 |
| **核心贡献** | 20M+ 人形机器人姿态数据集（Humanoid-X），从 163K 互联网视频挖掘 → 文本驱动机器人运动生成 |

**Pipeline**：
```
163K 互联网人类视频
  → 3D 姿态估计
  → Motion Retargeting to humanoid
  → RL 验证可行性
  → 文本标注（自动）
  → 20M+ Humanoid-X Dataset
  → Transformer (text → humanoid keypoints → joint control)
```

**对 HyMotion 的意义**：
- UH-1 证明了大规模数据对机器人运动的价值
- HyMotion 可以作为「数据放大器」：从 549K 训练样本生成无限多变体 → 扩充 UH-1 式数据集
- HyMotion 的文本条件生成 + UH-1 的 retargeting pipeline = 端到端 text → robot motion

---

#### 2.2.7 ProtoMotions — GPU-Accelerated Humanoid Simulation (NVIDIA, 2025)

| 项目 | 内容 |
|------|------|
| **机构** | NVIDIA Research |
| **代码** | https://github.com/NVlabs/ProtoMotions ✅ 已开源 (Apache 2.0) |
| **核心贡献** | 统一的 GPU 加速仿真训练框架：数字人 + 人形机器人，集成 MaskedMimic/KIMODO |

**关键特性**：
- 支持 IsaacGym / IsaacLab / NVIDIA Newton / MuJoCo
- 4×A100 12 小时训练 AMASS 全集（40+ 小时动作）
- 一键 AMASS → 机器人 retargeting (PyRoki)
- 集成 MaskedMimic（地形导航）+ KIMODO（文本生成）
- Sim-to-Real 部署（ONNX 导出 → Unitree G1）

**对 HyMotion 的意义**：
- **最佳集成平台**：如果要搭建「生成 → 物理验证 → 机器人部署」全链路，ProtoMotions 是最完整的基础设施
- 已集成 KIMODO → 替换为 HyMotion 的接口成本低
- MaskedMimic 的 masked inpainting 与 HyMotion 的 VACE 同构

---

#### 2.2.8 GR00T N1 — Foundation Model for Humanoid Robots (NVIDIA, 2025)

| 项目 | 内容 |
|------|------|
| **机构** | NVIDIA |
| **版本** | N1 (March 2025) → N1.5 (June 2025) → N1.6 (CoRL 2025) |
| **核心贡献** | 双系统架构（快思考 + 慢思考），合成数据管线 GR00T-Mimic |

**合成运动数据管线**：
- GR00T-Teleop: Apple Vision Pro 遥操作 → 数字孪生
- **GR00T-Mimic**: 11 小时生成 780K 合成轨迹（= 6500 人类小时），性能 +40%
- GR00T-Gen: Domain Randomization 增强多样性

**启示**：
- 大规模合成运动数据 = 核心竞争力
- HyMotion 的 549K 训练样本 + 条件控制能力 → 可构建类似的合成数据管线
- 证明了「运动生成 × 数据增强 × RL 训练」三者结合的商业价值

---

### 2.3 工作对比总结

| 工作 | 年份/会议 | 运动来源 | 物理仿真 | 真实机器人 | 代码 | 与HyMotion对接难度 |
|------|----------|---------|---------|-----------|------|------------------|
| PHC | ICCV 2023 | AMASS MoCap | IsaacGym | — | ✅ | 低（SMPL→PHC） |
| PULSE | ICLR 2024 | PHC 蒸馏 | IsaacGym | — | ✅ | 中（需训练潜空间） |
| OmniH2O | CoRL 2024 | AMASS retarget | IsaacGym | Unitree H1 | ✅ | 低（SMPL接口） |
| HumanPlus | CoRL 2024 | AMASS 40h | IsaacGym | Unitree H1 | ✅ | 低（SMPL接口） |
| ExBody2 | 2024 | AMASS/CMU | IsaacGym | Unitree G1/H1 | ✅ | 低 |
| **ASAP** | RSS 2025 | AMASS retarget | Multi-sim | Unitree G1 | ✅ | **低（GMR接口）** |
| **VideoMimic** | CoRL 2025 | RGB视频 | IsaacGym | Unitree G1 | ✅ | 中 |
| **PARC** | SIGGRAPH 2025 | 迭代增强 | IsaacGym | — | ✅ | **低（直接闭环）** |
| **GMR** | ICRA 2026 | SMPLX/BVH/FBX | — | 17+机器人 | ✅ | **极低（即插即用）** |
| **UH-1** | Humanoids 2025 | 互联网视频 | IsaacGym | — | ✅ | 低 |
| **ProtoMotions** | 2025 | AMASS | Multi-sim | Unitree G1 | ✅ | **低（已集成KIMODO）** |
| RoboGhost | 2025 | Motion Latent | IsaacGym | Unitree G1 | ❌ | 高 |
| DreamControl | 2025 | Diffusion Prior | — | Unitree G1 | ❌ | 中 |
| BeyondMimic | 2025 | MoCap | — | Unitree G1 | ❌ | 中 |
| MaskedMimic | SIGA 2024 | MoCap | — | — | ❌(ProtoMotions) | 中 |

---

## 3. 方向 B：机器人/具身智能训练 → 提升动作生成物理真实性

### 3.1 技术路线分类

| 路线 | 思路 | 代表工作 | 推理时需仿真器 | 修改生成模型 |
|------|------|---------|:---:|:---:|
| **B1: 仿真投影 (Post-processing)** | 每步去噪后物理仿真修正 | PhysDiff | ✅ | ❌ |
| **B2: 物理条件化 (Conditioning)** | 物理参数作为生成条件 | SimDiff | ❌ | 轻量适配器 |
| **B3: RL 微调 (RLHF-style)** | 物理仿真奖励 → RL 更新模型 | RLPF, RobotMDM | 训练时 | 微调全模型 |
| **B4: 数据增强闭环** | 物理修正后回流训练数据 | PARC | ❌ | 重训/继续训 |
| **B5: 端到端物理感知** | 物理约束嵌入训练损失 | POMP, 显式物理loss | ❌ | 修改损失函数 |
| **B6: 相位流形对齐** | 仿真修正 + 相位先验对齐 | POMP | ✅(每帧) | 端到端训练 |

### 3.2 核心工作详解（新增）

#### 3.2.1 PARC（双向，既是方向A也是方向B）

PARC 的迭代闭环是目前**最接近统一框架**的工作：
```
┌─────────────────────────────────────────────────────────────┐
│  PARC 迭代闭环                                              │
│                                                             │
│  Motion Generator (可替换为HyMotion)                        │
│       │                                                     │
│       ▼ 生成运动                                            │
│  Physics Tracker (RL, IsaacGym)                             │
│       │                                                     │
│       ▼ 物理修正后的运动                                     │
│  Dataset Augmentation                                       │
│       │                                                     │
│       ▼ 扩充训练数据                                         │
│  重训 Motion Generator ← ─ ─ ─ ─ ─ 循环                    │
│                                                             │
│  每轮迭代：能力边界扩展，物理质量提升                           │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2.2 RLPF — RL from Physical Feedback (Under Review ICLR 2026)

| 项目 | 内容 |
|------|------|
| **机构** | PKU + BeingBeyond + WHU |
| **论文** | arXiv:2506.12769 |
| **代码** | https://github.com/BeingBeyond/RLPF （计划开源，暂未发布） |
| **核心贡献** | GRPO 微调运动生成模型，物理仿真器提供可行性奖励 |

**三阶段框架**：
```
Stage 1: Physics-Aware Evaluation
  生成运动 → IsaacGym 跟踪策略 → 物理可行性分数

Stage 2: Alignment Verification
  验证运动仍忠实于文本指令 → 语义对齐分数

Stage 3: RL Fine-Tuning (GRPO)
  reward = α * 物理可行性 + β * 语义对齐
  → Group Relative Policy Optimization 更新生成模型
```

**对 HyMotion 的直接适用性**：
- HyMotion 使用 Flow Matching → GRPO 对 continuous-time 模型的适用性需验证（原文用 DDPM）
- 但核心思路完全通用：**物理仿真器 = reward model**
- 可以与 SOAR（exposure bias correction）正交组合

---

#### 3.2.3 PhysDiff — Physics-Guided Motion Diffusion (ICCV 2023 Oral, NVIDIA)

**核心公式**：
```python
# 标准 DDPM 去噪
x_{t-1} = f(x_t, epsilon_theta(x_t, t))

# PhysDiff：每步加物理投影
x_hat_0 = predict_x0(x_t, epsilon_theta)           # 预测 clean motion
x_hat_0_phys = PhysSimulator.project(x_hat_0)      # MuJoCo 物理仿真修正
x_{t-1} = posterior(x_t, x_hat_0_phys, t)          # 用修正后的 x0 计算 x_{t-1}
```

**适配 Flow Matching (HyMotion) 的变体**：
```python
# Flow Matching ODE: dx/dt = v_theta(x_t, t)
# 物理投影可以嵌入 ODE 步：
x_{t+dt} = x_t + v_theta(x_t, t) * dt              # 正常 ODE step
x_{t+dt} = PhysSimulator.project(x_{t+dt})          # 物理修正
```

**局限**：每步都需要物理仿真器 → 推理速度慢（~5x）

---

#### 3.2.4 SimDiff — Simulator-Constrained Diffusion (2025)

**核心创新**：推理时不需要仿真器

```
训练时：
  MuJoCo 仿真（多种物理条件：重力g, 风速w, 摩擦f）
  → 生成 domain-randomized 训练数据
  → 冻结 MDM backbone + 训练 Sim Encoder + Motion Adapters
  → 物理参数作为 classifier-free guidance 条件

推理时：
  指定物理参数（如 g=9.8, w=0, f=标准）
  → 条件化生成 → 无需仿真器即可输出物理可行运动
```

**适配 HyMotion 方案**：
- 冻结 HyMotion MMDiT backbone
- 添加轻量 Physics Adapter（类似 LoRA/adapter）
- 物理参数条件化注入（与现有 text/mask conditioning 正交）
- 训练数据：IsaacGym/MuJoCo 仿真生成的物理修正运动

---

#### 3.2.5 POMP — Physics-Consistent Motion through Phase Manifolds (CVPR 2025)

**三模块架构**：
```
┌─────────────────────────────────────────────────────────┐
│                        POMP                              │
│                                                         │
│  Module 1: Diffusion Kinematic Module                   │
│    → 逐帧生成运动学姿态（类似标准 motion diffusion）       │
│                                                         │
│  Module 2: Simulation Dynamic Module                    │
│    → 物理仿真修正（碰撞响应、重力、地形适应）              │
│    → 但会引入累积误差和偏移                              │
│                                                         │
│  Module 3: Phase Encoding Module                        │
│    → 将仿真结果投影回运动学先验的相位流形                  │
│    → 防止仿真修正导致运动「失去自然度」                    │
│    → 关键创新：语义对齐 + 物理约束 的平衡                 │
└─────────────────────────────────────────────────────────┘
```

**代码状态**：未开源，但 CVPR 论文有充分实现细节

---

#### 3.2.6 Scalable Motion In-Betweening via Diffusion + Physics (April 2025)

| 项目 | 内容 |
|------|------|
| **论文** | arXiv:2504.09413 |
| **代码** | 未开源 |
| **核心贡献** | 两阶段：character-agnostic diffusion 生成 + RL controller 适配到目标角色 |

**启示**：
- Canonical skeleton 上训练 → 任意角色适配（类似 GMR 的通用性理念）
- Stage 2 的 RL adapter 修正足部滑动等物理伪影
- 与 HyMotion 的 SMPL 输出 + GMR retargeting 方案高度互补

---

### 3.3 对比总结

| 方案 | 推理开销 | 训练改动 | 物理质量提升 | HyMotion适配难度 | 代码可用性 |
|------|---------|---------|:---:|---------|---------|
| **PhysDiff 式仿真投影** | 高（5x） | 无 | ⭐⭐⭐ | 低 | 可自行实现 |
| **SimDiff 式适配器** | 零 | 轻量 | ⭐⭐⭐ | 低-中 | 需自行实现 |
| **RLPF 式 RL 微调** | 零 | 微调全模型 | ⭐⭐⭐⭐ | 中 | 待开源 |
| **PARC 式数据闭环** | 零 | 重训/继续训 | ⭐⭐⭐⭐ | 低 | ✅ |
| **显式物理损失** | 零 | 修改 loss | ⭐⭐ | 极低 | 已有经验 |
| **POMP 相位流形** | 中 | 端到端 | ⭐⭐⭐⭐ | 高 | ❌ |
| **MoGenDIT 后处理** | 低 | 无 | ⭐⭐ | 极低 | ✅ 已有 |

---

## 4. 统一闭环框架

基于调研，提出以下统一闭环框架设计：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Motion-Physics Unified Loop                         │
│                                                                      │
│   ┌───────────────┐                                                  │
│   │ Text / Cond   │                                                  │
│   └───────┬───────┘                                                  │
│           ▼                                                          │
│   ┌───────────────────────┐                                          │
│   │   HyMotion M2M        │◄──── RLPF/PARC 反馈更新（方向B）          │
│   │   (Flow Matching)     │                                          │
│   └───────────┬───────────┘                                          │
│               │ SMPL motion sequence                                 │
│               ▼                                                      │
│   ┌───────────────────────┐                                          │
│   │   GMR Retargeting     │ (CPU 实时, 支持17+机器人)                  │
│   └───────────┬───────────┘                                          │
│               │ Robot joint trajectory                                │
│               ▼                                                      │
│   ┌───────────────────────────────────────────┐                      │
│   │   ProtoMotions / IsaacGym                  │                      │
│   │   (Physics Simulation + RL Training)       │                      │
│   │                                            │                      │
│   │   ┌─────────────┐    ┌──────────────┐     │                      │
│   │   │ PHC/ASAP    │    │ ExBody2      │     │                      │
│   │   │ Tracking    │    │ Whole-body   │     │                      │
│   │   └──────┬──────┘    └──────────────┘     │                      │
│   │          │                                 │                      │
│   │          ▼                                 │                      │
│   │   Physics-corrected motion                 │                      │
│   │   + Feasibility reward                     │                      │
│   └───────────┬───────────────────────────────┘                      │
│               │                                                      │
│        ┌──────┴──────┐                                               │
│        ▼             ▼                                               │
│   方向A:          方向B:                                              │
│   Deploy to       RLPF reward / PARC data loop                       │
│   Real Robot      → Update HyMotion                                  │
│   (Unitree G1)                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. 已下载/待下载代码仓库索引

### 已在 ref_repo/ 的仓库

| 目录 | 项目 | 关联方向 |
|------|------|---------|
| `ref_repo/PHC/` | PHC: Perpetual Humanoid Control | A1 运动跟踪 |
| `ref_repo/OmniH2O/` | OmniH2O: Human-to-Humanoid | A1 遥操作 |
| `ref_repo/HumanPlus/` | HumanPlus: Humanoid Imitation | A4 数据扩展 |
| `ref_repo/TeSMo/` | TeSMo: Scene-Aware Motion | B5 场景约束 |
| `ref_repo/MotionLCM/` | MotionLCM: Real-time Generation | 实时生成 |
| `ref_repo/MDM/` | MDM: Motion Diffusion Model | 基线 |

### 🆕 建议新增下载

| 优先级 | 项目 | GitHub | 理由 |
|--------|------|--------|------|
| **P0** | **PARC** | https://github.com/mshoe/PARC | 直接实现「生成→物理修正→数据增强」闭环 |
| **P0** | **GMR** | https://github.com/YanjieZe/GMR | HyMotion→机器人的关键桥梁，即插即用 |
| **P0** | **ProtoMotions** | https://github.com/NVlabs/ProtoMotions | 最完整的仿真训练框架，已集成KIMODO |
| **P1** | **ASAP** | https://github.com/LeCAR-Lab/ASAP | Sim2Real Delta Model，含多仿真器支持 |
| **P1** | **VideoMimic** | https://github.com/hongsukchoi/VideoMimic | 完整视频→机器人部署pipeline |
| **P1** | **ExBody2** | https://github.com/jimazeyu/exbody2 | Teacher-Student全身控制 |
| **P1** | **UH-1** | https://github.com/sihengz02/UH-1 | 大规模数据集+文本控制 |
| **P2** | **RLPF** | https://github.com/BeingBeyond/RLPF | RL微调生成模型（待开源） |

### 未开源但需持续关注

| 项目 | 状态 | 跟踪链接 |
|------|------|---------|
| RoboGhost | 项目页 | https://arxiv.org/abs/2510.14952 |
| DreamControl | 项目页 | https://arxiv.org/abs/2509.14353 |
| BeyondMimic | 项目页 | https://arxiv.org/abs/2508.08241 |
| SimDiff | 论文 | https://arxiv.org/abs/2509.20927 |
| POMP | CVPR 2025 | CVPR Open Access |
| MotionGlot | 待开源 | https://ivl.cs.brown.edu/research/motionglot.html |

---

## 6. 可行实践方案（按优先级排序）

### 方案 1（P0，短期 1-2 周）：HyMotion → GMR → 物理仿真验证

**目标**：验证 HyMotion 输出在物理仿真中的可行性，建立 baseline

**步骤**：
```
1. 下载 GMR → 配置 SMPL→Unitree G1 retargeting
2. HyMotion 生成 100 条运动序列（覆盖 walk/run/dance/sit 等）
3. GMR retarget → IsaacGym/ProtoMotions 中 PHC 跟踪
4. 统计物理可行性指标：
   - 跟踪成功率（PHC tracking success rate）
   - 足部滑动距离
   - 地面穿透深度
   - 能量消耗合理性
5. 输出 baseline 报告
```

**产出**：
- 定量回答「HyMotion 生成的运动有多少比例是物理可行的」
- 识别主要物理问题类别及严重程度

---

### 方案 2（P0，短期 2-4 周）：PARC 式数据闭环（最推荐的统一方案）

**目标**：建立「生成 → 物理修正 → 数据增强 → 重训」迭代闭环

**步骤**：
```
Iteration 0:
  - HyMotion (当前模型) 生成 10K 运动序列
  - ProtoMotions/PHC 跟踪 → 物理修正
  - 保留跟踪成功 + 质量达标的运动 → 加入训练集
  - 统计物理修正前后的质量差异

Iteration 1:
  - 用增强数据继续训练 HyMotion（或 fine-tune）
  - 重新生成 10K → 跟踪 → 修正 → 增强
  - 对比 Iteration 0 vs 1 的物理可行性指标

Iteration N:
  - 直到收敛
```

**核心依赖**：
- PARC 代码（框架参考）
- ProtoMotions（仿真环境）
- GMR（retargeting）
- HyMotion（生成器）

**预期收益**：
- 物理可行性逐轮提升
- 无需修改 HyMotion 架构
- 生成的增强数据同时提升机器人训练质量

---

### 方案 3（P1，中期 4-8 周）：SimDiff 式物理适配器

**目标**：零推理开销的物理约束注入

**步骤**：
```
1. 搭建 MuJoCo/IsaacGym SMPL 仿真环境
2. Domain-Randomized 仿真数据生成（标准重力/地形/摩擦）
3. 冻结 HyMotion MMDiT backbone
4. 设计 Physics Adapter 模块（参考 SimDiff 的 Sim Encoder + Motion Adapters）
5. 训练：输入 = HyMotion 中间特征 + 物理参数条件；输出 = 物理修正后的运动
6. 推理时：指定标准物理参数 → 自动生成物理可行运动
```

**技术细节**：
- Adapter 可以是 LoRA 式（在 attention 层加低秩矩阵）或 channel-wise adapter
- 物理参数编码：gravity (3D), friction (1D), terrain_type (categorical)
- CFG 权重控制物理约束强度

---

### 方案 4（P1，中期 4-8 周）：RLPF 式 RL 微调

**目标**：用物理仿真奖励直接微调 HyMotion

**步骤**：
```
1. 训练 PHC/ASAP 跟踪策略作为物理评估器
2. 定义奖励函数：
   R = α * tracking_success_rate    # 物理可行性
     + β * text_alignment_score     # 语义对齐（FID/R-precision）
     + γ * motion_quality_score     # 运动质量（jitter/smoothness）
3. GRPO/PPO 采样：HyMotion 生成 K 条运动 → 评分 → 排序
4. 梯度更新 HyMotion（需适配 Flow Matching 的 RL 微调方式）
```

**技术挑战**：
- Flow Matching 的 RL 微调不如 DDPM 成熟（RLPF 原文用 DDPM）
- 可参考 SOAR 的 on-policy rollout 思路适配 rectified flow
- 需要大量采样（GPU 开销大）

---

### 方案 5（P2，长期 2-3 月）：完整 Text→Robot 部署 Pipeline

**目标**：端到端演示「文本指令 → HyMotion 生成 → 物理跟踪 → 真实机器人执行」

**步骤**：
```
1. 方案 1-3 的产出作为基础
2. 搭建 Unitree G1 仿真环境（ProtoMotions/IsaacLab）
3. GMR retarget → ExBody2/ASAP 训练跟踪策略
4. Sim2Real: Domain Randomization + ASAP Delta Model
5. 在真实 Unitree G1 上验证
```

---

### 方案 6（P2，持续）：显式物理损失 + 现有 MoGenDIT 后处理

**目标**：最低成本改进，立即可执行

**步骤**：
```
训练时：
  - foot_contact_loss: 预测足部接触 → 接触帧脚速度=0
  - ground_penetration_loss: FK 计算关节位置 → 惩罚 y<0
  - joint_limit_loss: 惩罚超出人体关节极限的角度

推理时：
  - 已有 MoGenDIT repair pipeline 后处理
  - 可加 StableMotion 式 foot-lock classifier guidance
```

---

## 7. 参考文献

### 方向 A：动作生成 → 机器人

[1] Zhengyi Luo et al. "PHC: Perpetual Humanoid Control." ICCV 2023. https://github.com/ZhengyiLuo/PHC
[2] Zhengyi Luo et al. "PULSE: Universal Humanoid Motion Representations." ICLR 2024. https://arxiv.org/abs/2310.04582
[3] Tairan He et al. "OmniH2O." CoRL 2024. https://github.com/LeCAR-Lab/human2humanoid
[4] Zipeng Fu et al. "HumanPlus." CoRL 2024. https://github.com/MarkFzp/humanplus
[5] RoboGhost. 2025. https://arxiv.org/abs/2510.14952
[6] DreamControl. 2025. https://arxiv.org/abs/2509.14353
[7] BeyondMimic. 2025. https://arxiv.org/abs/2508.08241
[8] MaskedMimic. SIGGRAPH Asia 2024. https://research.nvidia.com/labs/par/maskedmimic/
[9] SuperPADL. 2024. https://arxiv.org/abs/2407.10481
[10] **PARC. SIGGRAPH 2025. https://github.com/mshoe/PARC** ← 核心参考
[11] **ASAP. RSS 2025. https://github.com/LeCAR-Lab/ASAP** ← 核心参考
[12] **VideoMimic. CoRL 2025 Best Student Paper. https://github.com/hongsukchoi/VideoMimic**
[13] **ExBody2. 2024. https://github.com/jimazeyu/exbody2**
[14] **GMR. ICRA 2026. https://github.com/YanjieZe/GMR** ← 关键桥梁
[15] **UH-1. Humanoids 2025 Oral. https://github.com/sihengz02/UH-1**
[16] **ProtoMotions. NVIDIA 2025. https://github.com/NVlabs/ProtoMotions** ← 基础设施
[17] **GR00T N1. NVIDIA 2025. (商业产品)**
[18] MotionGlot. 2025. https://arxiv.org/abs/2410.16623

### 方向 B：物理仿真 → 提升生成质量

[19] Ye Yuan et al. "PhysDiff." ICCV 2023 Oral. https://nvlabs.github.io/PhysDiff/
[20] SimDiff. 2025. https://arxiv.org/abs/2509.20927
[21] **RLPF. Under Review ICLR 2026. https://github.com/BeingBeyond/RLPF** ← 核心参考
[22] RobotMDM. SIGGRAPH Asia 2024. Disney Research.
[23] POMP. CVPR 2025. https://openaccess.thecvf.com/CVPR2025
[24] Scalable Motion In-Betweening. 2025. https://arxiv.org/abs/2504.09413
[25] StableMotion. SIGGRAPH Asia 2025. (已在 ref_repo)
[26] SOAR. 2026. (已在 ref_repo)

---

## 附录：下载命令参考

```bash
# P0 优先下载
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo

# PARC (SIGGRAPH 2025)
git clone https://github.com/mshoe/PARC.git PARC

# GMR (ICRA 2026)
git clone https://github.com/YanjieZe/GMR.git GMR

# ProtoMotions (NVIDIA)
git clone https://github.com/NVlabs/ProtoMotions.git ProtoMotions

# P1 下载
# ASAP (RSS 2025)
git clone https://github.com/LeCAR-Lab/ASAP.git ASAP

# VideoMimic (CoRL 2025 Best Paper)
git clone https://github.com/hongsukchoi/VideoMimic.git VideoMimic

# ExBody2
git clone https://github.com/jimazeyu/exbody2.git ExBody2

# UH-1 (Humanoids 2025)
git clone https://github.com/sihengz02/UH-1.git UH-1
```

---

> **下一步行动**：
> 1. 执行下载命令，将 P0 仓库 clone 到 ref_repo/
> 2. 阅读 PARC 和 ProtoMotions 代码，评估与 HyMotion 对接的工程量
> 3. 执行方案 1（HyMotion → GMR → 物理验证 baseline）
