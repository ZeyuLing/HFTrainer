# 调研报告：动作生成大模型与仿真强化学习、具身智能的交叉融合

> **日期**: 2026-04-16
> **范围**: 动作生成大模型（如 HyMotion、KIMODO）↔ 具身智能 / 人形机器人 ↔ 仿真强化学习
> **存放位置**: `docs/temp/survey_motion_generation_embodied_intelligence.md`

---

## 目录

1. [调研背景与问题定义](#1-调研背景与问题定义)
2. [方向一：动作生成大模型 → 具身智能 / 人形机器人](#2-方向一动作生成大模型--具身智能--人形机器人)
   - 2.1 [总体思路：运动学先验作为机器人控制的基础](#21-总体思路运动学先验作为机器人控制的基础)
   - 2.2 [关键工作详解](#22-关键工作详解)
   - 2.3 [技术路线对比表](#23-技术路线对比表)
   - 2.4 [对 HyMotion / KIMODO 的启示](#24-对-hymotion--kimodo-的启示)
3. [方向二：仿真强化学习 → 提升动作生成的物理真实性](#3-方向二仿真强化学习--提升动作生成的物理真实性)
   - 3.1 [总体思路：物理仿真为生成模型注入物理约束](#31-总体思路物理仿真为生成模型注入物理约束)
   - 3.2 [关键工作详解](#32-关键工作详解)
   - 3.3 [技术路线对比表](#33-技术路线对比表)
   - 3.4 [对 HyMotion M2M 的启示](#34-对-hymotion-m2m-的启示)
4. [两个方向的统一框架展望](#4-两个方向的统一框架展望)
5. [已下载代码与文献索引](#5-已下载代码与文献索引)
6. [参考文献完整列表](#6-参考文献完整列表)

---

## 1. 调研背景与问题定义

当前，动作生成大模型（如 HyMotion M2M、KIMODO、UMO 等）与具身智能/人形机器人控制是两个快速发展但相对独立的研究领域：

- **动作生成大模型**：基于 Diffusion/Flow Matching 的生成模型，从文本/约束条件生成高质量人体运动序列。核心关注运动的**多样性**、**语义对齐**、**条件控制精度**。输出为运动学层面的关节角/位置序列，**不保证物理可行性**（如可能存在浮空、穿地、足部滑动等问题）。

- **具身智能/人形机器人**：通过强化学习在物理仿真器（IsaacGym、MuJoCo）中训练控制策略，使机器人执行各种任务。核心关注**物理可行性**、**稳定性**、**Sim-to-Real 迁移**。但动作多样性和自然度常受限于奖励函数设计。

**两个方向的互补关系**：

```
┌──────────────────────┐         ┌──────────────────────┐
│   动作生成大模型       │         │  仿真 RL / 具身智能    │
│                      │         │                      │
│  优势：多样性、自然度、 │ ──①──→ │  需求：丰富的运动先验   │
│  语义控制、大规模数据   │         │  帮助机器人掌握通用动作  │
│                      │         │                      │
│  短板：物理不真实      │ ←──②── │  优势：物理真实性、      │
│  浮空/穿地/滑动       │         │  接触力学、稳定控制     │
└──────────────────────┘         └──────────────────────┘
```

- **方向①**：动作生成大模型为机器人提供通用运动先验（Motion Prior），帮助人形机器人快速掌握多样化动作
- **方向②**：物理仿真和强化学习反馈帮助动作生成模型生成更符合物理规律的运动

---

## 2. 方向一：动作生成大模型 → 具身智能 / 人形机器人

### 2.1 总体思路：运动学先验作为机器人控制的基础

动作生成大模型在人形机器人领域的核心价值在于提供**大规模人类运动先验**。传统 RL 方法需要为每个任务从零设计奖励函数并训练策略，而运动生成模型可以：

1. **提供参考动作（Reference Motion）**：生成模型输出的运动序列作为 RL 策略的跟踪目标
2. **构建运动潜空间（Motion Latent Space）**：将生成模型的隐空间作为 RL 的动作空间，约束探索范围
3. **直接控制（Retargeting-Free）**：跳过中间步骤，将语言/文本指令通过运动隐表示直接映射为机器人动作
4. **作为扩散先验（Diffusion Prior）**：在 RL 训练中用生成模型的概率分布作为自然度正则化

目前该方向的主要技术路线如下：

```
                    ┌─────────────────────┐
                    │  Human Motion Data   │
                    │  (MoCap / AMASS)     │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Motion Generation   │
                    │  Model Training      │
                    │  (Diffusion/Flow)    │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      路线 A:           路线 B:           路线 C:
      Reference         Latent Space      Retargeting-
      Motion Tracking   as Action Space   Free Control
      ┌───────────┐    ┌───────────┐    ┌───────────┐
      │ 生成参考动作│    │ 训练运动潜 │    │ 端到端：    │
      │ → RL跟踪   │    │ 空间 → RL  │    │ 语言→隐表示 │
      │ → 物理执行  │    │ 在潜空间探│    │ → 机器人动作│
      └───────────┘    │ 索          │    └───────────┘
      代表：           └───────────┘    代表：
      PHC/PULSE,       代表：           RoboGhost,
      BeyondMimic,     PULSE,          HumanPlus
      MaskedMimic      SuperPADL
```

### 2.2 关键工作详解

#### 2.2.1 PHC: Perpetual Humanoid Control（ICCV 2023）

| 项目 | 内容 |
|------|------|
| **机构** | CMU (Zhengyi Luo) |
| **论文** | Perpetual Humanoid Control for Real-time Simulated Avatars |
| **代码** | https://github.com/ZhengyiLuo/PHC （已下载至 `ref_repo/PHC`） |
| **核心贡献** | 首个能在**全部 AMASS 数据集**上达到 100% 跟踪成功率的物理仿真人体控制器 |

**技术要点**：
- **Progressive Multiplicative Control Policy (PMCP)**：渐进式训练策略，动态扩展网络容量以学习 10,000+ 运动片段而不发生灾难性遗忘
- **Fail-State Recovery**：控制器能从失败状态（如摔倒）自主恢复，无需外部稳定力
- **与运动生成模型的接口**：提供了 MDM（Motion Diffusion Model）→ PHC 的 demo，即「语言 → 运动生成 → 物理跟踪」全链路

**与动作生成大模型的关系**：
- PHC 本质上是一个「运动学到动力学的桥梁」——接收任意运动学序列（如 HyMotion 的输出），将其转化为物理仿真中的稳定控制
- PHC 的 100% 跟踪成功率意味着**生成模型的质量瓶颈直接成为机器人动作质量的瓶颈**
- 隐含假设：生成的运动需要足够自然，否则物理控制器虽然能跟踪但姿态不自然

---

#### 2.2.2 PULSE: Universal Humanoid Motion Representations（ICLR 2024 Spotlight）

| 项目 | 内容 |
|------|------|
| **机构** | CMU (Zhengyi Luo) |
| **论文** | Universal Humanoid Motion Representations for Physics-Based Control |
| **代码** | https://github.com/ZhengyiLuo/PULSE |
| **核心贡献** | 将 PHC 的能力蒸馏为**32维运动潜空间**，覆盖 99.8% AMASS 动作 |

**技术要点**：
- **Motion Imitation Distillation**：先训练 PHC 作为运动模仿器，然后通过 encoder-decoder + variational information bottleneck 蒸馏为低维潜空间
- **Proprioceptive Prior**：基于本体感受的先验分布，在潜空间中采样即可生成自然运动
- **Hierarchical RL**：下游任务（导航、VR跟踪）的 RL 策略在 PULSE 的潜空间中探索，比原始关节空间更高效

**对动作生成大模型的意义**：
- PULSE 证明了**运动生成模型的隐表示可以作为机器人控制的动作空间**
- 如果将 HyMotion/KIMODO 的 Diffusion/Flow 潜空间与 PULSE 类似的物理蒸馏结合，可以构建「生成 + 物理可行」的统一运动空间

---

#### 2.2.3 OmniH2O: Universal Human-to-Humanoid Teleoperation（CoRL 2024）

| 项目 | 内容 |
|------|------|
| **机构** | CMU + 上海交大 |
| **论文** | OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning |
| **代码** | https://github.com/LeCAR-Lab/human2humanoid （已下载至 `ref_repo/OmniH2O`） |
| **核心贡献** | 使用 SMPL 运动学姿态作为通用接口，实现 VR/RGB/语言多模态人形机器人遥操作 |

**技术要点**：
- **Retargeting Pipeline**：AMASS 大规模人体运动数据 → SMPL 表示 → 梯度优化 retarget 到 Unitree H1 → 过滤不可行动作 → RL 训练
- **Privileged Teacher → Student Distillation**：先训练有完整状态的教师策略，再蒸馏为仅使用部分观测的学生策略
- **多模态输入**：支持 VR 头盔、RGB 相机、甚至 GPT-4o 语言指令

**与动作生成大模型的关系**：
- OmniH2O 的 retargeting 流程直接使用了 AMASS 等运动数据集 —— **这些数据集同样是 HyMotion/KIMODO 等模型的训练数据源**
- 如果将动作生成模型接入 OmniH2O 的 pipeline，可以实现「文本 → 运动生成 → SMPL pose → 机器人执行」的完整链路
- OmniH2O 发布了 **OmniH2O-6 数据集**（6个日常任务），可以作为动作生成模型在具身场景的评测基准

---

#### 2.2.4 HumanPlus: Humanoid Shadowing and Imitation from Humans（CoRL 2024）

| 项目 | 内容 |
|------|------|
| **机构** | Stanford (Zipeng Fu) |
| **论文** | HumanPlus: Humanoid Shadowing and Imitation from Humans |
| **代码** | https://github.com/MarkFzp/humanplus （已下载至 `ref_repo/HumanPlus`） |
| **核心贡献** | 单 RGB 相机实时驱动 Unitree H1 全身运动，40 小时 AMASS 数据 zero-shot 迁移到真实机器人 |

**技术要点**：
- **Humanoid Shadowing Transformer**：基于 PPO 的底层 RL 策略，在 AMASS 40 小时数据上训练，用于将 SMPL 姿态实时转化为机器人关节命令
- **Humanoid Imitation Transformer (HIT)**：高层视觉策略，基于遥操作数据学习自主技能
- **20-40 次示教达到 60-100% 成功率**：穿鞋、叠衣服、打字、搬运等日常任务

**对动作生成大模型的意义**：
- HumanPlus 证明了**大规模运动学数据（如 AMASS）可以直接训练机器人底层策略**
- 如果运动生成模型能产生高质量、多样化的合成运动数据，可以大幅扩展 HumanPlus 的训练集规模
- HumanPlus 的 SMPL 接口天然兼容 HyMotion/KIMODO 的输出表示

---

#### 2.2.5 RoboGhost: Retargeting-Free Humanoid Control via Motion Latent Guidance（2025）

| 项目 | 内容 |
|------|------|
| **机构** | 多机构 |
| **论文** | RoboGhost: Retargeting-Free Humanoid Control via Motion Latent Guidance |
| **代码** | https://gentlefress.github.io/roboghost-proj/ |
| **核心贡献** | 跳过运动解码和 retargeting，直接从**语言-运动隐表示**控制人形机器人 |

**技术要点**：
- **两阶段训练**：
  - Stage 1：连续自回归运动生成器，产出 motion latents
  - Stage 2：基于 Diffusion 的学生策略，从 MoE 教师策略蒸馏，直接从 motion latents 去噪为可执行动作
- **延迟降低 67%**：从 17.85s 降至 5.84s，成功率提升 5%
- **多模态支持**：文本、图像、音乐等都可作为输入

**对动作生成大模型的意义**：
- RoboGhost 是**动作生成模型与机器人控制最直接的融合**——模型的潜空间直接成为控制空间
- 证明了「运动生成模型 ≠ 只产出关节序列」，其潜表示本身就是有价值的控制信号
- 对 HyMotion 的启示：Flow Matching 的隐空间（velocity field）可能可以直接用于类似的 latent-guided 控制

---

#### 2.2.6 DreamControl: Human-Inspired Whole-Body Humanoid Control（2025）

| 项目 | 内容 |
|------|------|
| **机构** | 多机构 |
| **论文** | DreamControl: Human-Inspired Whole-Body Humanoid Control for Scene Interaction via Guided Diffusion |
| **链接** | https://arxiv.org/abs/2509.14353 |
| **核心贡献** | 将扩散模型训练的人类运动先验作为 RL 的引导信号，实现自然的场景交互 |

**技术要点**：
- **Diffusion Prior + RL**：扩散模型提供「人类会怎么动」的先验分布，RL 在此基础上针对具体任务（开抽屉、拾取物体）优化
- **自然度提升**：相比纯 RL，运动更加接近人类习惯
- **Unitree G1 验证**：在真实机器人上完成导航 + 操作的全身任务

**与动作生成大模型的关系**：
- DreamControl 的 diffusion prior 与 HyMotion/KIMODO 的生成模型本质相同 —— 都是从人类运动数据学到的概率分布
- 启示：HyMotion 的 Flow Matching 模型可以作为类似的 prior，用于引导 RL 策略

---

#### 2.2.7 BeyondMimic: From Motion Tracking to Versatile Humanoid Control（2025）

| 项目 | 内容 |
|------|------|
| **机构** | UC Berkeley + Stanford |
| **论文** | BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion |
| **链接** | https://arxiv.org/abs/2508.08241 |
| **核心贡献** | 统一的 pipeline：MoCap → 高质量运动跟踪 → guided diffusion 实现零样本任务泛化 |

**技术要点**：
- **Scalable Motion Tracking**：单一超参/MDP 训练所有动作（跳跃旋转、短跑、侧翻等高难度动作）
- **Guided Diffusion Policy**：在推理时通过简单 cost function（航点导航、避障）引导 diffusion policy 合成新动作，无需重训
- **Sim-to-Real**：在 Unitree G1 上实现，用户偏好自然度调查中获 70.8% 好评

---

#### 2.2.8 MaskedMimic: Unified Physics-Based Character Control（SIGGRAPH Asia 2024）

| 项目 | 内容 |
|------|------|
| **机构** | NVIDIA Research |
| **论文** | MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting |
| **链接** | https://research.nvidia.com/labs/par/maskedmimic/ |
| **核心贡献** | 将物理角色控制统一为**masked motion inpainting** 问题，VR 跟踪/文本驱动/手柄控制共用一个控制器 |

**技术要点**：
- **Masked Inpainting 视角**：VR 跟踪 = 给定头/手位置的 inpainting；文本控制 = 给定语义约束的 inpainting
- **Stage 1**：RL 训练全身运动跟踪；**Stage 2**：Teacher-Student 蒸馏，从 partial/masked 输入预测动作
- **98.1% VR 跟踪成功率**，95% 不规则地形成功率

**与 HyMotion M2M 的关系**：
- MaskedMimic 的 masked inpainting 思想与 HyMotion M2M 的 VACE conditioning 高度同构
- HyMotion M2M 在运动学层面做 mask-based completion，MaskedMimic 在物理仿真层面做 mask-based control
- **潜在融合方向**：HyMotion 生成 + MaskedMimic 物理执行 = 端到端的条件动作生成 + 物理控制

---

#### 2.2.9 SuperPADL: Scaling Language-Directed Physics-Based Control（2024）

| 项目 | 内容 |
|------|------|
| **机构** | 多机构 |
| **论文** | SuperPADL: Scaling Language-Directed Physics-Based Control with Progressive Distillation |
| **链接** | https://arxiv.org/abs/2407.10481 |
| **核心贡献** | 5000+ 运动片段的大规模物理控制器，支持自然语言接口 |

**技术要点**：
- **Progressive Distillation**：RL 专家 → 混合 RL/监督学习 → 统一策略
- **语言接口**：文本指令 → 动作选择与过渡
- **消费级 GPU 实时运行**

---

### 2.3 技术路线对比表

| 工作 | 年份 | 运动来源 | 控制方式 | 物理仿真器 | 机器人硬件 | 运动生成模型角色 |
|------|------|---------|---------|-----------|-----------|---------------|
| **PHC** | 2023 | AMASS MoCap | RL 跟踪 | IsaacGym | — (仿真) | 可接 MDM 输出 |
| **PULSE** | 2024 | PHC 蒸馏 | 潜空间 RL | IsaacGym | — (仿真) | 潜空间可替换为生成模型隐空间 |
| **OmniH2O** | 2024 | AMASS retarget | Teacher-Student RL | IsaacGym | Unitree H1 | SMPL 接口兼容 |
| **HumanPlus** | 2024 | AMASS 40h | PPO + 行为克隆 | IsaacGym | Unitree H1 | 可扩展训练集 |
| **RoboGhost** | 2025 | 运动潜表示 | Diffusion Policy | IsaacGym/MuJoCo | Unitree G1 | **隐表示直接控制** |
| **DreamControl** | 2025 | Diffusion Prior | RL + Diffusion 引导 | — | Unitree G1 | **作为 RL 先验** |
| **BeyondMimic** | 2025 | MoCap | RL + Guided Diffusion | — | Unitree G1 | **Diffusion Policy** |
| **MaskedMimic** | 2024 | MoCap | RL + Masked Inpainting | — | — (仿真) | **Masked 生成 = 控制** |
| **SuperPADL** | 2024 | 5000+ clips | Progressive Distillation | — | — (仿真) | 语言→动作选择 |

### 2.4 对 HyMotion / KIMODO 的启示

#### 2.4.1 HyMotion M2M 的优势与机会

1. **VACE Conditioning ↔ MaskedMimic 同构**：HyMotion M2M 的 per-dim mask 机制（T×135）与 MaskedMimic 的 masked inpainting 理念高度一致。如果在 HyMotion 输出的基础上加物理跟踪层，可以直接获得「条件生成 + 物理执行」能力。

2. **Flow Matching 隐空间的控制潜力**：类似 RoboGhost 的 latent guidance 思路，HyMotion 的 velocity field 可以直接作为机器人策略的条件输入，跳过运动解码步骤。

3. **大规模数据优势**：HyMotion 训练在 MotionHub（549K 样本）上，远超多数 RL 方法使用的数据规模。如果输出质量足够高，可以作为 HumanPlus/OmniH2O 等方法的数据增强源。

#### 2.4.2 当前差距

1. **缺少物理可行性验证**：HyMotion 输出未经物理仿真验证，可能存在浮空、穿地等问题
2. **缺少 xyz position 约束**：KIMODO 支持直接约束 xyz position（333-dim 表示含 position），而 HyMotion 仅有 rotation（135-dim），无法直接约束末端位置——这在机器人操作任务中是关键需求
3. **缺少 retargeting 接口**：需要将 SMPL 输出 retarget 到具体机器人骨架

#### 2.4.3 建议探索方向

| 优先级 | 方向 | 具体做法 | 参考工作 |
|--------|------|---------|---------|
| P0 | 物理可行性评估 | 将 HyMotion 输出接入 PHC/PULSE，统计跟踪成功率和物理指标 | PHC, PULSE |
| P1 | 添加 position 维度 | 扩展到 201-dim（含 22 joints xyz），支持末端位置约束 | KIMODO, UMO |
| P1 | 运动潜空间与 RL 对接 | 探索 Flow Matching 的 latent 作为 RL action space | RoboGhost, PULSE |
| P2 | Retargeting Pipeline | 实现 SMPL → Unitree H1/G1 的自动 retarget | OmniH2O |
| P2 | Diffusion Prior for RL | 将 HyMotion 作为 DreamControl 式的 prior | DreamControl |

---

## 3. 方向二：仿真强化学习 → 提升动作生成的物理真实性

### 3.1 总体思路：物理仿真为生成模型注入物理约束

纯运动学的动作生成模型（包括 HyMotion M2M）面临的核心问题是：**生成的运动不保证物理可行性**。常见问题包括：

| 问题 | 表现 | 原因 |
|------|------|------|
| 足部滑动 (foot sliding) | 脚与地面相对滑动 | 未建模接触力学 |
| 浮空 (floating) | 人体漂浮在空中 | 未建模重力 |
| 穿地 (ground penetration) | 脚穿过地面 | 无碰撞检测 |
| 关节超限 | 关节角度超出物理范围 | 未约束关节极限 |
| 不合理加速 | 瞬间速度变化不符合惯性 | 未建模动力学 |

物理仿真和 RL 可以通过以下方式帮助解决这些问题：

```
方式 A: 后处理（Post-Processing）
  生成模型输出 → 物理仿真器投影 → 修正后的运动
  代表：PhysDiff

方式 B: 训练时物理约束（Physics-Guided Training）
  训练数据 + 物理仿真数据 → 联合训练
  代表：SimDiff, POMP

方式 C: RL 微调（RL Fine-Tuning）
  预训练生成模型 → 物理仿真器评估 → RL 优化
  代表：RLPF, RobotMDM

方式 D: 端到端物理感知（Physics-Aware Generation）
  将物理约束作为损失/奖励直接嵌入生成模型
  代表：POMP, DreamControl
```

### 3.2 关键工作详解

#### 3.2.1 PhysDiff: Physics-Guided Human Motion Diffusion Model（ICCV 2023）

| 项目 | 内容 |
|------|------|
| **机构** | NVIDIA Labs (Ye Yuan) |
| **论文** | PhysDiff: Physics-Guided Human Motion Diffusion Model |
| **链接** | https://nvlabs.github.io/PhysDiff/ |
| **核心贡献** | 首次在扩散过程中嵌入物理仿真器，在每个去噪步将运动投影到物理可行空间 |

**技术要点**：
- **Physics-Based Motion Projection**：每个去噪步 t，先用 DDPM 预测 x_0，然后用 MuJoCo 仿真器将 x_0 投影为物理可行的 x_0'，再用 x_0' 计算 x_{t-1}
- **迭代修正**：物理投影的结果会影响后续去噪步，形成「去噪 ↔ 物理修正」的闭环
- **效果**：物理可行性提升 78%+，足部滑动和穿地显著减少

**核心公式**：
```
标准 DDPM:     x_{t-1} = f(x_t, ε_θ(x_t, t))
PhysDiff:      x_{t-1} = f(x_t, ε_θ(x_t, t), PhysSim(x̂_0))
                                                ↑ 物理仿真投影
```

**局限与后续**：
- 每个去噪步都需要调用物理仿真器，推理速度慢
- SimDiff（2025）将此方法重新解释为 classifier guidance，实现了无需仿真器的推理

---

#### 3.2.2 SimDiff: Simulator-Constrained Diffusion Model（2025）

| 项目 | 内容 |
|------|------|
| **论文** | SimDiff: Simulator-Constrained Diffusion Model for Physically Plausible Motion Generation |
| **链接** | https://arxiv.org/abs/2509.20927 |
| **核心贡献** | 将物理环境参数（重力、风速等）编码为条件，推理时**无需调用仿真器**即可生成物理可行运动 |

**技术要点**：
- **Domain-Randomized Simulation Data**：用 MuJoCo 在不同物理条件（变重力、变风速）下生成训练数据
- **Sim Encoder + Motion Adapters**：冻结预训练 diffusion backbone（如 MDM），仅训练轻量适配器注入物理条件
- **Classifier-Free Guidance**：将物理参数作为条件，通过 CFG 控制物理约束强度
- **组合泛化**：能适应训练时未见过的环境参数组合（如月球重力）

**对 HyMotion M2M 的启示**：
- SimDiff 的「冻结 backbone + 适配器」范式完全适用于 HyMotion——可以在 HyMotion 的 MMDiT 上加物理适配器
- 不需要修改 HyMotion 的核心训练流程

---

#### 3.2.3 RLPF: RL from Physical Feedback（2025，投稿 ICLR 2026）

| 项目 | 内容 |
|------|------|
| **论文** | RL from Physical Feedback: Aligning Large Motion Models with Humanoid Control |
| **链接** | https://arxiv.org/abs/2506.12769 |
| **代码** | https://github.com/BeingBeyond/RLPF (计划开源) |
| **核心贡献** | 用 RL 微调运动生成模型，使输出同时满足语义对齐和物理可行性 |

**技术要点**：
- **三阶段框架**：
  1. **Physics-Aware Evaluation**：将生成的运动送入 IsaacGym 物理仿真器，由运动跟踪策略评估可行性
  2. **Alignment Verification**：验证运动是否仍然忠实于文本指令
  3. **RL Fine-Tuning (GRPO)**：使用 Group Relative Policy Optimization 微调生成模型，平衡物理可行性和语义对齐
- **在 HumanML3D/AMASS 上验证**，成功部署在人形机器人上

**核心思想**：
```
              ┌─────────────────┐
              │  Motion Gen Model│ ← GRPO 更新
              │  (e.g. HyMotion) │
              └────────┬────────┘
                       │ 生成运动
                       ▼
              ┌─────────────────┐     ┌──────────────────┐
              │ Physics Simulator│────→│ Feasibility Score │
              │  (IsaacGym)     │     │  (物理可行性奖励)  │
              └─────────────────┘     └──────────────────┘
                                              │
              ┌─────────────────┐             │
              │ Text Alignment  │────→ 语义对齐奖励
              │  Checker        │             │
              └─────────────────┘             │
                                              ▼
                                      RL 微调梯度 → 更新模型
```

**对 HyMotion M2M 的意义**：
- **最直接的改进路径**：不需要修改 HyMotion 的架构，只需在训练后用 RLPF 做物理对齐微调
- GRPO 方法已在 LLM 对齐中验证（类似 RLHF 的思路），迁移到运动生成领域
- 可以与现有的 HyMotion 质量检查系统（`motion_annot_web/m2m_database` 的 quality checkers）结合

---

#### 3.2.4 RobotMDM: Motion Diffusion Model + RL Tracking（SIGGRAPH Asia 2024）

| 项目 | 内容 |
|------|------|
| **机构** | Disney Research + ETH Zurich |
| **论文** | RobotMDM (Serifi et al.) |
| **核心贡献** | Diffusion 生成 + RL 跟踪 + Reward Surrogate 反向微调 |

**技术要点**：
- **Reward Surrogate Model**：训练一个可微的代理模型，预测 RL 跟踪控制器的性能
- **Physical Alignment**：代理模型提供可微损失函数，反向传播微调 diffusion model
- **结果**：在走路、跑步、拳击等任务上超越 MDM 和 PhysDiff 基线

**思路**：
```
Diffusion Model → 生成运动 → RL Controller 跟踪
                    ↑                    │
                    │      Reward Surrogate (可微)
                    └──── 梯度反传 ◄──────┘
```

---

#### 3.2.5 POMP: Physics-Consistent Motion Generation through Phase Manifolds（CVPR 2025）

| 项目 | 内容 |
|------|------|
| **机构** | 上海交大 + 浙大 |
| **论文** | POMP: Physics-Consistent Motion Generative Model through Phase Manifolds |
| **链接** | https://openaccess.thecvf.com/content/CVPR2025/html/Ji_POMP_Physics-consistent_Motion_Generative_Model_through_Phase_Manifolds_CVPR_2025_paper.html |
| **核心贡献** | 运动学生成 + 物理仿真修正 + 相位流形对齐，实现长程物理一致性 |

**技术要点**：
- **三模块架构**：
  1. Diffusion-based Kinematic Module：逐帧生成运动学姿态
  2. Simulation-based Dynamic Module：物理仿真修正（碰撞响应、地形适应）
  3. Phase Encoding Module：将仿真结果投影回运动学先验的相位流形，防止累积误差
- **相位流形对齐**：通过语义对齐确保仿真修正后的运动仍然「自然」

---

#### 3.2.6 TeSMo: Text-Controlled Scene-Aware Motion Generation（ECCV 2024）

| 项目 | 内容 |
|------|------|
| **机构** | NVIDIA |
| **代码** | https://github.com/nv-tlabs/tesmo （已下载至 `ref_repo/TeSMo`） |
| **核心贡献** | 文本驱动的场景感知运动生成（如在椅子上坐下） |

**与物理约束的关系**：
- 场景几何作为隐式物理约束（不能穿过椅子、需要接触座面等）
- 使用 SDF (Signed Distance Field) 和接触约束引导 diffusion 过程

---

#### 3.2.7 MotionLCM: Real-time Controllable Motion Generation（ECCV 2024）

| 项目 | 内容 |
|------|------|
| **代码** | https://github.com/LinghaoChan/MotionLCM （已下载至 `ref_repo/MotionLCM`） |
| **核心贡献** | 基于 Latent Consistency Model 的实时运动生成（30ms/序列），支持轨迹控制 |

**对物理约束的启示**：
- 1-4 步推理实现接近多步 diffusion 的质量
- 实时性对于 RL 训练循环中的运动生成至关重要（RLPF 等方法需要大量采样）
- Motion ControlNet 架构可用于注入物理约束

---

### 3.3 技术路线对比表

| 工作 | 年份 | 物理约束注入方式 | 推理时需要仿真器 | 修改生成模型 | 物理可行性提升 |
|------|------|----------------|----------------|------------|-------------|
| **PhysDiff** | 2023 | 每步仿真投影 | ✅ 需要 | ❌ 不需要 | 78%+ |
| **SimDiff** | 2025 | 条件化 + 适配器 | ❌ 不需要 | 轻量适配器 | 超越 PhysDiff |
| **RLPF** | 2025 | RL 微调 (GRPO) | 训练时需要 | 微调全模型 | 显著提升 |
| **RobotMDM** | 2024 | Reward 代理梯度 | 训练时需要 | 微调 diffusion | 超越 MDM |
| **POMP** | 2025 | 相位流形对齐 | ✅ 每帧需要 | 端到端训练 | 长程一致 |
| **TeSMo** | 2024 | 场景 SDF 约束 | ❌ | 条件化 | 场景感知 |

### 3.4 对 HyMotion M2M 的启示

#### 3.4.1 直接可用的改进方案

| 方案 | 难度 | 效果预期 | 实施路径 |
|------|------|---------|---------|
| **A: RLPF 式 RL 微调** | 中 | 最直接提升物理可行性 | 搭建 IsaacGym 物理评估环境 → 训练跟踪策略 → 设计物理奖励 → GRPO 微调 HyMotion |
| **B: SimDiff 式适配器** | 低 | 零推理成本，轻量训练 | 冻结 HyMotion MMDiT → 加 Sim Encoder + Motion Adapters → 物理条件化训练 |
| **C: PhysDiff 式仿真投影** | 低 | 无需重训，后处理 | MuJoCo/IsaacGym 搭建 SMPL 仿真环境 → 每个 Flow Matching 步投影 |
| **D: 显式物理损失** | 低-中 | 改善特定问题 | 训练时加 foot contact loss、ground penetration loss、joint limit loss |
| **E: MoGenDIT 修复后处理** | 已有 | 改善局部问题 | 利用现有 MoGenDIT repair pipeline 修复浮空/穿地 |

#### 3.4.2 推荐优先级

1. **P0（短期）**：方案 D — 在 HyMotion M2M 训练中添加显式物理损失（foot contact、ground penetration），不改变架构
2. **P0（短期）**：方案 E — 利用现有 MoGenDIT pipeline 作为后处理修复
3. **P1（中期）**：方案 C — 实现 PhysDiff 式的仿真投影（在 Flow Matching 步中嵌入物理修正）
4. **P1（中期）**：方案 B — SimDiff 式适配器（冻结 MMDiT + 轻量物理适配器）
5. **P2（长期）**：方案 A — RLPF 式 RL 微调（需要搭建完整的物理评估和 RL 训练环境）

---

## 4. 两个方向的统一框架展望

两个方向正在快速融合，形成「动作生成 ↔ 物理仿真 ↔ 机器人控制」的闭环：

```
┌──────────────────────────────────────────────────────────────┐
│                    统一框架愿景                                │
│                                                              │
│   ┌────────────┐                      ┌────────────────┐     │
│   │ 文本/语言   │                      │ 物理仿真环境    │     │
│   │ 指令        │                      │ (IsaacGym等)   │     │
│   └──────┬─────┘                      └───────┬────────┘     │
│          │                                    │              │
│          ▼                                    │ 物理奖励     │
│   ┌────────────────┐    运动先验/潜表示     │              │
│   │ 运动生成大模型  │◄───────────────────────┘              │
│   │ (HyMotion/Flow │                                        │
│   │  Matching)     │──────────┐                             │
│   └────────────────┘          │                             │
│                               │ 运动序列/潜表示              │
│                               ▼                             │
│                    ┌────────────────┐                        │
│                    │ 物理控制策略    │                        │
│                    │ (PHC/PULSE/    │                        │
│                    │  MaskedMimic)  │                        │
│                    └───────┬────────┘                        │
│                            │ 关节力矩                        │
│                            ▼                                 │
│                    ┌────────────────┐                        │
│                    │ 人形机器人      │                        │
│                    │ (Unitree G1等) │                        │
│                    └────────────────┘                        │
│                                                              │
│   核心闭环：                                                  │
│   1. 生成模型提供多样运动先验 → 丰富机器人动作库               │
│   2. 物理仿真验证并反馈 → 提升生成质量                        │
│   3. 机器人执行结果 → 新的训练数据                            │
└──────────────────────────────────────────────────────────────┘
```

**关键趋势**：

1. **Motion as Language**：将运动 token 化，像 LLM 处理语言一样处理运动（MotionGlot），支持跨具身体（quadruped、humanoid）泛化
2. **Retargeting-Free Control**：跳过运动解码和 retargeting，直接用潜表示控制（RoboGhost）
3. **RL from Physical Feedback**：借鉴 RLHF 思路，用物理仿真器作为「人类反馈」（RLPF）
4. **Diffusion Policy**：扩散模型不仅用于生成运动，还直接作为控制策略（BeyondMimic、DreamControl）
5. **Foundation Model for Motion**：NVIDIA GR00T N1.6 等尝试构建通用运动基础模型，整合全身 RL、VLA、合成数据

---

## 5. 已下载代码与文献索引

### 已下载至 `ref_repo/` 的代码仓库

| 目录 | 项目 | 来源 | 关联方向 |
|------|------|------|---------|
| `ref_repo/PHC/` | PHC: Perpetual Humanoid Control | https://github.com/ZhengyiLuo/PHC | 方向① 运动跟踪 |
| `ref_repo/OmniH2O/` | OmniH2O: Human-to-Humanoid | https://github.com/LeCAR-Lab/human2humanoid | 方向① 遥操作 |
| `ref_repo/HumanPlus/` | HumanPlus: Humanoid Imitation | https://github.com/MarkFzp/humanplus | 方向① 模仿学习 |
| `ref_repo/TeSMo/` | TeSMo: Scene-Aware Motion | https://github.com/nv-tlabs/tesmo | 方向② 场景约束 |
| `ref_repo/MotionLCM/` | MotionLCM: Real-time Generation | https://github.com/LinghaoChan/MotionLCM | 方向② 实时生成 |
| `ref_repo/MDM/` | MDM: Motion Diffusion Model | https://github.com/GuyTevet/motion-diffusion-model | 基线参考 |
| `ref_repo/KIMODO/` | KIMODO (NVIDIA) | 已有 | 动作生成基线 |
| `ref_repo/UMO/` | UMO | 已有 | 动作生成基线 |
| `ref_repo/MoGenDiT/` | MoGenDiT (内部) | 已有 | 修复管线 |

### 未下载但已调研的重要工作（代码未开源或需特殊权限）

| 项目 | 原因 | 论文链接 |
|------|------|---------|
| PhysDiff | GitHub 认证问题 | https://nvlabs.github.io/PhysDiff/ |
| RLPF | 计划开源但未发布 | https://arxiv.org/abs/2506.12769 |
| RoboGhost | 项目页，代码状态不明 | https://arxiv.org/abs/2510.14952 |
| DreamControl | 项目页 | https://arxiv.org/abs/2509.14353 |
| BeyondMimic | 项目页 | https://arxiv.org/abs/2508.08241 |
| MaskedMimic | NVIDIA 内部 | https://research.nvidia.com/labs/par/maskedmimic/ |
| SimDiff | — | https://arxiv.org/abs/2509.20927 |
| POMP | — | CVPR 2025 Open Access |
| SuperPADL | — | https://arxiv.org/abs/2407.10481 |
| RobotMDM | Disney Research 内部 | SIGGRAPH Asia 2024 |
| PULSE | 与 PHC 同源 | https://arxiv.org/abs/2310.04582 |

---

## 6. 参考文献完整列表

### 方向一：动作生成大模型 → 具身智能

[1] Zhengyi Luo et al. "Perpetual Humanoid Control for Real-time Simulated Avatars." ICCV 2023. https://github.com/ZhengyiLuo/PHC

[2] Zhengyi Luo et al. "Universal Humanoid Motion Representations for Physics-Based Control." ICLR 2024 Spotlight. https://arxiv.org/abs/2310.04582

[3] Tairan He et al. "OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning." CoRL 2024. https://arxiv.org/abs/2406.08858

[4] Zipeng Fu et al. "HumanPlus: Humanoid Shadowing and Imitation from Humans." CoRL 2024. https://arxiv.org/abs/2406.10454

[5] RoboGhost: Retargeting-Free Humanoid Control via Motion Latent Guidance. 2025. https://arxiv.org/abs/2510.14952

[6] DreamControl: Human-Inspired Whole-Body Humanoid Control for Scene Interaction via Guided Diffusion. 2025. https://arxiv.org/abs/2509.14353

[7] BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion. 2025. https://arxiv.org/abs/2508.08241

[8] Chen Tessler et al. "MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting." SIGGRAPH Asia 2024. https://research.nvidia.com/labs/par/maskedmimic/

[9] SuperPADL: Scaling Language-Directed Physics-Based Control with Progressive Distillation. 2024. https://arxiv.org/abs/2407.10481

[10] MotionGlot: Motion as a Language for cross-embodiment control. TechXplore 2025. https://techxplore.com/news/2025-05-ai-motion-kinds-robots.html

### 方向二：仿真强化学习 → 提升物理真实性

[11] Ye Yuan et al. "PhysDiff: Physics-Guided Human Motion Diffusion Model." ICCV 2023. https://nvlabs.github.io/PhysDiff/

[12] SimDiff: Simulator-Constrained Diffusion Model for Physically Plausible Motion Generation. 2025. https://arxiv.org/abs/2509.20927

[13] Junpeng Yue et al. "RL from Physical Feedback: Aligning Large Motion Models with Humanoid Control." Submitted ICLR 2026. https://arxiv.org/abs/2506.12769

[14] Serifi et al. "RobotMDM." SIGGRAPH Asia 2024. Disney Research + ETH Zurich.

[15] Bin Ji, Ye Pan et al. "POMP: Physics-Consistent Motion Generative Model through Phase Manifolds." CVPR 2025.

[16] NVIDIA. TeSMo: Text-Controlled Scene-Aware Motion Generation. ECCV 2024. https://github.com/nv-tlabs/tesmo

[17] Wenxun Dai et al. "MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model." ECCV 2024. https://github.com/LinghaoChan/MotionLCM

### 基础参考

[18] Guy Tevet et al. "Human Motion Diffusion Model." ICLR 2023. https://github.com/GuyTevet/motion-diffusion-model

[19] NVIDIA. KIMODO: Kinematics-Aware Motion Diffusion Transformer. 2026. (已在 ref_repo/KIMODO)

[20] UMO: Unified Motion Operations. 2026. (已在 ref_repo/UMO)

[21] NVIDIA. GR00T N1.6: Foundation Model for Humanoid Robots. 2025-2026.

[22] NVIDIA Isaac GR00T Blueprint for humanoid robot development.

---

> **附注**: 本报告所有临时文档按项目规范存放于 `docs/temp/`。已下载的代码仓库统一存放于 `ref_repo/`。
