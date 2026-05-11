# 调研报告 v3：动作生成大模型 ↔ 具身智能 — 双向融合研究与实践方案

> **日期**: 2026-05-12（基于 v2 2026-05-08 全面升级，新增五问深度分析和最新文献）
> **课题**: (1) 动作生成大模型如何指导机器人/具身智能训练；(2) 具身智能训练方式如何提升动作生成模型的物理真实性
> **存放位置**: `docs/temp/survey_motion_gen_embodied_v3_20260512.md`

---

## 目录

1. [研究背景与核心矛盾](#1-研究背景与核心矛盾)
2. [五问深度分析](#2-五问深度分析)
3. [关键文献与代码索引](#3-关键文献与代码索引)
4. [统一闭环框架设计](#4-统一闭环框架设计)
5. [可行实践方案](#5-可行实践方案)
6. [参考文献](#6-参考文献)

---

## 1. 研究背景与核心矛盾

### 1.1 两大领域的现状

**动作生成大模型**（如 HyMotion M2M / T2M 1.0、MDM、MotionDiffuse、MoMask 等）：
- ✅ 擅长：语义控制（文本/音频→动作）、多样性、大规模数据训练（549K+ 样本）
- ❌ 短板：生成动作不保证物理可行性 — 浮空、滑动、穿地、关节扭曲、地面反力不合理

**具身智能/机器人训练**（RL + 物理仿真，如 IsaacGym/MuJoCo + PPO/SAC）：
- ✅ 擅长：物理可行性、接触力学、Sim2Real 部署、稳定平衡控制
- ❌ 短板：动作多样性受限、数据稀缺、探索效率低、不具备语义理解能力

### 1.2 核心矛盾

```
┌───────────────────────────────────────┐
│  动作生成大模型                          │
│  (HyMotion M2M / Flow Matching)        │
│                                       │
│  ✅ 多样性、语义控制、大规模数据            │
│  ❌ 物理不可行（浮空/滑动/穿地）            │
└────────────┬──────────────────────────┘
             │                    ▲
    方向A：运动先验              方向B：物理反馈
    指导机器人训练               提升生成质量
             │                    │
             ▼                    │
┌───────────────────────────────────────┐
│  物理仿真 / 强化学习 / 机器人             │
│  (IsaacGym, MuJoCo, Unitree G1)       │
│                                       │
│  ✅ 物理真实性、接触力学、Sim2Real         │
│  ❌ 动作多样性受限、数据稀缺               │
└───────────────────────────────────────┘
```

**两个方向互惠**：
- **方向 A**：动作生成模型输出丰富运动先验 → 降低 RL 探索成本、扩大动作库
- **方向 B**：物理仿真提供可行性反馈 → 反向修正/微调生成模型，使输出物理可行

---

## 2. 五问深度分析

### 问题 1：为什么之前的方法无法做好该问题？

**核心原因：运动学与动力学的割裂**

之前的动作生成方法（MDM、MoMask、MotionDiffuse、T2M-GPT 等）存在以下系统性缺陷：

| 缺陷类别 | 具体表现 | 根本原因 |
|----------|---------|---------|
| **训练数据纯运动学** | 模型学习的是 MoCap 关节角轨迹，不含力/力矩信息 | HumanML3D/AMASS 数据集只有运动学（kinematics），没有动力学（dynamics） |
| **无物理约束** | 生成动作可能浮空、穿地、滑动、关节超限 | 损失函数只监督关节位置/角度重建，不包含物理约束项 |
| **无接触建模** | 不知道何时触地、何时施力 | 没有足-地接触模型（contact model），没有地面反力（GRF） |
| **评估指标脱离物理** | FID/R-precision 不反映物理可行性 | 评估只看分布匹配和语义对齐，不看"能不能在物理世界执行" |
| **Sim2Real gap** | 运动学轨迹无法直接在真实机器人上执行 | 缺少力矩映射、平衡控制、接触适应 |

**为什么不能简单加物理损失？**
- 在训练时加 `ground_penetration_loss`、`foot_contact_loss` 等只能缓解部分问题
- 这些是**弱约束**（soft constraint），无法保证物理一致性（hard constraint）
- 真正的物理约束需要**闭环仿真**（forward dynamics），而不是几何近似

**之前两个领域各自为战的原因**：
1. **动作生成社区**关注的指标（FID、Diversity、R-precision）不包含物理可行性
2. **机器人社区**不需要"多样性"，只需要稳定可靠的控制策略
3. **技术栈不同**：动作生成 = PyTorch + diffusion/flow；机器人 = IsaacGym/MuJoCo + PPO
4. **数据格式不同**：动作生成用 HumanML3D (263-dim)；机器人用关节角 + PD 控制
5. **缺少桥梁工具**：直到 GMR (2026) 才有通用实时 SMPL→Robot retargeting

---

### 问题 2：动作生成模型如何作用于具身智能模型训练？

**五条技术路线**：

#### 路线 A1: Reference Motion Tracking（最成熟 ⭐⭐⭐）
```
动作生成模型 → SMPL motion → GMR retarget → Robot joint trajectory
                                                    ↓
                                              RL tracking policy (PHC/ASAP)
                                                    ↓
                                              物理可行的机器人运动
```
- **代表**: PHC → OmniH2O → HumanPlus → ASAP → VideoMimic → BeyondMimic
- **原理**: 生成运动序列作为 reference，RL 策略学习跟踪执行
- **HyMotion 对接**: 输出 SMPL → GMR retarget → ProtoMotions/PHC 跟踪。**最低对接成本**

#### 路线 A2: Motion Latent as Action Space
```
动作生成模型(encoder) → 潜空间 z → RL 在 z 空间优化 → decoder → 运动
```
- **代表**: PULSE, RoboGhost
- **原理**: 用生成模型的隐空间约束 RL 搜索空间，避免无效探索
- **优势**: RL 搜索空间大幅缩小，探索效率提升

#### 路线 A3: Diffusion Policy（直接作为控制策略）
```
观测 (state + goal) → 动作生成模型 → 动作序列 → 执行
```
- **代表**: DreamControl-v2, PDP (Physics-Based Diffusion Policy)
- **原理**: 扩散/流匹配模型直接输出控制动作，不再需要单独的 RL 策略
- **挑战**: 需要在仿真环境中训练，推理速度要求高

#### 路线 A4: Data Augmentation（合成数据扩充训练集 ⭐⭐⭐）
```
少量种子数据 → 动作生成模型 → 大量合成运动 → RL 训练数据
```
- **代表**: PARC (SIGGRAPH 2025), GR00T-Mimic (NVIDIA), UH-1
- **PARC 的迭代闭环**: 生成→物理修正→数据增强→重训→生成更好的运动
- **GR00T-Mimic**: 11小时生成 780K 合成轨迹（= 6500 人类小时），性能 +40%
- **HyMotion 价值**: 549K 训练样本 + 条件控制 → 可构建类似合成数据管线

#### 路线 A5: Language-Guided Robot Control
```
自然语言指令 → 动作生成模型 → 目标动作 → RL 策略跟踪
```
- **代表**: Humanoid-LLA, UH-1, CLAW
- **原理**: 动作生成模型作为"语言→运动"的翻译器
- **HyMotion T2M 的天然优势**: 已具备文本→运动生成能力

**核心价值总结**：
| 作用方式 | 价值 | 例子 |
|----------|------|------|
| 运动先验 | 约束 RL 搜索空间 | PHC 跟踪 HyMotion 输出 |
| 数据放大 | 合成海量训练数据 | PARC 式数据飞轮 |
| 语义桥接 | 文本指令→机器人动作 | HyMotion T2M → GMR → Robot |
| 风格迁移 | 将人类动作风格迁移到机器人 | ExBody2 表现力控制 |

---

### 问题 3：动作生成模型对于具身智能的训练有什么必须性？之前的具身智能算法是用什么数据训练的？

#### 之前具身智能的数据来源

| 数据来源 | 代表工作 | 规模 | 局限性 |
|----------|---------|------|--------|
| **光学 MoCap** | PHC (AMASS 40h), KIMODO (700h) | 数十~数百小时 | 成本极高（$500-2000/小时），场景受限 |
| **视频重建** | VideoMimic, UH-1 (163K videos) | 数十万视频 | 3D重建噪声大，精度有限 |
| **遥操作** | GR00T-Teleop, HumanPlus | 数小时~数十小时 | 需要硬件设备，采集效率低 |
| **RL 自我探索** | Legged Gym, AnySkill | 仿真中无限 | 动作多样性受限于奖励函数设计 |
| **手工设计** | 传统步态规划 | 有限 | 无法覆盖复杂/创意动作 |

#### 动作生成模型的必须性论证

**必须性 1：数据规模的瓶颈**
- MoCap 采集成本 ~$1000/小时，AMASS 数据集 40 小时 ≈ $40K+
- KIMODO 700 小时光学 MoCap 仅一家机构（NVIDIA）可负担
- 视频重建质量不稳定（噪声、遮挡、深度歧义）
- **动作生成模型可以 zero-cost 产出无限多样训练数据**

**必须性 2：覆盖长尾动作**
- MoCap 数据集主要覆盖行走/跑步/基础动作
- 舞蹈、武术、体操、日常交互等长尾动作数据稀缺
- **动作生成模型通过文本条件可以生成任意语义的动作**

**必须性 3：可控数据增强**
- PARC 证明：迭代式「生成→物理修正→增强」可以指数级扩展能力边界
- GR00T-Mimic 证明：合成数据带来 +40% 性能提升
- **动作生成模型提供的不是随机增强，而是有条件、有语义的增强**

**必须性 4：跨具身体（cross-embodiment）泛化**
- 不同机器人形态（双足/四足/轮式）需要不同运动数据
- 动作生成模型 → GMR retarget 可适配 17+ 机器人
- **一个模型服务多种机器人，而不是每种机器人都需要独立采集**

**对比表**：

| 维度 | 无动作生成模型 | 有动作生成模型 |
|------|------------|------------|
| 数据成本 | ~$1000/小时 MoCap | ~0（GPU 推理成本可忽略） |
| 动作多样性 | 受限于采集场景 | 文本条件控制，理论上无限 |
| 长尾覆盖 | 极差 | 好（训练数据已含长尾） |
| 新机器人适配 | 需重新采集 | GMR retarget 即可 |
| 语义控制 | 无 | 文本/条件驱动 |
| 实时增广 | 不可能 | 在线生成 |

---

### 问题 4：具身智能的训练方式对于动作生成模型的训练有什么可借鉴的地方？为什么之前的动作生成模型没有普遍使用？

#### 4.1 可借鉴的训练方式

| 具身智能技术 | 可借鉴之处 | 对应方案 | 代表工作 |
|------------|-----------|---------|---------|
| **物理仿真 reward** | 用物理可行性作为奖励信号微调生成模型 | RLPF 式 RL 微调 | RLPF, RobotMDM |
| **Domain Randomization** | 仿真中随机物理参数训练 → 模型学会适应多种物理条件 | SimDiff 式条件化 | SimDiff |
| **Teacher-Student 蒸馏** | 特权信息(物理仿真)→蒸馏到只接受运动学输入的学生模型 | VIRAL 式蒸馏 | VIRAL, ExBody2 |
| **Curriculum Learning** | 从简单到复杂逐步训练 | 可应用于物理难度递增 | MotionLab |
| **迭代数据增强** | 物理修正后的运动回流训练集 | PARC 数据飞轮 | PARC |
| **可微分仿真** | 端到端梯度通过物理仿真器 | DynaFlow 式端到端训练 | DynaFlow |

#### 4.2 最关键的借鉴：RL 式奖励反馈

```
传统动作生成训练：
  数据 → 模型 → 损失(重建误差) → 更新
  ↑ 纯监督学习，只看"数据分布匹配"

借鉴具身智能后：
  数据 → 模型 → 生成动作 → 物理仿真评估 → 奖励信号 → RL更新
  ↑ 加入"物理可行性"作为优化目标
```

**RLPF (RL from Physical Feedback)** 是最直接的体现：
1. 生成模型生成 K 条运动
2. 物理仿真器（IsaacGym + PHC）评估每条运动的可行性
3. 结合语义对齐分数，用 GRPO 更新生成模型
4. 重复 → 物理质量逐步提升

#### 4.3 为什么之前的动作生成模型没有在机器人领域普遍使用？

| 阻碍因素 | 具体说明 | 现在是否已解决 |
|----------|---------|:---:|
| **表示不兼容** | HumanML3D (263-dim) ≠ Robot joint angles | ✅ GMR (2026) 解决 |
| **物理质量不足** | 浮空/滑动/穿地无法直接执行 | 🟡 部分解决（RLPF/PARC） |
| **无实时性** | 扩散模型推理太慢 | ✅ MotionLCM 实时 |
| **无 retargeting** | SMPL→Robot 缺少通用工具 | ✅ GMR 支持 17+ 机器人 |
| **评估标准不同** | FID vs 物理可行性 | 🟡 PARC/RLPF 引入物理评估 |
| **技术栈割裂** | PyTorch vs IsaacGym | ✅ ProtoMotions 统一 |
| **缺少端到端验证** | 没人证明生成→执行全链路可行 | ✅ VideoMimic, ASAP |

**关键转折点（2025-2026）**：
- GMR (ICRA 2026) 提供了通用 SMPL→Robot retargeting
- ProtoMotions (NVIDIA 2025) 统一了仿真训练基础设施
- PARC/RLPF 建立了生成模型↔物理仿真的闭环
- VideoMimic 验证了完整的视频/运动→机器人部署链路

**现在是使用动作生成模型指导机器人训练的最佳时机。**

---

### 问题 5：之前的方案是否将动作生成模型和具身智能模型的训练统一起来了？如果没有的话为什么不能这样做？为什么我们的方案可以这样做？

#### 5.1 现有方案的"统一"程度

| 工作 | 统一程度 | 局限 |
|------|---------|------|
| **PARC** (SIGGRAPH 2025) | ⭐⭐⭐⭐ 最接近 | 迭代闭环，但生成模型和物理跟踪器是**分开训练**的 |
| **RLPF** (2025) | ⭐⭐⭐ | RL 微调生成模型，但**物理评估器(PHC)需预训练** |
| **DynaFlow** (2025) | ⭐⭐⭐⭐ | 可微分仿真嵌入生成，但**只在四足机器人验证**，未推广到人形 |
| **PhysDiff** (ICCV 2023) | ⭐⭐ | 每步投影，**推理时需要仿真器** |
| **POMP** (CVPR 2025) | ⭐⭐⭐ | 相位流形对齐，但**架构特异性强** |

#### 5.2 为什么之前不能真正统一？

**技术障碍**：
1. **梯度不可传播**：传统物理仿真器（MuJoCo/IsaacGym）的碰撞响应不可微 → 无法端到端训练
2. **表示空间不同**：动作生成模型工作在运动学空间（关节角/位置），机器人控制工作在力矩/PD目标空间
3. **时间尺度不同**：动作生成 ~30fps，RL 控制 ~100-200fps
4. **训练范式不同**：生成模型用监督学习/去噪训练，机器人用 RL 的试错学习
5. **评估标准不同**：FID/R-precision vs tracking success rate/stability

**本质原因**：两个领域的优化目标不同
- 动作生成：最小化 `||x_generated - x_real||`（分布匹配）
- 机器人控制：最大化 `E[R(τ)]`（累积奖励）

#### 5.3 为什么我们的方案可以统一？

**HyMotion 的独特优势**：

1. **Flow Matching 架构 → 天然适配 RLPF/SOAR 微调**
   - HyMotion 使用 Rectified Flow，SOAR（已在 ref_repo）提供了 on-policy rollout 微调方案
   - RLPF 的 GRPO 框架可以适配到 continuous-time 模型
   - 已有 SOAR post-training 代码基础 → 加入物理 reward 的边际成本低

2. **VACE 条件化机制 → 自然支持物理条件注入**
   - HyMotion 的 VACE（mask-based conditioning）可以把物理约束编码为额外条件
   - 类似 SimDiff 的物理参数条件化，但不需要修改架构

3. **201-dim 表示 → 包含 FK 可计算的关节位置**
   - 201-dim = pose(22×6 rot6d) + joint(22×3) + trans(3)
   - 可以直接计算 FK → 获取关节位置 → 评估物理约束（穿地、滑动等）
   - 不需要额外的运动学计算管线

4. **GMR 的出现 → SMPL→Robot 的即插即用桥梁**
   - GMR (ICRA 2026) 支持 SMPLX 输入 → 17+ 机器人输出
   - CPU 实时运行 → 可嵌入 RL 训练循环
   - HyMotion 输出 SMPL → GMR → IsaacGym → PHC/ASAP 跟踪

5. **ProtoMotions 已集成 KIMODO → 替换为 HyMotion 的成本低**
   - NVIDIA ProtoMotions 框架已集成 KIMODO 的 mask-based generation
   - HyMotion M2M 的 VACE conditioning 与 MaskedMimic 同构
   - 可以直接替换生成器组件

**我们的统一方案**：

```
┌──────────────────────────────────────────────────────────────┐
│              HyMotion + Physics Unified Training Loop          │
│                                                              │
│  Phase 1: Forward (生成→执行)                                 │
│  ┌─────────┐   SMPL    ┌─────┐  Robot   ┌──────────────┐    │
│  │HyMotion │──────────→│ GMR │────────→│ ProtoMotions  │    │
│  │  M2M    │           │     │         │ (IsaacGym)    │    │
│  └────▲────┘           └─────┘         │ PHC Tracking  │    │
│       │                                └──────┬───────┘    │
│       │                                       │             │
│  Phase 2: Backward (反馈→更新)                  │             │
│       │                                       ▼             │
│       │                              ┌─────────────────┐    │
│       │                              │ Physics Reward:  │    │
│       │                              │ • Track success  │    │
│       │                              │ • Foot contact   │    │
│       │                              │ • Energy         │    │
│       │                              │ • Stability      │    │
│       │                              └────────┬────────┘    │
│       │                                       │             │
│       │              ┌────────────────────────┘             │
│       │              │                                      │
│       │         ┌────┴────┐                                 │
│       │         │ 选择路径 │                                 │
│       │         └────┬────┘                                 │
│       │              │                                      │
│       ├──────────────┤ 路径1: RLPF (GRPO微调HyMotion)       │
│       │              │ 路径2: PARC (修正数据→重训HyMotion)    │
│       │              │ 路径3: SimDiff (物理Adapter)          │
│       │              │ 路径4: SOAR+PhysReward (已有基础)      │
│       │              │                                      │
│  Phase 3: Deploy (部署)                                      │
│       └──────→ Sim2Real (ASAP Delta Model) → Unitree G1     │
│                                                              │
│  ──────────────────────────────────────────────              │
│  关键：每个组件都有开源实现，且对接HyMotion的接口成本低          │
└──────────────────────────────────────────────────────────────┘
```

**与已有方案的差异对比**：

| 维度 | PARC | RLPF | DynaFlow | 我们的方案 |
|------|------|------|----------|-----------|
| 生成模型 | 通用(可替换) | MDM | Flow Matching | HyMotion M2M (Flow Matching) |
| 物理反馈方式 | 数据增强 | RL微调 | 端到端可微 | **多路径可选（灵活）** |
| Retarget | 内置 | — | 内置 | GMR (17+机器人) |
| 仿真框架 | IsaacGym | IsaacGym | 自定义 | ProtoMotions (多仿真器) |
| 条件控制 | 无 | 文本 | 无 | **VACE多任务（文本/轨迹/编辑/补全）** |
| 现有代码基础 | ✅ | 🟡 | ❌ | **✅ HyMotion + SOAR + ProtoMotions** |
| 人形机器人 | ❌ 数字人 | ❌ | ❌ 四足 | **✅ 人形(SMPL→Unitree)** |

---

## 3. 关键文献与代码索引

### 3.1 已下载到 ref_repo/ 的仓库

| 目录 | 项目 | 关联方向 | 状态 |
|------|------|---------|------|
| `ref_repo/PHC/` | PHC: Perpetual Humanoid Control (ICCV 2023) | A1 运动跟踪 | ✅ 已有 |
| `ref_repo/OmniH2O/` | OmniH2O: Human-to-Humanoid (CoRL 2024) | A1 遥操作 | ✅ 已有 |
| `ref_repo/HumanPlus/` | HumanPlus: Humanoid Imitation (CoRL 2024) | A4 数据扩展 | ✅ 已有 |
| `ref_repo/PARC/` | **PARC: Physics-based Augmentation (SIGGRAPH 2025)** | A4+B4 数据闭环 | ✅ 新增 |
| `ref_repo/GMR/` | **GMR: General Motion Retargeting (ICRA 2026)** | 桥梁工具 | ✅ 新增 |
| `ref_repo/ProtoMotions/` | **ProtoMotions: GPU-Accelerated Sim (NVIDIA 2025)** | 仿真基础设施 | ✅ 新增 |
| `ref_repo/ASAP/` | **ASAP: Aligning Sim and Real Physics (RSS 2025)** | Sim2Real | ✅ 新增 |
| `ref_repo/VideoMimic/` | **VideoMimic: Visual Imitation (CoRL 2025 Best Paper)** | 端到端部署 | ✅ 新增 |
| `ref_repo/UH-1/` | **UH-1: Universal Humanoid (Humanoids 2025 Oral)** | 大规模数据 | ✅ 新增 |
| `ref_repo/SOAR/` | SOAR: On-Policy Rollout Post-Training | B3 微调 | ✅ 已有 |
| `ref_repo/KIMODO/` | KIMODO: Large-Scale MoCap (NVIDIA) | 基线对比 | ✅ 已有 |
| `ref_repo/StableMotion/` | StableMotion: Quality-Aware (SIGGRAPH Asia 2025) | B4 质量提升 | ✅ 已有 |
| `ref_repo/MoGenDiT/` | MoGenDiT: Motion Repair (内部) | B1 后处理 | ✅ 已有 |

### 3.2 建议持续关注（未开源或待开源）

| 项目 | 会议/年份 | 关键贡献 | 状态 |
|------|----------|---------|------|
| RLPF | ICLR 2026 | GRPO 微调 + 物理 reward | 待开源 |
| DynaFlow | arXiv 2025 | 可微分仿真 + Flow Matching | 待开源 |
| DreamControl-v2 | 2025 | Diffusion 直接控制人形机器人 | 未开源 |
| BeyondMimic | 2025 | Guided diffusion + RL tracking | 未开源 |
| RoboGhost | 2025 | Retargeting-free latent control | 未开源 |
| SimDiff | 2025 | 物理参数条件化 adapter | 未开源 |
| POMP | CVPR 2025 | 相位流形对齐 | 未开源 |
| GenMimic | UC Berkeley 2025 | 视频→机器人轨迹 | 部分开源 |
| Humanoid-LLA | ShanghaiTech 2025 | 语言→人形机器人 | 待查 |
| Kinema4D | NTU 2026 | 4D 时空仿真训练 | 待查 |
| TraceGen | 2025 | 3D trace space world modeling | 待查 |

### 3.3 最新发现的重要工作（2025-2026 新增）

| 工作 | 核心贡献 | 与课题的关系 |
|------|---------|------------|
| **DynaFlow** (arXiv 2509.19804) | 可微分物理仿真嵌入 Flow Matching → 保证动力学可行 | **B 方向核心参考**：与 HyMotion 同为 Flow Matching，最直接的物理嵌入方案 |
| **GenMimic** (UC Berkeley) | 人类视频→机器人轨迹，4D 重建 + RL | A 方向完整 pipeline |
| **RoboPerform** (2026) | 音频驱动人形机器人表现力运动 | A3 扩散策略控制 |
| **CLAW** (2025) | Composable 语言→全身运动生成 | A5 语言控制 |
| **FlexMotion** (2025) | 轻量 diffusion + 物理感知控制 | B5 物理约束 |
| **Morph** (2025) | Motion-free 物理优化 | B1 后处理替代 |
| **HERMES** (2025) | 统一 RL 移动双臂操作 | A 方向操作任务 |
| **Humanoid-LLA** (ShanghaiTech 2025) | 自由形式语言→人形运动 | A5 语言控制 |

---

## 4. 统一闭环框架设计

### 4.1 分阶段实施路线图

```
Phase 1 (Week 1-2): Baseline 建立
  HyMotion → GMR → ProtoMotions PHC tracking
  输出：物理可行性 baseline 报告

Phase 2 (Week 3-6): PARC 式数据闭环
  迭代：生成 → 物理修正 → 数据增强 → 重训
  输出：物理质量逐轮提升曲线

Phase 3 (Week 7-10): RLPF/SOAR 物理微调
  SOAR post-training + Physics reward
  输出：生成模型内化物理约束

Phase 4 (Week 11-14): SimDiff 式 Adapter
  冻结 backbone + Physics Adapter
  输出：零推理开销的物理可行生成

Phase 5 (Week 15+): 端到端部署
  ASAP Sim2Real → Unitree G1
  输出：真实机器人执行演示
```

### 4.2 技术路线选择矩阵

| 方案 | 时间成本 | 代码可用性 | 论文贡献度 | 推荐度 |
|------|---------|-----------|-----------|--------|
| Phase 1: GMR baseline | 1-2 周 | ✅ 全部开源 | 低（验证性） | ⭐⭐⭐⭐⭐ |
| Phase 2: PARC 闭环 | 4 周 | ✅ 全部开源 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Phase 3: SOAR+Physics | 4 周 | 🟡 需适配 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Phase 4: Physics Adapter | 4 周 | ❌ 需实现 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Phase 5: Sim2Real | 4+ 周 | 🟡 需硬件 | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## 5. 可行实践方案

### 方案 1（P0，短期 1-2 周）：HyMotion → GMR → 物理仿真验证 Baseline

**目标**：定量评估 HyMotion 生成运动的物理可行性

**步骤**：
1. 配置 GMR SMPL→Unitree G1 retargeting
2. HyMotion 生成 100 条运动序列（walk/run/dance/sit/jump）
3. GMR retarget → ProtoMotions PHC 跟踪
4. 统计物理可行性指标：
   - 跟踪成功率（PHC tracking success rate）
   - 足部滑动距离（foot skating）
   - 地面穿透深度（ground penetration）
   - 能量消耗合理性（energy consumption）
   - 关节力矩超限率（joint torque limit violation）

**产出**：Baseline 报告 — "HyMotion 生成运动的 X% 在物理仿真中可行"

---

### 方案 2（P0，中期 2-6 周）：PARC 式数据闭环（最推荐 ⭐⭐⭐⭐⭐）

**目标**：建立「生成 → 物理修正 → 数据增强 → 重训」迭代闭环

**步骤**：
```
Iteration 0:
  HyMotion (当前模型) → 生成 10K 运动
  → GMR retarget → ProtoMotions/PHC 跟踪 → 物理修正
  → 保留成功+质量达标的运动 → 加入训练集
  → 统计修正前后质量差异

Iteration 1:
  用增强数据 fine-tune HyMotion
  → 重新生成 10K → 跟踪 → 修正 → 增强
  → 对比 Iter 0 vs 1 的物理可行性

Iteration N:
  直到物理可行性收敛
```

**核心依赖**：PARC (框架参考) + ProtoMotions (仿真) + GMR (retarget) + HyMotion (生成)

**预期产出**：
- 物理可行性从 ~30% 逐步提升到 ~80%+
- 无需修改 HyMotion 架构
- 生成的数据同时可用于机器人训练

---

### 方案 3（P1，中期 4-8 周）：SOAR + Physics Reward 微调

**目标**：在 SOAR post-training 框架中加入物理 reward

**步骤**：
```
1. 基于已有 HyMotion-SOAR post-training 代码
2. 扩展 reward 函数：
   R = α * exposure_bias_correction (SOAR)
     + β * physics_feasibility (PHC tracking success)
     + γ * text_alignment (R-precision)
3. GRPO/PPO 采样→评分→更新
4. 与 Phase 2 的数据增强互补
```

**优势**：已有 SOAR 代码基础，边际开发成本低

---

### 方案 4（P1，中期 4-8 周）：SimDiff 式物理 Adapter

**目标**：零推理开销的物理约束注入

**步骤**：
```
1. 搭建 MuJoCo/IsaacGym SMPL 仿真环境
2. Domain-Randomized 仿真数据生成
3. 冻结 HyMotion MMDiT backbone
4. 设计 Physics Adapter（LoRA 或 channel-wise）
5. 训练：物理参数条件 + HyMotion 特征 → 物理修正运动
6. 推理：指定标准物理参数 → 自动物理可行
```

---

### 方案 5（P2，长期 3+ 月）：DynaFlow 式可微分仿真嵌入

**目标**：将可微分物理仿真器嵌入 HyMotion 的 Flow Matching 训练

**创新点**：
- DynaFlow 验证了可微分仿真 + Flow Matching 的可行性（四足机器人）
- 将其推广到 **人形全身运动**（22 关节 SMPL）是全新贡献
- 挑战：人形的接触模式比四足复杂得多

---

### 方案 6（P2，持续）：Text → Robot 端到端部署

**目标**：演示 "文本指令 → HyMotion → 物理跟踪 → Unitree G1 执行"

---

## 6. 参考文献

### 方向 A：动作生成 → 机器人训练

[1] Zhengyi Luo et al. "PHC: Perpetual Humanoid Control." ICCV 2023. https://github.com/ZhengyiLuo/PHC
[2] Tairan He et al. "OmniH2O." CoRL 2024. https://github.com/LeCAR-Lab/human2humanoid
[3] Zipeng Fu et al. "HumanPlus." CoRL 2024. https://github.com/MarkFzp/humanplus
[4] **PARC. SIGGRAPH 2025. https://github.com/mshoe/PARC** ← 数据闭环核心
[5] **ASAP. RSS 2025. https://github.com/LeCAR-Lab/ASAP** ← Sim2Real
[6] **VideoMimic. CoRL 2025 Best Student Paper. https://github.com/hongsukchoi/VideoMimic**
[7] **ExBody2. 2024. https://github.com/jimazeyu/exbody2**
[8] **GMR. ICRA 2026. https://github.com/YanjieZe/GMR** ← 关键桥梁
[9] **UH-1. Humanoids 2025 Oral. https://github.com/sihengz02/UH-1** ← 大规模数据
[10] **ProtoMotions. NVIDIA 2025. https://github.com/NVlabs/ProtoMotions** ← 仿真基础设施
[11] GR00T N1. NVIDIA 2025. (商业产品)
[12] DreamControl-v2. 2025. arXiv:2509.14353
[13] BeyondMimic. 2025. arXiv:2508.08241
[14] MaskedMimic. SIGGRAPH Asia 2024.
[15] SuperPADL. 2024. arXiv:2407.10481
[16] VIRAL. 2025. arXiv:2511.15200
[17] Humanoid-LLA. ShanghaiTech 2025. arXiv:2511.22963
[18] CLAW. 2025. arXiv:2604.11251
[19] RoboPerform. 2026. arXiv:2512.23650
[20] GenMimic. UC Berkeley 2025. arXiv:2512.05094
[21] RoboGhost. 2025. arXiv:2510.14952
[22] Kinema4D. NTU 2026.

### 方向 B：物理仿真 → 提升生成质量

[23] Ye Yuan et al. "PhysDiff." ICCV 2023 Oral. NVIDIA.
[24] SimDiff. 2025. arXiv:2509.20927
[25] **RLPF. Under Review ICLR 2026. arXiv:2506.12769** ← RL微调核心
[26] RobotMDM. SIGGRAPH Asia 2024. Disney Research.
[27] POMP. CVPR 2025.
[28] **DynaFlow. 2025. arXiv:2509.19804** ← 可微分仿真+FM
[29] FlexMotion. 2025. arXiv:2501.16778
[30] Morph. 2025. arXiv:2411.14951v3
[31] Scalable Motion In-Betweening. 2025. arXiv:2504.09413
[32] StableMotion. SIGGRAPH Asia 2025. (已在 ref_repo)
[33] SOAR. NUS/Alibaba 2026. (已在 ref_repo)
[34] PhyInter. 2025. ScienceDirect.

### 综合/Survey

[35] "Grounding Intelligence in Movement." UPenn 2025. arXiv:2507.02771
[36] RoboScape. Tsinghua/Manifold AI 2025. arXiv:2506.23135
[37] SPIRAL. 2026. Reflective Planning Framework.
[38] TraceGen. 2025. arXiv:2511.21690

---

## 附录 A：下载命令

```bash
# 已完成下载 ✅
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo

# PARC (SIGGRAPH 2025) ✅
git clone --depth 1 https://github.com/mshoe/PARC.git PARC

# GMR (ICRA 2026) ✅
git clone --depth 1 https://github.com/YanjieZe/GMR.git GMR

# ProtoMotions (NVIDIA) ✅
git clone --depth 1 https://github.com/NVlabs/ProtoMotions.git ProtoMotions

# ASAP (RSS 2025) ✅
git clone --depth 1 https://github.com/LeCAR-Lab/ASAP.git ASAP

# VideoMimic (CoRL 2025) ✅
git clone --depth 1 https://github.com/hongsukchoi/VideoMimic.git VideoMimic

# UH-1 (Humanoids 2025) ✅
git clone --depth 1 https://github.com/sihengz02/UH-1.git UH-1

# 待下载（需网络或认证修复）
# ExBody2
git clone --depth 1 https://github.com/jimazeyu/exbody2.git ExBody2
```

## 附录 B：ref_repo 当前总览

```
ref_repo/
├── PARC/          ← 🆕 SIGGRAPH 2025, 生成→物理修正→数据增强闭环
├── GMR/           ← 🆕 ICRA 2026, SMPL→17+机器人通用retarget
├── ProtoMotions/  ← 🆕 NVIDIA 2025, GPU加速仿真训练框架
├── ASAP/          ← 🆕 RSS 2025, Sim2Real Delta Model
├── VideoMimic/    ← 🆕 CoRL 2025 Best Paper, 视频→机器人部署
├── UH-1/          ← 🆕 Humanoids 2025, 20M+姿态数据集
├── PHC/           ← ICCV 2023, 物理人形控制
├── OmniH2O/       ← CoRL 2024, 人→机器人遥操作
├── HumanPlus/     ← CoRL 2024, 人形模仿学习
├── KIMODO/        ← NVIDIA, 大规模MoCap基线
├── SOAR/          ← 2026, On-policy后训练
├── StableMotion/  ← SIGGRAPH Asia 2025, 质量通道
├── MoGenDiT/      ← 内部, 运动修复
├── MDM/           ← 基线
├── Momask/        ← CVPR 2024, RVQ+masked T2M
├── MotionStreamer/ ← ICCV 2025, 自回归流
├── MotionLab/     ← ICCV 2025, 统一Gen+Edit
├── MotionLCM/     ← 实时生成
├── CondMDI/       ← 条件扩散补帧
├── OmniControl/   ← 任意关节控制
├── BrushNet/      ← 图像修复参考
├── LODGE/         ← 长序列舞蹈
├── TeSMo/         ← 场景感知运动
├── UnderPressure/ ← 足部接触力
├── UMO/           ← 统一运动操作
├── WanVACE/       ← Wan2.1+VACE
├── Mamoda2.5/     ← 外部工具
└── HY-SOAR/       ← 内部SOAR适配
```
