# UMO — 参考工作分析

## 基本信息

- **论文**：UMO: Unified In-Context Learning Unlocks Motion Foundation Model Priors
- **作者**：Xiaoyan Cong*, Zekun Li* 等（Brown University / MIT / Meta Reality Lab / MPI / HKU）
- **时间**：2026-3-16（arXiv:2603.15975v1）
- **代码**：论文中提到"Code and models will be publicly available"，**当前尚未开源**
- **主页**：https://oliver-cong02.github.io/UMO.github.io/

---

## 论文核心内容

### 问题定位

现有运动生成方法大多为单任务专用架构，不能共享知识、跨任务泛化。大型 Text-to-Motion（T2M）基础模型（如 HY-Motion）学到了强大的生成先验，但只支持 T2M，如何高效把这个先验扩展到更多下游任务是核心问题。UMO 的答案是：**用极少量额外参数的轻量 adapter（temporal fusion），通过统一的 in-context 表述，将 T2M 先验解锁到多样下游任务**。

### 核心 Insight

任何 motion 任务中，每一帧相对于 source motion context 都恰好落在三种互斥关系之一：
1. **Preserve（保留）**：这帧直接保留，不做修改
2. **Generate（生成）**：这帧从头生成，无 source reference
3. **Edit（编辑）**：这帧基于 source motion 进行修改

这三种关系完备且互斥，任何任务的每一帧 intent 都能归入其中之一。

### 主要创新点

#### 1. Frame-Level Meta-Operation Embeddings（帧级元操作嵌入）

引入三个可学习的嵌入向量 `[preserve] (P)`, `[generate] (G)`, `[edit] (E)`，每个维度 = motion_dim (R^201)。

每帧分配一个 meta-operation embedding `τ_i ∈ {P, G, E}`，与 source motion `s_i` 相加：
```
s̃_i = s_i + Emb(τ_i)
```
- Preserve/Edit: `s_i = m_i`（原始 motion）
- Generate: `s_i = 0`（无 source）

#### 2. Lightweight Temporal Fusion（轻量时间融合）

四种 conditioning 架构对比选型（以 Keyframe Infilling 为测试任务）：

| 架构 | 额外参数 | 额外 FLOPs | 额外延迟 | [P]-MPJPE | FID |
|------|----------|-----------|---------|-----------|-----|
| Temporal Fusion | **0.207M** | **0.140G** | **0.01s** | **0.95** | **0.476** |
| AdaLN | 4.4M | 1.66G | 0.02s | 11.1（最差） | 8.86 |
| Sequential Concat | 0.207M | 198.6G | 0.89s | 2.04 | 11.77 |
| ControlNet | 234M | 85.12G | 0.49s | 5.19 | 6.52 |

**Temporal Fusion 获胜**：在 backbone 的输入嵌入层做 element-wise 加法：
```
x'_t = E_in(x_t) + E_ctx(s̃)
```
- `E_ctx`：MLP encoder，初始化为 pretrained input encoder `E_in` 的复制（继承预训练表示能力）
- 只在 input level 融合，0.207M 参数，~0 延迟
- AdaLN 压缩全序列为单向量，丢失帧级粒度，[P]-MPJPE 最差（无法 preserve 约束帧）

#### 3. Unified Language Conditioning（统一语言 conditioning）

所有任务条件统一用文本编码，无需 task-specific 模块：

| 类型 | 模板 | 示例 |
|------|------|------|
| Motion description | `<motion description>` | "A man kicks something with his left leg." |
| Editing instruction | `<editing instruction>` | "Speed up your motion." |
| Parameterized trajectory | `{type:<curve_type>, params:{...}}` | 含起止点、控制点的参数化曲线 |
| Spatial constraint | "A person walks from (x1,y1) to (x2,y2). Avoiding N obstacles at (x,y,r)..." | |

几何约束（轨迹跟踪、避障）直接 serialize 为结构化文本，利用 LLM 已有的数学/坐标预训练能力。

#### 4. 任务统一表述（Task Instantiation via Composition）

所有任务由 `(s_i, τ_i)` 配置 + language condition 完全定义：

| 任务 | source s | meta-op τ |
|------|---------|-----------|
| Text-to-Motion / Trajectory / Obstacle | 0 | G（全部帧） |
| Instruction Editing / Reaction / Stylization | m_i | E（全部帧） |
| Motion Prediction（续写） | `[m_1…m_k, *, …, *]` | `[P, …P, *, …*]` |
| Motion Backcasting（反推前缀） | `[*, …, *, m_{T-k+1}…m_T]` | `[*, …, *, P, …P]` |
| In-betweening（填中间） | `[m, …P, *, …, *, m, …P]` | `[P…, G…, P…]` |
| Keyframe Infilling（稀疏关键帧） | 关键帧 m_k | 关键帧 P，其余 G |

### 网络架构

- **Backbone**：HY-Motion-Lite（460M），MMDiT 架构，与我方 HunyuanMotion 完全一致
- **文本编码**：Qwen3-8B（LLM encoder）+ CLIP encoder（与 HY-Motion 相同）
- **额外参数**：仅 0.207M（E_ctx MLP + 3 个 meta-operation embeddings）
- **无架构修改**：backbone 的所有 MMDiT blocks、attention 机制均不改变

### 动作表示

与 HY-Motion 完全一致：
- **201 dims per frame**：global root translation (3D) + root orientation (6D) + 21 local joint rotations (21×6D) + 22 local joint positions (22×3D)
- `3 + 6 + 126 + 66 = 201`
- **SMPL 骨骼**，30 fps，zero-mean unit-variance 归一化

### 训练

- **Flow Matching**（rectified flow）：`x_t = (1-t)x_0 + tx_1`，`v_t = x_1 - x_0`
- **数据集**：HumanML3D + MotionFix + Inter-X + InterHuman + 合成几何约束数据（各 2000 序列）
- **硬件**：4 NVIDIA B200 GPUs，batch=256，lr=5e-5
- **Steps**：统一模型 100k steps；单任务专家模型 6k steps
- **推理**：50-step Euler ODE，CFG scale=2.0

### 支持任务与评估结果

| 任务 | vs. Previous SOTA | 关键数字 |
|------|------------------|---------|
| Text-to-Motion (HumanML3D) | **SOTA FID** | FID=9.46（vs MotionStreamer 11.79） |
| Temporal Inpainting (4 subtasks) | 全面超越 CondMDI, MotionLab | In-betweening MPJPE=8.55 |
| Instruction-based Editing (MotionFix) | **大幅超越** | R@3(batch)=100.0%（vs PartMotionEdit 90.21%） |
| Trajectory Following | 速度比 OmniControl 快 90× | Traj.Err=18.78cm |
| Obstacle Avoidance | 成功率=95% | 比优化方法快 200× |
| Reaction Generation | **SOTA FID** | FID=2.055（vs InterMask 2.99） |

---

## 与我们自己工作的对比

### HyMotion M2M (我方) vs UMO

| 维度 | UMO | HyMotion M2M (我方) |
|------|-----|---------------------|
| **Backbone** | HY-Motion-Lite (460M) MMDiT | HunyuanMotion MMDiT (0.46B/1.5B) — **实质上相同** |
| **动作表示** | 201-dim (global transl 3D + root orient 6D + 21 local rot 126D + 22 local pos 66D)，30fps | 138-dim (abs_rel transl 6D + 22 joint rot_6d 132D)，30fps，**无 local joint positions** |
| **Conditioning 机制** | Frame-level [P]/[G]/[E] meta-op embeddings + **temporal fusion（element-wise add to input embedding）** | **VACE**：`[x_t; inactive; reactive; src_mask]` 拼接，src_mask 作为额外输入通道 |
| **额外参数量** | **0.207M**（极轻量，E_ctx 初始化为预训练权重复制） | VACE conditioning 整套（input_encoder shape 扩大：201→552 dims input） |
| **条件粒度** | **帧级（whole-body per frame）**，无 per-joint 控制 | **逐帧逐维度（T×138）**，支持 per-joint 级别 mask |
| **任务建模** | 3 meta-operations 组合描述任意任务，[edit] 需要 source motion；新任务只需定义新 `(s,τ)` 模板 | 统一二值 mask（M1-M6 策略），mask=0 已知，mask=1 生成；任务建模更简单 |
| **Source motion 角色** | [P]/[E] 帧必须提供 source motion，[G] 帧 s_i=0 | src_motion 可以全零（退化为 unconditional generation） |
| **编辑任务** | ✅ Instruction-based editing（[edit] + editing instruction text）；整帧级别编辑 | ❌ 当前未专门实现 instruction-based editing |
| **几何约束** | ✅ 轨迹跟踪 + 避障，通过结构化文本 prompt，零推理 overhead | ❌ 不支持；但 translation mask=0 可以做 path conditioning（待实现） |
| **多人/反应生成** | ✅ Dual-identity reaction generation（两人）| ❌ 不支持 |
| **Part-level 控制** | ❌ 论文明确指出为 limitation（整帧操作，无关节级别区分） | ✅ T×138 mask 支持任意关节子集 |
| **文本编码** | Qwen3-8B + CLIP（与 HY-Motion 相同） | Qwen3-8B + CLIP-L（相同） |
| **生成范式** | Flow Matching（rectified flow，预测 velocity） | Flow Matching（velocity 或 x1/JiT） |
| **代码开源** | 承诺开源，当前未发布 | 否（内部） |

### 核心设计理念差异

#### 1. In-Context 表述 vs Binary Mask

**UMO**：三种语义操作 [P]/[G]/[E] 显式区分"保留/生成/编辑"三种 intent，source motion 是 context 的一部分。特别是 [edit] 操作：source motion 是 condition，模型生成的是"在 source 基础上按指令修改的结果"，而非"从 mask 位置从头生成"。

**我方 M2M**：统一二值 mask（mask=0 已知，mask=1 生成）。这覆盖了 UMO 的 [P]（mask=0 对应已知帧）和 [G]（mask=1 对应需要生成的帧），但**没有显式的 [edit] 概念**——如果 src_motion 和 tgt_motion 不同（即编辑场景），当前 M2M 的 M4/M5 等策略并没有捕捉这种"soft editing"语义。

#### 2. Temporal Fusion vs VACE

**UMO Temporal Fusion**：
```python
x'_t = E_in(x_t) + E_ctx(s̃)   # element-wise add to input embedding
```
极简，仅在 input level 叠加，backbone 所有 blocks 不变。代价是 E_ctx 和 backbone 的所有 cross-attention 机制没有直接交互，依赖 backbone 自身 self-attention 来消化 conditioning 信息。

**我方 VACE**：
```python
x_input = concat([x_t, inactive, reactive, src_mask], dim=-1)  # channel-wise concat
```
src_motion 以 channel-wise concat 方式注入，input encoder 的 weight 需要适配更大 input dim（201→552），并且 inactive/reactive 分离允许模型区分"我看到的是什么"和"我需要生成什么"。粒度更细（逐 dim），但 input encoder 参数变多，初始化时 shape mismatch。

#### 3. Part-Level Control 的缺失

UMO 的论文第 5 节（Limitations）明确指出：**三种帧级元操作只能做 whole-body 级别的控制，不支持 part-level（关节级别）控制**，并提到这是未来工作方向。这正是我方 M2M 的核心优势之一：T×138 的逐维度 mask 允许"第 t 帧只编辑右手臂关节，其余关节保持不变"。

#### 4. 几何约束的 Language-Only 编码

UMO 将轨迹/障碍物等几何约束序列化为结构化文本，用同一 LLM 编码，无需专用 spatial encoder。这很优雅且可扩展，但精度受限于 LLM 对坐标的理解（Traj.Err=18.78cm vs 优化方法的 2.93cm）。我方目前不支持几何约束，但 M2M 的 translation mask 机制天然可以做 path conditioning（将 translation 维 mask=0，提供路径约束）——但还没有实现。

#### 5. 参数效率极限

UMO 只加 0.207M 参数就覆盖了 6+ 种任务，展示了预训练 T2M 先验的强大迁移性。我方 M2M 改动更大（input encoder 重新初始化，VACE 整套），能换来更细粒度的控制（per-joint mask），两者是不同的 tradeoff 选择。

### 可借鉴的点

1. **[Edit] 操作概念**：当前 M2M 只有 preserve/generate 的二值 mask，缺少 [edit] 语义——即"我提供了 source motion，但我想要的不是原样保留（mask=0），而是在它基础上按文本指令修改"。这在 instruction-based editing、style transfer 等场景非常有用，可以考虑扩展 mask 的语义（如引入 3-class label 而非 2-class binary mask）。

2. **Temporal Fusion 作为轻量 adapter**：对于希望快速支持新任务（如几何约束、multi-person）而不大改架构的场景，temporal fusion（element-wise add to input）是极简高效的选择。

3. **几何约束 via 结构化文本**：将轨迹/障碍物等序列化为 JSON 格式的文本，用 LLM encoder 编码，不需要专用模块。如果要支持 path-conditioned generation，这是最低 engineering cost 的实现路径（前提是 LLM 能理解坐标格式，Qwen3-8B 应该可以）。

4. **统一多任务训练的 FID 提升**：UMO 的结果显示 Unified 模型在几乎所有任务上优于 Expert 模型（Tab. 4-8），说明多任务 joint training 有 synergy 效应。M2M 已经在做多任务（M1-M6 策略混合训练），应当继续保持这个方向。

5. **E_ctx 初始化为 E_in 的复制**：in-context motion encoder 初始化为预训练 input encoder 的权重复制，而不是随机初始化。我方当前 input_encoder 在 M2M 初始化时是随机初始化（shape mismatch）。UMO 的 E_ctx 能继承预训练的 motion representation 能力，加速收敛。
