# HyMotion M2M 下一代方案：条件解耦编排流匹配 (CDO-FM)

**Condition-Decoupled Orchestration Flow Matching**

文档版本: 1.0 | 日期: 2026-05-11 | 状态: 方案设计

---

## 目录

1. [问题诊断](#1-问题诊断)
2. [设计目标与约束](#2-设计目标与约束)
3. [核心方案概述](#3-核心方案概述)
4. [架构设计](#4-架构设计)
5. [训练策略](#5-训练策略)
6. [质量增强：接地感知生成](#6-质量增强接地感知生成)
7. [实验计划](#7-实验计划)
8. [与前沿方法对比及新颖性分析](#8-与前沿方法对比及新颖性分析)
9. [顶会论文定位](#9-顶会论文定位)
10. [风险与备选方案](#10-风险与备选方案)
11. [实施路线图](#11-实施路线图)

---

## 1. 问题诊断

### 1.1 核心症状

当前 HyMotion M2M 带 caption 的模型在 T2M 任务上几乎无法理解 text 输入，输出动作质量甚至比 unconditional 更差。

### 1.2 根因分析

经过对完整代码库的深度审计，发现问题由**实现 bug** 和**架构局限性**共同导致：

#### 1.2.1 实现 Bug（已确认 / 高度疑似）

| # | Bug | 严重程度 | 状态 | 代码位置 |
|---|-----|---------|------|---------|
| B1 | **Bundle-level Parameter 不训练/不保存/不同步** | P0-Critical | 已修复 2026-03-27 | `hftrainer/models/base_bundle.py` |
| B2 | **null_vtxt_feat 每次加载为全零** | P0-Critical | 已修 (B1的后果) | `hymotion_m2m/bundle.py:115` |
| B3 | **VACE reactive 通道泄露 target 信息** | P0-Critical | 已修复 2026-03-25 | `hymotion_m2m_trainer.py` |
| B4 | **M2M base bundle `cond_mask_prob=1.0`** | **P0-疑似未修** | **待确认** | `hymotion_m2m/bundle.py:74` |
| B5 | **Text token长度分布 OOD** | P1 | 已修复 2026-04-20 | text encoder padding |
| B6 | **Null embedding 分布不匹配** | P1 | 已修复 2026-04-21 | null 统计量对齐 |

**关键发现 — B4**: base M2M bundle 的 `cond_mask_prob` 设为 **1.0**（100% 文本 dropout），意味着训练时模型**从未见过真实文本**。这是 text conditioning 完全失效的直接原因。T2M bundle 使用 `cond_mask_prob=0.1`（10% dropout），CRFM v3 trainer 可能覆盖此值，但 base M2M 训练路径下文本被完全屏蔽。

#### 1.2.2 架构局限性

| # | 局限 | 影响 |
|---|------|------|
| A1 | **Text 与 Motion condition 信号竞争** | Motion condition 信息密度远高于 text，模型注意力自然偏向 motion，text 信号被淹没 |
| A2 | **VACE 固定 4 通道编码缺乏条件类型感知** | inactive/reactive/mask 不区分 keyframe、trajectory、joint constraint 等条件类型 |
| A3 | **CFG 对 text 的控制力不足** | 当 motion condition 很强（dense mask）时，有无 text 的输出差异极小，CFG 失效 |
| A4 | **训练数据质量** | 15.5% 低质量数据（滑步/抖动/关节跳变）拉低上限 |
| A5 | **无显式物理约束** | 生成结果不保证脚-地面接触的物理合理性 |

### 1.3 核心矛盾

> 当前架构将 text 和 motion 条件混在同一个 conditioning pipeline 中，但两者的**信息密度天然不对称**：一个 dense keyframe condition 包含的约束信息量远大于一句 caption。模型在 multi-task 训练中自然学到"有 motion condition 时忽略 text"的捷径策略（shortcut learning）。

这不是 TAP/TAL 等梯度 trick 能根本解决的问题，需要**架构层面的条件解耦**。

---

## 2. 设计目标与约束

### 2.1 功能需求

| # | 需求 | 优先级 |
|---|------|--------|
| R1 | 支持任意 condition-target pattern（稀疏/稠密、position/rotation、任意关键帧数量/位置） | P0 |
| R2 | 同时理解 motion condition 与 text condition，不互相干扰 | P0 |
| R3 | 兼容 T2M / 带条件 T2M / 语义编辑 / 动作修复 等全部任务 | P0 |
| R4 | 解决滑步等生成质量问题 | P1 |
| R5 | 方案具备顶会论文新颖性 | P1 |
| R6 | 实现可行，不引入过大额外计算量（< 1.5x 当前训练成本） | P2 |

### 2.2 非功能约束

- 沿用 MMDiT (dual-stream → single-stream) 架构骨架，最大化复用已有代码和预训练权重
- 保持 135-dim 运动表征和 SMPL-22 骨骼
- 训练规模 ≤ 32 GPU × V100 (80GB)
- 与现有 eval pipeline (E1-E15) 完全兼容

---

## 3. 核心方案概述

### 3.1 方案名称

**条件解耦编排流匹配 (Condition-Decoupled Orchestration Flow Matching, CDO-FM)**

### 3.2 一句话描述

> 将扩散过程的条件注入**分层解耦**为语义层（text → global motion intent）、结构层（motion condition → spatial-temporal constraints）和质量层（physics → contact-aware refinement），通过**密度感知路由机制**动态编排各层的相对贡献，使单一模型自适应地在纯文本生成与强条件补全之间平滑插值。

### 3.3 核心创新点

1. **Density-Modulated Dual-Stream Attention (DM-DSA)**: 基于条件密度的双流注意力门控，text stream 和 motion condition stream 的贡献由可学习的密度调制器动态平衡

2. **Typed Condition Canvas (TCC)**: 替代固定的 inactive/reactive/mask 三通道，引入条件类型嵌入（type embedding），让模型区分 keyframe / trajectory / joint constraint / boundary 等不同语义的空间条件

3. **Contact-Conditioned Flow Matching (CCFM)**: 将地面接触状态作为显式条件通道，训练模型在生成过程中感知并尊重脚-地面物理约束

4. **Progressive Condition Exposure (PCE)**: 三阶段渐进式训练策略，从纯 T2M 逐步引入 motion condition，防止 text 信号在 multi-task 训练中被淹没

---

## 4. 架构设计

### 4.1 整体架构

```
                         ┌─────────────────────────────────────┐
                         │          CDO-FM Architecture         │
                         └─────────────────────────────────────┘

  Text Input                                          Motion Condition Input
  "a person walks                                     [sparse keyframes + mask]
   forward and waves"                                 [trajectory + joint constraints]
       │                                                      │
       ▼                                                      ▼
  ┌─────────────┐                                    ┌──────────────────┐
  │ Text Encoder │ (frozen Qwen3 + CLIP-L)           │ Typed Condition  │
  │             │                                    │ Canvas Encoder   │
  └──────┬──────┘                                    │   (TCC-Enc)      │
         │                                           └────────┬─────────┘
   ctxt (B,S,4096)                                      tcc (B,L,D_cond)
   vtxt (B,1,768)                                    cond_type (B,L,N_types)
         │                                           density (B,L,J)
         │                                                  │
         │              ┌──────────────┐                    │
         │              │   Density    │◄───────────────────┘
         │              │  Modulator   │
         │              │   (DM)       │
         │              └──────┬───────┘
         │                     │ gate_text (B,1), gate_motion (B,L,J)
         │                     │
         ▼                     ▼
  ╔══════════════════════════════════════════════════════╗
  ║            MMDiT Backbone (reused)                   ║
  ║                                                      ║
  ║  ┌──────────────────────────────────────────────┐   ║
  ║  │  Dual-Stream Blocks (×N_double)              │   ║
  ║  │                                              │   ║
  ║  │  Motion Stream:                              │   ║
  ║  │    x_t + TCC features → self-attn → FFN     │   ║
  ║  │                                              │   ║
  ║  │  Text Stream:                                │   ║
  ║  │    ctxt → self-attn → FFN                    │   ║
  ║  │                                              │   ║
  ║  │  Cross-stream: gated joint attention          │   ║
  ║  │    gate_text · text_attn + gate_motion ·     │   ║
  ║  │    motion_attn                               │   ║
  ║  └──────────────────────────────────────────────┘   ║
  ║                                                      ║
  ║  ┌──────────────────────────────────────────────┐   ║
  ║  │  Single-Stream Blocks (×N_single)            │   ║
  ║  │    [motion_tokens; text_tokens] → attn → FFN │   ║
  ║  └──────────────────────────────────────────────┘   ║
  ║                                                      ║
  ╚══════════════════════════════════════════════════════╝
         │
         ▼
  ┌─────────────────┐     ┌──────────────────┐
  │  Flow Velocity  │     │  Contact Head    │
  │  Prediction     │     │  (auxiliary)     │
  │  v_θ (B,L,135) │     │  c_θ (B,L,2)    │
  └─────────────────┘     └──────────────────┘
         │                        │
         ▼                        ▼
  ┌─────────────┐          ┌────────────┐
  │  ODE Solve  │          │  Contact   │
  │  + Replace  │◄─────────│  Guided    │
  │  Guidance   │          │  IK Refine │
  └─────────────┘          └────────────┘
         │
         ▼
    Output Motion (B, L, 135)
```

### 4.2 模块详细设计

#### 4.2.1 Typed Condition Canvas (TCC)

**动机**: 当前 VACE 的 `[inactive, reactive, mask]` 是一种无类型的编码——模型只知道"这里有值/没值"，不知道"这是一个关键帧约束还是一个轨迹约束"。不同类型的条件应该有不同的影响方式。

**设计**:

```python
class TypedConditionCanvas(nn.Module):
    """
    将各种类型的 motion condition 编码为统一的条件画布。
    每种条件类型有独立的 type embedding，让模型区分条件语义。
    """
    # 条件类型枚举
    COND_TYPES = {
        'none':        0,   # 无条件（待生成区域）
        'keyframe':    1,   # 关键帧（完整 pose）
        'trajectory':  2,   # 轨迹约束（仅 translation）
        'joint_pos':   3,   # 关节位置约束
        'joint_rot':   4,   # 关节旋转约束
        'boundary':    5,   # 边界帧（用于过渡/衔接）
        'edit_source': 6,   # 编辑任务的源动作（低质量/待修改）
        'contact':     7,   # 接触状态约束
    }

    def __init__(self, motion_dim=135, num_types=8, type_embed_dim=64):
        super().__init__()
        # 可学习的条件类型嵌入
        self.type_embedding = nn.Embedding(num_types, type_embed_dim)

        # 条件值编码器（替代原 VACE 的 inactive/reactive 拼接）
        self.value_encoder = nn.Linear(motion_dim, motion_dim)

        # 类型-值融合
        self.fusion = nn.Sequential(
            nn.Linear(motion_dim + type_embed_dim, motion_dim),
            nn.SiLU(),
            nn.Linear(motion_dim, motion_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)  # zero-init for residual safety

    def forward(self, condition_motion, condition_mask, condition_type_ids):
        """
        Args:
            condition_motion: (B, L, 135) — 条件区域的运动值（非条件区域为 0）
            condition_mask:   (B, L, 135) — 二值 mask，1=待生成，0=已知条件
            condition_type_ids: (B, L, J=23) — 每个 (frame, joint-group) 的类型 ID
        Returns:
            tcc_features: (B, L, 135) — 编码后的条件画布特征
        """
        # 编码条件值
        cond_val = self.value_encoder(condition_motion)  # (B, L, 135)

        # 获取类型嵌入并扩展到 per-dim
        type_emb = self.type_embedding(condition_type_ids)  # (B, L, 23, type_embed_dim)
        type_emb = expand_joint_group_to_dim(type_emb, D=135)  # (B, L, 135, type_embed_dim)

        # 融合值 + 类型
        fused = self.fusion(torch.cat([
            cond_val.unsqueeze(-1).expand_as(type_emb[..., :1]),  # value broadcast
            type_emb
        ], dim=-1))  # (B, L, 135)

        # mask=0 的区域有条件特征，mask=1 的区域为 learned null
        tcc_features = fused * (1 - condition_mask) + self.null_cond * condition_mask

        return tcc_features
```

**关键改进 vs 当前 VACE**:
- **类型感知**: 模型知道每个位置提供的是什么类型的约束
- **统一编码**: 不再区分 inactive/reactive，统一为 value + type
- **可扩展**: 新增条件类型只需添加 type ID，无需改架构
- **编辑支持**: `edit_source` type 明确标记需要修复的区域

#### 4.2.2 Density-Modulated Dual-Stream Attention (DM-DSA)

**动机**: Text 和 motion condition 的信息密度天然不对称。当 motion condition 很丰富（如 dense keyframes）时，text 的边际信息量很低，但当 motion condition 很稀疏时，text 是唯一的语义指导。需要一个**自适应机制**来平衡两者。

**设计**:

```python
class DensityModulator(nn.Module):
    """
    基于条件密度计算 text/motion condition 的贡献权重。
    密度高 → 增强 motion stream、适当保留 text stream
    密度低 → 增强 text stream

    关键：不是简单的 hard routing，而是 soft gating，
    确保即使在高密度条件下 text 仍有一定影响（语义编辑需要）。
    """
    def __init__(self, dim=1024, min_text_gate=0.15):
        super().__init__()
        self.min_text_gate = min_text_gate  # text 贡献的下限，防止完全忽略

        # 多粒度密度特征
        self.frame_density_enc = SinusoidalEncoding(dim=dim//2)  # per-frame 密度
        self.global_density_enc = SinusoidalEncoding(dim=dim//2)  # 全局密度

        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 2),  # [gate_text, gate_motion]
        )
        # 初始化为均衡门控
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, torch.tensor([0.5, 0.5]).log())

    def forward(self, condition_mask, timestep_emb):
        """
        Args:
            condition_mask: (B, L, 135) — 1=生成，0=已知
            timestep_emb:   (B, D) — 时间步嵌入（扩散早期需要更多语义指导）
        Returns:
            gate_text:   (B, 1) — text stream 权重
            gate_motion: (B, 1) — motion condition stream 权重
        """
        # 计算多尺度密度
        frame_density = condition_mask.mean(dim=-1)          # (B, L)  per-frame
        global_density = condition_mask.mean(dim=(-1, -2))   # (B,)    global

        # 编码密度
        frame_feat = self.frame_density_enc(frame_density)   # (B, L, D/2)
        global_feat = self.global_density_enc(global_density) # (B, D/2)

        # 结合时间步（扩散早期=高噪声，更需要语义指导）
        density_feat = torch.cat([global_feat, timestep_emb[:, :self.dim//2]], dim=-1)

        # 计算门控
        gates = self.gate_net(density_feat).softmax(dim=-1)  # (B, 2)
        gate_text = gates[:, 0:1].clamp(min=self.min_text_gate)
        gate_motion = gates[:, 1:2]

        # 归一化
        total = gate_text + gate_motion
        gate_text = gate_text / total
        gate_motion = gate_motion / total

        return gate_text, gate_motion
```

**在 DiT Block 中的应用**:

```python
class DualStreamDiTBlock(nn.Module):
    """修改后的 dual-stream block，集成密度调制"""

    def forward(self, x_motion, x_text, gate_text, gate_motion, ...):
        # 各自的 self-attention
        motion_out = self.motion_self_attn(x_motion)
        text_out = self.text_self_attn(x_text)

        # 关键改动：密度调制的 cross-stream attention
        # 原始: joint_attn = concat_attn([motion, text])
        # 新: gated contribution

        # Motion 对 Text 的 cross-attention（获取语义指导）
        m2t_attn = self.cross_attn_m2t(q=motion_out, kv=text_out)
        # Text 对 Motion 的 cross-attention（获取空间上下文）
        t2m_attn = self.cross_attn_t2m(q=text_out, kv=motion_out)

        # 密度调制融合
        motion_out = motion_out + gate_text * m2t_attn + gate_motion * self.motion_cond_proj(tcc_features)
        text_out = text_out + t2m_attn

        # FFN
        motion_out = motion_out + self.motion_ffn(motion_out)
        text_out = text_out + self.text_ffn(text_out)

        return motion_out, text_out
```

**关键设计决策**:
- `min_text_gate=0.15`: 即使在最密集的条件下（如修复任务，只有少数帧待生成），text 至少保留 15% 的贡献。这确保了"语义编辑"类任务（如"让这个走路变得更高兴"）在高条件密度下仍然有效
- 门控权重与 timestep 耦合：扩散过程的早期（高噪声）更需要 text 的全局语义指导，后期（低噪声）更依赖 motion condition 的精确空间约束
- Soft gating 而非 hard routing：确保梯度可以流经两个 stream

#### 4.2.3 条件感知 AdaLN (Condition-Aware AdaLN)

**动机**: 当前 AdaLN 只接收 timestep embedding + 可选的 CDE。我们将其升级为同时感知条件密度和条件类型分布。

```python
class ConditionAwareAdaLN(nn.Module):
    """
    AdaLN 调制，同时考虑：
    1. Timestep t
    2. 条件密度 ρ (全局 + per-frame)
    3. 条件类型分布 τ (每种类型的覆盖比例)
    4. Text 存在标记 (有/无 text condition)
    """
    def __init__(self, dim, num_cond_types=8):
        super().__init__()
        # 条件类型分布编码
        self.type_dist_proj = nn.Linear(num_cond_types, dim)
        # text 存在标记
        self.text_flag_proj = nn.Linear(1, dim)
        # 融合 → AdaLN 参数
        self.adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim * 3, dim * 6),  # scale, shift for 3 sub-layers
        )

    def forward(self, timestep_emb, density_emb, type_distribution, has_text):
        """
        type_distribution: (B, num_types) — 每种条件类型占总条件的比例
        has_text: (B, 1) — 是否有 text condition
        """
        combined = timestep_emb + density_emb + \
                   self.type_dist_proj(type_distribution) + \
                   self.text_flag_proj(has_text.float())
        return self.adaln_proj(combined)
```

### 4.3 输入张量格式对比

| 维度 | 当前 VACE | CDO-FM (proposed) | 变化 |
|------|-----------|-------------------|------|
| 噪声状态 | x_t (B,L,135) | x_t (B,L,135) | 不变 |
| 条件值 | inactive (B,L,135) | tcc_value (B,L,135) | 语义相同，编码方式不同 |
| 反应通道 | reactive (B,L,135) | — (合并到 TCC) | **移除独立通道** |
| 掩码 | mask (B,L,135) | mask (B,L,135) | 不变 |
| 类型信息 | 无 | type_emb (B,L,135) | **新增** |
| 接触状态 | 无 | contact (B,L,2) | **新增** |
| **模型输入** | **(B,L,540)** → proj | **(B,L,540+2)** → proj | 接近不变 |

关键: TCC 将 inactive + reactive 统一为一个带类型信息的条件通道，总输入维度几乎不变（+2 用于 contact），不增加主干计算量。

### 4.4 CFG (Classifier-Free Guidance) 改进

当前 CFG 只有一级（drop text），在 motion condition 存在时 text 的 CFG 效果极弱。

**提议：二级解耦 CFG (Decoupled CFG)**

```
v_uncond     = model(x_t, null_text, null_motion_cond)   # 完全无条件
v_motion     = model(x_t, null_text, motion_cond)        # 仅 motion
v_text       = model(x_t, text,      null_motion_cond)   # 仅 text
v_full       = model(x_t, text,      motion_cond)        # 全条件

# 解耦引导
v_guided = v_uncond
         + w_motion * (v_motion - v_uncond)       # motion 条件引导
         + w_text   * (v_full   - v_motion)       # text 的增量引导（在 motion 基础上）
```

**优势**:
- `w_text` 控制的是 text 在 motion condition 基础上的**增量贡献**，而非绝对贡献
- 当 motion condition 很强时，`v_motion` 已经很好，`v_full - v_motion` 的差异主要体现在 text 要求的**风格/情感**变化上
- 用户可以独立调节 `w_motion` 和 `w_text`

**训练适配**: 需要三种 dropout 组合
- (drop_text=True, drop_motion=True): 训练 `v_uncond`
- (drop_text=True, drop_motion=False): 训练 `v_motion`
- (drop_text=False, drop_motion=False): 训练 `v_full`
- (drop_text=False, drop_motion=True): 可选，训练 `v_text`

推荐采样概率: `p(both_drop)=0.05, p(text_drop)=0.10, p(motion_drop)=0.05, p(no_drop)=0.80`

---

## 5. 训练策略

### 5.1 Progressive Condition Exposure (PCE)

这是本方案最关键的训练创新。核心思想：**不要从一开始就让模型面对所有条件类型的组合，而是渐进地引入条件复杂度**。

```
Phase 1: Text Foundation (T2M Warmup)
├─ 目标: 建立稳固的 text → motion 理解
├─ 数据: 100% 纯 T2M (mask = all 1, no motion condition)
├─ 训练: 标准 flow matching + text CFG (drop_prob=0.1)
├─ 时长: ~30% 总训练 steps
├─ 初始化: 从 T2M 1.0-Lite 预训练权重开始
└─ 验证: T2M FID/R-Precision 达到 T2M baseline 水平

Phase 2: Condition Introduction (渐进引入 motion condition)
├─ 目标: 在保持 text 理解的同时学习 motion condition
├─ 数据: 线性增加 motion condition 比例
│   ├─ Step 0:    80% T2M + 20% sparse condition (M6 keyframe only)
│   ├─ Step T/3:  50% T2M + 50% mixed condition (M3+M6)
│   └─ Step 2T/3: 30% T2M + 70% full condition mix (M1-M7)
├─ 训练: flow matching + 解耦 CFG + text awareness loss (TAL)
├─ 关键: TAL 确保 text 信号不被 motion condition 淹没
├─ 时长: ~40% 总训练 steps
└─ 验证: T2M 指标不下降 + M2M 指标稳步上升

Phase 3: Full Multi-Task Mastery
├─ 目标: 全面提升所有任务性能
├─ 数据: 100% 全 mask pattern (v3 Rank-K 采样器)
│   ├─ T2M (M5): 10-15% (维持 text 能力)
│   ├─ Condition mix (M1-M4, M6-M7): 70-80%
│   └─ Edit/Repair: 10-15%
├─ 训练: 全特性 (解耦 CFG + TAL + contact loss + quality filter)
├─ 时长: ~30% 总训练 steps
└─ 验证: 全部 E1-E15 评估任务
```

### 5.2 Anti-Shortcut Training Mechanisms

防止模型学到"有 motion condition 就忽略 text"的捷径：

#### 5.2.1 Text-Awareness Loss v2 (TAL-v2)

升级当前 TAL，增加更强的约束：

```python
def text_awareness_loss_v2(pred_with_text, pred_without_text, pred_with_wrong_text,
                            condition_mask, mask_density):
    """
    三重文本感知损失:
    1. 有 text vs 无 text 的差异应大于阈值 (原 TAL)
    2. 正确 text vs 错误 text 的差异应大于阈值 (新: 对比损失)
    3. 差异量级与 mask_density 负相关 (新: 密度自适应)
    """
    # 原始 TAL: 有 text 应与无 text 不同
    text_effect = (pred_with_text - pred_without_text).norm(dim=-1)
    min_effect = adaptive_threshold(mask_density)  # 密度越高，阈值越低但不为零
    tal_original = F.relu(min_effect - text_effect).mean()

    # 新增: 对比损失 — 正确 text 与错误 text 应产生不同结果
    correct_dist = (pred_with_text - pred_without_text).norm(dim=-1)
    wrong_dist = (pred_with_wrong_text - pred_without_text).norm(dim=-1)
    # 正确 text 的影响应大于错误 text
    tal_contrastive = F.relu(wrong_dist - correct_dist + margin).mean()

    return tal_original + 0.5 * tal_contrastive
```

#### 5.2.2 Text-Motion Condition Dropout 协调

```python
# 训练时的 dropout schedule
# 关键: 确保 (有text, 有motion condition) 的组合占多数
dropout_schedule = {
    'text_only':     0.15,  # drop motion condition, keep text → 强制 text 学习
    'motion_only':   0.10,  # drop text, keep motion condition → 标准 M2M
    'both':          0.05,  # drop both → 无条件 baseline
    'full':          0.70,  # keep both → 目标状态
}
```

### 5.3 数据策略

#### 5.3.1 质量过滤

从 549K 样本切换到 456K 高质量子集（`high_quality.json`），预期 +3-5% 质量提升。

#### 5.3.2 Text Augmentation

当前每条 motion 只有一个 caption。为了增强 text 理解的鲁棒性：
- 同一 motion 的多种描述（不同详细程度、不同重点）
- 通过 LLM 改写现有 caption（已有 `--use-rewritten` 支持）
- 负样本: 随机配对错误 caption 用于 TAL-v2 的对比损失

#### 5.3.3 Condition Type 标注

当前 mask 只有 0/1，不包含类型信息。TCC 需要类型标注：
- **自动生成**: 在 `PrepareM2MCondition` 中根据 mask pattern 生成器自动设置 type ID
- M5 (全 mask) → type=none
- M6 (keyframe) → type=keyframe
- M3 (时间连续) → type=boundary (边界帧)
- M4 (关节连续) → type=joint_rot
- Edit mode → type=edit_source
- 这不需要额外的人工标注

---

## 6. 质量增强：接地感知生成

### 6.1 Contact-Conditioned Flow Matching (CCFM)

**核心思想**: 将脚-地面接触状态作为显式条件，让模型在生成过程中直接"知道"哪些帧的脚应该落地。

#### 6.1.1 Contact Label 提取

从 GT motion 自动提取接触标签（训练时使用）：

```python
def extract_contact_labels(motion, skeleton, vel_thresh=0.01, height_thresh=0.03):
    """
    从 GT 动作中提取双脚接触标签。
    Args:
        motion: (T, 135) — 标准化前的原始动作
        skeleton: SMPL-22 骨骼定义
    Returns:
        contact: (T, 2) — [left_foot, right_foot] 接触状态 (0/1)
    """
    # 通过 FK 计算全局关节位置
    joint_positions = forward_kinematics(motion, skeleton)  # (T, 22, 3)

    # 脚踝关节索引 (SMPL-22: L_Ankle=7, R_Ankle=8)
    foot_pos = joint_positions[:, [7, 8], :]  # (T, 2, 3)
    foot_vel = torch.diff(foot_pos, dim=0, prepend=foot_pos[:1])  # (T, 2, 3)
    foot_speed = foot_vel.norm(dim=-1)  # (T, 2)
    foot_height = foot_pos[:, :, 1]  # (T, 2) — Y 轴为高度

    # 接触判定: 速度低 AND 高度低
    contact = (foot_speed < vel_thresh) & (foot_height < height_thresh)
    return contact.float()
```

#### 6.1.2 Contact 作为条件通道

```python
# 在 TCC 编码中追加 contact 通道
model_input = torch.cat([
    x_t,            # (B, L, 135) — 噪声状态
    tcc_features,   # (B, L, 135) — 类型化条件特征
    mask,           # (B, L, 135) — 生成 mask
    contact_cond,   # (B, L, 2)   — 双脚接触状态（新增）
], dim=-1)  # (B, L, 407) → 输入 projection
```

#### 6.1.3 Contact-Aware Loss

```python
def contact_aware_loss(pred_velocity, pred_motion, gt_contact, skeleton):
    """
    在预测接触的帧上惩罚脚部滑动。
    """
    # 通过 FK 获取脚部位置
    pred_foot_pos = fk_foot(pred_motion, skeleton)  # (B, T, 2, 3)
    pred_foot_vel = torch.diff(pred_foot_pos, dim=1)  # (B, T-1, 2, 3)

    # 接触帧的脚部速度应为零
    contact_mask = gt_contact[:, 1:, :]  # (B, T-1, 2)
    foot_skating_loss = (pred_foot_vel.norm(dim=-1) * contact_mask).mean()

    return foot_skating_loss
```

#### 6.1.4 Contact Head (辅助预测)

在推理时，如果用户未提供 contact 信息，模型需自行预测合理的接触模式：

```python
class ContactPredictionHead(nn.Module):
    """轻量级接触预测头，与主 velocity head 并行"""
    def __init__(self, dim, num_feet=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, num_feet),
        )

    def forward(self, motion_features):
        return self.proj(motion_features).sigmoid()
```

### 6.2 推理时 IK 精修

即使训练时有 contact loss，推理时仍可能有轻微滑步。增加一个轻量后处理：

```python
def ik_foot_lock(motion, predicted_contacts, skeleton, blend_frames=3):
    """
    对预测的接触帧执行 IK 锁脚。
    1. 找到每个接触段的第一帧，记录脚部位置
    2. 在接触段内，固定脚部位置
    3. IK 求解调整膝/踝关节旋转
    4. 在接触段边界做 blend_frames 帧的平滑过渡
    """
    # ... IK 实现
    return refined_motion
```

### 6.3 其他质量增强

#### 6.3.1 Velocity Consistency Loss

惩罚速度的不平滑变化（减少抖动）：

```python
vel_loss = smooth_l1(pred_velocity[:, 1:] - pred_velocity[:, :-1])
acc_loss = smooth_l1(pred_accel[:, 1:] - pred_accel[:, :-1])
```

#### 6.3.2 FK Consistency Loss

确保 rotation 和 position 的一致性：

```python
# 从预测的 rot6d 通过 FK 计算 position
pred_pos_from_rot = forward_kinematics(pred_rot6d, skeleton)
# 与预测的 position 对比
fk_loss = mse(pred_pos_from_rot, pred_joint_pos)
```

---

## 7. 实验计划

### 7.1 消融实验 (Ablation Study)

| 实验 | 配置变化 | 验证目标 | 预计 GPU 时长 |
|------|---------|---------|-------------|
| A0: Baseline Fix | 修复 B4 (cond_mask_prob=0.1) | 确认 bug fix 的效果 | 8×V100, 2天 |
| A1: + TCC | 替换 VACE 为 TCC | 条件类型感知的效果 | 8×V100, 3天 |
| A2: + DM-DSA | 加入密度调制双流注意力 | text-motion 平衡效果 | 8×V100, 3天 |
| A3: + PCE | 三阶段渐进训练 | 防止 text shortcut | 16×V100, 5天 |
| A4: + Decoupled CFG | 二级解耦 CFG | 推理时 text 控制力 | — (推理改动) |
| A5: + CCFM | contact-conditioned + contact loss | 滑步改善 | 16×V100, 5天 |
| A6: Full CDO-FM | 所有组件 | 最终性能 | 32×V100, 7天 |

### 7.2 评估指标

#### Text Conditioning 评估
| 指标 | 说明 | 目标 |
|------|------|------|
| **R-Precision (Top-1/2/3)** | Text-motion 匹配准确率 | > 0.45 / 0.65 / 0.75 |
| **MM-Dist** | Text-motion 特征距离 | < 3.5 |
| **Text Effect Ratio** | ‖pred_w_text - pred_wo_text‖ / ‖pred_wo_text‖ | > 0.15 |
| **FID** | 生成动作的分布质量 | < 0.5 |

#### Motion Condition 评估
| 指标 | 说明 | 目标 |
|------|------|------|
| **Condition MPJPE** | 条件帧处的关节位置误差 | < 1mm |
| **Boundary Smoothness** | 条件/生成边界的加速度跳变 | < 50 mm/s² |
| **Interpolation Quality** | 中间帧的插值准确度 | MPJPE < 20mm |

#### 质量评估
| 指标 | 说明 | 目标 |
|------|------|------|
| **Foot Skating** | 接触帧脚部滑动距离 | < 0.25 |
| **Jitter** | 关节位置高频抖动 | < 600 |
| **Physical Plausibility** | 脚穿地面、悬空等比例 | < 5% |

### 7.3 评估任务覆盖

全部 E1-E15 任务 + 新增:
- **E-Text**: 纯 T2M 质量评估
- **E-TextCond**: Text + sparse condition 联合评估
- **E-Edit**: 语义编辑（"让走路变得更高兴"）评估
- **E-Contact**: 接触质量专项评估

---

## 8. 与前沿方法对比及新颖性分析

### 8.1 方法对比矩阵

| 特性 | 当前 M2M | VACE (Wan2.1) | OmniGen2 | Seedance 2.0 | Step1X-Edit | **CDO-FM (Ours)** |
|------|---------|---------------|----------|--------------|-------------|-------------------|
| 条件类型感知 | ✗ (3通道固定) | ✗ (VCU通用) | ✗ | ✗ | ✓ (MLLM理解) | **✓ (TCC 类型嵌入)** |
| Text-Motion 平衡 | ✗ (竞争) | ✗ | ✓ (双路径) | ✓ (DB-DiT) | ✓ (MLLM路由) | **✓ (DM-DSA 密度调制)** |
| 解耦 CFG | ✗ | ✗ | ✗ | ? | ✗ | **✓ (二级解耦)** |
| 渐进训练 | ✗ | ✗ | ✓ (3阶段) | ? | ✓ (2阶段) | **✓ (PCE 3阶段)** |
| 物理约束 | ✗ | ✗ | ✗ | ✓ (physics-aware) | ✗ | **✓ (CCFM)** |
| 任务统一 | 部分 | ✓ | ✓ | ✓ | 编辑为主 | **✓ (全覆盖)** |
| 运动领域适配 | ✓ | ✗ (视频) | ✗ (图像) | ✗ (视频) | ✗ (图像) | **✓** |

### 8.2 新颖性论证

1. **Density-Modulated Dual-Stream Attention**: 不同于 UniCombine 的 LoRA switching 或 OmniGen2 的 hard dual-path，DM-DSA 实现了条件密度驱动的 soft routing，这是首次在 motion generation 中提出。其核心洞察——条件密度的变化需要动态调整语义（text）vs 结构（motion）信号的相对权重——具有普适性，可推广到 video/image inpainting。

2. **Typed Condition Canvas**: 不同于 VACE 的无类型 3 通道编码，TCC 赋予每个空间条件明确的语义类型。这使得模型能够用不同的策略处理 keyframe 约束（高置信度精确约束）和 edit source（低置信度参考信号），类似于 VACE 的 Concept Decoupling 但更精细。

3. **Contact-Conditioned Flow Matching**: 将物理接触状态集成到 flow matching 训练中，是首次在扩散式 motion generation 框架中实现端到端的接地感知生成（而非后处理）。

4. **Progressive Condition Exposure**: 受 LLM 指令微调中课程学习的启发，PCE 首次系统化地解决 multi-modal multi-task motion generation 中的 shortcut learning 问题。

### 8.3 与 VACE 的关系

CDO-FM 可以视为 VACE 框架在 motion generation 领域的**深度进化**:
- VACE 提出了 `V=[T;F;M]` 的统一编码 → 我们的 TCC 是其类型化扩展
- VACE 的 Context Adapter → 我们的 DM-DSA 是其密度感知版本
- VACE 的 Concept Decoupling → 我们的 TCC 条件类型实现了更细粒度的解耦

---

## 9. 顶会论文定位

### 9.1 推荐标题

**"CDO-FM: Condition-Decoupled Orchestration for Unified Text-and-Motion Conditioned Human Motion Generation"**

或更简洁:

**"MotionCanvas: Density-Aware Condition Orchestration for Universal Motion Generation"**

### 9.2 故事线

> 现有 motion generation 方法要么只做 text-to-motion (T2M)，要么只做 motion completion (M2M)，无法在一个模型中同时理解文本语义和空间运动约束。我们发现根因在于两类条件的**信息密度天然不对称**——一句话的信息量远低于 10 帧 dense keyframe。简单地将两者混入同一 conditioning pipeline 会导致模型学到忽略文本的 shortcut。
>
> 为此，我们提出 CDO-FM，通过 (1) 密度调制双流注意力实现 text/motion 条件的自适应平衡，(2) 类型化条件画布让模型区分不同语义的空间约束，(3) 渐进式条件暴露训练策略防止 shortcut learning，(4) 接触条件化 flow matching 实现端到端的物理感知生成。
>
> 在 XXX benchmark 上，CDO-FM 首次在单一模型中同时达到 T2M SOTA 和 M2M SOTA，且在 text-conditioned motion completion 这一新任务上显著优于所有 baseline。

### 9.3 目标会议

- **首选**: CVPR 2027 (DDL ~Nov 2026) / ICLR 2027 (DDL ~Oct 2026)
- **备选**: NeurIPS 2027 (DDL ~May 2027) / ECCV 2027 (DDL ~Mar 2027)

### 9.4 可能的审稿人关注点

| 审稿人质疑 | 预备回应 |
|-----------|---------|
| DM-DSA 是否过于复杂？简单 concat 是否足够？ | A2 消融实验直接对比 |
| TCC 的条件类型标注是否引入额外成本？ | 完全自动生成，零人工成本 |
| PCE 的阶段划分是否需要精细调参？ | 提供 sensitivity analysis |
| Contact loss 与现有 post-processing 的对比？ | A5 消融 + 后处理消融 |
| 135-dim motion representation 是否限制了方法通用性？ | 讨论扩展到 SMPL-X/手部 |

---

## 10. 风险与备选方案

### 10.1 风险评估

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| DM-DSA 训练不稳定 | 中 | 高 | 渐进引入 gate (先固定 0.5/0.5，再放开学习) |
| TCC 条件类型对性能提升有限 | 低-中 | 中 | 退化为标准 VACE (type_embed=0)，不影响其他组件 |
| PCE 三阶段训练成本过高 | 中 | 中 | Phase 1 可直接从 T2M checkpoint 初始化跳过 |
| Contact label 提取不准确 | 低 | 低 | 使用多阈值 + 时间窗平滑提高鲁棒性 |
| Decoupled CFG 推理开销 3x | 高 | 中 | 提供 standard CFG 回退；或 distillation 减少步数 |

### 10.2 降级方案

如果完整 CDO-FM 的某些组件效果不理想，可以独立使用其中任一子系统：

**最小可行方案 (MVP)**: 修复 B4 (cond_mask_prob) + PCE 训练策略 + Decoupled CFG
- 预期: 解决 text conditioning 失效的 80% 问题
- 成本: 最低，只需改训练配置和推理代码
- 时间: 1-2 周

**中间方案**: MVP + DM-DSA
- 预期: 进一步平衡 text/motion，达到论文水平
- 成本: 需要修改 DiT block，重新训练
- 时间: 3-4 周

**完整方案**: 全部 CDO-FM 组件
- 预期: 最佳性能，完整的论文故事
- 成本: 全面架构升级
- 时间: 6-8 周

---

## 11. 实施路线图

### Phase 0: 紧急 Bug 修复 (Week 1)

```
□ 确认 B4: 检查实际训练使用的 cond_mask_prob 值
□ 如果确实是 1.0，改为 0.1 重新训练一版 baseline
□ 同步确认所有已知 bug (B1-B6) 的修复状态
□ 切换训练数据到 high_quality.json (456K)
□ 建立 text conditioning 评估基准 (R-Precision, Text Effect Ratio)
```

### Phase 1: MVP — 训练策略改进 (Week 2-3)

```
□ 实现 PCE 三阶段训练 scheduler
□ 实现 Decoupled CFG (推理端)
□ 实现 TAL-v2 (对比损失)
□ 训练 A0: baseline fix
□ 训练 A3: PCE
□ 评估: 确认 text conditioning 恢复工作
```

### Phase 2: 架构升级 — TCC + DM-DSA (Week 4-6)

```
□ 实现 TypedConditionCanvas
□ 修改 PrepareM2MCondition 添加条件类型自动标注
□ 实现 DensityModulator
□ 修改 DualStreamDiTBlock 集成 DM-DSA
□ 训练 A1 + A2 消融
□ 训练 A6: 完整 CDO-FM (不含 contact)
```

### Phase 3: 质量增强 — CCFM (Week 6-8)

```
□ 实现 contact label 提取
□ 预处理: 对训练集生成 contact labels
□ 实现 ContactPredictionHead + contact_aware_loss
□ 实现 IK foot lock 后处理
□ 训练 A5: CCFM 消融
□ 训练最终版本
```

### Phase 4: 论文撰写 (Week 8-12)

```
□ 全面评估: E1-E15 + 新增评估任务
□ 可视化: 生成 demo videos, attention maps
□ 论文撰写: Introduction, Method, Experiments, Ablation
□ 用户研究 (可选)
□ 提交
```

---

## 附录 A: 关键代码修改清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `hftrainer/models/motion/hymotion_m2m/bundle.py` | 修改 | 增加 TCC, ContactHead; 修复 cond_mask_prob |
| `hftrainer/models/motion/hymotion_m2m/network/condition_routing.py` | 新增 | TypedConditionCanvas, DensityModulator |
| `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py` | 修改 | DualStreamDiTBlock 集成 DM-DSA |
| `hftrainer/trainers/motion/hymotion_m2m_cdofm_trainer.py` | 新增 | CDO-FM Trainer (PCE + TAL-v2 + contact loss) |
| `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 修改 | Decoupled CFG + contact-guided IK |
| `hftrainer/datasets/motion/transforms/` | 修改 | PrepareM2MCondition 增加类型标注 + contact extraction |
| `configs/hymotion_m2m/cdofm/` | 新增 | CDO-FM 各阶段训练配置 |

## 附录 B: 参考文献

1. **VACE** (Alibaba, 2025): All-in-One Video Creation and Editing — VCU 统一编码, Concept Decoupling
2. **OmniGen2** (BAAI, 2025): Decoupled dual-path, Omni-RoPE, reflection data
3. **Seedance 2.0** (ByteDance, 2026): DB-DiT, omnireference, physics-aware rendering
4. **Step1X-Edit** (StepFun, 2025): MLLM→DiT pipeline, noise-augmented training
5. **UniCombine** (Fudan/Tencent, 2025): Conditional MMDiT Attention, LoRA switching
6. **Flux Fill** (BFL, 2024): Mask concatenation inpainting, guided distillation
7. **MotionCLR** (2024): Attention-based motion diffusion, training-free editing
8. **MotionGPT-2** (NeurIPS 2024): VQ-tokenized motion + LLM

## 附录 C: 计算资源预算

| 实验 | GPU 配置 | 预计时长 | 总 GPU 小时 |
|------|---------|---------|------------|
| A0: Baseline Fix | 8×V100 | 48h | 384 |
| A1-A3: 消融 | 8×V100 × 3 | 72h each | 1,728 |
| A5: CCFM | 16×V100 | 120h | 1,920 |
| A6: Full CDO-FM | 32×V100 | 168h | 5,376 |
| 评估 + 其他 | 8×V100 | 48h | 384 |
| **总计** | | | **~9,792 GPU·h** |

---

*文档结束。方案核心: 通过密度感知的条件解耦编排，实现 text 和 motion condition 的自适应平衡，并以接触条件化 flow matching 提升生成质量。*
