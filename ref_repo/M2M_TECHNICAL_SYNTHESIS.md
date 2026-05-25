# HyMotion M2M — 综合技术分析与改进路线图

## Executive Summary

基于对 KIMODO、UMO、MotionLab 的深入分析，本文档识别了 **M2M 的四个关键技术差距** 和 **三阶段改进路线**。

### M2M 的核心竞争优势
1. **Per-dimension masking** (T×135)：全球唯一的维度级细粒度控制，支持任意关节子集编辑
2. **Flow Matching backbone**：与 UMO 同架构（HY-Motion 系列），天然对齐 foundation model 生态
3. **原生多任务学习**：从第一步就见 completion 任务，无冷启动问题

### M2M 的关键劣势（vs 竞争对手）

| 维度 | KIMODO | UMO | MotionLab | M2M 当前 |
|------|--------|-----|-----------|---------|
| **位置控制能力** | ✅ xyz 精确控制 | ❌ 无 position dims | ✅ xyz 精确控制 | ❌ 无 position dims |
| **指令编辑** | ❌ | ✅ instruction-based | ✅ instruction-based | ❌ 仅 part 重生成 |
| **风格迁移** | ❌ | ❌ | ✅ SRA 69.21 | ❌ |
| **多人反应** | ❌ | ✅ dual-identity | ❌ | ❌ |
| **训练调度** | ✅ 2阶段 phase | ✅ 多任务联合 | ✅ 7阶段 curriculum | ❌ 固定 M1-M6 比例 |
| **T2M 质量保护** | ✅ Phase 1 纯 T2M | ✅ backbone 冻结 | ⚠️ 部分稀释 | ⚠️ M5 仅 5% |

---

## Part 1: Motion Completion 三范式对比

### 1.1 KIMODO：推理时硬替换（Imputation）

**架构**：
```
训练:
  Phase 1 (500k steps): 纯 T2M，无约束
  Phase 2 (500k steps): random imputation + binary mask concat training
  
输入: concat([x_t, mask], dim=-1)  [333×2]
```

**推理流程**：
```python
# MIB 推理 (给定首尾帧的 position)
for t in range(T, 0, -1):
    x_t[t_start, pos_dims] = GT_start_pos     # ← 硬覆盖
    x_t[t_end, pos_dims] = GT_end_pos          # ← 硬覆盖
    mask[t_start, pos_dims] = 1
    mask[t_end, pos_dims] = 1
    
    model_input = concat([x_t, mask])
    pred_x0 = model(model_input)
    x_{t-1} = scheduler.step(pred_x0, x_t, t)
```

**特点**：
- ✅ Position 维度精确锁定（0 误差）
- ✅ 灵活支持任意维度约束（trajectory, heading 等）
- ❌ Position-only 约束时 rotation 可能不一致（靠 FK loss 学习相关性，有几 cm 误差）
- ❌ 推理时额外开销（每步硬替换 + mask concat）

### 1.2 UMO：软注入 via Element-wise Add（Temporal Fusion）

**架构**：
```
冻结 backbone，加轻量 adapter:
  E_ctx: MLP encoder (0.207M)
  Emb[P], Emb[G], Emb[E]: (3, 201)
  
注入: x'_t = E_in(x_t) + E_ctx(source_motion + Emb[τ_i])
```

**推理流程**：
```python
# MIB 推理 (给定首尾帧的完整 motion)
source = [first_frame_motion, 0, 0, ..., 0, last_frame_motion]
τ = [P, G, G, ..., G, P]
s̃ = source + Emb[τ]

for t in range(T, 0, -1):
    input_emb = E_in(x_t) + E_ctx(s̃)  # ← 软注入，无硬替换
    pred_x0 = backbone(input_emb)
    x_{t-1} = scheduler.step(pred_x0, x_t, t)
```

**特点**：
- ✅ 极轻量（0.207M），冻结 backbone 保留 T2M 质量
- ✅ 多任务统一（P/G/E 覆盖任意任务组合）
- ✅ [edit] 语义强大（source motion 作为 context 注入，不是硬约束）
- ❌ 帧级粒度，无法做 per-joint 控制
- ❌ P 帧不精确（[P]-MPJPE ≈ 0.95mm）
- ❌ 不支持 xyz position 维度

### 1.3 HyMotion M2M：软注入 via Channel Concat（VACE）

**架构**：
```
输入适配到 4× motion_dim:
  x_input = concat([x_t, inactive, reactive, src_mask], dim=-1)
  inactive = src_motion * (1 - src_mask)    # 已知部分的值
  reactive = src_motion * src_mask          # 待生成部分（split）
  src_mask: binary mask (T, 135)
```

**推理流程**：
```python
# MIB 推理 (给定首尾帧的完整 rotation)
src_motion = [first_rot, 0, 0, ..., 0, last_rot]
src_mask = [0, 1, 1, ..., 1, 0]
inactive = src_motion * (1 - src_mask)
reactive = src_motion * src_mask

for t in range(T, 0, -1):
    model_input = concat([x_t, inactive, reactive, src_mask])
    pred_x0 = model(model_input)
    x_{t-1} = scheduler.step(pred_x0, x_t, t)
```

**特点**：
- ✅ 最细粒度（T×135，per-dim）→ per-joint 控制唯一可用
- ✅ 原生支持任意 mask 组合（temporal/joint/keyframe/full）
- ✅ 训练推理分布一致（M1-M6 混合采样）
- ❌ input_encoder 参数 4× 扩大 → 初始化 shape mismatch
- ❌ T2M 质量可能被 completion 稀释（M5 仅 5%）
- ❌ 不支持 xyz position 维度
- ❌ 固定 M1-M6 比例，无法动态调整

---

## Part 2: M2M 的四个关键技术差距

### 2.1 P0 Gap: xyz Position 维度缺失

**当前**：M2M 用 135-dim 表示，不包含 joint positions
**竞争对手**：KIMODO (333D with pos) / MotionLab (263D with pos/vel/contact)

**实现成本**：2-3 weeks
**性能收益**：支持 trajectory constraints，精度 ~10-15cm

### 2.2 P0 Gap: Task Instruction Modulation 缺失

**当前**：M1-M6 mask 模式对模型不可见
**竞争对手**：MotionLab (CLIP task encoding + task token)

**实现成本**：1 week
**性能收益**：+2-5% FID

### 2.3 P1 Gap: Motion Curriculum Learning 缺失

**当前**：固定 M1-M6 采样比例
**竞争对手**：MotionLab (7-stage curriculum with FID-weighted resampling, 11.7× improvement)

**实现成本**：2-3 days
**性能收益**：+20-40% FID

### 2.4 P1 Gap: Instruction-based Editing 缺失

**当前**：M4 仅能做 joint-level 重生成
**竞争对手**：UMO ([edit] semantics) / MotionLab (instruction editing + style transfer)

**实现成本**：1-2 weeks
**性能收益**：新增 editing 能力 (SRA 60-65)

---

## Part 3: 改进路线图

### 阶段 1：快速胜利（2-3 周）

#### 1.1 Task Instruction Modulation（P0）
- 定义 6 个 task instructions
- 在 timestep_emb 加 task_token
- 预期：+2-5% FID

#### 1.2 Motion Curriculum Learning（P1）
- 实现 FID-weighted sampler
- 设计 curriculum schedule (7 stages)
- 预期：+20-40% FID

#### 1.3 E_ctx 权重初始化
- 从预训练权重复制而非随机初始化
- 预期：+3-5% 收敛速度

### 阶段 2：核心能力（4-6 周）

#### 2.1 Position 维度支持（P0）
- 135 → 201 dims (add 22×3D local positions)
- FK Loss 实现
- 合成 trajectory data 20-30%
- 预期：trajectory error ~10-15cm

#### 2.2 Instruction-based Editing（P1）
- 从 MotionFix 数据集复用
- Task Instruction 扩展
- 预期：SRA 60-65

### 阶段 3：探索性（8-12 周）

- Style Transfer (P2)
- 5-Modality Path architecture (P2)
- Multi-person / Reaction generation

---

## 技术决策

### Q1: 是否需要 Position 维度？
**答**：立即启动
- ROI 高（一次投入，后续沿用）
- P0 级功能缺失，竞争对手有
- 前置阻挡 trajectory/obstacle avoidance

### Q2: Curriculum Schedule 参数如何调优？
**建议**：
```
Epoch 1-100:    M5 only (100%)
Epoch 101-300:  M5 (60%) + M1 (40%)
Epoch 301-500:  M5 (40%) + M1/M3 各 30%
Epoch 501-700:  M1-M5 各 20%
Epoch 701+:     FID-weighted resampling
```

---

## 检查清单

### 阶段 1（2-3 周）
- [ ] Task Instruction Modulation
- [ ] Curriculum Learning (FID-weighted sampler)
- [ ] E_ctx 初始化优化
- [ ] Verify +2-5% FID & +20-40% FID

### 阶段 2（4-6 周）
- [ ] Position 维度: 135 → 201
- [ ] FK Loss 实现
- [ ] Instruction Editing 从 MotionFix
- [ ] Benchmark trajectory error & SRA

---

## 参考

- KIMODO/CLAUDE.md: Imputation-based conditioning
- UMO/CLAUDE.md: Temporal Fusion + Frame-level meta-ops
- MotionLab/CLAUDE.md: Task Instruction + Motion Curriculum + Aligned RoPE
- m2m_ablation_experiments.md: 25 实验设计

