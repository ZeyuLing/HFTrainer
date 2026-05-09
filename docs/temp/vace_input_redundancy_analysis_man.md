# HyMotion M2M `_man` 变体 VACE 输入通道冗余性分析

> 日期：2026-04-08
> 范围：仅分析 `_man`（mask-aware noise）变体，不涉及标准（non-MAN）变体

---

## 1. 当前输入设计回顾

模型输入为 4 个通道拼接：

```
x_input = cat([x_t, inactive, reactive, mask], dim=-1)  # (B, L, 4*135 = 540)
```

| 通道 | 维度 | 定义 | mask=0（已知区域） | mask=1（生成区域） |
|------|------|------|-------------------|-------------------|
| **x_t** | 135 | Flow matching 噪声样本 | **x_clean**（`_man` 训练时不加噪） | `(1-t)*noise + t*x_clean` |
| **inactive** | 135 | `src_motion * (1 - mask)` | x_clean（归一化后的真实运动） | **0** |
| **reactive** | 135 | `src_motion * mask` | **0** | **0**（completion 模式） |
| **mask** | 135 | 二值掩码 | 0 | 1 |

总输入维度：**540**，经 `input_encoder`（`nn.Linear(540, feat_dim)`）投影到 transformer 隐藏维度。

---

## 2. 逐通道分析

### 2.1 x_t — 必须保留

**结论：必须（ESSENTIAL）**

**原因**：
- x_t 是 flow matching 框架的核心。模型预测 velocity `v = x1 - x0`，用于 ODE 积分从噪声恢复 clean motion。移除 x_t 等于移除整个生成框架。
- 在 `_man` 变体中，x_t 承载了双重信息：
  - 已知区域（mask=0）：`x_t[known] = x_clean`，即 **clean motion 值**
  - 生成区域（mask=1）：`x_t[gen] = (1-t)*noise + t*x_clean`，即 **带噪运动**
- 这是模型预测的输入基础，不可移除。

---

### 2.2 inactive — 冗余（与 x_t 在已知区域信息完全重复）

**结论：冗余（REDUNDANT）**

**核心论据**：

在 `_man` 训练中，x_t 的已知区域被显式设为 x_clean：

```python
# trainer line 221-222
keep_mask = 1 - src_mask
x_t = x_t * src_mask + x1 * keep_mask   # x_t[known] = x_clean
```

而 inactive 的定义是：

```python
inactive = src_motion * (1 - mask)   # inactive[known] = src_motion[known] = x_clean
                                      # inactive[gen] = 0
```

因此 **在已知区域，x_t 和 inactive 包含完全相同的值（x_clean）**；在生成区域，inactive 固定为 0（无信息）。inactive 没有提供任何 x_t 不包含的信息。

**对比参考实现**：

| 项目 | 模型输入 | 有 inactive 吗？ | 效果 |
|------|---------|-----------------|------|
| **KIMODO** | `[x_t, mask]` (333+333=666) | 无 | SOTA keyframe/end-effector control |
| **MoGenDiT** | `[x_t, mask]` (201+201=402) | 无 | 最佳 repair 效果 |
| **M2M `_man`** | `[x_t, inactive, reactive, mask]` (135*4=540) | 有 | — |

KIMODO 和 MoGenDiT 都使用 mask-aware noise（x_t[known]=clean），都只用 `[x_t, mask]` 两通道，都不需要 inactive。它们的成功经验直接证明 inactive 不是必须的。

**可能的反对意见及反驳**：

| 反对意见 | 反驳 |
|---------|------|
| "inactive 是 timestep 无关的稳定锚点" | 在 `_man` 中 x_t[known] 也是 timestep 无关的（始终 = x_clean），inactive 没有提供额外的稳定性 |
| "inactive 让模型更容易学习" | KIMODO/MoGenDiT 无 inactive 也学得很好。多余输入增加 input_encoder 参数量（540→270 可省约 277K 参数），反而增加学习负担 |
| "保留 inactive 方便将来切回标准模式" | 如果只训 `_man`，inactive 是纯噪音。如果需要兼容标准模式，应该用独立 config 而非在 `_man` 中保留冗余通道 |
| "inactive 和 x_t 在训练早期有微小的数值差异（浮点精度）" | 差异 < 1e-7，不可能对学习产生有意义的影响 |

**冗余的代价**：
- input_encoder 从 `Linear(405, feat_dim)` 变为 `Linear(540, feat_dim)`，多出 `135 * feat_dim` 参数（feat_dim=1024 时多 138K 参数）
- 每帧增加 135 维无用输入，增加内存和计算量
- 模型需要学会忽略这个冗余通道，浪费表达容量

---

### 2.3 reactive — 完全冗余（全零通道）

**结论：完全冗余（COMPLETELY REDUNDANT）**

**原因**：

在 completion 模式下（当前 `_man` 所有训练配置使用的唯一模式），reactive 的构造为：

```python
# Trainer first zeros mask regions:
src_motion = src_motion * (1 - src_mask)   # src_motion[gen] = 0

# Then prepare_vace_input:
reactive = src_motion * src_mask            # reactive = 0 * mask = 0 everywhere
```

**reactive 在 completion 模式下永远是全零张量。** 这 135 维输入完全没有信息，等价于常量偏置。

**量化代价**：
- 135 维全零输入 → input_encoder 中 `135 * feat_dim` 参数完全浪费（feat_dim=1024 时 = 138K 参数）
- 每帧每步额外传输 135 个零值
- 模型必须学会对这 135 维权重输出零贡献

**reactive 何时有用**：
- **仅在 editing 模式下有用**：`reactive = LQ_motion * mask`，传入 mask=1 区域的退化/预编辑运动值
- 当前 `_man` 没有 editing 训练（`edit_mode` flag 未使用），所以 reactive 完全无用
- 如果未来加入 editing，reactive 才有意义——但那应该是一个独立的模型变体

---

### 2.4 mask (src_mask) — 必须保留

**结论：必须（ESSENTIAL）**

**原因**：

虽然在 `_man` 中 x_t 已经隐含了已知/生成区域的信息（已知区域 clean，生成区域 noisy），但显式 mask 仍然不可替代：

1. **时间步接近 1 时无法区分**：当 `t → 1`，生成区域 `x_t[gen] = (1-t)*noise + t*x_clean → x_clean`。此时 x_t 的已知和生成区域都接近 clean，模型无法从 x_t 的值分辨哪些需要生成。显式 mask 是唯一在所有 timestep 下都清晰的信号。

2. **任务结构的唯一编码**：mask pattern 定义了当前任务类型（M1-M7）。相同的运动数据 + 不同的 mask = 完全不同的任务（inbetween vs prediction vs joint completion）。模型需要明确知道任务结构才能正确生成。

3. **对齐 KIMODO/MoGenDiT 的最佳实践**：两者都保留了 mask 通道。mask 是所有 imputation-based 方案的标配。

4. **Loss 计算依赖**：`_man` 的 loss 只在 generation regions（mask=1）计算。mask 是训练框架的结构性组件。

---

## 3. 总结

### 冗余性判定表

| 通道 | 维度 | 判定 | 原因 |
|------|------|------|------|
| **x_t** | 135 | **必须** | Flow matching 核心输入，承载 clean（已知）+ noisy（生成）信息 |
| **inactive** | 135 | **冗余** | 在 `_man` 中与 x_t[known] 完全相同；KIMODO/MoGenDiT 无此通道也能正常工作 |
| **reactive** | 135 | **完全冗余** | Completion 模式下永远全零；仅 editing 模式有值，但 `_man` 不训练 editing |
| **mask** | 135 | **必须** | t→1 时无法从 x_t 区分已知/生成；任务结构的唯一编码；loss 计算依赖 |

### 最优输入设计（`_man` completion 专用）

```
x_input = cat([x_t, mask], dim=-1)  # (B, L, 2*135 = 270)
```

- 输入维度从 540 降为 **270**（减少 50%）
- input_encoder 参数量从 `540 * feat_dim` 降为 `270 * feat_dim`（减少 50%）
- 与 KIMODO、MoGenDiT 的输入设计完全一致
- 保留所有必要信息，不损失任何模型能力

### 如果需要兼容 editing 模式

```
x_input = cat([x_t, reactive, mask], dim=-1)  # (B, L, 3*135 = 405)
```

- 保留 reactive 为未来 editing 留接口
- inactive 仍然冗余（editing 模式下 x_t[known] = x_clean 也成立）
- 比当前设计省 135 维

### 实施建议

1. **新训练实验**应使用 `[x_t, mask]` 输入（需新 input_encoder 权重，无法复用旧 checkpoint）
2. **已训练的 `_man` checkpoint** 无法直接切换——input_encoder 的权重维度不同，需要从头训练
3. 如果需要量化 inactive 的实际效果，可以做消融实验：在现有 540 维 checkpoint 上，推理时将 inactive 通道置零，观察输出质量变化。如果质量无显著下降，则实锤 inactive 在 `_man` 中确实冗余

---

## 4. 附录：数据流对比图

```
当前 _man 设计 (540 维):
  x_t          [已知=clean, 生成=noisy]  ← 核心
  inactive     [已知=clean, 生成=0]      ← 与 x_t[已知] 重复
  reactive     [全零]                    ← 完全无用
  mask         [0/1]                     ← 核心
  ─────────────────────────────────────
  信息量 = x_t + mask = 270 维有效信息 + 270 维冗余

推荐 _man 设计 (270 维):
  x_t          [已知=clean, 生成=noisy]  ← 核心
  mask         [0/1]                     ← 核心
  ─────────────────────────────────────
  信息量 = 270 维，全部有效

参考: KIMODO (666 维 = 333+333):
  x_t          [已知=clean, 生成=noisy]  ← 核心
  mask         [0/1]                     ← 核心

参考: MoGenDiT (402 维 = 201+201):
  x_t          [已知=clean, 生成=noisy]  ← 核心
  mask         [0/1]                     ← 核心
```
