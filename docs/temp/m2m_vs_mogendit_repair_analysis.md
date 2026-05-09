# HyMotion M2M vs MoGenDIT 修复推理策略对比分析

## 1. 核心架构差异一览

| 维度 | MoGenDIT | HyMotion M2M |
|------|----------|--------------|
| 噪声模型 | DDPM (离散时间步) | Flow Matching (连续 ODE) |
| 模型输入 | `x_t` (201-dim) | `[x_t, inactive, reactive, src_mask]` (4×135 = 540-dim) |
| Mask 感知训练 | **是** — `q_sample` 中 `obs_mask=1` 的区域**不加噪** | 标准：**否**；`_man` 变体：**是** — `mask_aware_noise=True` |
| 已知区域信息来源 | 模型从 `x_t` 本身读取（因为已知区域是 clean 的） | 标准：从 VACE `inactive` 通道；`_man`：`x_t` + `inactive` 双通道 |
| 运动表征 | 201-dim: rot6d(132) + joint_pos(66) + trans(3) | 135-dim: trans(3) + rot6d(132)，**无 joint_pos** |
| rot6d 格式 | 列优先 (column-major) | 行优先 (row-major, SMPL format) |

---

## 2. 问题一：HyMotion M2M 多出的 inactive / reactive / src_mask 通道在 adaptive mask 修复时的内容

### 2.1 VACE 三通道定义

代码位置：`hftrainer/models/motion/hymotion_m2m/bundle.py:289-323`

```python
inactive = src_motion * (1 - src_mask)   # 已知区域的值，mask=1 区域为 0
reactive = src_motion * src_mask          # split_reactive 模式下 = mask=1 区域的值
# 或者
reactive = torch.zeros_like(src_motion)   # clean_zero_mask 模式下 = 全 0

vace_context = cat([inactive, reactive, src_mask], dim=-1)  # (B, L, 3×135)
```

### 2.2 Completion 模式（当前修复默认）

修复脚本 `eval_m2m_repair.py:308-311` 中：

```python
if not edit_mode:
    # Completion mode: zero masked regions
    motion_norm = motion_norm * (1 - msk)
```

调用者在传入 `prepare_vace_input` **之前**就把 `src_motion` 的 mask=1 区域清零了。因此：

| 通道 | mask=0 区域（已知/保留） | mask=1 区域（需要修复） |
|------|------------------------|------------------------|
| **inactive** | `src_motion * (1 - mask)` = **clean 归一化值** | `0 * (1 - 1)` = **0** |
| **reactive** | `0 * mask` = **0**（因为 src_motion 此处已是 0）或 `src_motion * 0` = **0** | `0 * 1` = **0** |
| **src_mask** | **0**（已知） | **1**（需生成） |

**结论**：Completion 模式下，reactive 通道**全为 0**。模型从 inactive 通道读取已知区域的 clean motion，从 src_mask 通道知道哪些区域要生成。

### 2.3 Editing 模式

```python
# edit_mode=True 时不清零，src_motion 保持原值
```

| 通道 | mask=0 区域（已知） | mask=1 区域（需修复） |
|------|--------------------|--------------------|
| **inactive** | clean 归一化值 | **0** |
| **reactive** | **0** | **LQ 原始值**（退化的运动） |
| **src_mask** | **0** | **1** |

**结论**：Editing 模式下，reactive 通道在 mask=1 区域携带**原始低质量运动信号**，模型可以参考它来"编辑"而非"从零生成"。

### 2.4 与 MoGenDIT 的对比

MoGenDIT **没有**这三个通道。它的信息传递完全通过 `x_t` 本身：
- 已知区域在 `q_sample` 时不加噪 → `x_t[known]` 就是 clean motion
- 每个去噪步骤开始时 `x_t[mask] = x_t_original[mask]` 再次强制恢复

M2M 的 VACE 是一种**显式条件注入**机制，等价于 MoGenDIT 隐式地通过 `x_t` 传递信息。

---

## 3. 问题二：noisy motion（x_t）通道在修复时是否等价于 MoGenDIT

### 3.1 你的期待

> MoGenDIT replace 为 GT 的部分，M2M 也要 replace 为 GT；MoGenDIT 加噪/重新生成的部分 M2M 也要重新生成。

### 3.2 MoGenDIT 的做法（基准）

**代码**: `EasyDiffusion/base_diffusion.py:384-440`

```
1. q_sample(x0, t=num_timesteps, obs_mask=mask)
   → 已知区域(mask=True): x_t = x0 (不加噪!)
   → 生成区域(mask=False): x_t = sqrt_alpha * x0 + sqrt_1_minus_alpha * noise

2. 每步去噪循环:
   if imputation_mode in ["all", "skip_last"]:
       x_t[mask] = x_t_original[mask]    # 强制替换为 clean GT
   x_t = ddim_sample(x_t, t)

3. 最终步:
   if imputation_mode == "all":
       x_t[mask] = x_t_original[mask]    # 最终一步也替换
   # "skip_last": 最终步不替换
```

**关键**：MoGenDIT 的训练时 `q_sample` 也用了 `obs_mask`，已知区域在训练时就不加噪。因此推理时在每步强制替换为 clean 值是**训练一致的**。

### 3.3 HyMotion M2M 的实际做法

**代码**: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py:200-317`

#### 3.3.1 初始 x_t 构造

```python
z = torch.randn(B, T, D)  # 纯噪声

if sdedit_strength > 0 and 'clean_motion' in batch:
    t_start = 1.0 - sdedit_strength
    y0 = (1 - t_start) * z + t_start * clean   # 部分噪声 + 部分 clean
else:
    y0 = z                                       # 纯噪声
```

**差异 1：初始噪声构造不区分 mask 区域。**

| | MoGenDIT | M2M (默认, sdedit=0) | M2M (sdedit>0) |
|---|---------|---------------------|----------------|
| 已知区域 x_t | **clean motion**（不加噪） | **纯噪声** | `(1-t)*noise + t*clean`（部分噪） |
| 生成区域 x_t | noisy motion | **纯噪声** | `(1-t)*noise + t*clean`（部分噪） |

⚠️ **已知区域和生成区域的初始噪声处理完全相同**，没有像 MoGenDIT 那样区分。

#### 3.3.2 每步替换（Replacement Guidance）

```python
# replacement_guidance 默认为 'none'！
if use_replacement:
    keep_mask = src_mask < 0.5              # True = 已知区域
    x_clean = src_motion                     # 归一化后、mask区域已清零的 motion

    for i in range(n_ode_steps):
        v = fn(t_curr, x)
        x = x + v * dt                      # Euler 步

        if rep_mode == 'skip_last' and not is_last_step:
            x = torch.where(keep_mask, x_clean, x)   # 替换已知区域
```

**差异 2：`replacement_guidance` 默认是 `'none'`**。

在 `eval_m2m_repair.py:265-274` 中：

```python
# 只有 _man 变体（mask-aware noise 训练的）才用 skip_last
is_man = "man" in model_name
if is_man:
    replacement_guidance = 'skip_last'
else:
    replacement_guidance = 'none'        # ← 标准模型不做替换!
```

#### 3.3.3 x_clean 的来源问题

即使开启了 replacement guidance，替换用的 `x_clean` 也有问题：

```python
x_clean = src_motion   # 这是已经被 (1-mask) 清零过的 motion!
```

- mask=0 区域：`x_clean = clean normalized motion` ✓
- mask=1 区域：`x_clean = 0` （被清零了）

所以 `torch.where(keep_mask, x_clean, x)` 只在 keep_mask=True（即 mask=0、已知区域）处替换为 clean 值，mask=1 区域保持模型生成结果。这个逻辑方向是对的。

**但问题是 x_clean 不是 flow-matching 路径上的值！** MoGenDIT 替换的是 q_sample 后的"clean 值"（因为训练时已知区域就是 clean），而 M2M 的训练时已知区域是 `(1-t)*noise + t*clean`（混合值），推理时却替换为纯 clean 值。

`flow_interp` 模式试图修正这一点：`x_target = (1-t_next)*z + t_next*x_clean`，但默认没启用。

### 3.4 核心差异总结表

| 操作 | MoGenDIT | M2M (默认 uncond_fm) | M2M (_man 变体) |
|------|----------|---------------------|-----------------|
| **初始 x_t 已知区域** | clean（不加噪） | 纯噪声 / SDEdit 混合 | 纯噪声 / SDEdit 混合 |
| **每步替换已知区域** | ✅ skip_last | ❌ none（不替换） | ✅ skip_last |
| **替换值** | clean motion | N/A | clean motion（训练一致因为 mask_aware_noise） |
| **训练时已知区域** | clean（不加噪） | noisy（均匀加噪） | clean（mask_aware_noise） |
| **训练-推理一致性** | ✅ 一致 | ✅ 一致（都不特殊处理） | ✅ 一致（都用 clean） |

---

## 4. 结论：`_man` 变体已实现 MoGenDIT 等价的修复策略（2026-04 更新）

### 4.1 标准 M2M (`uncond_fm`) 的实际行为

```
1. 初始化: y0 = 纯噪声（所有区域，包括已知区域）
2. ODE 循环:
   - 模型输入: [x_t, inactive, reactive=0, src_mask]
   - 模型从 inactive 通道读已知区域信息
   - x_t 所有区域自由演化，无任何替换
3. 最终: 后处理 blend
   combined = original * (1-mask) + model_output * mask
```

### 4.2 `_man` 变体 (`uncond_fm_man`) — 已修复（2026-04）

```
1. 初始化: y0[known] = clean_motion, y0[generate] = noise
   （训练一致：x_t[known] = x1）
2. ODE 循环:
   - 模型输入: [x_t, inactive, reactive=0, src_mask]
   - 模型从 x_t + inactive 双通道读已知区域信息
   - 每步（除最后一步）: x[known] = clean_motion（imputation）
3. 最终: 后处理 blend
   combined = original * (1-mask) + model_output * mask
```

**关键修复（2026-04）**：
- ✅ `clean_motion` 现在取自未清零的完整 normalized motion（之前错误地取自已清零的 `src_motion`）
- ✅ `y0` 初始化已知区域为 clean（之前全部为纯噪声，与 _man 训练不一致）
- ✅ SDEdit 已移除（概念与 imputation 冗余，实现有 bug）
- ✅ `flow_interp` 模式已移除（从未正确实现）
- ✅ `adaptive_mask_to_dense` 已修复 trans_mask 丢弃问题

### 4.3 训练-推理一致性对比

| 操作 | MoGenDIT | M2M (标准 uncond_fm) | M2M (_man 变体) |
|------|----------|---------------------|-----------------|
| **初始 x_t 已知区域** | clean（不加噪） | 纯噪声 | **clean**（训练一致） |
| **每步替换已知区域** | ✅ skip_last | ❌ none（不替换） | ✅ skip_last |
| **替换值** | clean motion | N/A | **clean_motion**（未清零的完整 normalized motion） |
| **训练时已知区域** | clean（不加噪） | noisy（均匀加噪） | clean（mask_aware_noise） |
| **训练-推理一致性** | ✅ 一致 | ✅ 一致（都不特殊处理） | ✅ 一致（都用 clean） |

### 4.4 已完成的实验

| 实验 | 训练 | 推理 | 状态 |
|------|------|------|------|
| Baseline | 均匀噪声 | none + blend | ✅ 已实现 |
| V4-full (_man) | mask_aware_noise | skip_last + blend | ✅ 已实现 |

---

## 附录 A：MoGenDIT ada_denoise 三阶段算法细节

**代码**: `MoGenDIT/motion_process/motion_refiner.py:327-373`

```
Stage 1 (探测):
  denoise(motion, mask=frame0_only, step=10, eta=1)
  → denoised_motion

Stage 2 (计算自适应 mask):
  change = |motion - denoised_motion|
  new_keep_mask = (change <= 0.1)  ← 逐元素、逐维度
  # 低变化区域保留，高变化区域重新生成

Stage 3 (选择性重新生成):
  denoise(original_motion, keep_mask=new_keep_mask, step=10, eta=1)
  # 每步: x_t[new_keep_mask] = x_t_original[new_keep_mask]
  # 只有高变化区域被重新生成
```

**注意**: ada_denoise 内部的 mask 是**逐维度**的 (1, T, 201)，而 `compute_adaptive_mask()` 输出的是**逐关节**的 (T, 22)。前者用于 MoGenDIT 内部的 Stage 3，后者用于输出给 M2M 使用。

## 附录 B：Adaptive Mask 从 MoGenDIT 到 M2M 的转换

`eval_m2m_repair.py:136-161` 中的 `adaptive_mask_to_dense()`:

```
MoGenDIT 输出: joint_mask (T, 22), trans_mask (T,)
    ↓
组合为 (T, 23) grid: [trans, joint0, ..., joint21]
    ↓
时序膨胀 (temporal_dilate=5): 每个 flagged 帧向前后扩展 5 帧
    ↓
展开为 (T, 135): trans→3dim, 每个 joint→6dim rot6d
    ↓
M2M 的 src_mask (T, 135)
```

⚠️ **已修复（2026-04）**：`adaptive_mask_to_dense` 现在正确将 MoGenDiT 的 `trans_mask` 写入 combined grid 第 0 列（`combined[:T, 0] = trans_mask[:T]`）。之前第 0 列始终为 0，导致 translation 从不被 mask。
