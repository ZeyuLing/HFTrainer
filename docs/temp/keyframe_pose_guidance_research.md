# Keyframe Pose Guidance for HyMotion M2M — 调研报告

> 需求：用户输入一个 **单帧 target pose**（135 维 rot6d + translation）和一个 **src motion**（完整动作序列），模型输出的动作在指定 keyframe 位置向该 target pose 靠近。

---

## 目录

1. [需求分析与问题定义](#1-需求分析与问题定义)
2. [方案一：纯推理 Imputation（无需训练）](#2-方案一纯推理-imputation无需训练)
3. [方案二：Keyframe Guidance Loss（推理时引导）](#3-方案二keyframe-guidance-loss推理时引导)
4. [方案三：训练方案 — ref_pose 条件注入](#4-方案三训练方案--ref_pose-条件注入)
5. [方案四：训练方案 — Mask-Aware Flow Matching + Imputation](#5-方案四训练方案--mask-aware-flow-matching--imputation)
6. [方案对比与推荐](#6-方案对比与推荐)
7. [现有代码基础分析](#7-现有代码基础分析)
8. [参考文献与相关工作](#8-参考文献与相关工作)

---

## 1. 需求分析与问题定义

### 1.1 输入输出定义

| 项目 | 说明 |
|------|------|
| **输入 1** | `src_motion`: (T, 135) — 源动作序列，包含 translation(3) + rot6d(22×6=132) |
| **输入 2** | `target_pose`: (1, 135) — 目标姿态（单帧），用户希望输出动作在某个 keyframe 靠近此 pose |
| **输入 3** | `keyframe_idx`: int — 用户指定的 keyframe 位置（如第 60 帧） |
| **输入 4（可选）** | `text_prompt`: str — 文本描述（可为空） |
| **输出** | `output_motion`: (T, 135) — 生成动作，其中 `output[keyframe_idx]` 尽可能接近 `target_pose` |

### 1.2 核心挑战

1. **M2M 的 VACE 架构限制**：HyMotion M2M 通过 VACE 侧通道（inactive channel）传递已知信息，x_t 全程 noisy。模型不从 x_t 读取已知区域信息，因此推理时简单替换 x_t 的某一帧 **效果有限**（见 CLAUDE.md Known-Region Conditioning 对比分析）。

2. **"靠近" vs "精确匹配"**：用户需求是"朝 target pose 靠近"，不一定要求零误差。这给了更多设计空间——可以用 soft guidance 而非 hard constraint。

3. **动态一致性**：输出动作需要在 keyframe 附近自然过渡，不能有跳变。

### 1.3 现有 M2M 能力盘点

当前 M2M 已支持的相关能力：

| 能力 | 实现方式 | 与本需求关系 |
|------|---------|-------------|
| Sparse Keyframe Interpolation | M6 策略：保留 K 个 keyframe，生成其余帧 | **直接相关**：如果 target_pose 作为 keyframe 插入 src_motion，然后用 M6 式 mask 让模型补全 |
| Motion In-Between | M3 策略：保留首尾，生成中间 | target_pose 可作为"尾帧"锚点 |
| Joint Completion | M4 策略：mask 特定关节 | 可以只约束部分关节 |
| Edit-Repair mode | reactive 通道传 LQ 值 | 可将"含 target_pose 的拼接 motion"作为 LQ 输入 |
| ref_pose plumbing | 已有代码骨架（见 §7） | 可扩展 |

---

## 2. 方案一：纯推理 Imputation（无需训练）

### 2.1 核心思路

利用现有 M2M checkpoint（completion 模式），将 target_pose 当作一个"已知 keyframe"，构建 mask 让模型补全其余帧。

### 2.2 推理管线

```
Step 1: 构造输入 motion
   composite_motion = src_motion.clone()   # (T, 135)
   composite_motion[keyframe_idx] = target_pose  # 将 keyframe 替换为 target_pose

Step 2: 构建 mask
   # 方案 A：仅 keyframe 周围需要重新生成（local editing）
   src_mask = zeros(T, 135)
   src_mask[keyframe_idx - W : keyframe_idx + W] = 1  # W 为过渡窗口
   src_mask[keyframe_idx] = 0  # keyframe 本身是已知的

   # 方案 B：全部重新生成，keyframe 作为约束
   src_mask = ones(T, 135)
   src_mask[keyframe_idx] = 0  # 只有 keyframe 已知

   # 方案 C：保留首尾 + keyframe（类似 M6 策略）
   src_mask = ones(T, 135)
   src_mask[0] = 0             # 保留第一帧
   src_mask[-1] = 0            # 保留最后一帧
   src_mask[keyframe_idx] = 0  # 保留 target keyframe

Step 3: 标准 M2M 推理
   normalized_motion = bundle.normalize_motion(composite_motion)
   normalized_motion = normalized_motion * (1 - src_mask)  # 清零 mask 区域
   vace_context = bundle.prepare_vace_input(normalized_motion, src_mask)
   # ODE integration...
   output = odeint(fn, noise, t=[0,1], method='midpoint')

Step 4: Post-hoc blend
   final = src_motion * (1 - src_mask) + denormalize(output) * src_mask
   # keyframe 精确保持（mask=0）
```

### 2.3 优势与局限

| 优势 | 局限 |
|------|------|
| ✅ **零训练成本**，立即可用 | ❌ **边界跳变**：VACE 架构的 post-hoc blend 在 mask 边界会有不连续 |
| ✅ 利用 M6 keyframe sparse 训练策略，模型已学过此 pattern | ❌ **keyframe 精度取决于 blend**：mask=0 的帧精确保持，但周围过渡可能不自然 |
| ✅ 代码改动极小（仅推理脚本） | ❌ **不支持"软靠近"**：要么精确保持(mask=0)，要么完全重新生成(mask=1) |

### 2.4 改进变体：Replacement Guidance

利用 pipeline 已实现的 `replacement_guidance` 模式：

```python
# 推理时在每步 ODE 积分中替换 keyframe 帧
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    replacement_guidance='flow_interp',  # 或 'skip_last'
)
```

但 **当前训练未使用 mask-aware noise**，replacement guidance 效果有限（见 pipeline 注释）。如果先做 V4（mask-aware flow matching）训练实验，此方案会大幅改善。

### 2.5 推理管线代码示例

```python
def keyframe_pose_imputation(
    bundle, pipeline, src_motion, target_pose, keyframe_idx,
    mode='keyframe_only', window=30, text=None,
):
    """纯推理方案：将 target_pose 作为已知 keyframe 进行 M2M completion。

    Args:
        mode: 'keyframe_only' — 只保留 keyframe，全部重生成
              'local_edit' — 保留大部分帧，仅在 keyframe 附近重生成
              'anchor_inbetween' — 保留首帧 + keyframe + 尾帧
    """
    T, D = src_motion.shape
    composite = src_motion.clone()
    composite[keyframe_idx] = target_pose

    src_mask = torch.zeros(T, D)

    if mode == 'keyframe_only':
        src_mask[:] = 1.0
        src_mask[keyframe_idx] = 0.0

    elif mode == 'local_edit':
        half_w = window // 2
        start = max(0, keyframe_idx - half_w)
        end = min(T, keyframe_idx + half_w)
        src_mask[start:end] = 1.0
        src_mask[keyframe_idx] = 0.0

    elif mode == 'anchor_inbetween':
        src_mask[:] = 1.0
        src_mask[0] = 0.0
        src_mask[-1] = 0.0
        src_mask[keyframe_idx] = 0.0

    # Normalize and zero mask regions
    normalized = bundle.normalize_motion(composite.unsqueeze(0))
    normalized = normalized * (1 - src_mask.unsqueeze(0))

    batch = {
        'src_motion': normalized,
        'src_mask': src_mask.unsqueeze(0),
        'src_length': [T],
        'tgt_length': [T],
    }
    if text:
        batch.update(bundle.encode_text([text]))

    result = pipeline(batch)
    output_denorm = bundle.denormalize_motion(result['latent'])

    # Blend
    mask_3d = src_mask.unsqueeze(0)
    final = composite.unsqueeze(0) * (1 - mask_3d) + output_denorm * mask_3d
    return final.squeeze(0)
```

---

## 3. 方案二：Keyframe Guidance Loss（推理时引导，无需训练）

### 3.1 核心思路

类似 classifier guidance / loss guidance：在 ODE 积分过程中，每步计算当前 x_t 在 keyframe 位置与 target_pose 的 loss 梯度，用梯度修正 ODE 速度场方向。

### 3.2 数学推导

标准 flow matching ODE：
```
dx/dt = v_θ(x_t, t, c)
```

加入 keyframe guidance 后：
```
dx/dt = v_θ(x_t, t, c) - λ · ∇_x L(x_t[keyframe_idx], target_pose)
```

其中 `L` 是 keyframe 匹配 loss：
```python
def keyframe_loss(x_t, target_pose_norm, keyframe_idx):
    pred_kf = x_t[:, keyframe_idx, :]  # (B, D)
    return F.mse_loss(pred_kf, target_pose_norm)
```

### 3.3 推理管线修改

```python
def fn_with_guidance(t, x):
    # 标准 velocity
    x.requires_grad_(True)
    v = original_fn(t, x.detach())

    # Guidance: 计算 keyframe loss 的梯度
    # 从 x 反推当前 predicted x1: x1_pred = x + (1-t) * v
    x1_pred = x + (1 - t) * v
    kf_pred = x1_pred[:, keyframe_idx, :]
    loss = F.mse_loss(kf_pred, target_pose_norm)
    grad = torch.autograd.grad(loss, x, create_graph=False)[0]

    # 修正 velocity
    return v - guidance_scale * grad
```

### 3.4 优势与局限

| 优势 | 局限 |
|------|------|
| ✅ **零训练成本**，利用现有 checkpoint | ❌ **需要反向传播**：每步 ODE 需计算梯度，推理速度约慢 2-3x |
| ✅ **软约束**：可通过 `guidance_scale` 控制靠近程度 | ❌ **可能不稳定**：大 guidance_scale 会导致 ODE 发散 |
| ✅ 自然过渡，无 hard blend 跳变 | ❌ **keyframe 精度有限**：取决于 guidance_scale 和步数 |
| ✅ 可与方案一结合使用 | ❌ 需要调参（guidance_scale、guidance 的开始/结束 timestep） |

### 3.5 实际操作建议

- `guidance_scale` 建议从 1.0 开始尝试，逐步增大到 10-50
- 只在 t ∈ [0.3, 0.9] 区间施加 guidance（早期 x_t 太 noisy 梯度无意义，末期已收敛）
- 可与方案一的 replacement guidance 结合：imputation 提供 hard anchor，gradient guidance 提供 soft 平滑

---

## 4. 方案三：训练方案 — ref_pose 条件注入

### 4.1 核心思路

在 VACE 通道中显式加入 ref_pose 作为额外条件。训练时随机采样一帧作为 ref_pose，模型学习在生成过程中"参考"该帧的姿态。

### 4.2 现有代码基础

**bundle.py 和 trainer.py 已有 ref_pose 的完整 plumbing**（见 §7 详细分析）：

1. `bundle.prepare_vace_input(src_motion, ref_pose, src_mask)` — ref_pose 以额外 token 形式 prepend 到 VACE context 序列前面
2. `bundle.prepare_padding(src_motion, tgt_motion, tgt_length, src_mask, src_length, ref_pose)` — padding mask 正确处理 ref_pose 的额外长度
3. `trainer.train_step` — ref_pose 从 batch 中读取并传入 VACE 构建
4. `pipeline._inference` — ref_pose 从 batch 中读取并用于推理

### 4.3 VACE 中 ref_pose 的注入方式

当前实现将 ref_pose 作为额外的 1 帧 token prepend 到 motion 序列：

```python
# bundle.py prepare_vace_input 中 ref_pose 处理：
if ref_pose is not None:
    _, L_ref, _ = ref_pose.shape        # L_ref = 1 (单帧)
    # ref_pose 的 VACE context：inactive = ref_pose, reactive = zeros
    ref_vace = cat([ref_pose, zeros_like(ref_pose)], dim=-1)  # (B, 1, 2*D)
    # ref_pose 的 mask = 0（已知）
    ref_mask = zeros(B, 1, D)
    src_mask = cat([ref_mask, src_mask], dim=1)
    vace_context = cat([ref_vace, vace_context], dim=1)  # prepend
```

即：模型看到的序列是 `[ref_pose_token, motion_frame_0, ..., motion_frame_T-1]`，其中 ref_pose_token 的 mask=0（已知），模型通过 self-attention 让所有帧都能"看到"这个 target pose。

**流目标（x1）也 prepend 了 ref_pose**：
```python
x1 = tgt_motion
if ref_pose is not None:
    x1 = cat([ref_pose, x1], dim=1)  # (B, 1+T, D)
```

这意味着模型在 flow matching 中也需要预测 ref_pose 帧的 velocity/x1，但 ref_pose 帧本身是已知的，loss 权重可以降低或忽略。

### 4.4 训练数据构建

**核心问题：如何构建 (src_motion, target_pose, output_motion) 数据对？**

#### 方案 3A：从同一动作序列采样 ref_pose（Self-Supervised）

```python
# 在 PrepareM2MUniversalMask 之后增加 transform
class PrepareRefPose(BaseTransform):
    def __init__(self, ref_pose_prob=0.3):
        """以 ref_pose_prob 概率为样本构造 ref_pose。"""
        self.ref_pose_prob = ref_pose_prob

    def transform(self, results):
        if random() > self.ref_pose_prob:
            results['ref_pose'] = None
            return results

        motion = results['tgt_motion']  # (T, D) — 完整动作
        T = motion.shape[0]

        # 从 mask=1 区域随机选一帧作为 ref_pose
        # 这帧的 GT 值就是 target_pose
        src_mask_grid = results['src_mask']  # (T, D)
        masked_frames = (src_mask_grid.sum(dim=-1) > 0).nonzero().squeeze(-1)
        if len(masked_frames) == 0:
            results['ref_pose'] = None
            return results

        kf_idx = masked_frames[randint(0, len(masked_frames))]
        ref_pose = motion[kf_idx:kf_idx+1]  # (1, D)
        results['ref_pose'] = ref_pose
        return results
```

**训练语义**：
- `src_motion`: 源动作（mask 区域清零）
- `ref_pose`: 从目标动作中采样的一帧（作为 guidance）
- `tgt_motion`: 完整目标动作
- 模型学习：在 completion 过程中，让生成的帧"靠近" ref_pose（因为 ref_pose 就来自 GT）

**优势**：
- 不需要额外数据，完全自监督
- 训练信号清晰：ref_pose 就是 GT 的一帧
- 与现有 mask 策略兼容

#### 方案 3B：从不同动作中采样 ref_pose（Cross-Motion）

更贴近实际应用场景——用户的 target_pose 可能来自另一个动作：

```python
class PrepareRefPoseFromPool(BaseTransform):
    def __init__(self, ref_pose_prob=0.3, same_motion_ratio=0.5):
        self.ref_pose_prob = ref_pose_prob
        self.same_motion_ratio = same_motion_ratio

    def transform(self, results):
        if random() > self.ref_pose_prob:
            results['ref_pose'] = None
            return results

        if random() < self.same_motion_ratio:
            # 从同一动作采样（方案 3A）
            ref_pose = results['tgt_motion'][randint(0, T)]
        else:
            # 从其他动作采样（需要 dataset 提供 random_motion_frame）
            ref_pose = results.get('random_ref_pose')

        results['ref_pose'] = ref_pose.unsqueeze(0) if ref_pose is not None else None
        return results
```

**注意**：当 ref_pose 来自其他动作时，输出不应精确匹配 ref_pose（因为物理上可能不兼容），而应在"风格"或"部分关节"上靠近。此时 loss 设计更复杂。

### 4.5 网络结构调整

**当前 HunyuanMotionMMDiT 的 input_dim = 540 = 135 + 3×135**，已经包含了 VACE 通道。ref_pose 作为序列维度的额外 token（prepend），**不改变 input_dim**，因此：

> **网络结构不需要修改**。ref_pose 通过 prepend token 的方式注入，模型的 self-attention 自然能处理变长序列。

唯一需要确认的是：
1. **Positional encoding (RoPE)**：ref_pose 的位置编码是否合理？当前 prepend 在 position 0，原始 motion 从 position 1 开始。这是合理的——ref_pose 是"条件 token"，位于序列开头。
2. **padding mask**：已在 `prepare_padding` 中正确处理（ref_mask = True for ref_pose token）。

### 4.6 Loss 设计

ref_pose 帧本身的 loss 可以特殊处理：

```python
# 在 M2MLoss 中增加 ref_pose_weight
if ref_pose is not None:
    # ref_pose 帧的 velocity target 是 ref_pose - noise[0]（trivial）
    # 可以给 ref_pose 帧的 loss 降权或忽略
    # 但也可以保留——让模型学会"准确重建 ref_pose 帧"
    pass
```

建议保留 ref_pose 帧的 loss（权重 1.0），因为模型确实需要在该位置输出与 ref_pose 一致的值。

### 4.7 推理流程

```python
def keyframe_pose_with_ref_pose(
    bundle, pipeline, src_motion, target_pose, keyframe_idx, text=None,
):
    """训练方案推理：ref_pose 作为条件 token。"""
    T, D = src_motion.shape

    # 构建 mask：keyframe 附近需要重生成
    src_mask = torch.zeros(T, D)
    # 可以用 local_edit 或 full 模式
    src_mask[:] = 1.0  # 全部重生成，ref_pose 作为唯一约束
    # 或保留首尾帧作为锚点

    # Normalize
    normalized = bundle.normalize_motion(src_motion.unsqueeze(0))
    normalized = normalized * (1 - src_mask.unsqueeze(0))

    # ref_pose 也需要 normalize
    ref_pose_norm = bundle.normalize_motion(target_pose.unsqueeze(0).unsqueeze(0))  # (1, 1, D)

    batch = {
        'src_motion': normalized,
        'src_mask': src_mask.unsqueeze(0),
        'ref_pose': ref_pose_norm,
        'src_length': [T],
        'tgt_length': [T],
    }
    result = pipeline(batch)
    return bundle.denormalize_motion(result['latent']).squeeze(0)
```

### 4.8 优势与局限

| 优势 | 局限 |
|------|------|
| ✅ 与 M2M 架构天然兼容（代码骨架已存在） | ❌ 需要训练（但可在现有训练上 finetune） |
| ✅ 无 hard blend 跳变 | ❌ "靠近程度"取决于训练数据分布 |
| ✅ 推理时无需梯度计算，速度不降 | ❌ ref_pose 信号通过 attention 传递，可能被稀释 |
| ✅ 自然过渡 | ❌ 如果 ref_pose 和 src_motion 差异极大，可能导致生成质量下降 |

---

## 5. 方案四：训练方案 — Mask-Aware Flow Matching + Imputation

### 5.1 核心思路

先实现 V4（mask-aware flow matching），让模型训练时已知区域在 x_t 中保持 clean。这样推理时可以直接 impute（像 KIMODO/MoGenDiT 那样），keyframe 位置精确保持 target_pose。

### 5.2 与当前方案的关系

这不是一个独立方案，而是对方案一（imputation）的**训练增强**：

```
当前状态：
  训练: x_t[所有区域] = noisy → 推理时 impute 无效（train-infer mismatch）

V4 改进后：
  训练: x_t[keep] = clean, x_t[gen] = noisy → 推理时 impute 有效（train-consistent）
```

### 5.3 训练修改（已在 trainer.py 中实现）

```python
# 已有代码（trainer.py line 220-222）
if self.mask_aware_noise and src_mask is not None:
    keep_mask = 1 - src_mask
    x_t = x_t * src_mask + x1 * keep_mask
```

只需在 config 中设置 `mask_aware_noise=True`：
```python
trainer = dict(
    type='HyMotionM2MTrainer',
    mask_aware_noise=True,  # 启用 V4
)
```

### 5.4 推理修改

启用 replacement guidance（已在 pipeline 中实现）：
```python
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    replacement_guidance='flow_interp',  # 与 V4 训练一致
)
```

推理流程与方案一相同，但因为训练时模型已习惯看到 x_t 中的 clean 区域，replacement guidance 现在是 train-consistent 的。

### 5.5 数据对构建

无需额外数据构建——使用现有 7 种 mask 策略，只是加噪方式改变。

### 5.6 优势与局限

| 优势 | 局限 |
|------|------|
| ✅ Keyframe 精确保持（零误差） | ❌ 需要从头训练（不能复用当前 checkpoint） |
| ✅ 平滑过渡（模型在训练时就见过边界） | ❌ 可能影响 T2M（全 mask）能力 |
| ✅ 代码改动最小（config 参数切换） | ❌ 需要 loss mask（已知区域不计 loss，已实现） |
| ✅ 理论最优（KIMODO/MoGenDiT 验证过） | ❌ 训练收敛可能需要调整 |

---

## 6. 方案对比与推荐

### 6.1 总体对比

| 维度 | 方案一：Imputation | 方案二：Gradient Guidance | 方案三：ref_pose 训练 | 方案四：V4 + Imputation |
|------|-------------------|------------------------|---------------------|----------------------|
| **是否需要训练** | ❌ 不需要 | ❌ 不需要 | ⚠️ 需要 finetune | ⚠️ 需要从头训练 |
| **Keyframe 精度** | 精确（hard blend） | 近似（soft guidance） | 近似（attention 传递） | 精确（train-consistent impute） |
| **过渡平滑度** | ⚠️ 边界跳变 | ✅ 自然过渡 | ✅ 自然过渡 | ✅ 自然过渡 |
| **推理速度** | 正常 | 慢 2-3x（需反向传播） | 正常 | 正常 |
| **代码改动** | 小（推理脚本） | 中（pipeline 修改） | 小（利用已有骨架） | 最小（config 参数） |
| **可控性** | binary（mask 0/1） | 连续（guidance_scale） | 隐式（model 决定） | binary（mask 0/1） |
| **与其他任务兼容** | ✅ 不影响 | ✅ 不影响 | ⚠️ 需要 ref_pose dropout | ⚠️ 可能影响 T2M |

### 6.2 推荐策略

**短期（立即可用）**：方案一 + 方案二组合

```
1. 方案一（imputation）作为 baseline：
   - src_motion 中 keyframe_idx 帧替换为 target_pose
   - mask 方案 C（anchor_inbetween）：保留首帧 + target keyframe + 尾帧
   - 快速验证效果

2. 如果边界跳变严重，叠加方案二（gradient guidance）：
   - 在方案一的 mask=1 区域内，额外施加 keyframe loss guidance
   - 或不做 hard imputation，纯用 gradient guidance
```

**中期（需要 finetune）**：方案三（ref_pose 训练）

```
1. 在现有训练中增加 PrepareRefPose transform（30% 概率采样 ref_pose）
2. 无需改网络结构（利用已有 ref_pose plumbing）
3. Finetune 现有 checkpoint 若干 epoch
4. 同时支持有/无 ref_pose 的推理（30% dropout → 模型不依赖 ref_pose）
```

**长期（最优方案）**：方案四（V4 + Imputation）

```
1. 启用 mask_aware_noise=True 重新训练
2. 推理时用 replacement_guidance='flow_interp'
3. Keyframe 精确保持 + 平滑过渡 + 推理速度不降
```

### 6.3 方案组合矩阵

方案之间不互斥，可以组合：

| 组合 | 效果 |
|------|------|
| 方案一 alone | 快速验证，边界可能跳变 |
| 方案一 + 方案二 | imputation anchor + gradient smoothing，效果最好的免训练方案 |
| 方案三 alone | 自然过渡但 keyframe 精度略低 |
| 方案三 + 方案二 | ref_pose 条件 + gradient 精调 |
| 方案四 alone | 理论最优，需要完整训练 |
| 方案四 + 方案三 | 极致方案：mask-aware 训练 + ref_pose 条件 + imputation |

---

## 7. 现有代码基础分析

### 7.1 ref_pose 代码骨架

代码中已有完整的 ref_pose 支持（虽然当前没有 dataset transform 生成 ref_pose）：

| 文件 | 相关代码 | 功能 |
|------|---------|------|
| `bundle.py` L279-313 | `prepare_vace_input(src_motion, ref_pose, src_mask)` | ref_pose prepend 到 VACE context |
| `bundle.py` L226-277 | `prepare_padding(..., ref_pose)` | padding mask 处理 ref_pose |
| `trainer.py` L107-111 | `ref_pose = batch.get('ref_pose')` | 从 batch 读取 ref_pose |
| `trainer.py` L116 | `prepare_padding(..., ref_pose)` | 传递给 bundle |
| `trainer.py` L203-204 | `x1 = cat([ref_pose, x1], dim=1)` | flow target prepend ref_pose |
| `trainer.py` L227 | `prepare_vace_input(src_motion, ref_pose, src_mask)` | VACE 构建 |
| `pipeline.py` L114-118 | `ref_pose = batch.get('ref_pose')` | 推理时读取 ref_pose |
| `pipeline.py` L137-141 | `prepare_vace_input(src_motion, ref_pose, src_mask)` | 推理时 VACE 构建 |

**结论**：方案三的网络和训练/推理管线代码已基本就绪，只缺 dataset transform 生成 ref_pose。

### 7.2 replacement_guidance 代码

Pipeline 中已实现 4 种 replacement guidance 模式（L36-58, L186-256）：
- `none`: 标准 ODE
- `all`: 每步替换 known regions 为 clean values
- `skip_last`: 同 all 但跳过最后一步
- `flow_interp`: 替换为 flow matching 插值（train-consistent for V4）

### 7.3 mask_aware_noise 代码

Trainer 中已实现 mask-aware noise（L220-222）：
```python
if self.mask_aware_noise and src_mask is not None:
    keep_mask = 1 - src_mask
    x_t = x_t * src_mask + x1 * keep_mask
```

以及对应的 loss mask（L246-248）：
```python
if self.mask_aware_noise and src_mask is not None:
    generation_mask = src_mask
```

### 7.4 需要新增的代码

| 组件 | 文件位置 | 内容 |
|------|---------|------|
| `PrepareRefPose` transform | `hftrainer/datasets/motion/motionhub/transforms/ref_pose.py` | 从 GT 采样 ref_pose |
| `keyframe_guidance_fn` | `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 方案二的 gradient guidance |
| 推理入口脚本 | `tools/infer_m2m_keyframe.py` | 统一的 keyframe pose guidance 推理入口 |
| config | `configs/hymotion_m2m/hymotion_m2m_ref_pose_*.py` | 方案三的训练 config |

---

## 8. 参考文献与相关工作

### 8.1 直接相关

| 项目 | 方法 | 与本需求关系 |
|------|------|-------------|
| **KIMODO** (NVIDIA) | Imputation + Phase 2 训练 | keyframe impute 是其核心能力，Phase 2 训练让 impute train-consistent |
| **MoGenDiT** (内部) | Mask-aware noise + 每步替换 | 已验证 mask-aware noise + impute 的有效性 |
| **UMO** (Brown/MIT/Meta) | Adapter add + meta-operation | [P] Preserve tag = keyframe 保留，但仅帧级别 |
| **OmniControl** (NeurIPS 2024) | ControlNet-style 引导 | 在 keyframe 位置提供 joint position 约束 |
| **GMD** (ICLR 2024) | Guided Motion Diffusion | 在 diffusion 采样时施加关节约束 loss guidance |
| **PriorMDM** (ICLR 2024) | DoubleTake / composition | 多段 motion 在 keyframe 处拼接、迭代去噪平滑 |

### 8.2 通用方法论

| 方法 | 适用性 |
|------|--------|
| **Classifier-Free Guidance** | 已用于文本条件，可扩展到 pose 条件 |
| **Classifier Guidance / Loss Guidance** | 方案二的理论基础 |
| **Inpainting in Diffusion Models** | 方案一的理论基础（RePaint, Palette） |
| **Conditional Token Prepend** | 方案三的做法，类似 ViT 的 [CLS] token |
| **VACE (All-in-One Video Creation)** | M2M 的基础框架，支持任意 mask pattern |

### 8.3 关键洞见

1. **KIMODO 证明了**：在 flow matching/diffusion 框架下，keyframe imputation 是最直接有效的方法，前提是训练时 x_t 在已知区域保持 clean（Phase 2 / mask-aware noise）。

2. **GMD 证明了**：即使不修改训练，推理时的 gradient guidance 也能有效引导 joint position 约束，但存在速度/稳定性 trade-off。

3. **UMO 证明了**：轻量 adapter / token-level 条件注入可以在不改 backbone 的情况下实现 motion conditioning，但精度不如 imputation。

4. **M2M 的独特优势**：已有 per-joint per-frame 的 mask 粒度（7 种策略），以及 VACE 的 reactive 通道（可传递 LQ/reference 值），使得上述所有方案都可以实现。

---

## 附录 A：HyMotion M2M 135 维布局

```
dims [0:3]     — translation (3 absolute)
dims [3:9]     — joint 0:  Pelvis (root orientation, rot6d)
dims [9:15]    — joint 1:  L_Hip
dims [15:21]   — joint 2:  R_Hip
dims [21:27]   — joint 3:  Spine1
dims [27:33]   — joint 4:  L_Knee
dims [33:39]   — joint 5:  R_Knee
dims [39:45]   — joint 6:  Spine2
dims [45:51]   — joint 7:  L_Ankle
dims [51:57]   — joint 8:  R_Ankle
dims [57:63]   — joint 9:  Spine3
dims [63:69]   — joint 10: L_Foot
dims [69:75]   — joint 11: R_Foot
dims [75:81]   — joint 12: Neck
dims [81:87]   — joint 13: L_Collar
dims [87:93]   — joint 14: R_Collar
dims [93:99]   — joint 15: Head
dims [99:105]  — joint 16: L_Shoulder
dims [105:111] — joint 17: R_Shoulder
dims [111:117] — joint 18: L_Elbow
dims [117:123] — joint 19: R_Elbow
dims [123:129] — joint 20: L_Wrist
dims [129:135] — joint 21: R_Wrist
```

## 附录 B：ref_pose VACE 注入细节

```
标准 VACE context (无 ref_pose):
  序列: [frame_0, frame_1, ..., frame_T-1]
  每帧: [inactive(D), reactive(D), mask(D)] = 3*D = 405 维
  总 model input: [x_t(135), inactive(135), reactive(135), mask(135)] = 540 维

带 ref_pose 的 VACE context:
  序列: [ref_pose, frame_0, frame_1, ..., frame_T-1]  (长度 1+T)
  ref_pose token:
    inactive = ref_pose 值 (D=135)
    reactive = 0 (D=135)
    mask = 0 (D=135, 表示"已知")
  model input 维度不变: 540

  x1 (flow target): [ref_pose, tgt_frame_0, ..., tgt_frame_T-1]  (长度 1+T)
  padding_mask: [True, True, ..., True, False, ...]  (ref_pose 位永远 True)
```
