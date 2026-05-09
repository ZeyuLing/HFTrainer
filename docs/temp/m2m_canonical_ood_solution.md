# HyMotion M2M: Canonical Pose OOD Problem in Transition Tasks — Analysis & Solution

> Created: 2026-03-31
> Status: Proposal (pending implementation)
> Related: `hftrainer/models/motion/CLAUDE.md`, `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py`

---

## 1. Problem Statement

### 1.1 Current Training Pipeline

HyMotion M2M 的所有训练数据在 `LoadSmplx55` 阶段经过 canonical 化处理：

- **首帧面朝 Z 轴正方向**（root orientation 的 yaw 被归零或对齐）
- **首帧 XZ 坐标为 (0, 0)**（translation 在 XZ 平面被平移到原点）
- **数据增强**：`LoadSmplx55` 中的 `transl_aug_prob=0.75`，以 75% 概率对整段动作做随机 yaw 旋转（[-180, +180] 度）+ XZ 偏移（std=1.0），但这只是**对整段动作施加统一变换**，不改变动作内部的相对结构

因此，模型在训练时看到的数据分布有一个隐含假设：**每段动作的起始状态都经过了 canonical 归一化**。

### 1.2 过渡任务 (Transition) 的 OOD 问题

当需要做两段动作之间的过渡（Transition / In-Between）时：

```
动作 A（canonical）→ [过渡区间，需要生成] → 动作 B（canonical）
```

**理想做法**：
1. 动作 A 正常播放，结尾时人物位于世界坐标 (x_A, 0, z_A)，面朝 yaw_A 方向
2. 过渡区间从 A 的结尾状态平滑过渡到 B 的开始状态
3. 动作 B 接在过渡之后播放

**问题**：
- 动作 B 也是 canonical 的（首帧在原点、面朝 Z 正方向）
- 当将 A 的结尾帧和 B 的首帧作为 known condition 输入 M2M 时：
  - A 的结尾帧：translation = (x_A, y_A, z_A)，root yaw = yaw_A
  - B 的首帧：translation = (0, 0, 0)，root yaw = 0
- **这种"跳跃"在训练数据中从未出现过** — 训练时 unmasked 区域的动作都是时间连续的，不会出现 translation 和朝向的突变
- 模型面对这种 OOD 输入，生成的过渡动作质量严重下降

### 1.3 更一般的 OOD 场景

问题不仅限于双段过渡，还包括：

| 场景 | OOD 表现 |
|------|---------|
| **长序列拼接**：A -> B -> C 三段串联 | B、C 段的 known condition 偏离 canonical |
| **指定世界坐标起点**：从任意 (x, z, yaw) 开始生成 | 首帧 translation/root_rot 偏离训练分布 |
| **Motion Prediction with context**：给定上一段结尾帧做 prediction | 上一段结尾不在原点 |
| **交互场景**：两人交互，B 人的 canonical 坐标系与 A 人不同 | B 人的 condition 帧 OOD |

---

## 2. Root Cause Analysis

### 2.1 为什么训练数据是 canonical 的

```
LoadSmplx55._process_one_person():
    1. 读取 NPZ -> abs_trans (T, 3), poses (T, 165)
    2. 若 do_aug: abs_trans = abs_trans @ R_y.T + offset  # 全局旋转+平移
    3. transl = process_transl(abs_trans, 'abs')  # 保持绝对坐标
    4. pose = process_smplx_pose(poses, 'rotation_6d', 'smpl_22')  # 含旋转增强
```

原始 NPZ 数据本身是 canonical 的（MotionHub 标准），增强只做统一变换，不改变动作序列内部的相对结构。因此模型学到的 translation 分布是：**首帧大致在 offset 附近（offset std=1.0），后续帧从首帧开始连续运动**。

### 2.2 为什么 data augmentation 不足以解决问题

当前 `LoadSmplx55` 的增强策略：
- `transl_aug_yaw_deg = 180.0`：随机 yaw 旋转 -> 模型已经见过各种朝向
- `transl_aug_offset_std = (1.0, 0.0, 1.0)`：随机 XZ 偏移 -> 模型已经见过非零首帧

**但是**，增强是**对整段动作统一施加的**。模型学到的是"整段动作可以在任意起点、任意朝向开始，但内部必须连续"。过渡任务的问题不在于首帧偏离原点，而在于：

1. **unmasked 区域内存在空间不连续**：A 结尾帧 -> [gap] -> B 首帧，中间有 translation 和 yaw 的突变
2. **两段 known condition 不在同一个 canonical 坐标系下**：A 帧在 A 的世界坐标系，B 帧在 B 的 canonical 坐标系（原点）

### 2.3 为什么不能只增大 augmentation offset

增大 `transl_aug_offset_std` 从 1.0 到 10.0 会让模型见到更大的绝对坐标，但这**不解决**问题：

1. 增强仍然是**全序列统一的** — 所有帧同步偏移
2. 模型仍然不会见到**序列内部的坐标跳变**
3. 更大的偏移可能恶化 normalization 质量（Mean/Std 基于 canonical 数据计算）

**核心洞察**：问题不在坐标的**幅度**，而在 condition 区域之间的**不连续性**。

---

## 3. Solution Design

解决方案分为两层，**兼容并互补**，建议同时实施：

- **方案 A（推理时）**：坐标系对齐 — 零训练改动，立即可用
- **方案 B（训练时）**：分布增强 — 让模型从根本上适应不连续 condition

长期方向还有方案 C（Relative Representation）和方案 D（Root Heading Decomposition），在本文档末尾讨论。

---

### 3.1 方案 A：推理时坐标系对齐 (Runtime Canonicalization)

**核心思想**：在推理时，将整个过渡问题变换到 canonical 坐标系下，让模型看到的输入与训练分布一致，生成完成后再变换回世界坐标系。

#### 3.1.1 Transition 场景标准流程

```
输入：
  - motion_A: 前段动作 (T_A, 135)，世界坐标系
  - motion_B: 后段动作 (T_B, 135)，canonical 坐标系（首帧原点、面朝Z+）
  - N_overlap_A: A 的尾部 overlap 帧数（作为 known condition）
  - N_overlap_B: B 的头部 overlap 帧数（作为 known condition）
  - T_transition: 过渡区间总帧数（含 overlap）

步骤：
  1. 提取 A 的结尾状态：
     - trans_A_end = motion_A[-1, 0:3]
     - root_rot_A_end = motion_A[-1, 3:9]
     - 从 root_rot_A_end 提取 yaw_A（绕 Y 轴旋转角）

  2. 将 B 的首帧从 canonical 变换到 A 结尾的坐标系：
     - B' = ApplyRigidTransform(motion_B, R_yaw=yaw_A, offset=trans_A_end)
     （使 B 的首帧对齐到 A 结尾附近，仅改变 root 朝向和 translation）

  3. 构建过渡片段的 canonical 版本：
     - 取 A 尾部 N_overlap_A 帧 + 中间空白 T_gen 帧 + B' 头部 N_overlap_B 帧
     - 对整个过渡片段做 inverse canonicalization：
       a. trans_canonical_origin = A 尾部第一帧的 translation
       b. yaw_canonical = A 尾部第一帧的 root yaw
       c. 将整个片段旋转到面朝 Z+、平移到 XZ 原点
     - 现在这个片段看起来就像一段普通的 canonical 动作，中间有 masked 区域

  4. 送入 M2M 模型做 In-Between 补全：
     - src_mask: A 尾部帧 mask=0, 中间帧 mask=1, B' 头部帧 mask=0
     - 模型在 canonical 分布下正常工作

  5. 将生成结果逆变换回世界坐标系：
     - ApplyRigidTransform(result, R_yaw=yaw_canonical, offset=trans_canonical_origin)

  6. 拼接：motion_A[:-N_overlap_A] + transition_result + motion_B'[N_overlap_B:]
```

#### 3.1.2 数学表述

设 A 尾部第一个 condition 帧（即过渡片段的首帧）在世界坐标系下的状态为：
- 位置：`p_0 = (x_0, y_0, z_0)`
- Root 朝向 yaw：`theta_0`

**Canonicalization 变换**：
```python
R_canon = R_y(-theta_0)            # 绕 Y 轴旋转 -theta_0（消去 yaw）
t_canon = -R_canon @ p_0           # 平移到原点（只取 XZ，保留 Y）

# 对过渡片段每帧 t:
transl'[t] = R_canon @ transl[t] + t_canon    # XZ 归零
root_rot'[t] = R_canon @ root_rot_matrix[t]   # root 朝向归零
body_rot'[t] = body_rot[t]                     # 非根关节不变（local rotation）
```

**逆变换**（生成完成后）：
```python
R_decanon = R_y(theta_0)
t_decanon = p_0

transl[t] = R_decanon @ transl'[t] + t_decanon
root_rot_matrix[t] = R_decanon @ root_rot'[t]
```

#### 3.1.3 Yaw 提取方法

从 rot6d (row-major) 提取 yaw 角：

```python
def extract_yaw_from_root_rot6d(root_rot6d: Tensor) -> Tensor:
    """从 root rot6d (6,) 提取绕 Y 轴的 yaw 角 (radians)。

    使用 geometry.py 的 rot6d_to_rotation_matrix（row-major native）：
      rot6d -> 3x3 matrix -> 提取 yaw

    对于 Y-up 坐标系 (SMPL convention):
      R_y(yaw) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
      sin_yaw = R[0,2], cos_yaw = R[2,2]
      yaw = atan2(R[0,2], R[2,2])
    """
    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        rot6d_to_rotation_matrix,
    )
    R = rot6d_to_rotation_matrix(root_rot6d)  # (..., 3, 3)
    yaw = torch.atan2(R[..., 0, 2], R[..., 2, 2])
    return yaw
```

**注意**：此 yaw 提取假设 pitch/roll 角较小。对于大部分人体运动（行走、跑步、坐、站），这个假设成立。对于体操等极端动作，需要使用完整的 Euler 分解。

#### 3.1.4 刚性变换函数

```python
def apply_rigid_transform_to_motion(
    motion: Tensor,           # (T, 135)
    R_yaw: Tensor,            # (3, 3)
    offset: Tensor,           # (3,)
) -> Tensor:
    """对 motion 的 translation 和 root rotation 施加 Y 轴刚性变换。

    - translation: t' = R @ t + offset
    - root rotation (dims 3:9): R_root' = R @ R_root
    - body rotations (dims 9:135): 不变（local rotation 不受世界坐标系影响）
    """
    motion = motion.clone()

    # Transform translation
    transl = motion[:, 0:3]  # (T, 3)
    motion[:, 0:3] = (R_yaw @ transl.unsqueeze(-1)).squeeze(-1) + offset

    # Transform root rotation
    root_rot6d = motion[:, 3:9]  # (T, 6)
    root_mat = rot6d_to_rotation_matrix(root_rot6d)       # (T, 3, 3)
    root_mat_new = R_yaw.unsqueeze(0) @ root_mat           # (T, 3, 3)
    motion[:, 3:9] = rotation_matrix_to_rot6d(root_mat_new)  # (T, 6)

    # Body joints (dims 9:135) are local rotations - unchanged
    return motion
```

**与 `LoadSmplx55` 的一致性**：上述变换逻辑与 `LoadSmplx55._process_one_person()` 中的增强完全一致：
- `abs_trans = abs_trans @ R_y.T + offset` 对应 translation 变换
- `apply_root_yaw_to_axis_angle(poses, R_y)` 对应 root rotation 变换
- body pose 不变

唯一区别：`LoadSmplx55` 在 axis-angle 空间做变换，推理时在 rot6d 空间做。需要使用 `geometry.py` 的 row-major 函数（与 `fk_utils.py` 的 torch path 一致）。

#### 3.1.5 实现位置

| 文件 | 改动 |
|------|------|
| **新建** `hftrainer/pipelines/motion/transition_utils.py` | `extract_yaw_from_root_rot6d()`, `build_yaw_rotation_matrix()`, `apply_rigid_transform_to_motion()`, `canonicalize_transition()`, `decanonicalize_transition()` |
| **修改** `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | `__call__()` 新增 `transition_mode` 分支 |
| **新建** `tools/m2m_transition.py` | CLI 工具：两个 NPZ + overlap 参数 -> 过渡 NPZ |
| **新建** `tests/smoke/test_m2m_transition.py` | 单元测试：round-trip, mask 构建 |

---

### 3.2 方案 B：训练时坐标系不连续增强 (Transition Augmentation)

**目标**：让模型在训练时就见过"condition 帧在不同坐标系"的情况。

#### 3.2.1 核心 Transform

```python
@TRANSFORMS.register_module()
class TransitionCoordAugmentation(BaseTransform):
    """对 inbetween 类型 mask 的后段 context 做坐标系变换。

    必须在 PrepareM2MUniversalMask 之后执行。
    仅当 src_mask 呈现 inbetween 模式（前段 unmask, 中间 mask, 后段 unmask）时生效。

    效果：后段 known condition 被随机旋转+平移，模拟"B 段来自不同坐标系"。
    tgt_motion 也做同样变换 — 模型需要学会生成坐标系过渡。

    Args:
        prob: 触发概率（默认 0.3）
        yaw_range_deg: yaw 旋转范围 (degrees)，默认 [-180, 180]
        offset_std: XZ 偏移标准差 (meters)，默认 2.0
    """

    def __init__(self, prob=0.3, yaw_range_deg=180.0, offset_std=2.0):
        self.prob = prob
        self.yaw_range_deg = yaw_range_deg
        self.offset_std = offset_std
```

#### 3.2.2 Inbetween 模式检测

```python
def _detect_inbetween(self, src_mask: Tensor) -> Optional[Tuple[int, int]]:
    """检测 src_mask 是否为 inbetween 模式。

    Returns:
        (past_end, future_start) 或 None

    Inbetween 模式特征：
      - frames [0, past_end) 全部 mask=0
      - frames [past_end, future_start) 全部 mask=1
      - frames [future_start, T) 全部 mask=0
    """
    T = src_mask.shape[0]
    # 每帧的 mask ratio
    frame_mask = src_mask.mean(dim=-1)  # (T,)
    is_masked = frame_mask > 0.5

    # 找第一个 masked 帧
    masked_indices = torch.where(is_masked)[0]
    if len(masked_indices) == 0:
        return None

    past_end = masked_indices[0].item()
    if past_end == 0:
        return None  # 没有前段 context

    # 找最后一个 masked 帧
    future_start = masked_indices[-1].item() + 1
    if future_start >= T:
        return None  # 没有后段 context

    # 验证中间全部 masked
    if not is_masked[past_end:future_start].all():
        return None  # 不是连续 mask

    return (past_end, future_start)
```

#### 3.2.3 变换逻辑

```python
def transform(self, results):
    src_mask = results.get('src_mask')
    if src_mask is None or np.random.rand() > self.prob:
        return results

    bounds = self._detect_inbetween(src_mask)
    if bounds is None:
        return results

    past_end, future_start = bounds
    T = src_mask.shape[0]

    # 采样随机变换
    yaw_deg = np.random.uniform(-self.yaw_range_deg, self.yaw_range_deg)
    yaw_rad = np.deg2rad(yaw_deg)
    offset = np.array([
        np.random.normal(0, self.offset_std),
        0.0,  # Y 轴不变
        np.random.normal(0, self.offset_std),
    ], dtype=np.float32)

    R_yaw = build_Ry_from_rad(yaw_rad)  # (3, 3)

    # 对后段 [future_start:T] 的 src_motion 和 tgt_motion 做变换
    for key in ['src_motion', 'tgt_motion']:
        motion = results[key]
        future_part = motion[future_start:T]
        future_transformed = apply_rigid_transform_numpy(
            future_part.numpy(), R_yaw, offset
        )
        results[key] = torch.cat([
            motion[:future_start],
            torch.from_numpy(future_transformed),
        ], dim=0)

    return results
```

#### 3.2.4 Config 修改

```python
# configs/hymotion_m2m/base_m2m.py
dataset = dict(
    pipeline=[
        dict(type='LoadSmplx55', ...),
        dict(type='RandomCropPadding', clip_len=360, ...),
        dict(type='PrepareM2MUniversalMask', ...),
        # NEW: 30% 概率对 inbetween 后段做坐标系变换
        dict(type='TransitionCoordAugmentation',
             prob=0.3,
             yaw_range_deg=180.0,
             offset_std=2.0),
    ],
)
```

#### 3.2.5 关键设计决策

**Q: 为什么对 tgt_motion 也做变换？**
A: tgt_motion 是模型的预测目标。如果只变 src_motion（condition）而不变 tgt_motion，模型会学到"condition 帧坐标系跳变时，target 仍在原始坐标系"，这不是我们想要的。我们希望模型学到"后段的 target 也在后段的坐标系下"，即**生成的过渡动作应该连接两个不同坐标系**。

**Q: 这会不会影响 loss？**
A: 会。被变换后的 tgt_motion 后段在新坐标系下，模型需要学会在 masked 区域生成平滑过渡。这正是 transition 任务的本质。初期 loss 可能略增（因为任务更难了），但长期有助于 transition 生成质量。

**Q: 仅对 30% 的 inbetween 样本生效，够吗？**
A: M3 temporal_contiguous 占总训练的 23%，其中 inbetween 是 5 种子模式之一（~4.6%）。30% 的增强概率意味着约 1.4% 的训练样本会有坐标系不连续。这个比例不大，不会显著影响正常训练，但足以让模型学到 transition 能力。如果效果不足，可以增大 prob 或扩展到 prediction/prefix 模式。

---

### 3.3 方案 C：Relative Translation Representation (长期方向)

**思路**：将 translation 从 absolute `(x, y, z)` 改为 relative `(dx, dy, dz)`（帧间位移）。

| 属性 | 当前 (abs) | 提议 (rel) |
|------|-----------|-----------|
| Translation dims [0:3] | 世界绝对坐标 | 帧间位移 |
| 首帧值 | 依赖 canonical + 增强 | 恒定 `(0, 0, 0)` |
| 训练分布 | 依赖首帧位置 | **天然平移不变** |
| Transition OOD | 存在 | 大幅缓解（但 yaw 仍需处理） |

**优势**：
- 从根本上消除 translation 维度的 OOD
- 与 HyMotion T2M 1.0 对齐（也用 `transl_type='rel'`）
- 减少对绝对坐标 normalization 的依赖

**劣势**：
- 需要重训模型（表示维度不变但语义变了），重新统计 Mean/Std
- 首帧绝对坐标信息丢失，推理时需要额外记录
- Root yaw 的 OOD 问题仍在（yaw 仍是绝对的）
- `LoadSmplx55` 已支持 `transl_type='rel'`，但 M2M 的 bundle/trainer/pipeline 需要相应调整

**实现路径**：
1. Config 改 `transl_type='rel'`
2. 重新统计 `_stats_rel/Mean.npy` 和 `Std.npy`
3. Bundle `decode_motion_from_latent()` 添加 rel -> abs 重建（需首帧 abs 坐标）
4. Pipeline 在 transition 模式下处理 rel 表示的拼接

**预计工作量**：1 周 + 重训

---

### 3.4 方案 D：Root Heading Decomposition (研究方向)

**思路**：将 root rotation 分解为 heading (yaw) + body rotation，heading 使用 relative 表示（帧间 yaw 角速度）。

```
当前 135 dim: [abs_transl(3), root_rot6d(6), body_rot6d(126)]
提议 137 dim: [rel_transl(3), heading_vel(1), root_body_rot6d(6), body_rot6d(126), heading(1)]
```

- **heading_vel**：帧间 yaw 角变化量，天然不依赖绝对朝向
- **root_body_rot6d**：去掉 heading 后的 root 旋转（pitch+roll），方差更小

参考：MDM, MotionDiffuse, HumanMAC 等工作都有类似设计。

**评估**：改动大，需要完全重定义 motion representation，属于下一代模型的设计方向。

---

## 4. Implementation Plan

### Phase 1: 推理时 Canonicalization (方案 A) — 立即可用

| 步骤 | 文件 | 改动 | 工作量 |
|------|------|------|--------|
| 1 | 新建 `hftrainer/pipelines/motion/transition_utils.py` | `extract_yaw_from_root_rot6d()`, `build_yaw_rotation_matrix()`, `apply_rigid_transform_to_motion()`, `canonicalize_transition()`, `decanonicalize_transition()` | 0.5d |
| 2 | 修改 `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | `__call__()` 新增 `transition_mode` 分支，调用 transition_utils | 0.5d |
| 3 | 新建 `tools/m2m_transition.py` | CLI 工具：输入两个 NPZ + overlap 参数，输出过渡 NPZ | 0.5d |
| 4 | 新建 `tests/smoke/test_m2m_transition.py` | round-trip 测试、mask 构建测试、yaw 提取测试 | 0.5d |

**总计**：~2 天

### Phase 2: 训练时增强 (方案 B) — 需要重训

| 步骤 | 文件 | 改动 | 工作量 |
|------|------|------|--------|
| 1 | 新建 `hftrainer/datasets/motion/motionhub/transforms/transition_coord_aug.py` | `TransitionCoordAugmentation` transform | 1d |
| 2 | 修改 `universal_mask.py` | 在 results 中记录 mask sub-mode 方便下游判断 | 0.5d |
| 3 | 修改 config | dataset pipeline 添加 `TransitionCoordAugmentation` | 0.5d |
| 4 | 新建测试 | 验证增强后数据分布、可视化 | 0.5d |
| 5 | 重训 + 评估 | 对比 transition 生成质量 | 视训练时间 |

**总计**：~3 天 + 训练时间

### Phase 3: Relative Translation (方案 C) — 长期

| 步骤 | 工作量 |
|------|--------|
| Mean/Std 重统计 | 0.5d |
| Bundle/Trainer/Pipeline 适配 | 2d |
| Config 调整 + 测试 | 1d |
| 重训 + FID/diversity 评估 | 视训练时间 |

**总计**：~1 周 + 训练时间

---

## 5. Validation Plan

### 5.1 Phase 1 验证

| 测试 | 方法 | 预期 |
|------|------|------|
| Canonicalization round-trip | `canon -> decanon` 应恢复原始 motion | max error < 1e-5 |
| Yaw 提取准确性 | 构造已知 yaw 的旋转矩阵，验证提取 | exact match |
| Transition 质量（主观） | 两段不同 yaw 动作的过渡，可视化 | 平滑、无跳变 |
| Transition 质量（定量） | canonicalized vs raw 输入的过渡 MPJPE | canonicalized 更低 |
| 边界连续性 | 过渡区与 A/B 交界处的速度/加速度跳变 | 跳变 < 阈值 |

### 5.2 Phase 2 验证

| 测试 | 方法 | 预期 |
|------|------|------|
| 增强数据分布 | 可视化增强前后的 translation/yaw 分布 | 后段 context 有坐标系跳变 |
| 训练收敛 | loss curve 对比（有 vs 无增强） | 不应显著上升 |
| Transition（不做 canonicalization） | 直接用非 canonical 输入推理 | 质量应接近 Phase 1 |
| 常规任务回归 | FID/diversity 对比 | 不应退化 |

---

## 6. Risk Assessment

| 风险 | 影响 | 缓解 |
|------|------|------|
| Yaw 提取有误（pitch 非零时） | Canonicalization 不完全 | 使用完整 Euler 分解而非简化公式 |
| Mean/Std 在 canonical 假设下计算 | 非 canonical 输入的 normalize 不准 | Phase 1 中 canonicalize 后再 normalize，所以安全 |
| Phase 2 增强影响训练稳定性 | Loss spike | 控制增强概率（30%），仅对 inbetween 生效 |
| B 段的 Y 坐标（高度）与 A 不匹配 | 生成时高度不连续 | 可选 Y 轴也做对齐 |
| 长序列多段拼接的误差累积 | 后段越来越偏 | 每段独立 canonicalize |
| `geometry.py` row-major vs `rotation_convert.py` col-major 混用 | 旋转结果错误 | 推理侧只用 `geometry.py`（row-major native），与 `fk_utils.py` torch path 一致 |

---

## 7. Cross-Reference

- 本方案的 rotation 变换逻辑与 `LoadSmplx55` (`load_smplx.py`) 中的增强完全一致
- rot6d 转换路径与 `fk_utils.py` 中的 torch path 一致（使用 `geometry.py` row-major native）
- VACE conditioning 不受影响 — canonicalization 在 normalize 之前完成
- 与 V4 (Mask-Aware Flow Matching) 方案正交兼容
- 与 V5 (Global Rotation Space) 方案正交兼容

---

## Appendix A: Current Data Augmentation Analysis

`LoadSmplx55` augmentation parameters:
```python
transl_aug_prob = 0.75       # 75% chance of augmentation
transl_aug_yaw_deg = 180.0   # yaw rotation range: [-180, 180] degrees
transl_aug_offset_std = (1.0, 0.0, 1.0)  # XZ offset std=1.0m, Y=0
```

Distribution implications:
- 75% of samples: first frame at random (x, z) with std=1.0m, random yaw in [-180, 180]
- 25% of samples: first frame at canonical origin, facing Z+
- **All samples**: internal motion is continuous (no intra-sequence coordinate breaks)

## Appendix B: Related Work Comparison

| Method | Representation | Transition Handling |
|--------|---------------|-------------------|
| **MDM** | Relative root + joint positions | Natural via relative repr |
| **MotionDiffuse** | Root velocity + heading vel | Natural via velocity repr |
| **KIMODO** | Global rotation + global position | Imputation handles any start pose |
| **MoGenDiT** | Local rotation + local position | Mask-aware noise + replacement |
| **UMO** | Local rotation (201-dim) | Frame-level [P]/[G]/[E] meta-ops |
| **HyMotion M2M (ours)** | Local rotation + abs translation | **Current: OOD; Proposed: A+B** |

MDM 和 MotionDiffuse 通过 relative representation 天然避免了此问题。KIMODO 通过 global representation + imputation 处理。MoGenDiT 通过 mask-aware noise 使得推理时 replacement 有效。UMO 不直接支持 transition（frame-level only）。

我们的方案 A（runtime canonicalization）类似于 KIMODO 的世界坐标系对齐思路，方案 B（training augmentation）类似于 MoGenDiT 的 noise-augmented 训练，方案 C（relative repr）对齐 MDM/MotionDiffuse 的设计哲学。
