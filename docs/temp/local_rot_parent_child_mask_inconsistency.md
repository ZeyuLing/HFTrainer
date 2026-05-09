# Local Rotation下父子节点Mask不一致问题分析与解决方案

## 1. 问题描述

### 1.1 现象

HyMotion M2M 在 local rotation 表示下执行修复时，adaptive mask 基于 MoGenDIT 去噪前后的逐关节差异独立判定每个关节是否需要修复。**父节点被 mask 但子节点不被 mask** 的情况大量存在。

在 local rotation 表示中，每个关节的旋转是相对于其父节点的。当我们修改父节点的 local rotation 但保持子节点不变时：
- 子节点的 **local rotation 不变**（因为不在 mask 中）
- 但子节点的 **world-space 姿态发生变化**（因为 FK 链中父节点变了）

这意味着即使子节点本身没有质量问题，修复父节点后子节点的全局姿态也会被间接改变，可能引入新的视觉偏差。

### 1.2 量化数据

在 CJGame 95 个问题样本上的统计：

| 指标 | 数值 |
|------|------|
| 父节点 masked、子节点未 masked 的总次数 | 39,634 / 73,797 = **53.7%** |
| 每帧平均受影响的未 masked 后代节点数 | **2.4 / 22** |

按运动链深度分布：

| 深度 | 父 masked 子未 masked 比例 |
|------|--------------------------|
| 1 (Hip/Spine1) | 67.5% |
| 2 (Knee/Spine2) | 39.8% |
| 3 (Ankle/Spine3) | 71.4% |
| 4 (Foot/Neck/Collar) | 53.0% |
| 5 (Shoulder) | 47.1% |
| 6 (Elbow) | 58.0% |
| 7 (Wrist) | 50.6% |

典型 case：`Base_Stand_Lobby_Performance01_023_360_720`
- Spine3 (j=9)：124 帧被 masked
- Neck (j=12)：0 帧被 masked（Spine3 的子节点）
- 124 帧中 Spine3 被修改但 Neck 保持原始 local rotation → Neck 的 world-space 姿态偏移

### 1.3 根因

Adaptive mask 的计算方式（MoGenDIT `compute_adaptive_mask`）是：
1. 对整个动作做轻度去噪
2. 比较去噪前后每个关节的 **axis-angle 差异**
3. 差异 > 0.15 rad 的关节标记为需要修复

这个过程对每个关节**独立**判定，**完全没有考虑 local rotation 的运动链依赖关系**。

## 2. 问题本质

在 local rotation 表示下，关节 j 的 world rotation 为：

```
R_world(j) = R_world(parent(j)) @ R_local(j)
```

如果 parent 被修改（Δparent），但 child 保持原始 R_local(child)：

```
R_world_new(child) = (R_world(parent) + Δparent) @ R_local(child)
                   = R_world_old(child) + Δparent @ R_local(child)
```

即 child 的 world rotation 被隐式修改了 `Δparent @ R_local(child)`。这个偏移会沿运动链向下传播并累积。

**对比 global rotation**：如果使用 global rotation 表示，每个关节的旋转独立于其父节点，修改一个关节不会影响其他任何关节。但 global rotation 有自身的问题（方差膨胀、SMPL 不兼容等），不是银弹。

## 3. 解决方案

### 方案 A：Mask 扩展 — 自动 mask 所有受影响的后代节点（推荐）

**核心思路**：如果一个关节被 mask，其所有后代节点也应该被 mask（至少在该帧上）。这样模型有机会为整个子树重新生成一致的 local rotation。

**实现**：

```python
def expand_mask_to_descendants(joint_mask):
    """Expand mask: if parent is masked, all descendants are also masked.

    Args:
        joint_mask: (T, 22) bool array
    Returns:
        expanded_mask: (T, 22) bool array
    """
    PARENT = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
    # Build children map
    children = {j: [] for j in range(22)}
    for j in range(22):
        if PARENT[j] >= 0:
            children[PARENT[j]].append(j)

    # BFS from each masked joint to all descendants
    expanded = joint_mask.copy()
    T = joint_mask.shape[0]
    for t in range(T):
        for j in range(22):
            if not joint_mask[t, j]:
                continue
            # BFS to mark all descendants
            stack = list(children[j])
            while stack:
                c = stack.pop()
                expanded[t, c] = True
                stack.extend(children[c])
    return expanded
```

**优点**：
- 实现简单，只需在 adaptive mask 计算后加一步后处理
- 保证运动链一致性：整个子树由模型重新生成
- 不需要修改模型架构或训练

**缺点**：
- Mask 比例增大（从 ~15% 可能增大到 ~30-40%），模型需要生成更多内容
- 如果 Pelvis (root) 被 mask，会导致全身所有关节都被 mask（退化为全身重新生成）

**缓解措施**：
- 对 Pelvis 特殊处理：如果 Pelvis 被 mask，不传播到所有后代，而是用 translation-only 修复
- 设置 mask 扩展的最大深度（如最多传播 2 层）
- 设置总 mask ratio 上限（如 50%），超过则不扩展

### 方案 B：分层修复 — 从根到叶依次修复

**核心思路**：按运动链从上到下分层修复。先修复根部关节（如 Pelvis, Spine），再修复中间关节（Shoulder, Knee），最后修复末端（Wrist, Foot）。每层修复后，将修复结果作为下一层的 known condition。

**实现**：

```python
DEPTH_GROUPS = [
    [0],              # depth 0: Pelvis
    [1, 2, 3],        # depth 1: L_Hip, R_Hip, Spine1
    [4, 5, 6],        # depth 2: L_Knee, R_Knee, Spine2
    [7, 8, 9],        # depth 3: L_Ankle, R_Ankle, Spine3
    [10, 11, 12, 13, 14],  # depth 4: Feet, Neck, Collars
    [15, 16, 17],     # depth 5: Head, Shoulders
    [18, 19],         # depth 6: Elbows
    [20, 21],         # depth 7: Wrists
]

def hierarchical_repair(motion, joint_mask, bundle, pipeline, device):
    """Repair layer by layer, root to leaf."""
    current_motion = motion.clone()

    for depth_joints in DEPTH_GROUPS:
        # Build mask for this layer: only joints at this depth that need repair
        layer_mask = np.zeros_like(joint_mask)
        for j in depth_joints:
            layer_mask[:, j] = joint_mask[:, j]

        if layer_mask.sum() == 0:
            continue

        # Run M2M repair with this layer's mask
        # current_motion already has parent joints fixed from previous layers
        repaired = run_single_repair(current_motion, layer_mask, bundle, pipeline, device)

        # Update current_motion with repaired values for masked joints
        mask_135 = expand_mask_to_135d(layer_mask, np.zeros(T))
        current_motion = current_motion * (1 - mask_135) + repaired * mask_135

    return current_motion
```

**优点**：
- 每层修复时，父节点已经是修复后的正确值
- 子节点修复时可以参考已修复的父节点上下文
- Mask 比例不会过大（每层只 mask 少量关节）

**缺点**：
- 需要多次前向推理（最多 8 层 = 8 次），速度慢
- 每层修复是独立的，层间没有全局一致性约束
- 实现较复杂

### 方案 C：后处理子节点补偿（最轻量）

**核心思路**：修复后，对于"父节点被修改但子节点未被 mask"的情况，计算父节点的 world rotation 变化量 Δ，并将子节点的 local rotation 做逆向补偿，使其 world-space 姿态保持不变。

**实现**：

```python
def compensate_children(original_135d, repaired_135d, joint_mask):
    """Post-hoc: adjust unmask children's local rot to preserve world pose.

    When parent is modified but child is not masked, the child's world pose
    shifts. We counter-rotate child's local rot to cancel the parent's change.
    """
    from scipy.spatial.transform import Rotation

    PARENT = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
    T = original_135d.shape[0]
    result = repaired_135d.copy()

    for t in range(T):
        for j in range(22):
            parent = PARENT[j]
            if parent < 0:
                continue
            # If parent was masked (modified) but child was NOT masked
            if joint_mask[t, parent] and not joint_mask[t, j]:
                # Get parent's original and repaired local rotations
                p_start = 3 + parent * 6
                orig_p_rot6d = original_135d[t, p_start:p_start+6]
                rep_p_rot6d = repaired_135d[t, p_start:p_start+6]

                # Convert to rotation matrices
                R_orig = rot6d_to_matrix(orig_p_rot6d)
                R_rep = rot6d_to_matrix(rep_p_rot6d)

                # Delta = R_rep @ R_orig^T (parent's world rotation change)
                delta = R_rep @ R_orig.T

                # Child's new local rot = delta^T @ child's original local rot
                # This cancels the parent's change in world space
                c_start = 3 + j * 6
                R_child = rot6d_to_matrix(result[t, c_start:c_start+6])
                R_child_compensated = delta.T @ R_child
                result[t, c_start:c_start+6] = matrix_to_rot6d(R_child_compensated)

    return result
```

**优点**：
- 不需要额外模型推理，纯几何运算
- 子节点的 world-space 姿态精确保持不变
- 实现简单

**缺点**：
- 只保证子节点的 world pose 不变，但修复后的 local rotation 可能不自然
- 如果父节点修复得不好，补偿后的子节点 local rotation 可能有物理不合理的值
- 只补偿直接子节点，多层传播需要递归处理
- 不能改善修复质量，只是避免退化

### 方案 D：混合方案（推荐实施方案）

**结合方案 A + C 的优势**：

1. **浅层扩展**（方案 A 的简化版）：如果父节点被 mask，自动扩展 mask 到**直接子节点**（不递归到所有后代）。这样模型有机会为父子一起重新生成。
2. **深层补偿**（方案 C）：对于扩展后仍然出现"祖先 masked、后代 unmasked"的情况（间隔超过 1 层），做后处理补偿。

```python
def smart_mask_expand(joint_mask, max_expand_depth=1):
    """Expand mask to direct children only (1 level)."""
    PARENT = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
    children = {j: [] for j in range(22)}
    for j in range(22):
        if PARENT[j] >= 0:
            children[PARENT[j]].append(j)

    expanded = joint_mask.copy()
    for depth in range(max_expand_depth):
        new_expanded = expanded.copy()
        for j in range(22):
            if expanded[:, j].any():
                for c in children[j]:
                    new_expanded[:, c] |= expanded[:, j]
        expanded = new_expanded
    return expanded

def repair_with_compensation(motion, joint_mask, trans_mask, bundle, pipeline, device):
    """Full repair pipeline with mask expansion + post-hoc compensation."""
    # Step 1: Expand mask to direct children
    expanded_mask = smart_mask_expand(joint_mask, max_expand_depth=1)

    # Step 2: Run M2M repair with expanded mask
    repaired = run_m2m_repair(motion, expanded_mask, trans_mask, bundle, pipeline, device)

    # Step 3: Post-hoc compensation for remaining unmasked descendants
    repaired = compensate_children(motion, repaired, expanded_mask)

    return repaired
```

**优势**：
- 直接子节点被模型一起修复（最重要的一层一致性）
- 更远的后代通过几何补偿保持 world pose
- Mask 扩展幅度可控（只扩展 1 层，不会膨胀到全身）
- 不需要多次前向推理

## 4. 实施建议

**推荐方案 D（混合方案）**，分两步实施：

### Phase 1：mask 扩展（改 `repair_eval_cjgame.py`）

在 `run_m2m_repair` 函数中，adaptive mask 加载后、expandto 135d 之前，插入 `smart_mask_expand`。

### Phase 2：后处理补偿（可选）

在保存 NPZ 之前，对 combined motion 做 `compensate_children`。

### Phase 3：训练侧改进（长期）

在 `PrepareM2MUniversalMask` 中增加 M7 策略的变体：生成散点 mask 后自动扩展到子节点，确保训练分布覆盖"父子同时 masked"的 pattern。

## 5. 对比：Global Rotation 下此问题不存在

在 global rotation 表示下，每个关节的旋转是世界坐标系绝对旋转，修改任何一个关节不会影响其他关节。因此：
- **不存在**父子 mask 不一致的问题
- Adaptive mask 可以对每个关节独立判定
- 这是 global rotation 的核心优势之一

但 global rotation 有方差膨胀问题（远端关节 Std 增大 2-6 倍），需要权衡。

## 6. 预期效果

| 指标 | 当前（无扩展） | 方案 D（1层扩展+补偿） |
|------|---------------|---------------------|
| 父 masked 子未 masked 比例 | 53.7% | ~15%（只剩间隔2+层的） |
| Mask 总比例 | ~15% | ~22%（增加约50%） |
| 需要额外推理 | 否 | 否 |
| 子节点 world pose 一致性 | ❌ | ✅ |
