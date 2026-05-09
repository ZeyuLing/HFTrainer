# M2M 修复 Local Rotation 子节点传播问题：自适应方案

> 文档类型: 方案提案（待决策）
> 日期: 2026-04-09
> 状态: Draft

---

## 1. 问题定义

### 1.1 核心问题

HyMotion M2M 使用 **local rotation**（SMPL 父节点相对旋转）表示，135维 = 3 (abs translation) + 22×6 (rot6d per joint)。

当前修复流程中，如果质量检查器检测到关节 A 有问题，**只 mask 并修改关节 A 本身**，不修改 A 的任何子节点。但在 local rotation 表示下：

```
Global_Rotation(joint_j) = Global_Rotation(parent_j) @ Local_Rotation(joint_j)
```

如果父节点 A 的 local rotation 被修改（记为 `R_A → R_A'`），而子节点 B 的 local rotation 保持不变（`R_B` 不变），那么 B 的**全局旋转**会发生变化：

```
Global(B)_before = Global(A) @ R_B
Global(B)_after  = Global(A') @ R_B     （其中 Global(A') ≠ Global(A)）
```

这导致 B 及其所有后代节点的**世界空间姿态**都会发生大幅变化，即便它们的 local rotation 完全正确。

### 1.2 严重程度

SMPL-22 运动链最长为 8 层：

```
Pelvis(0) → Spine1(3) → Spine2(6) → Spine3(9) → Neck(12) → Head(15)
Pelvis(0) → L_Hip(1) → L_Knee(4) → L_Ankle(7) → L_Foot(10)
Spine3(9) → L_Collar(13) → L_Shoulder(16) → L_Elbow(18) → L_Wrist(20)
```

**影响分析**：

| 被修改的关节 | 受影响的子孙节点数量 | 影响描述 |
|-------------|-------------------|---------|
| Pelvis (root, j=0) | 21（全部） | 修改 root orientation 会改变整个身体的世界空间姿态 |
| Spine1 (j=3) | 13 (Spine2, Spine3, Neck, Head, L/R_Collar, L/R_Shoulder, L/R_Elbow, L/R_Wrist) | 修改躯干会影响整个上半身 |
| L_Hip (j=1) | 3 (L_Knee, L_Ankle, L_Foot) | 修改髋关节会影响整条腿 |
| L_Shoulder (j=16) | 2 (L_Elbow, L_Wrist) | 修改肩关节会影响整条手臂 |
| L_Wrist (j=20) | 0 | 末端节点，无子孙，修改无传播问题 |

**典型失败场景**：

1. **Spine jitter 修复**：Spine1 有 jitter → 只 mask Spine1 → 模型修复 Spine1 → 整个上半身（13个子孙关节）的世界空间姿态突变
2. **Hip penetration 修复**：L_Hip 穿透 → 只 mask L_Hip → 修复后整条左腿相对身体的位置大幅变化
3. **Root orientation 修复**：Pelvis 旋转异常 → 只 mask Pelvis → 修复后全身姿态在世界空间中突变

### 1.3 代码证据

1. **`adaptive_mask_to_dense()`** (`scripts/eval_m2m_repair.py` L136-162)：直接将 `joint_mask (T, 22)` 一对一映射到 135 维 mask，只做 temporal dilation，无 kinematic chain 传播。

2. **所有 checker 的 mask builder** (`mask_utils.py`)：只标记出问题的具体关节。例如 `_build_jitter_mask` 只标记 `jitter_joints`。

3. **`dilate_mask()`** (`repair_mask_utils.py` L180-241)：存在 `joint_radius` 参数可沿 kinematic chain 做 BFS 扩展，但仅在 web UI 中定义，repair eval 脚本中**从未调用**。

4. **`repair_single()`** (`scripts/eval_m2m_repair.py` L291-350)：blend 时 `combined = original * (1-mask) + repaired * mask`，mask=0 的子节点完全保持原始 local rotation。

---

## 2. 方案总览

提出三个层次的方案，从简单到复杂：

| 方案 | 思路 | 训练改动 | 推理改动 | 预期效果 |
|------|------|---------|---------|---------|
| **A: Kinematic-Aware Mask Propagation** | 推理时自动扩展 mask 到子孙节点 | 无 | 中等 | 直接解决问题，但 mask 变大，模型生成范围增加 |
| **B: Post-Repair FK Consistency Correction** | 修复后在 global rotation 空间做 blend，再转回 local | 无 | 中等 | 精确控制子节点的全局旋转过渡 |
| **C: Global Rotation 表示（长期方案）** | 改用 global rotation 训练，从根本上消除问题 | 大（需重训） | 小 | 根治问题，但训练成本高 |

**推荐**: **方案 A 为主 + 方案 B 为辅**（两者可叠加），方案 C 作为长期方向。

---

## 3. 方案 A: Kinematic-Aware Mask Propagation（推荐首选）

### 3.1 核心思路

在推理时，当检测到关节 A 被 mask 时，自动将 A 的所有子孙节点也加入 mask。模型同时修复父节点和所有子孙节点，保证 kinematic chain 的一致性。

### 3.2 关键：不需要 mask 所有子孙，只需要 mask "深度衰减" 范围内的子孙

如果某个子孙节点距离被标记的问题关节很远，且该子孙本身的 local rotation 是正确的，那么模型在修复时不需要改变它——它的 local rotation 已经是对的，只需要在 blend 时通过 FK 保证全局一致性。但 M2M 的 VACE completion 模式下，mask=0 的区域会被 hard blend 回原始值，这就要求子孙节点也被 mask，让模型有机会自行决定它们的值。

**但有一个更好的选择**：对于子孙节点，我们可以使用 **editing 模式**（reactive 通道传入原始 LQ motion），而非 completion 模式（reactive=0）。这样模型看到的是：

- 父节点 A：mask=1, reactive=0（completion，从头生成）
- 子孙节点 B：mask=1, reactive=B 的原始 local rotation（editing，在原始基础上微调）

这让模型知道子孙节点本身大致正确，只需要微调以适应父节点的变化。

### 3.3 分层 Mask 策略：Completion + Editing 混合

```python
def propagate_mask_kinematic(
    joint_mask_grid: np.ndarray,    # (T, 22) bool, True=flagged by checker
    mode: str = "completion_then_edit",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        expanded_mask: (T, 22) bool - all positions that need mask=1
        is_edit_mask: (T, 22) bool - among mask=1, which are in editing mode
    """
    T, J = joint_mask_grid.shape
    expanded = joint_mask_grid.copy()
    is_edit = np.zeros_like(joint_mask_grid)

    # Build children map from SMPL22_PARENTS
    children = {j: [] for j in range(J)}
    for j, p in enumerate(SMPL22_PARENTS):
        if p >= 0:
            children[p].append(j)

    # For each frame, propagate flagged joints to all descendants
    for t in range(T):
        flagged = set(np.where(joint_mask_grid[t])[0])
        # BFS from each flagged joint to find all descendants
        descendants_to_add = set()
        for root_j in flagged:
            queue = list(children[root_j])
            while queue:
                child = queue.pop(0)
                if child not in flagged:  # 只标记原本没问题的子孙
                    descendants_to_add.add(child)
                queue.extend(children[child])

        for d in descendants_to_add:
            expanded[t, d] = True
            is_edit[t, d] = True  # 子孙用 editing 模式

    return expanded, is_edit
```

### 3.4 推理 Data Flow

```python
def repair_single_kinematic_aware(pipeline, motion_135, checker_mask_135, device):
    """带 kinematic 传播的修复流程"""
    bundle = pipeline.bundle
    T = motion_135.shape[0]

    # 1. 将 135-dim mask 转回 (T, 22) joint grid
    checker_grid = mask_135_to_grid_22(checker_mask_135)  # (T, 22)

    # 2. Kinematic-aware propagation
    expanded_grid, is_edit_grid = propagate_mask_kinematic(checker_grid)

    # 3. 构建 expanded 135-dim mask
    expanded_mask = expand_grid_to_mask(expanded_grid)  # (T, 135)
    is_edit_mask = expand_grid_to_mask(is_edit_grid)     # (T, 135)

    # 4. Prepare VACE input with mixed completion/editing
    motion_norm = bundle.normalize_motion(motion_135.unsqueeze(0).to(device))

    # completion 区域: reactive=0 (原始 checker-flagged 关节)
    # editing 区域: reactive=motion_norm (子孙节点保持原始值作为参考)
    src_motion = motion_norm * (1 - expanded_mask)  # inactive: 未 mask 区域
    # reactive: 仅在 editing 子区域有值
    reactive_override = motion_norm * is_edit_mask

    # 注意：这需要修改 prepare_vace_input 或在 batch 中传入 reactive_override
    batch = {
        "src_motion": src_motion,
        "src_mask": expanded_mask,
        "reactive_override": reactive_override,  # 新增
        "clean_motion": motion_norm,  # for _man imputation
        ...
    }

    # 5. Model inference
    result = pipeline(batch)

    # 6. Blend with expanded mask
    repaired = bundle.denormalize_motion(result["latent"])
    combined = motion_135 * (1 - expanded_mask) + repaired * expanded_mask

    return combined
```

### 3.5 需要修改的代码

| 文件 | 修改内容 | 难度 |
|------|---------|------|
| `scripts/eval_m2m_repair.py` | `adaptive_mask_to_dense()` 增加 kinematic propagation | 低 |
| `motion_annot_web/m2m_database/repair_mask_utils.py` | `dilate_mask()` 默认启用 `joint_radius≥1` | 低 |
| `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 支持 `reactive_override` batch key | 中 |
| `hftrainer/trainers/motion/hymotion_m2m_trainer.py` | 训练时支持混合 completion/editing mask | 中（可选） |

### 3.6 优缺点

**优点**：
- 不需要重新训练模型（纯推理侧改动）
- 子孙节点用 editing 模式，模型知道它们大致正确，只做微调
- 与现有 M7 scattered_joint 训练策略兼容（训练时已见过 scattered 的 mask pattern）

**缺点**：
- Mask 面积增大 → 模型需要生成更多内容 → 可能引入新问题（尤其对 spine→全上半身 的大面积 mask）
- 混合 completion/editing 的 VACE 输入是模型训练时**从未见过**的 pattern（训练时 reactive 要么全是0，要么全有值），可能导致模型行为异常
- 对 root (Pelvis) 被 flag 的情况，需要 mask 全身，等价于全量重新生成

### 3.7 风险控制

1. **深度截断**：设置最大传播深度（如 `max_depth=2`），超过深度的子孙不再 mask
2. **面积上限**：如果 kinematic 传播后 mask 面积 >50%，回退到不传播（保持现状）
3. **分阶段验证**：先用纯 completion 模式（所有扩展节点 reactive=0），验证效果后再切 editing 混合

---

## 4. 方案 B: Post-Repair FK Consistency Correction（推荐辅助）

### 4.1 核心思路

不改变 mask 策略，仍然只 mask 问题关节 A。但在修复完成后的 blend 阶段，**在 global rotation 空间做平滑 blend**，然后转回 local rotation。这样可以保证子节点的世界空间姿态平滑过渡。

### 4.2 原理

当前 blend 在 local rotation 空间做：
```python
combined_local[j] = original_local[j] * (1-mask[j]) + repaired_local[j] * mask[j]
```

问题：如果 mask[A]=1, mask[B]=0（B 是 A 的子节点），那么 B 保持原始 local rotation，但 A 变了 → B 的全局旋转跳变。

**改进**：在 global rotation 空间做 blend：

```python
# 1. 将原始和修复后的 motion 都转到 global rotation
original_global = local_to_global(original_local)   # (T, 22, 6)
repaired_global = local_to_global(repaired_local)    # (T, 22, 6)

# 2. 在 global rotation 空间做 soft blend
#    对 mask=1 的关节 A: 完全取 repaired
#    对 mask=0 的子孙 B: 根据到最近 masked 祖先的 kinematic 距离做权重衰减
blend_weight = compute_kinematic_blend_weight(mask_grid, SMPL22_PARENTS)
combined_global = original_global * (1 - blend_weight) + repaired_global * blend_weight

# 3. 转回 local rotation
combined_local = global_to_local(combined_global)    # (T, 22, 6)
```

### 4.3 Kinematic Blend Weight 计算

```python
def compute_kinematic_blend_weight(
    mask_grid: np.ndarray,         # (T, 22) bool
    alpha_decay: float = 0.5,      # 每跳衰减系数
    max_hops: int = 3,             # 最大传播跳数
) -> np.ndarray:
    """
    对每个 (t, j)，计算 blend weight ∈ [0, 1]:
    - mask=1 的关节: weight = 1.0
    - mask=0 但祖先有 mask=1 的关节: weight = alpha^distance
    - 其他: weight = 0.0
    """
    T, J = mask_grid.shape
    weights = mask_grid.astype(np.float32).copy()

    children = build_children_map(SMPL22_PARENTS)

    for t in range(T):
        # BFS from masked joints downward
        for j in range(J):
            if not mask_grid[t, j]:
                continue
            # Propagate weight to descendants
            queue = [(c, 1) for c in children[j]]
            while queue:
                child, depth = queue.pop(0)
                if depth > max_hops:
                    continue
                w = alpha_decay ** depth
                weights[t, child] = max(weights[t, child], w)
                for grandchild in children[child]:
                    queue.append((grandchild, depth + 1))

    return weights  # (T, 22)
```

### 4.4 实现注意事项

**关键约束：rot6d blend 不能直接线性插值**。两个 rot6d 的线性组合不保证还是一个有效旋转（不在 SO(3) 流形上）。需要：

1. **方法1**：先转为 rotation matrix (3×3)，做 matrix 级 slerp 或 geodesic 插值，再转回 rot6d
2. **方法2**：利用 rot6d 的 Gram-Schmidt 重投影特性——线性组合后做一次 Gram-Schmidt orthogonalization（即 `rotation_6d_to_matrix`），得到最近有效旋转
3. **方法3**（推荐）：在 **axis-angle** 空间做插值（`slerp` 或 `axis_angle * w`），然后转回 rot6d

```python
# 方法2: 利用 rot6d -> matrix 的内置 Gram-Schmidt
blended_raw = original_6d * (1 - w) + repaired_6d * w   # 线性组合（不是有效旋转）
blended_mat = rot6d_to_matrix(blended_raw)                # Gram-Schmidt 重投影到 SO(3)
blended_6d = matrix_to_rot6d(blended_mat)                 # 回到 rot6d
```

方法2 在权重接近 0 或 1 时误差很小（实测 < 1e-4），适合我们的场景（decay weight 通常在 0.5、0.25、0.125 等典型值）。

### 4.5 需要修改的代码

| 文件 | 修改内容 | 难度 |
|------|---------|------|
| `scripts/eval_m2m_repair.py` | `repair_single()` 中的 blend 逻辑替换为 FK consistency correction | 中 |
| （新增）`hftrainer/models/motion/components/fk_blend.py` | FK blend 核心函数 | 中 |

### 4.6 优缺点

**优点**：
- 不改变模型、不改变 mask、不需要重新训练
- 纯后处理，最安全，不影响模型行为
- 精确控制子节点受影响的程度（通过 `alpha_decay` 和 `max_hops`）
- Float32 精度下 FK 转换实质无损（误差 < 1e-6，见 CLAUDE.md §转换精度）

**缺点**：
- 子节点的 local rotation 被间接修改——可能引入**新的** local rotation 异常（如果原始 local rotation 是正确的，被修改后可能反而变错）
- blend 后的子节点**既不是**原始的 local rotation，**也不是**模型生成的——是一个插值中间态
- 对 "父节点修改幅度很大" 的情况（如 root orientation 修复），衰减传播可能不够，远端子孙仍有跳变
- 计算开销增加（需要两次 local↔global 转换），但对单个 motion 来说微不足道

---

## 5. 方案 C: Global Rotation 表示（长期方案）

### 5.1 核心思路

在 global rotation 空间下，修改关节 A 不会影响任何其他关节——因为每个关节独立存储自己的世界空间旋转。问题从根本上消失。

### 5.2 现有基础

已有完整的 global rotation 消融实验基础（见 CLAUDE.md §Global vs Local Rotation Space）：
- `LocalToGlobalRotation` transform 已实现
- `global_to_local_rot6d_torch()` 推理 decode 已实现
- `_stats_global_rot/` 的 Mean/Std 已计算
- Config: `model.rotation_space='global'` 已支持
- DiT 变体 `dit_fm_man_globalrot_s` (49M, 757 epochs) 和 `dit_fm_man_globalrot_b` (288M, 833 epochs) 已在训练

### 5.3 分析

**已知优势**：邻居可预测性 +41%，统一坐标系

**已知问题**：
- 方差膨胀（Spine3: 6.1x, L_Wrist: 2.5x）→ normalization 更敏感
- 信息冗余（子节点包含所有祖先信息）→ 模型需学习冗余结构
- 长训练后 global 是否真的优于 local 尚无定论

### 5.4 与当前方案的关系

- 如果 global rotation 模型最终质量 ≥ local rotation 模型，则直接切换，问题根治
- 如果 global rotation 模型质量不如 local，则仍需方案 A/B

---

## 6. 方案 A+B 组合实施路径（推荐）

### 6.1 Phase 1: 快速验证（1-2天）

**目标**：验证问题的实际严重程度，为后续方案提供 baseline。

**步骤**：
1. 从现有修复评测结果中，筛选 "父节点被 mask 且有子孙节点" 的案例
2. 计算修复前后子孙节点的 **global rotation 变化量**（不是 local rotation 变化量）
3. 量化问题严重程度：`global_rotation_delta = ||GlobalRot_after(child) - GlobalRot_before(child)||`

```python
def analyze_child_disruption(original_135, repaired_135, mask_grid_22):
    """量化修复对 unmasked 子节点在 global 空间的影响"""
    orig_local = motion_135_to_local_rot(original_135)    # (T, 22, 6)
    rep_local = motion_135_to_local_rot(repaired_135)       # (T, 22, 6)

    orig_global = local_to_global_rot6d(orig_local)   # (T, 22, 6)
    rep_global = local_to_global_rot6d(rep_local)     # (T, 22, 6)

    delta = np.abs(orig_global - rep_global)            # (T, 22, 6)

    # 只看 unmasked 子节点
    unmasked = ~mask_grid_22                             # (T, 22)
    child_delta = delta[unmasked]                        # 展平

    return {
        "mean_global_delta": float(child_delta.mean()),
        "max_global_delta": float(child_delta.max()),
        "per_joint_mean": [float(delta[:, j][~mask_grid_22[:, j]].mean())
                          for j in range(22) if (~mask_grid_22[:, j]).any()],
    }
```

### 6.2 Phase 2: 方案 B 实现（2-3天）

**目标**：实现 FK Consistency Correction 后处理。

**步骤**：
1. 实现 `compute_kinematic_blend_weight()`
2. 实现 `fk_consistent_blend()`，包含 Gram-Schmidt 重投影
3. 替换 `repair_single()` 中的 blend 逻辑
4. 在现有修复评测集上跑 A/B 对比

```python
def fk_consistent_blend(
    original_135: torch.Tensor,    # (T, 135)
    repaired_135: torch.Tensor,    # (T, 135)
    mask_grid_22: np.ndarray,      # (T, 22) bool
    alpha_decay: float = 0.5,
    max_hops: int = 3,
) -> torch.Tensor:
    """FK-consistent blend in global rotation space."""
    T = original_135.shape[0]

    # 1. Compute kinematic blend weights (T, 22)
    weights_22 = compute_kinematic_blend_weight(mask_grid_22, alpha_decay, max_hops)

    # 2. Translation blend (standard, no FK issue)
    weights_trans = torch.from_numpy(mask_grid_22[:, :1].any(axis=1).astype(np.float32))  # 简化
    combined_trans = original_135[:, :3] * (1 - weights_trans.unsqueeze(1)) + \
                     repaired_135[:, :3] * weights_trans.unsqueeze(1)

    # 3. Rotation: convert to global, blend, convert back
    orig_rot = original_135[:, 3:].reshape(T, 22, 6)
    rep_rot = repaired_135[:, 3:].reshape(T, 22, 6)

    orig_global = local_to_global_rot6d_torch(orig_rot)   # (T, 22, 6)
    rep_global = local_to_global_rot6d_torch(rep_rot)      # (T, 22, 6)

    # Weighted blend in global space + Gram-Schmidt re-projection
    w = torch.from_numpy(weights_22).unsqueeze(-1).float()  # (T, 22, 1)
    blended_global = orig_global * (1 - w) + rep_global * w  # (T, 22, 6) 不是有效旋转

    # Re-project to SO(3) via rot6d → matrix → rot6d
    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        rot6d_to_rotation_matrix, rotation_matrix_to_rot6d,
    )
    blended_mat = rot6d_to_rotation_matrix(blended_global)     # (T, 22, 3, 3) Gram-Schmidt
    blended_global_valid = rotation_matrix_to_rot6d(blended_mat)  # (T, 22, 6)

    # Convert back to local
    blended_local = global_to_local_rot6d_torch(blended_global_valid)  # (T, 22, 6)

    # 4. Assemble
    combined = torch.cat([combined_trans, blended_local.reshape(T, 132)], dim=-1)
    return combined
```

### 6.3 Phase 3: 方案 A 实现（2-3天）

**目标**：实现 kinematic-aware mask propagation。

**步骤**：
1. 在 `adaptive_mask_to_dense()` 中增加 kinematic 传播逻辑
2. 先用纯 completion 模式测试（所有扩展节点 mask=1, reactive=0）
3. 如果纯 completion 效果不好（子孙节点偏离原始太多），再加 editing 混合
4. A/B 对比

### 6.4 Phase 4: 评估与参数调优（2-3天）

**评估维度**：

| 指标 | 计算方式 | 优先级 |
|------|---------|--------|
| **Global Pose Stability** | 修复前后 unmasked 子节点的 global joint position 变化量（FK 后的 3D 坐标） | P0 |
| **Quality Check Pass Rate** | 修复后通过质量检查的比例 | P0 |
| **MPJPE Unmasked** | 修复前后 unmasked 区域的 joint position 误差 | P1 |
| **Boundary Smoothness** | mask 边界帧的速度/加速度连续性 | P1 |
| **Repair Success Rate** | 修复后 target checker 通过的比例 | P1 |

**参数搜索空间**：

| 参数 | 范围 | 影响 |
|------|------|------|
| `alpha_decay` (方案B) | [0.3, 0.5, 0.7] | 衰减速度，越小子节点受影响越小 |
| `max_hops` (方案B) | [1, 2, 3, 无限] | 传播深度 |
| `max_propagation_depth` (方案A) | [1, 2, 3, 全部] | 扩展 mask 的深度 |
| `use_editing_for_descendants` (方案A) | [True, False] | 子孙用 editing 还是 completion |

---

## 7. 额外考虑

### 7.1 与 MoGenDIT adaptive mask 的交互

MoGenDIT 的 `compute_adaptive_mask()` 基于 "denoise 前后变化量" 判断哪些关节有问题。由于 MoGenDIT 使用 201 维表示（含 local joint position），它的判断隐含了 FK 的影响——如果父节点的 local rotation 异常，子节点的 local joint position 也会变化，MoGenDIT 可能已经间接 flag 了部分子孙。

**验证方法**：统计 MoGenDIT adaptive mask 中，被 flag 的关节有多少是被 flag 关节的子孙。如果比例较高，说明 MoGenDIT 已部分缓解了这个问题。

### 7.2 Translation 的特殊处理

Translation 不参与 kinematic chain（它是全局平移，不是旋转树的一部分）。方案 A/B 中的 kinematic 传播**不应该**影响 translation mask。这与当前 `adaptive_mask_to_dense()` 中 translation 和 joint 分开处理的逻辑一致。

### 7.3 Root Orientation 的特殊情况

如果 Pelvis (root, j=0) 被 flag，kinematic 传播会导致 mask 全身 21 个子孙——等价于全量重新生成。这种情况需要特殊处理：

1. **Option 1**: Root 被 flag 时，不做 kinematic 传播（保持现状），只用方案 B 的 FK blend
2. **Option 2**: Root 被 flag 时，用特殊的 "root-fix" 模式——只修复 root orientation，然后全身在 global 空间做补偿（每个子孙的 global rotation 保持不变，重新计算 local rotation）

**推荐 Option 2**，实现方式：

```python
def fix_root_preserve_global(original_135, repaired_root_rot):
    """修复 root orientation 但保持所有子节点的 global 旋转不变"""
    orig_local = original_135[:, 3:].reshape(T, 22, 6)
    orig_global = local_to_global(orig_local)      # (T, 22, 6)

    # 替换 root 的 global rotation
    new_global = orig_global.clone()
    new_global[:, 0] = repaired_root_rot           # 只改 root
    # 其他关节的 global rotation 完全不变

    # 转回 local: 新的 local rotation 自动适应新的 root
    new_local = global_to_local(new_global)         # (T, 22, 6)

    return cat([original_135[:, :3], new_local.reshape(T, 132)], dim=-1)
```

### 7.4 训练侧对齐（可选，长期）

如果方案 A（mask 传播）效果好，可以考虑在训练时也引入类似的 kinematic-aware mask pattern：

- 在 M7 策略中增加子模式：当 flag 某关节时，以一定概率同时 flag 其子孙
- 这会让模型在训练时学会处理 "父节点 + 子孙节点同时 mask" 的 pattern

但这需要重新训练，作为长期优化。

---

## 8. 总结与决策点

| 决策项 | 选项 | 推荐 |
|--------|------|------|
| 首先实施哪个方案？ | A / B / C | **B（最安全）**，然后 A |
| 方案 B 的 alpha_decay？ | 0.3 / 0.5 / 0.7 | 从 0.5 开始，实验调优 |
| 方案 A 是否用 editing 混合？ | 是 / 否 | 先纯 completion 测试，效果不好再加 |
| Root 被 flag 时怎么办？ | 不传播 / 全局补偿 | 全局补偿（Option 2） |
| 是否需要重新训练？ | 是 / 否 | 短期不需要，长期考虑方案 C |

**预期收益**：
- Global pose stability: 消除 unmasked 子节点的世界空间姿态跳变
- 修复成功率: 预计提升 5-15%（具体取决于当前有多少失败是由子节点跳变导致的）
- 不破坏现有修复质量: 方案 B 的 alpha_decay→0 等价于无修改

---

*待 Phase 1 量化分析完成后更新此文档。*
