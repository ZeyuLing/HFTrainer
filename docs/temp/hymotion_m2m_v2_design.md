# HyMotion M2M v2：统一运动生成/编辑/修复/约束 — 完整设计方案

> 存放位置：`docs/temp/hymotion_m2m_v2_design.md`
> 日期：2026-04-11
> 状态：方案设计（WIP）

---

## 1. 目标

构建一个**统一模型**，支持任意关节、任意帧粒度下的：

| 能力       | 说明                          | 覆盖的现有工作                   |
| -------- | --------------------------- | ------------------------- |
| **生成**   | 从文本/无条件生成完整动作               | HyMotion T2M              |
| **补全**   | 给定部分帧/关节，补全其余               | HyMotion M2M, KIMODO, UMO |
| **编辑**   | 给定原动作 + 编辑指令，修改动作           | UMO [Edit], M2M reactive  |
| **修复**   | 给定含噪/缺陷动作，修复为高质量            | MoGenDIT, M2M edit-repair |
| **位置约束** | 给定任意关节的世界空间3D位置，生成严格满足约束的动作 | KIMODO end-effector       |

**核心约束**：模型输出的 rotation 和 position 通道必须**物理一致**（FK 自洽）。

---

## 2. 运动表示：198-dim

### 2.1 维度布局

| 段              | 维度          | 内容                            | 大小      |
| -------------- | ----------- | ----------------------------- | ------- |
| Translation    | `[0:3]`     | SMPL translation 参数           | 3       |
| Rotation       | `[3:135]`   | 22 关节 × 6D rot6d              | 132     |
| Joint Position | `[135:198]` | 21 关节 × 3D (**XZ 相对 pelvis，Y 绝对世界**，pelvis 除外) | 63      |
| **总计**         |             |                               | **198** |

> **Pelvis position 被移除**：Scheme D 下 pelvis pos = `[0, pelvis_y, 0]`，XZ 恒零（无信息量），Y 与 translation 冗余（差常数 `J_template[0][1]`）。Pelvis 的世界位置由 translation 通道完整表达。

Joint position 索引：joint j (j=1..21) 在 `[135 + (j-1)*3 : 135 + j*3]`。

### 2.2 Position Reference Frame：XZ_rel + Y_abs（Scheme D）

#### 2.2.1 编码公式

```python
joint_pos_world = FK(local_rotation, translation, bone_offsets)  # 必须用 local rotation
pelvis_world = joint_pos_world[:, 0, :]

# 只保留 joint 1~21（跳过 pelvis）
joint_pos = joint_pos_world[:, 1:, :].clone()  # (T, 21, 3)
joint_pos[..., 0] -= pelvis_world[..., 0:1]  # X: 相对 pelvis
joint_pos[..., 2] -= pelvis_world[..., 2:3]  # Z: 相对 pelvis
# Y: 保持绝对世界坐标
```

**重要**：FK 必须使用 local rotation，即使 rotation 通道存储的是 global rotation。

#### 2.2.2 选择理由

| 维度 | Scheme A (全相对 pelvis) | Scheme D (XZ_rel+Y_abs) |
|------|:---:|:---:|
| **XZ normalization** | ✅ bounded | ✅ bounded |
| **Foot contact** | ❌ Y 从 -0.52~-0.86 变化 | ✅ Y≈0（常数） |
| **Y 方向约束** | 需知 pelvis_y | 直接写入 |

不用 ReprV1（Y 相对 root）、ReprV2（root_motion 参考系太复杂）、全绝对世界（XZ 无界增长）。

#### 2.2.3 Translation 与 Pelvis 的关系

`pelvis_world = translation + J_template[0]`（差一个已知常量）。Translation 通道完整表达 pelvis 世界位置，无需在 position 通道重复。

### 2.3 Rotation Convention

| 方案         | 含义 | 优势 | 劣势 |
| ---------- | --- | --- | --- |
| **local**  | SMPL 父节点相对旋转 | 生态兼容 | 邻居不可直接插值 |
| **global** | 世界坐标系绝对旋转 | KIMODO 对齐 | 方差膨胀 |

Config 驱动，共享 198-dim 布局。推理输出统一转回 local。

### 2.4 Rot6d Convention

**Row-major**：`[R00, R01, R10, R11, R20, R21]`，与 v1 一致。

### 2.5 Normalization

Per-dimension z-score，Std < 1e-3 clamp 到 1。

| Stats | 路径 | 状态 |
|-------|------|------|
| Local rotation | `_stats_201dim/` | ✅ 完成（407K samples, 85.4M frames） |
| Global rotation | `_stats_201dim_global_rot/` | 🔄 修复 FK bug 后重算中（第二次） |

验证（local）：Pelvis XZ = 精确 0 ✅，Pelvis Y = 0.863 ✅，Ankle Y ≈ 0.09 ✅

### 2.6 自回归长动作

每段在 canonical frame (XZ=0, 面朝Z) 中生成，推理时做 world↔canonical 变换，不需要训练增强。

---

## 3. 模型架构

### 3.1 Backbone

沿用 `HunyuanMotionMMDiT`（0.46B / 1.5B）。

### 3.2 输入

```
x_input = [x_t(198), reactive(198), condition_mask(198)] = 594-dim
```

去掉 inactive 通道（MAN 下冗余）。

### 3.3 Condition Mask

**(T, 198) binary tensor**：mask=0 = 动画师给定的条件（已知），mask=1 = 模型生成。

约束粒度：
- **translation [0:3]**：可逐维（X/Y/Z 独立）
- **rotation [3:135]**：必须按关节（6 维一体）
- **position [135:198]**：可逐维（X/Y/Z 独立）

### 3.4 Reactive 通道

- **Completion 模式**（无原动作参考）：reactive = 全零
- **Editing 模式**（有原动作参考）：reactive = 原动作 / 降质动作

---

## 4. 训练策略

### 4.1 核心理念

训练策略的本质是回答：**动画师会给模型什么条件？**

一个 condition 由**三个正交轴 + 一个模式开关**自由组合：

```
condition = Mode × Temporal × Spatial × Channel
```

- **Mode**：是否有原动作参考（reactive 通道）
- **Temporal**：哪些帧是已知的
- **Spatial**：哪些关节是已知的
- **Channel**：已知关节的哪些通道已知

任意组合都是合法的。训练时从各轴独立采样，组合出覆盖面极广的 condition pattern。

### 4.2 Mode（模式开关）

| Mode | reactive 通道 | 场景 |
|------|-------------|------|
| **Completion** | 全零 | 生成、补全、位置约束 — 无原动作参考 |
| **Editing** | 原动作 | 编辑（修改部分内容）、修复（原动作有缺陷） |

> **修复不是独立的推理范式**。修复是一个目标，可通过两种范式实现：
> - **Inpainting 式修复**：mask 掉缺陷区域（mask=1），completion 模式重新生成
> - **Editing 式修复**：将缺陷动作填入 reactive 通道，模型输出修复后版本
>
> 训练时，editing 式修复通过合成损坏（jitter, sliding, joint_jump 等）构造"缺陷动作→干净动作"数据对。

### 4.3 Condition 采样：两层架构

**设计目标**：
1. **完备覆盖**：任何合法的推理时 condition pattern，在训练时都有非零概率被采样到
2. **加速收敛**：对高频使用的动画任务，提供更多训练曝光
3. **无需手工枚举**：底层靠连续参数控制，不是 if-else case list

两层架构：

| 层 | 占比 | 职责 | 采样方式 |
|----|------|------|---------|
| **Tier 1: 参数化随机** | ~60% | 覆盖任意 condition pattern | 三个轴各用连续参数随机生成 |
| **Tier 2: 高频任务加速** | ~40% | 针对常见动画任务加速收敛 | 从预定义的高频 pattern 中采样 |

#### 4.3.1 Tier 1：参数化随机（~60%）

**核心思想**：不枚举 case，而是用连续参数空间控制 mask 的生成。理论上可以覆盖任何合法的 `(T, 198)` binary mask。

**三个轴独立采样**：

##### Temporal 轴（哪些帧已知）

用 2-state Markov chain 生成帧级 known/generate 序列：

```python
def sample_temporal_markov(T, rng):
    """Markov chain: 生成 (T,) binary 序列, 1=generate, 0=known。

    参数：
        p_start: 第一帧为 known 的概率 (0.0~1.0)
        p_stay_known: known 帧之后继续 known 的概率 (控制连续已知段长度)
        p_stay_gen: generate 帧之后继续 generate 的概率 (控制连续生成段长度)

    Markov chain 的好处：
        - p_stay_known≈1, p_stay_gen≈0.1 → 大段已知 + 零星缺失 (dense_with_gaps)
        - p_stay_known≈0.1, p_stay_gen≈1 → 稀疏关键帧 (sparse_keyframes)
        - p_stay_known=p_stay_gen≈0.5 → 随机交替
        - 极端: p_stay_gen=1.0, p_start=0 → 全 generate (纯生成)
        - 极端: p_stay_known=1.0, p_start=1 → prefix / suffix
    """
    p_start = rng.uniform(0.0, 1.0)
    p_stay_known = rng.beta(2, 2)    # 偏中间，覆盖各种段长
    p_stay_gen = rng.beta(2, 2)

    seq = np.zeros(T, dtype=np.int32)
    seq[0] = 0 if rng.random() < p_start else 1
    for i in range(1, T):
        if seq[i-1] == 0:  # known
            seq[i] = 0 if rng.random() < p_stay_known else 1
        else:              # generate
            seq[i] = 1 if rng.random() < p_stay_gen else 0
    return seq
```

关键性质：**连续参数 `(p_start, p_stay_known, p_stay_gen)` 控制 mask 形态**，无需枚举 case。不同参数组合自然涌现出 prefix、suffix、inbetween、sparse keyframe、dense with gaps 等所有 pattern。

##### Spatial 轴（哪些关节已知）

对每个 known 帧，用 **per-joint 独立 Bernoulli** 决定哪些关节被条件化：

```python
def sample_spatial_bernoulli(rng):
    """Per-joint 独立采样，保证任意关节子集都有非零概率。

    p_joint ~ Beta(1, 6)，E ≈ 0.14 ≈ 3 joints：
        - 实际使用中，用户多数时候约束 1-3 个关节
          （一只手到某位置、双脚接地、头朝某方向）
        - 4-5 个关节已较多（四肢末端同时约束）
        - 全身已知的 pattern（keyframe、in-between）由 Tier 2 覆盖
        - p_joint→1 仍有非零概率，完备性不受影响
    """
    p_joint = rng.beta(1, 6)  # E=0.14, 中位数≈0.11, ~2-3 joints
    selected = [j for j in range(22) if rng.random() < p_joint]

    # 至少选 1 个关节
    if len(selected) == 0:
        selected = [rng.randint(0, 22)]

    return selected
```

关键性质：
- **完备覆盖**：任意关节子集都有非零概率
- **偏向稀疏**：`Beta(1,2)` 使多数样本约束 3-8 个关节，反映实际使用（约束太密则生成空间不足）
- 不假设拓扑连续 — 任意关节组合均可被采样
- 高频 pattern（全身、末端等）由 Tier 2 加速覆盖

##### Channel 轴（已知关节的哪些通道）

对每个被条件化的关节，独立决定 rot/pos 的 mask：

```python
def sample_channel(rng):
    """Per-joint channel mask。

    rot6d (6-dim): 整体 keep 或 generate（6维一体）
    pos (3-dim): 逐维独立 Bernoulli（支持 Y-only, XZ-only 等）
    """
    rot_keep = rng.random() < 0.6

    pos_keep_prob = rng.beta(2, 1)
    pos_x_keep = rng.random() < pos_keep_prob
    pos_y_keep = rng.random() < pos_keep_prob
    pos_z_keep = rng.random() < pos_keep_prob

    # 至少一个通道被保留
    if not rot_keep and not any([pos_x_keep, pos_y_keep, pos_z_keep]):
        pos_y_keep = True

    return rot_keep, (pos_x_keep, pos_y_keep, pos_z_keep)
```

**Translation [0:3] 独立于 joint 0 position 采样。**

在 Scheme D 下，pelvis position [135:138] = `[0, pelvis_y, 0]`（XZ 恒为 0，无信息量），而 translation [0:3] 才是世界位置。两者不是绑定关系：

| 通道 | X | Y | Z |
|------|---|---|---|
| translation [0:3] | 世界水平位置 | 世界高度 | 世界水平位置 |
| pelvis pos [135:138] | 恒 0 | ≈ trans_y + 常数 | 恒 0 |

因此 translation 作为**第 23 个独立采样单元**，复用同一套 `sample_channel` 逻辑：

```python
# Translation 视为独立控制单元，与 22 个关节并列
# 在 spatial 采样后，独立决定 translation 是否被约束
def sample_translation(known_frames, mask, rng):
    """Translation [0:3] 独立采样，复用 channel 逻辑。"""
    trans_keep = rng.random() < 0.2  # 约束轨迹的场景比约束关节更常见于 Tier 2
    if not trans_keep:
        return

    # 逐维独立（XZ 常一起用于轨迹，Y 偶尔单独用于高度）
    pos_keep_prob = rng.beta(2, 1)
    tx = rng.random() < pos_keep_prob
    ty = rng.random() < pos_keep_prob
    tz = rng.random() < pos_keep_prob
    if not any([tx, ty, tz]):
        tx, tz = True, True  # fallback: 至少 XZ

    # 可选叠加 heading（root rotation [3:9]）
    heading_keep = rng.random() < 0.3

    for f in known_frames:
        if tx: mask[f, 0] = 0
        if ty: mask[f, 1] = 0
        if tz: mask[f, 2] = 0
        if heading_keep:
            mask[f, 3:9] = 0  # root rot6d
```

典型轨迹场景的覆盖：

| 场景 | Temporal | Translation channel | 效果 |
|------|----------|-------------------|------|
| 完整轨迹 | 全帧 known | XZ keep | 所有帧水平位置已知 |
| 部分轨迹 | 连续段 known | XZ keep | 一段路径已知 |
| 稀疏途径点 | 稀疏帧 known | XYZ keep | 几个位置点已知 |
| 轨迹 + heading | 稀疏/密集 | XZ + root rot keep | 位置 + 朝向 |
| 仅高度 | 稀疏帧 | Y only | 站在平台上 |

##### 完整 Tier 1 采样

```python
def sample_tier1(T, rng):
    """参数化随机采样：生成一个 (T, 198) condition mask。"""
    mask = np.ones((T, 198))  # 全 generate

    # 1. Temporal
    temporal_seq = sample_temporal_markov(T, rng)
    known_frames = np.where(temporal_seq == 0)[0]
    if len(known_frames) == 0:
        return mask

    # 2. Spatial (22 joints)
    per_frame_spatial = rng.random() < 0.1
    shared_joints = sample_spatial_bernoulli(rng) if not per_frame_spatial else None

    for f in known_frames:
        joints = sample_spatial_bernoulli(rng) if per_frame_spatial else shared_joints
        for j in joints:
            rot_keep, (px, py, pz) = sample_channel(rng)
            # rotation [3+j*6 : 3+(j+1)*6]（所有 22 关节都有 rotation）
            if rot_keep:
                mask[f, 3+j*6 : 3+(j+1)*6] = 0
            # position（仅 joint 1..21，pelvis 无 position 通道）
            if j > 0:
                pos_base = 135 + (j - 1) * 3
                if px: mask[f, pos_base] = 0
                if py: mask[f, pos_base + 1] = 0
                if pz: mask[f, pos_base + 2] = 0

    # 3. Translation (独立于 joints)
    sample_translation(known_frames, mask, rng)

    return mask
```

#### 4.3.2 Tier 2：高频任务加速（~40%）

**核心思想**：Tier 1 虽然能覆盖任意 pattern，但对高频使用的具体任务，收敛速度不够（因为这些 pattern 只是参数空间中的一个点/小区域）。Tier 2 显式构造这些高频 pattern，增加训练曝光。

| 编号 | Pattern | 占比 | 描述 |
|------|---------|------|------|
| **T2-1** | 纯生成 | 8% | mask 全 1，保持 text-to-motion 能力 |
| **T2-2** | In-between | 8% | 首尾 K 帧全身 rot+pos 已知，中间全 generate |
| **T2-3** | Prefix | 5% | 前 N 帧全身已知，续写后面 |
| **T2-4** | Sparse keyframes | 5% | K 帧（几何分布）全身 rot+pos 已知 |
| **T2-5** | End-effector position | 5% | 手腕/脚踝的 position 约束（rot 不约束） |
| **T2-6** | Trajectory | 4% | translation XZ（+ 可选 root heading）在多帧已知 |
| **T2-7** | Foot grounding | 3% | 脚踝 pos_y = 0 在接地帧（物理约束） |
| **T2-8** | Editing + Repair | 2% | 三类修复数据混合（详见 §4.6） |

```python
def sample_tier2(T, gt_motion, rng):
    """从高频 pattern 中采样。"""
    pattern = rng.choice(
        ['pure_gen', 'inbetween', 'prefix', 'keyframes',
         'end_effector', 'trajectory', 'foot_ground', 'edit_repair'],
        p=[0.20, 0.20, 0.125, 0.125, 0.125, 0.10, 0.075, 0.05],
    )

    mask = np.ones((T, 198))
    reactive = np.zeros((T, 198))

    if pattern == 'pure_gen':
        pass  # mask 全 1

    elif pattern == 'inbetween':
        # 首尾各取 1~5 帧
        n_start = rng.randint(1, 6)
        n_end = rng.randint(1, 6)
        for f in range(n_start):
            mask[f, :] = 0  # 全通道 keep
        for f in range(T - n_end, T):
            mask[f, :] = 0

    elif pattern == 'prefix':
        n_keep = rng.randint(1, max(2, T // 2))
        mask[:n_keep, :] = 0

    elif pattern == 'keyframes':
        K = min(rng.geometric(p=0.15), T)
        frames = sorted(rng.choice(T, size=K, replace=False))
        for f in frames:
            mask[f, :] = 0  # 全身 rot+pos

    elif pattern == 'end_effector':
        # 手腕(20,21) 和/或 脚踝(7,8) 的 position
        ee_joints = rng.choice([7, 8, 20, 21],
                                size=rng.randint(1, 5), replace=False)
        K = min(rng.geometric(p=0.1), T)
        frames = sorted(rng.choice(T, size=K, replace=False))
        for f in frames:
            for j in ee_joints:
                pos_base = 135 + (j - 1) * 3
                mask[f, pos_base : pos_base + 3] = 0  # position XYZ

    elif pattern == 'trajectory':
        # translation XZ on dense/sparse frames
        K = rng.randint(max(1, T // 10), T)
        frames = sorted(rng.choice(T, size=K, replace=False))
        for f in frames:
            mask[f, 0] = 0  # trans X
            mask[f, 2] = 0  # trans Z
            if rng.random() < 0.4:
                mask[f, 3:9] = 0  # root rotation (heading)

    elif pattern == 'foot_ground':
        # 脚踝 Y=0 约束
        ankle_joints = [7, 8]  # L_Ankle, R_Ankle
        K = rng.randint(max(1, T // 5), T)
        frames = sorted(rng.choice(T, size=K, replace=False))
        for f in frames:
            for j in ankle_joints:
                pos_base = 135 + (j - 1) * 3
                mask[f, pos_base + 1] = 0  # pos_Y only

    elif pattern == 'edit_repair':
        corrupted = apply_corruption(gt_motion, rng)
        reactive = corrupted
        # 损坏区域 mask=1（生成），其余 mask=0
        mask = build_corruption_mask(corrupted, gt_motion)

    return mask, reactive
```

#### 4.3.3 完整采样入口

```python
def sample_condition(T, gt_motion, step, rng):
    """Phase 2 condition 采样：两层架构。"""

    # 选层
    use_tier2 = rng.random() < 0.4

    if use_tier2:
        mask, reactive = sample_tier2(T, gt_motion, rng)
    else:
        mask = sample_tier1(T, rng)
        reactive = np.zeros((T, 198))

    # 叠加层（两层都可能叠加）
    # 25% 概率额外叠加轨迹约束
    if rng.random() < 0.25:
        overlay_trajectory(mask, T, rng)

    # Editing mode: 15% 概率（仅 Tier 1 的 completion 样本）
    if not use_tier2 and rng.random() < 0.15:
        corrupted = apply_corruption(gt_motion, rng)
        reactive = corrupted

    return mask, reactive
```

### 4.4 三个轴的推理覆盖清单

以下列出推理时的典型 condition pattern，以及训练时的覆盖方式：

| 推理任务 | Temporal | Spatial | Channel | 训练覆盖 |
|---------|----------|---------|---------|---------|
| 纯文生 | 全 generate | — | — | T2-1 + Tier1(p_start→1) |
| In-between | 首尾帧已知 | 全身 | rot+pos | T2-2 |
| 续写 | prefix 已知 | 全身 | rot+pos | T2-3 |
| 关键帧动画 | K帧稀疏 | 全身 | rot+pos | T2-4 |
| 末端位置约束 | 稀疏帧 | 手腕/脚踝 | pos only | T2-5 |
| 路径跟随 | 密集帧 | root | trans_xz | T2-6 |
| 脚接地 | 接地帧 | ankle | pos_y | T2-7 |
| 编辑/修复 | 全帧/局部 | 全身/局部 | rot+pos | T2-8 + Tier1(editing) |
| 保持上身重做下肢 | 全帧 | 上身子集 | rot+pos | Tier1(BFS from spine) |
| 肘弯曲角度约束 | 稀疏帧 | 单关节 | rot only | Tier1(rot_keep=True, pos_keep=False) |
| 手贴墙面 | 稀疏帧 | 手腕 | pos_x only | Tier1(pos_x_keep=True, others=False) |
| 复合约束（路径+手位置） | 混合 | root+手腕 | trans_xz+pos_xyz | Tier1 + overlay_trajectory |

> **完备性保证**：Tier 1 的参数化随机在 `(p_start, p_stay_known, p_stay_gen) × (p_joint) × (rot_keep, pos_keep_prob)` 连续空间上采样。三个轴全部使用独立 Bernoulli 或 Markov chain，不引入拓扑/语义偏置，任意合法的 `(T, 198)` binary mask 都有非零概率被生成。Tier 2 额外加速了高频区域的收敛。

### 4.5 两阶段训练

#### Phase 1：纯生成预训练

- 只使用全 mask=1（纯 text-to-motion），建立基础生成质量
- FK consistency loss 此阶段开始 warmup
- 预计 500k steps

#### Phase 2：条件训练

- 使用 §4.3 的两层 condition 采样
- Phase 2 **关闭文本 dropout**（借鉴 KIMODO：避免 drop 掉约束相关的语义信息）
- Curriculum：Phase 2 初期 Tier 2 占比更高（60%），后期 Tier 1 占比增大（70%），逐渐让模型见到更复杂的随机组合

```python
# Phase 2 curriculum
tier2_ratio = max(0.30, 0.60 - 0.30 * step / 300000)
# step=0:      60% Tier 2, 40% Tier 1
# step=300k:   30% Tier 2, 70% Tier 1
```

### 4.6 Editing 模式训练：修复与编辑

Editing 模式的核心：reactive 通道填入原动作（或缺陷动作），模型学习输出修正后的版本。

#### 4.6.1 三类修复训练数据

| 类别 | 数据来源 | 规模 | 特点 |
|------|---------|------|------|
| **A. 在线合成（M2M corruptors）** | 5 个现有 corruptor 在线对 GT 施加 | 无限（与 GT 数据等量） | 有精确的 (T,J) corruption mask；与 checker 对偶 |
| **B. 在线合成（MoGenDiT-style）** | 8 种 degradation 按段施加 | 无限 | 段级施加，更贴近真实分布；含 translation 缺陷 |
| **C. 真实 LQ-HQ 数据对** | `fixed_rule_manifest.jsonl` | 7,212 对（已验证） | 真实分布，无合成偏差；但数量有限 |

三类数据**同时使用**，在每个 editing batch 中混合采样。

#### 4.6.2 类别 A：M2M Corruptors（在线合成）

现有 5 个 corruptor（`hftrainer/utils/data_corruptor/`）：

| Corruptor | 缺陷类型 | 对应 Checker |
|-----------|---------|-------------|
| `JitterCorruptor` | 高频抖动（Perlin noise + 时域量化） | `JitterChecker` |
| `SlidingCorruptor` | 脚滑（root/leg 速度不匹配） | `FootSlidingChecker` |
| `JointJumpCorruptor` | 关节突变（offset, burst, freeze, stutter） | `JointJumpChecker` |
| `WristCandyWrapperCorruptor` | 手腕糖纸旋转（palm flip 180°, arm twist 360°） | `CandyWrapperChecker` |
| `LimbCandyWrapperCorruptor` | 肢体糖纸旋转（四肢 180°/360°） | `CandyWrapperChecker` |

特点：
- 每个 corruptor 输出精确的 **(T, J) corruption mask**，标识受影响的帧和关节
- 有 3 个强度等级（low/medium/high），训练时随机选择
- 与 quality checker 系统对偶 — 可验证修复效果

```python
def apply_m2m_corruption(gt_motion, rng):
    """从 5 个 corruptor 中随机选 1-2 个施加。"""
    corruptors = [JitterCorruptor, SlidingCorruptor, JointJumpCorruptor,
                  WristCandyWrapperCorruptor, LimbCandyWrapperCorruptor]
    n_corrupt = rng.choice([1, 2], p=[0.7, 0.3])
    selected = rng.choice(corruptors, size=n_corrupt, replace=False)

    corrupted = gt_motion.copy()
    corruption_mask = np.zeros((T, 22), dtype=bool)  # (T, J)

    for C in selected:
        intensity = rng.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        corrupted, mask_tj = C(intensity).corrupt(corrupted, rng)
        corruption_mask |= mask_tj

    return corrupted, corruption_mask
```

#### 4.6.3 类别 B：MoGenDiT-style Degradation（在线合成）

8 种 degradation（参考 `MoGenDiT/motion_degradation.py`）：

| Degradation | 概率 | 描述 |
|------------|------|------|
| Joint orientation pops | 9.5% | ±90° 随机旋转，5-50% 关节 |
| Joint rotation pops | 9.5% | 类似，但沿 FK chain 累积 |
| Pose twist | 9.5% | ±60° 持续扭曲，25-50% 关节 |
| Candy wrapper twist | 9.5% | IK 歧义模拟，±180° 球关节 |
| Frozen frame | 9.5% | 重复首帧（模拟追踪丢失），段长 10-50% |
| Translation drift | 9.5% | 平移漂移累积（0.12-0.32x） |
| Translation distortion | 9.5% | 轴缩放 + 偏航旋转 |
| Identity（干净） | 33.3% | 不施加缺陷 |

施加方式：
- 序列切分为 **10-30 帧的段**，每段独立选一种 degradation
- Translation drift 跨段累积（模拟真实 mocap 漂移）
- 50% 概率对 joint pops / drift 做时域平滑

```python
def apply_mogendit_degradation(gt_motion, rng):
    """按段施加 MoGenDiT-style degradation。"""
    T = gt_motion.shape[0]
    seg_len = rng.randint(10, 31)
    corrupted = gt_motion.copy()
    corruption_mask = np.zeros((T, 22), dtype=bool)

    for start in range(0, T, seg_len):
        end = min(start + seg_len, T)
        deg_type = sample_degradation_type(rng)  # 含 33% identity
        if deg_type != 'identity':
            corrupted[start:end], mask = apply_segment_degradation(
                corrupted[start:end], deg_type, rng)
            corruption_mask[start:end] |= mask

    return corrupted, corruption_mask
```

**与类别 A 的互补**：
- A 类对应已知的 **5 种特定缺陷**，correction mask 精确到关节级
- B 类覆盖 **A 类不包含的缺陷**（translation drift/distortion、frozen frame、FK chain 累积的 rotation pop）
- B 类的段级施加更贴近真实 mocap 数据的缺陷分布（缺陷往往是局部连续段）

#### 4.6.4 类别 C：真实 LQ-HQ 数据对

来源：`data/hymotion_m2m_refine_data/fixed_rule_manifest.jsonl`

| 修复类型 | 数量 | 说明 |
|---------|------|------|
| joint_twist | 11,570 → 5,XXX verified | 解剖学角度超限 |
| jitter | 2,794 → X,XXX verified | 高频抖动 |
| joint_jump | 856 → XXX verified | 关节位置突变 |
| **总计** | 15,220 → 7,212 verified | 通过 quality checker 验证 |

使用方式：
- **只使用 passed=true 的 7,212 对**（质量有保证）
- LQ motion → reactive 通道，HQ motion → target
- corruption mask 由 checker 在线计算（对 LQ motion 跑 checker 得到受影响帧/关节）

```python
def load_real_lq_hq_pair(manifest_entry):
    """加载真实 LQ-HQ 数据对。"""
    lq_motion = load_npz(manifest_entry['lq'])
    hq_motion = load_npz(manifest_entry['hq'])

    # 用 checker 计算 corruption mask（哪些帧/关节有缺陷）
    checker = MotionQualityChecker()
    report = checker.check(lq_motion)
    corruption_mask = report.get_frame_joint_mask()  # (T, 22)

    return lq_motion, hq_motion, corruption_mask
```

**类别 C 的独特价值**：
- 合成 corruption（A/B）不可能完全覆盖真实缺陷分布
- 真实数据对弥补了 distribution gap
- 数量有限（7K），但通过与合成数据混合训练可以发挥杠杆效应

#### 4.6.5 Condition / Reactive 划分与鲁棒性

**核心约束**：MAN imputation 在 mask=0 处每步硬替换，这些值在去噪过程中不可被模型修改。因此 **mask=0 处必须是干净的，不能有任何缺陷**。

这决定了三个通道的语义：

| 通道 | 内容 | 可否修改 |
|------|------|---------|
| mask=0 (condition) | 确定干净的帧/关节 | ❌ imputation 锁死 |
| mask=1 (generate) | 可能有问题的区域 | ✅ 模型重新生成 |
| reactive | 完整 corrupted motion | 仅供参考，不锁死 |

##### 训练时的 imputation 值

```python
# Repair (editing mode):
# mask=0 处填 GT（干净值）— 训练时有 GT 可用
x_t[mask == 0] = gt_clean_motion

# reactive 通道填完整的 corrupted motion（给模型参考上下文）
reactive = corrupted_motion   # 含缺陷，但不被 imputation 锁死
```

推理时，mask=0 处填的是 corrupted motion 中**确认干净的部分**。
训练/推理的 mismatch 通过 mask 扰动来缓解（见下文）。

##### Mask 扰动机制：偏向 Over-mask

鲁棒性的核心策略是**确保 mask 足够大**，把所有可能有问题的区域都划进 mask=1。
训练时通过 mask 扰动让模型适应各种 mask 粒度，但**禁止 under-mask**（不能让有缺陷的帧留在 condition 里）。

```python
def perturb_corruption_mask(mask_tj, T, rng):
    """对 (T, 22) corruption mask 施加随机扰动。

    只允许 mask 不变或变大（over-mask），禁止变小（under-mask）。
    因为 mask=0 处被 imputation 锁死，缺陷帧绝不能留在 condition 里。
    """
    perturb_type = rng.choice(
        ['precise', 'dilated_small', 'dilated_large', 'joint_expand', 'full_seq'],
        p=[0.25, 0.25, 0.15, 0.20, 0.15],
    )

    if perturb_type == 'precise':
        # 25%: 保留精确 mask（理想情况）
        return mask_tj

    elif perturb_type == 'dilated_small':
        # 25%: 时域小膨胀（前后各 1-3 帧）
        # 模拟 checker 保守地多标记了边界帧
        d = rng.randint(1, 4)
        return temporal_dilate(mask_tj, d)

    elif perturb_type == 'dilated_large':
        # 15%: 时域大膨胀（前后各 5-15 帧）
        # 模拟不确定缺陷范围时的保守策略
        d = rng.randint(5, 16)
        return temporal_dilate(mask_tj, d)

    elif perturb_type == 'joint_expand':
        # 20%: 空间膨胀 → 扩展到相邻关节
        # 模拟缺陷可能影响了邻近关节的情况
        expanded = mask_tj.copy()
        for j in range(22):
            if mask_tj[:, j].any():
                parent = SMPL22_PARENTS[j]
                if parent >= 0:
                    expanded[:, parent] |= mask_tj[:, j]
                for child in SMPL22_CHILDREN[j]:
                    expanded[:, child] |= mask_tj[:, j]
        # 可叠加时域膨胀
        if rng.random() < 0.5:
            d = rng.randint(1, 5)
            expanded = temporal_dilate(expanded, d)
        return expanded

    else:  # 'full_seq'
        # 15%: 全序列 mask=1
        # 模拟"整条动作有问题，全部重做"
        return np.ones_like(mask_tj)
```

##### 扰动策略设计意图

| 扰动类型 | 占比 | 效果 | 推理时对应场景 |
|---------|------|------|-------------|
| `precise` | 25% | mask 精确匹配缺陷 | checker 检测完美 |
| `dilated_small` | 25% | mask 比缺陷稍大 | checker + 小膨胀（最常用） |
| `dilated_large` | 15% | mask 远大于缺陷 | 不确定时的保守策略 |
| `joint_expand` | 20% | 扩展到相邻关节 | 缺陷可能跨关节传播 |
| `full_seq` | 15% | 全序列重做 | 不用 checker，直接全修 |

**关键原则**：所有扰动只会让 mask 变大或不变，绝不会变小。这保证了 condition（mask=0）中不会有缺陷残留。

##### 推理时的 Over-mask 策略

```python
# 推理时：checker 检测 + 膨胀 → 确保 mask 覆盖所有缺陷
checker_mask = quality_checker.check(corrupted_motion).get_mask()  # (T, 22)
# 保守膨胀
inference_mask = temporal_dilate(checker_mask, d=5)
inference_mask = spatial_dilate(inference_mask, SMPL22_PARENTS)
```

##### 完整 Repair 数据流

```
训练时：
  GT (clean) + corrupted_motion + precise_mask
      ↓
  perturbed_mask = perturb_corruption_mask(precise_mask)  # 只允许 over-mask
      ↓
  reactive = corrupted_motion                     # 完整缺陷动作（参考）
  condition_mask = expand_to_198dim(perturbed_mask)
  x_t[mask=0] = GT                               # condition 处填干净 GT
  x_t[mask=1] = (1-t)*noise + t*GT               # 生成区域正常加噪
  target = GT                                     # loss 对齐干净动作

推理时：
  corrupted_motion + checker_mask（over-mask 后）
      ↓
  reactive = corrupted_motion                     # 完整缺陷动作（参考）
  condition_mask = expand_to_198dim(over_masked)
  x_t[mask=0] = corrupted_motion[干净部分]        # 确认干净的区域
  x_t[mask=1] = noise                            # 从噪声开始
  → model 输出修复后的动作
```

**训练/推理的 mismatch**：mask=0 处训练时是 GT、推理时是 corrupted motion 的干净部分。
因为 over-mask 策略确保推理时 mask=0 处确实是干净的，所以 mismatch 很小。
此外，over-mask 训练（dilated/joint_expand/full_seq 共 75%）让模型习惯了 mask 远大于实际缺陷的情况，进一步降低了对 mask=0 处值精确度的敏感性。

#### 4.6.6 混合采样策略

```python
def sample_editing_data(gt_motion, manifest_pool, rng):
    """Editing 模式的数据采样。

    A/B 类在线合成（无限量），C 类真实数据（7K，控制过采样）。
    比例为可调超参数，初始值基于以下考量：
        - A/B 在线合成是主体（数量无限，覆盖面广）
        - C 真实数据量小（7K），过高比例会导致过拟合
        - A 与 B 的缺陷类型互补，大致均分
    """
    # 可调超参数
    p_A = 0.45   # M2M corruptors
    p_B = 0.40   # MoGenDiT-style degradation
    p_C = 0.15   # 真实 LQ-HQ 对
    # 注意：以上比例需要根据实验效果调整

    source = rng.choice(['A', 'B', 'C'], p=[p_A, p_B, p_C])

    if source == 'A':
        corrupted, mask_tj = apply_m2m_corruption(gt_motion, rng)
        target = gt_motion
    elif source == 'B':
        corrupted, mask_tj = apply_mogendit_degradation(gt_motion, rng)
        target = gt_motion
    else:  # C
        entry = rng.choice(manifest_pool)
        corrupted, target, mask_tj = load_real_lq_hq_pair(entry)

    # 构造 reactive 和 condition mask
    reactive = encode_198dim(corrupted)
    condition_mask = expand_tj_to_198dim(mask_tj)  # (T, 22) → (T, 198)
    return reactive, condition_mask, target
```

**C 类过采样控制**：C 类只有 ~7K 对，若总 editing batch 为 100K（Phase 2 的 15%），
p_C=0.15 意味着 C 类被采样 ~15K 次，即每条平均重复 ~2 次，尚在合理范围。
若后续 C 类数据量增长（更多修复对通过验证），可相应提高 p_C。

#### 4.6.7 Corruption Mask → Condition Mask 转换

Corruptor 输出的 `(T, J)` mask 需要展开为 `(T, 198)` condition mask：

```python
def expand_tj_to_198dim(mask_tj, include_translation=True):
    """(T, 22) bool → (T, 198) condition mask。

    受损关节：rotation + position 通道都标记为 generate (mask=1)
    未受损关节：保持 keep (mask=0)
    """
    T = mask_tj.shape[0]
    mask_198 = np.zeros((T, 198))

    for j in range(22):
        if mask_tj[:, j].any():
            affected_frames = mask_tj[:, j]
            # rotation: dims [3+j*6 : 3+(j+1)*6]
            mask_198[affected_frames, 3+j*6 : 3+(j+1)*6] = 1
            # position (joint 1-21): dims [135+(j-1)*3 : 135+j*3]
            if j > 0:
                pos_base = 135 + (j - 1) * 3
                mask_198[affected_frames, pos_base : pos_base + 3] = 1

    # Translation: 如果 pelvis (j=0) 受损，或 translation 类缺陷
    if include_translation and mask_tj[:, 0].any():
        mask_198[mask_tj[:, 0], 0:3] = 1

    return mask_198
```

#### 4.6.8 与 MoGenDiT 修复方案的对比分析

##### 两种修复路径

| | MoGenDiT（x₀ 替换式） | M2M v2（reactive 通道式） |
|---|---|---|
| **缺陷动作进入方式** | 直接作为 x₀，与扩散噪声融合 | 放入 reactive 通道，独立于 x_t |
| **干净区域保护** | mask=1 处不加噪，直接保留 | mask=0 处 imputation 锁死 |
| **模型任务** | 同时去噪 + 去缺陷 | 参考 reactive 上下文，在 mask=1 区域生成干净动作 |
| **train/infer mismatch** | **无** | **有**（mask=0 训练=GT，推理=corrupted 干净部分） |
| **mask 精度要求** | 低（全让模型判断） | 高（mask=0 必须干净） |

##### MoGenDiT 在纯修复上的优势

1. **零 mismatch** — 训练和推理数据流完全一致
2. **无 mask 精度问题** — 不需要区分干净/缺陷区域，全交给模型
3. **去噪与去缺陷天然融合** — 缺陷被视为"另一种噪声"

##### M2M v2 的不可替代优势（MoGenDiT 做不到的）

1. **修复 + 约束组合** — 修复脚滑的同时保持手在指定位置。MoGenDiT 无法在 x₀ 中同时表达"这里有缺陷"和"这里有硬约束"
2. **指令式编辑** — "把走改成跑"需要 reactive 提供原动作 + text 指定意图。x₀ 替换式没有通道传递编辑语义
3. **精确保护** — mask=0 处 imputation 锁死 = 数学精确保护。MoGenDiT 的 mask 保护只是"不加噪"，模型仍可能微调这些区域
4. **修复范围精确控制** — 通过 mask 精确指定修哪些帧/关节。MoGenDiT 只能通过噪声强度间接控制
5. **统一架构** — 同一个模型、同一套通道同时处理生成/补全/编辑/修复/约束

##### 核心论点

> **我们选择 reactive 通道式，不是因为它在纯修复上更好，而是因为它是唯一能支撑统一框架的方案。** 纯修复质量可能略逊于 MoGenDiT（因为 mismatch），但换来的是所有任务共享一个模型。

##### 缩小 mismatch 的措施

为确保纯修复质量不显著下降，采取以下措施：

| 措施 | 作用 |
|------|------|
| Over-mask 训练（75% 样本 mask 大于实际缺陷） | 推理时 mask=0 处几乎一定是干净的，缩小与训练时填 GT 的差距 |
| full_seq 模式（15%） | 模型学会不依赖 mask=0 区域，极端情况下无 condition 也能修复 |
| Reactive 通道提供完整上下文 | 即使 mask=0 较少，模型仍可从 reactive 获取全局动作结构 |
| 三类训练数据混合 | C 类真实数据弥补合成数据的 distribution gap |
| 推理时 checker + 保守膨胀 | 确保 mask=0 处确实干净 |

##### 验证计划

纯修复质量需与 MoGenDiT 做 A/B 对比：

| 指标 | 说明 |
|------|------|
| 修复精度 | 修复后 vs GT 的 MPJPE（mm） |
| 伪影引入率 | 修复后新增的 quality checker 告警数 |
| 边界过渡质量 | mask 边界处的速度连续性 |
| 主观评价 | 双盲 A/B test |

如果纯修复质量确实显著低于 MoGenDiT，备选方案：
- 为 repair 任务增加专门的训练比例
- 推理时对修复任务采用类 MoGenDiT 的 partial denoise 策略（降低 mask=1 区域的初始噪声水平，让模型更多依赖 reactive 信息而非从纯噪声开始）

#### 4.6.9 非修复的编辑任务

除修复外，editing 模式还支持：
- **风格迁移**：reactive = 原风格动作，target = 目标风格动作（需要配对数据或在线转换）
- **幅度放大/缩小**：reactive = 原动作，target = 幅度调整后的 GT
- **指令编辑**：reactive = 原动作，text condition = 编辑指令

这些暂不在 v2 初版实现，但 reactive 通道的设计天然支持扩展。

### 4.7 MAN Flow Matching

```python
noise = randn_like(x_clean)
x_t_gen = (1 - t) * noise + t * x_clean    # 生成区域
x_t_keep = x_clean                           # 条件区域
x_t = where(mask == 0, x_t_keep, x_t_gen)   # 组合

model_input = cat([x_t, reactive, mask], dim=-1)  # 594-dim
v_pred = model(model_input, t, text_cond)          # 预测 velocity
```

### 4.8 FK Consistency Loss

#### 问题

模型在训练时预测的是 velocity（噪声到数据的方向），不是去噪后的动作。FK loss 需要作用于去噪后的预测。

#### 方案

从 velocity 预测推导出对 x_clean（x₁）的估计：

```python
# Flow matching: x_t = (1-t)*noise + t*x_clean
# velocity: v = x_clean - noise
# 因此: x_1_hat = x_t + (1 - t) * v_pred
x_1_hat = x_t + (1 - t).unsqueeze(-1) * v_pred   # (B, T, 198)

# 只对 mask=1（生成区域）施加 FK loss
pred_rot = x_1_hat[..., 3:135]
pred_pos = x_1_hat[..., 135:198]
pred_trans = x_1_hat[..., 0:3]

# Denormalize（FK 需要真实物理值）
pred_rot_denorm = pred_rot * std[3:135] + mean[3:135]
pred_pos_denorm = pred_pos * std[135:198] + mean[135:198]
pred_trans_denorm = pred_trans * std[0:3] + mean[0:3]

# FK: rotation → position（Scheme D 参考系，跳过 pelvis）
fk_world = differentiable_FK(pred_rot_denorm, pred_trans_denorm, bone_offsets)
pelvis = fk_world[..., 0:1, :]
fk_pos_D = fk_world[:, :, 1:, :].clone()  # 跳过 pelvis
fk_pos_D[..., 0] -= pelvis[..., 0]
fk_pos_D[..., 2] -= pelvis[..., 2]

# Re-normalize for loss computation
fk_pos_norm = (fk_pos_D.reshape(B, T, 63) - mean[135:198]) / std[135:198]

# t² 加权：t 接近 0 时 x_1_hat 极不准确，FK loss 无意义
weight = (t ** 2).unsqueeze(-1)  # (B, 1)
loss_fk = weight * SmoothL1(pred_pos, fk_pos_norm)
```

#### 为什么用 t² 加权

| t | x_1_hat 质量 | FK loss 意义 | t² 权重 |
|---|---|---|---|
| 0.0 | 极差（几乎纯噪声） | 无意义 | 0.00 |
| 0.3 | 较差 | 弱 | 0.09 |
| 0.7 | 较好 | 有意义 | 0.49 |
| 1.0 | 准确（≈GT） | 完全有意义 | 1.00 |

#### Loss 总权重

```
λ_fk = λ_fk_base * warmup(step) * t²
```

- `λ_fk_base = 0.1`
- `warmup(step) = min(1.0, step / 50000)`：前 50k steps 从 0 增到 1
- `t²`：每个样本的时间步加权

### 4.9 Loss 总结

```
L_total = L_main(v_pred, v_gt)
        + λ_fk * warmup * t² * L_fk(pred_pos, FK(pred_rot))
        + λ_edit * L_edit(...)   [仅 editing 模式 batch]
```

| Loss | 权重 | 说明 |
| ---- | --- | --- |
| `L_main` | 1.0 | SmoothL1(velocity pred, velocity GT)，198-dim 全通道 |
| `L_fk` | 0.1 × warmup × t² | FK 一致性 |
| `L_edit` | 0.3 | 仅 editing 模式 batch |

---

## 5. 推理

### 5.1 Completion 推理（生成、补全、约束）

```python
y0[keep] = clean_motion
for step in ode_steps:
    v = model(t, x)
    x = x + v * dt
    x[keep] = clean_motion  # MAN imputation
```

### 5.2 位置约束推理

用户输入 `[(frame, joint, target_xyz)]`：

```python
# joint j (j=1..21)，position 在 [135+(j-1)*3 : 135+j*3]
idx = 135 + (j - 1) * 3
# Y：直接写入绝对高度
clean_motion[f, idx + 1] = target_y
# XZ：相对 pelvis
clean_motion[f, idx + 0] = target_x - pelvis_x
clean_motion[f, idx + 2] = target_z - pelvis_z
# mask: position=0, rotation=1
# 若约束 pelvis 位置，直接约束 translation [0:3]
```

### 5.3 Editing 推理（编辑、修复）

reactive 填入原动作/缺陷动作，mask 标记修改区域。

---

## 6. KIMODO 借鉴与超越

### 借鉴

| KIMODO | v2 |
|--------|-----|
| 两阶段训练（Phase 1 生成 → Phase 2 条件） | ✅ |
| Keyframe 数量渐进 + 偏向少量 | ✅（几何分布，不设上限） |
| 25% 多约束混合 | ✅（三轴组合天然支持） |
| Root XZ 和 Y 分开约束 | ✅ Scheme D |

### 超越

| 维度 | KIMODO | v2 |
|------|--------|-----|
| Position 逐维约束 | ❌ | ✅ Y-only, XZ-only |
| Rotation-only 约束 | ❌ | ✅ |
| 编辑/修复 | ❌ | ✅ reactive 通道 |
| FK 一致性 | ❌ | ✅ FK loss |
| Condition 组合空间 | 5 种预定义类型 | Temporal×Spatial×Channel 自由组合 |

### 暂不引入

| 设计 | 理由 |
|------|------|
| 双阶段 denoiser | MAN 下 root 天然稳定 |
| Smooth root | 与 198-dim 不兼容 |
| Foot contact / velocity 通道 | FK loss + pos_y_only 替代 |

---

## 7. 数据准备

### 7.1 198-dim 编码

```python
# FK 必须用 local rotation（即使 rotation_space='global'）
world_pos = FK(local_rotation, translation, bone_offsets)
joint_pos_D = scheme_D_encode(world_pos[:, 1:, :])  # 跳过 pelvis，XZ rel pelvis, Y absolute

# rotation 通道使用目标空间（local 或 global）
motion_198 = cat([translation, rotation_channel, joint_pos_D.reshape(T, 63)])
```

### 7.2 Stats

已完成：`tools/compute_201dim_stats.py`（16 进程并行，407K HQ samples）。需适配为 198-dim（去除 pelvis position 3 维）后重算。

### 7.3 数据质量

使用 HQ 过滤后的 407K 高质量样本。

---

## 8. 能力覆盖对比

| 能力 | KIMODO | MoGenDIT | UMO | M2M v1 | **M2M v2** |
| --- | --- | --- | --- | --- | --- |
| Text-to-Motion | ✅ | ❌ | ✅ | ✅ | ✅ |
| In-Between | ✅ | ✅ | ✅ | ✅ | ✅ (temporal) |
| Per-joint completion | ✅ | ✅ | ❌ | ✅ | ✅ (spatial) |
| End-effector position | ✅ | ❌ | ❌ | ❌ | ✅ (channel:pos) |
| Position 逐维约束 | ❌ | ❌ | ❌ | ❌ | ✅ (pos_y_only等) |
| Rotation-only 约束 | ❌ | ❌ | ❌ | ✅ | ✅ (channel:rot) |
| Trajectory | ✅ | ❌ | ✅ | ✅ | ✅ (spatial:root) |
| Editing | ❌ | ❌ | ✅ | ✅ | ✅ (editing mode) |
| Repair | ❌ | ✅ | ❌ | ✅ | ✅ (editing/inpaint) |
| FK 一致性 | ❌ | ❌ | N/A | N/A | ✅ |

---

## 9. 实施路线

### Phase 1：数据与表示（1 周）

1. ~~201-dim Mean/Std 统计~~ → 需更新为 198-dim（去掉 pelvis position）
2. 198-dim 在线数据转换
3. Condition pattern 采样器（两层架构：Tier 1 参数化随机 + Tier 2 高频加速）
4. FK → position 一致性验证

### Phase 2：模型改造（1 周）

1. input_encoder 405 → 594
2. final_layer 135 → 198
3. FK consistency loss（含 t² 加权 + warmup）
4. v1 checkpoint warm-start

### Phase 3：训练（2-3 周）

1. Phase 1 训练：纯生成，500k steps
2. Phase 2 训练：三轴组合条件 + editing，500k steps
3. local vs global rotation 对比

### Phase 4：评估（1 周）

1. 22 关节 position constraint 精度
2. 补全/生成/修复质量
3. 可视化 demo

---

## 10. 关键设计决策

| 决策 | 理由 |
| --- | --- |
| 去掉 inactive 通道 | MAN 下冗余 |
| 198-dim XZ_rel+Y_abs | foot Y≈0，Y 约束无依赖，去除冗余 pelvis position |
| FK loss (t² weighted) | 从 velocity prediction 推导 x_1_hat，低 t 降权 |
| 两层 condition 采样（参数化随机 + 高频加速） | 替代手工枚举 case list，Tier 1 保证覆盖任意 pattern，Tier 2 加速高频任务 |
| 修复通过 editing/inpainting 两种范式实现 | 不是独立 pattern，是目标 |
| 两阶段训练 | 借鉴 KIMODO，先建生成质量再学条件 |
| Keyframe 几何分布不设上限 | 覆盖任意输入形态 |
| row-major rot6d | 与 v1 一致 |
