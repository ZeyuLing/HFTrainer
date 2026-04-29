# KIMODO — 参考工作分析

## 基本信息

- **论文**：Kimodo: Scaling Controllable Human Motion Generation
- **作者**：Davis Rempe*, Mathis Petrovich* 等（NVIDIA）
- **时间**：2026-3-16（Tech Report）
- **代码**：https://github.com/nv-tlabs/kimodo（**已开源**，本目录 `kimodo/` 子目录）
- **主页**：https://research.nvidia.com/labs/sil/projects/kimodo

---

## 论文核心内容

### 问题定位

训练高质量、可灵活控制的运动生成模型。现有方法受限于公开 mocap 数据集规模小（如 HumanML3D 仅 30 小时）导致质量、控制精度和泛化能力不足。Kimodo 的答案是：**用 700 小时 optical mocap（Bones Rigplay）从头训练**。

### 主要创新点（Paper Claim）

1. **两阶段 Transformer Denoiser**：将 root motion 和 body motion 解耦，stage-1 预测全局 root，stage-2 以 local root 为条件预测 body。两阶段 interleaved 训练（每个 denoising step 都同时做两阶段，不是先 root 全程再 body）。
2. **Smooth Root 表示**：对 pelvis 的水平分量做重度平滑，作为稳定的参考坐标系。相比直接用 pelvis 投影，减少脚步滑动，且更贴近动画师使用直线/曲线约束的习惯。
3. **Global Joint Rotation 表示**：关节旋转用世界坐标系下的 6D rotation，而非 SMPL 风格的局部相对旋转。好处是可以直接通过 imputation 施加稀疏 global rotation 约束（无需 FK chain）。
4. **Imputation-based Condition**：约束通过直接覆盖 noisy motion 的对应位置来施加（`x̃_t = m ⊙ x_tgt + (1-m) ⊙ x_t`），concat mask 作为输入额外通道。这支持 full-body keyframe、end-effector position/rotation、2D waypoint/path、foot contact 等各类约束。
5. **两阶段训练 Curriculum**：Phase 1（500k steps）纯 text-to-motion，Phase 2（500k steps）加入约束 conditioning。
6. **Separated CFG**：推理时 text guidance 和 constraint guidance 解耦，分别控制权重 `w_text` 和 `w_constr`。
7. **700h Optical Mocap**：Bones Rigplay 数据集，生产级别质量，人工标注文本。

### 动作表示（代码验证）

见 `kimodo/motion_rep/reps/kimodo_motionrep.py`，每帧包含：

| 分量 | 大小 | 含义 |
|------|------|------|
| `smooth_root_pos` | 3 | 全局 smooth root 位置（xzy，水平平滑） |
| `global_root_heading` | 2 | 全局朝向 `[cos ψ, sin ψ]` |
| `local_joints_positions` | J×3 | 关节位置，水平分量相对于 smooth root，y 为绝对高度 |
| `global_rot_data` | J×6 | 全局关节旋转（6D continuous rotation，世界坐标系） |
| `velocities` | J×3 | 全局关节速度 |
| `foot_contacts` | 4 | 足部接触标志 `{L heel, L toe, R heel, R toe}` |

- **Native 27-joint skeleton**：Bones Rigplay 标准骨骼（实验用）
- **SOMA 30-joint skeleton**：retarget 版本（public release 用）
- 另外支持 Unitree G1 robot、SMPL-X 的 retarget 版本
- fps：实验用 20fps，正式 release 用 30fps
- 向量总维度视 J 而定（27 joints: `3+2+27×3+27×6+27×3+4 = 3+2+81+162+81+4 = 333 dims`）

### 网络架构（代码验证）

见 `kimodo/model/twostage_denoiser.py` 和 `kimodo/model/backbone.py`：

- **Backbone**：`TransformerEncoderBlock` = 标准 Transformer Encoder，prefix 模式（text tokens + timestep token 拼接在 pose tokens 前面）
- **两个 Transformer**：root_model 和 body_model，参数相同（16 layers, 8 heads, latent_dim=1024），总 282M 参数
- **文本编码**：LLM2Vec（bidirectional LLaMA）取代 CLIP/T5，输出 4096-dim，直接 linear 投影到 latent_dim
- **额外 register tokens**：49 个全零 token 作为 prefix 增强模型表达能力
- **初始 heading token**：`c_dir = [cos(ψ_0), sin(ψ_0)]`，因为不做 canonicalize，需要告知模型初始朝向
- **Constraint 施加方式**：imputation `x̃_t = m ⊙ x_tgt + (1-m) ⊙ x_t`，然后 `concat(x̃_t, m)` 作为输入

两阶段 forward（`TwostageDenoiser.forward`）：
```
1. Stage1: root_model(x_extended, text, t) → root_pred (global, 5 dims)
2. convert global root → local root: [angular_vel, planar_vel_xz, abs_height_y]
3. Stage2: body_model([local_root; body_features], text, t) → body_pred
4. output = concat(root_pred, body_pred)
```
注意：训练时 stage2 的 local root 做 detach（避免梯度穿透），推理时不做。

### 训练

- DDPM 框架，1000 diffusion steps，DDIM 推理 100 steps
- Loss: 各分量的 smooth L1 加权求和（position/velocity/rotation/foot contact + FK 约束项），权重 `γ1=γ3=γ5=10, γ2=2, γ4=3, γ6=4, γ7=5`
- Optimizer: Adam-atan2, lr=2e-5
- Batch size: 2048（16 NVIDIA A100 SXM4-80GB）
- Text dropout 10% 用于 CFG
- EMA decay=0.995，every 10 steps

### 推理

- DDIM 100 steps
- Separated CFG: `x̂_0 = D_∅ + w_text(D_text - D_∅) + w_constr(D_constr - D_∅)`，默认 `w_text=w_constr=2`
- Multi-prompt 生成：顺序生成，用重叠帧 + full-body keyframe 约束保证过渡，做线性 blend
- Post-processing（可选）：foot lock + IK 修复足部滑动；短优化确保 exact constraint 满足

### 支持任务

| 任务 | 实现方式 |
|------|---------|
| Text-to-Motion | 无 constraint，纯 text 引导 |
| Full-body keyframe | 指定帧 imputation 全部关节 |
| End-effector (hand/foot) | 指定帧 imputation 手脚关节 position+rotation |
| 2D Waypoints | 指定稀疏帧的 root 2D 位置 |
| 2D Dense Path | 连续帧的 root 2D 轨迹约束 |
| Foot contact pattern | 指定帧的 foot contact 标志 |
| In-betweening | 首尾 full-body keyframe |
| Multi-prompt | 顺序生成 + 过渡混合 |
| Robotics (G1) | retarget 到 Unitree G1 骨骼 |

---

## 与我们自己工作的对比

### HyMotion M2M (我方) vs KIMODO

| 维度 | KIMODO | HyMotion M2M (我方) |
|------|--------|---------------------|
| **模型架构** | 标准 Transformer Encoder（2 stage，prefix 模式） | HunyuanMotion MMDiT（双流+单流 transformer，0.46B） |
| **生成范式** | DDPM diffusion，预测 x0 | Flow Matching，预测 velocity (v=x1-x0) 或 x1（JiT） |
| **动作表示** | 全局关节旋转 6D + smooth root pos + heading + local joint pos + velocity + foot contact，native 27-joint | SMPL-22，rotation_6d + abs_rel translation，138 dims，无 foot contact |
| **关节坐标系** | **世界坐标系（Global）** joint rotation，不 canonicalize | 局部相对旋转（SMPL local rotation），通过 abs_rel transl 分离根轨迹 |
| **Root 处理** | Smooth root（对 pelvis 水平方向平滑），root/body 两阶段分离 | abs_rel translation（前 6 dims），root 与 body 联合生成 |
| **Condition 施加** | Imputation：直接覆盖 noisy motion + concat binary mask | VACE conditioning：`[x_t; inactive; reactive; src_mask]` 拼接，mask 逐维度 |
| **Condition 粒度** | 帧级，支持关节级（end-effector），mask 为 `(T, J)` 网格 | **逐帧逐维度（T×138）**，语义为 6-dim joint group 粒度 |
| **任务建模** | 所有任务统一为 imputation（observed_motion + motion_mask），训练 curriculum 分两阶段 | 所有任务统一为 universal mask（M1-M6 策略），单阶段训练 |
| **文本编码** | LLM2Vec（bidirectional LLaMA，4096-dim），单一编码 | 双编码器：Qwen3-8B（ctxt, 4096-dim）+ CLIP-L（vtxt, 768-dim），MMDiT 双流融合 |
| **数据规模** | 700h 光学 mocap（Bones Rigplay，高质量人工标注） | MotionHub（多来源，含重建数据和 mocap） |
| **Foot contact** | 显式建模为输出特征（4-dim），loss 有 foot contact 项，有 post-process foot lock | 不显式建模，无 foot contact 损失 |
| **FK 约束** | Loss 中含 FK 一致性项（`FK(ĵ_a) - j_p`） | 无显式 FK loss |
| **多人/反应** | 不支持 | 不直接支持（src_mask 支持拼接但无 inter-person attention） |
| **支持任务** | T2M、keyframe、end-effector、trajectory、multi-prompt | T2M（M5）、temporal inpainting（M1-M4、M6）、joint editing（M4）、keyframe（M6）——通用 completion |
| **推理速度** | DDIM 100 steps，RTX 3090 上 2-5 秒/clip | Euler ODE 50 steps，类似速度 |
| **代码开源** | 是 | 否（内部） |

### 核心设计理念差异

1. **Global vs Local 坐标系**：KIMODO 全局关节旋转使得稀疏约束（如"第 50 帧右手在世界坐标 xyz 位置"）可以直接 imputation，无需 IK。我方用局部旋转（SMPL），稀疏 global 约束需要通过 FK 转换。

2. **Imputation vs VACE**：
   - KIMODO：`x̃_t = m ⊙ x_tgt + (1-m) ⊙ x_t`，在 diffusion 中用 GT 值直接替换 noisy motion 对应位置，简洁有效
   - 我方：VACE 把 observed/unobserved 分别 encode 成 inactive/reactive，与 noisy motion concat 后送入模型，不直接替换 noisy motion。这给模型更多"知道什么地方是已知的"的上下文，且不限制 noisy motion 的取值范围

3. **Smooth Root vs abs_rel transl**：KIMODO 的 smooth root 专门为 "animator 画直线/曲线" 的使用场景优化，这是面向 game/robotics production 工作流的；我方的 abs_rel 更通用但在约束精度上稍弱。

4. **两阶段 Denoiser vs 单模型**：KIMODO 的两阶段设计显著减少 foot skating（从 one-stage 的 7.59 降到 3.87 cm/s），原理是 body motion 以 root motion 为条件生成，自然降低对 root 运动的过拟合。我方单模型需要靠 loss 设计（不同分量加权）来平衡 root 和 body。

5. **数据策略**：KIMODO 只用高质量 optical mocap，通过 scaling（700h）来提升泛化。我方 MotionHub 更多样但质量参差不齐，需要靠模型来 denoise。

### 可借鉴的点

- **Global joint rotation 表示**：对需要支持世界坐标系约束的场景（如 end-effector world-space 控制）有优势，值得考虑作为可选表示
- **Smooth root**：比直接 pelvis 对 trajectory following 场景更稳定，如果要支持 path-conditioned generation 可以考虑
- **Separated CFG**：text 和 constraint guidance 分别控制，比单一 CFG scale 更灵活
- **Foot contact 显式建模**：在 loss 中加入 foot contact 项、post-process 时用 contact prediction 做 foot lock，减少足部滑动
- **两阶段训练 curriculum**：Phase 1 纯 T2M 预训练，Phase 2 加约束，可以避免从零训 constraint following 的难度


---

## KIMODO 约束系统详细分析

> 以下内容来自对 KIMODO 代码的深入分析（2026-03-23）

### Feature Layout (`KimodoMotionRep`)

**Per-frame components (concatenated):**

```python
size_dict = {
    "smooth_root_pos":        [3],       # dims [0:3]     — smoothed pelvis (x, z plane smoothed, y absolute)
    "global_root_heading":    [2],       # dims [3:5]     — [cos(ψ), sin(ψ)]
    "local_joints_positions": [27×3],   # dims [5:86]    — joint positions relative to smooth root (xz relative, y absolute)
    "global_rot_data":        [27×6],   # dims [86:248]  — 6D continuous global joint rotations (world-frame)
    "velocities":             [27×3],   # dims [248:329] — global joint velocities
    "foot_contacts":          [4],      # dims [329:333] — binary foot contact flags [L_heel, L_toe, R_heel, R_toe]
}
```

**Total: 3 + 2 + 81 + 162 + 81 + 4 = 333 dims**

### Key Design Points

1. **Global Joint Rotations** (NOT local SMPL-style):
   - All joint rotations are in **world coordinate frame** (6D continuous rotation)
   - This is crucial—allows direct imputation of end-effector constraints without IK
   - Conversion: matrix ↔ 6D via `cont6d_to_matrix()` / `matrix_to_cont6d()`

2. **Smooth Root** (not standard pelvis):
   - Pelvis horizontal (x,z) components are heavily smoothed via `get_smooth_root_pos()`
   - Y-axis remains raw (absolute height)
   - Designed to match animator workflow (they constrain straight lines, not noisy trajectories)

3. **Feature Slicing**:
   - `slice_dict` maps feature name → `slice(start, end)` for indexing
   - Example: `slice_dict["global_rot_data"] = slice(86, 248)`

---

## 2. Constraint Types & Imputation Mechanism

KIMODO supports **5 constraint types** (implemented in `constraints.py`):

### 2.1 Root2DConstraintSet

**What it controls:**
- 2D root trajectory (x, z plane only)
- Optional global heading (rotation around Y-axis)

**Implementation** (`constraints.py:75-180`):
```python
class Root2DConstraintSet:
    frame_indices: Tensor        # which frames [T1, T2, ...]
    smooth_root_2d: Tensor      # target (x,z) positions [N, 2]
    global_root_heading: Optional[Tensor]  # [cos(ψ), sin(ψ)] per frame [N, 2]
```

**How applied** (`create_conditions()` lines 237-246):
```python
if "smooth_root_2d" in index_dict:
    indices, smooth_root_2d = get_unique_index_and_data(...)
    f_sliced = observed_motion[:, slice_dict["smooth_root_pos"]]  # [T, 3]
    f_sliced[indices, 0] = smooth_root_2d[:, 0]  # set x
    f_sliced[indices, 2] = smooth_root_2d[:, 1]  # set z
    m_sliced = motion_mask[:, slice_dict["smooth_root_pos"]]
    m_sliced[indices, [0, 2]] = True  # mark as constrained
```

**Dims affected:** dims [0, 2] of the motion vector (x and z of smooth_root_pos)

---

### 2.2 Root Y Height Constraint

**What it controls:**
- Absolute height of pelvis (Y-axis only)

**Implementation** (`create_conditions()` lines 248-255):
```python
if "root_y_pos" in index_dict:
    indices, root_pos_Y = get_unique_index_and_data(...)
    f_sliced[indices, 1] = root_pos_Y  # set y-component
    m_sliced[indices, 1] = True
```

**Dims affected:** dim [1] of smooth_root_pos

---

### 2.3 Global Root Heading Constraint

**What it controls:**
- Global heading angle ψ (rotation around Y-axis)
- Stored as [cos(ψ), sin(ψ)]

**Implementation** (`create_conditions()` lines 257-264):
```python
if "global_root_heading" in index_dict:
    indices, global_root_heading = get_unique_index_and_data(...)
    f_sliced = observed_motion[:, slice_dict["global_root_heading"]]  # dims [3:5]
    f_sliced[indices] = global_root_heading
    m_sliced[indices] = True
```

**Dims affected:** dims [3, 4] (global_root_heading)

---

### 2.4 Global Joint Rotations Constraint

**What it controls:**
- Full-body or per-joint global rotations
- 6D continuous representation

**Implementation** (`create_conditions()` lines 266-277):
```python
if "global_joints_rots" in index_dict:
    indices_lst, global_joints_rots = get_unique_index_and_data(...)
    # indices_lst shape: [N_constrained, 2] = [[t1, j1], [t2, j2], ...] (frame, joint)
    global_rot_data = matrix_to_cont6d(global_joints_rots)  # convert to 6D
    
    f_sliced = observed_motion[:, slice_dict["global_rot_data"]]  # [T, J×6]
    # Create masking for affected (frame, joint) pairs
    masking = torch.zeros([T×J, 6], dtype=bool)
    masking[indices_lst.T[0] * nbjoints + indices_lst.T[1]] = True
    masking = masking.reshape(T, J*6)
    f_sliced[masking] = global_rot_data.flatten()
    m_sliced[masking] = True
```

**Dims affected:** 6 dims per constrained joint (within dims [86:248])

**Use cases:**
- Full-body keyframes: all 27 joints × 6 dims = 162 dims per frame
- End-effector control: 6 dims × num_endeffectors per frame

---

### 2.5 Global Joint Positions Constraint

**What it controls:**
- Global 3D joint positions (x, y, z in world frame)
- Stored as [nbjoints, 3]

**Implementation** (`create_conditions()` lines 279-297):
```python
if "global_joints_positions" in index_dict:
    indices_lst, global_joints_positions = get_unique_index_and_data(...)
    
    # Requirement: smooth root must already be constrained!
    # (Otherwise cannot convert global → local-relative positions)
    _test = motion_mask[T_indices, slice_dict["smooth_root_pos"]]
    if not _test[:, [0, 2]].all():
        raise ValueError("For constraining global positions, smooth root should also be constrained")
    
    # Get constrained smooth root position for reference
    smooth_root_pos = observed_motion[T_indices, slice_dict["smooth_root_pos"]].clone()
    local_reference = smooth_root_pos.clone()
    local_reference[..., 1] = 0.0  # zero out Y for XZ-plane relative calculation
    
    # Convert global → local-relative positions
    local_joints_positions = global_joints_positions - local_reference
    
    f_sliced = observed_motion[:, slice_dict["local_joints_positions"]]  # [T, J×3]
    masking = torch.zeros([T×J, 3], dtype=bool)
    masking[indices_lst.T[0] * nbjoints + indices_lst.T[1]] = True
    masking = masking.reshape(T, J*3)
    f_sliced[masking] = local_joints_positions.flatten()
    m_sliced[masking] = True
```

**Dims affected:** 3 dims per constrained joint (within dims [5:86])

**Key constraint:** Global positions require smooth_root_2d to be constrained first!

---

## 3. Constraint Set Classes

KIMODO provides high-level constraint classes that wrap raw data:

### FullBodyConstraintSet
**Constrains all 27 joints' positions and rotations on keyframes**

```python
class FullBodyConstraintSet:
    frame_indices: Tensor                 # which frames to constrain
    global_joints_positions: Tensor      # [N, 27, 3]
    global_joints_rots: Tensor          # [N, 27, 3, 3] (rotation matrices)
    smooth_root_2d: Tensor              # [N, 2] (inferred from pelvis if not given)
    root_y_pos: Tensor                  # [N,]
    global_root_heading: Tensor         # [N, 2]
```

**What gets applied:**
1. `global_joints_positions` → all 27 joints, all 3 dims (81 dims per frame)
2. `global_joints_rots` → stored but NOT used during imputation! 
   - Only global positions are actually applied
   - (Rotations can be reconstructed from positions via FK if needed, but typically not)
3. Automatically derives `smooth_root_2d`, `root_y_pos`, `global_root_heading` from the full-body pose
4. All get applied via `update_constraints()` which populates data_dict/index_dict

---

### EndEffectorConstraintSet
**Constrains selected end-effectors (hands, feet)**

```python
class EndEffectorConstraintSet:
    frame_indices: Tensor
    global_joints_positions: Tensor  # [N, 3] for selected end-effectors
    global_joints_rots: Tensor       # [N, 3, 3]
    joint_names: list[str]           # which joints? auto-expanded by skeleton
    smooth_root_2d, root_y_pos, global_root_heading  # derived like FullBody
```

**Subclasses:**
- `LeftHandConstraintSet(joint_names=["LeftHand"])`
- `RightHandConstraintSet(joint_names=["RightHand"])`
- `LeftFootConstraintSet(joint_names=["LeftFoot"])`
- `RightFootConstraintSet(joint_names=["RightFoot"])`

---

## 4. Imputation During Diffusion

### Key Code Path: `twostage_denoiser.py:98-103`

```python
if self.motion_mask_mode == "concat":
    if motion_mask is None or observed_motion is None:
        motion_mask = torch.zeros_like(x)
        observed_motion = torch.zeros_like(x)
    
    # DIRECT IMPUTATION: Replace noisy motion with ground truth at constrained dims
    x = x * (1 - motion_mask) + observed_motion * motion_mask
    # Element-wise: where motion_mask=1, use observed_motion; where =0, use noisy x
    
    x_extended = torch.cat([x, motion_mask], axis=-1)  # Append mask as extra channel
```

**What happens:**
1. `observed_motion`: 333-dim vector with GT values filled in at constrained indices, zeros elsewhere
2. `motion_mask`: 333-dim binary mask (1 = constrained, 0 = not constrained)
3. **Before each forward pass**, noisy x is forcibly replaced: constrained dims become exactly the GT value
4. Both modified `x` and `motion_mask` are concatenated (666 dims) and fed to transformer

**Critical insight:** The mask is passed as **extra input channel**, not used to weight the loss. The model learns:
- "When I see motion_mask=1 at position (t,d), the value at x[t,d] is ground truth—don't denoise it"
- "When motion_mask=0, treat x[t,d] as noisy and predict the denoised value"

### Constraint Application Timing

Constraints are applied **at every denoising step**:
1. Scheduler produces `x_t` (noisy motion at timestep t)
2. If constraints provided, directly impute: `x_t = x_t * (1 - mask) + observed * mask`
3. Feed imputed (x_t, mask) to transformer for one denoising step
4. Get prediction of x_0 (clean motion)
5. Sampler computes next timestep x_{t-1}
6. **Repeat from step 2** with new x_{t-1}

---

## 5. Dimension Coverage by Constraint Type

| Constraint Type | Affected Dims | Description |
|---|---|---|
| `smooth_root_2d` | [0, 2] | Root X, Z (2 dims total, out of 3 in smooth_root_pos) |
| `root_y_pos` | [1] | Root Y (absolute height) |
| `global_root_heading` | [3:5] | [cos(ψ), sin(ψ)] |
| `global_joints_rots` | [86:248] | 27 joints × 6 dims per constrained joint |
| `global_joints_positions` | [5:86] | 27 joints × 3 dims per constrained joint |
| *implicit: velocities* | [248:329] | NOT constrained (computed from positions during inverse) |
| *implicit: foot_contacts* | [329:333] | NOT constrained (computed from positions/velocities during inverse) |

---

## 6. Constraint Metadata Structure

### For JSON Serialization

Each constraint type has a `get_save_info()` method that exports to JSON:

**Root2DConstraintSet:**
```json
{
    "type": "root2d",
    "frame_indices": [10, 20, 30],
    "smooth_root_2d": [[1.0, 2.0], [1.5, 2.1], [1.2, 2.2]],
    "global_root_heading": [[1.0, 0.0], [0.99, 0.1], [0.97, 0.25]]  // optional
}
```

**FullBodyConstraintSet:**
```json
{
    "type": "fullbody",
    "frame_indices": [50],
    "local_joints_rot": [[[...], [...], ...], ...],  // 77-joint axis-angle (for retarget compatibility)
    "root_positions": [[1.0, 1.0, 2.0]],
    "smooth_root_2d": [[1.0, 2.0]]
}
```

**EndEffectorConstraintSet:**
```json
{
    "type": "end-effector",
    "frame_indices": [10, 20],
    "local_joints_rot": [...],
    "root_positions": [...],
    "smooth_root_2d": [...],
    "joint_names": ["LeftHand", "RightFoot"]  // optional, base class stores this
}
```

**Subclass constraints (LeftHandConstraintSet, etc.):**
- Same as EndEffectorConstraintSet but without `joint_names` (hardcoded in class)

---

## 7. Complete Example: Full-Body Keyframe @ Frame 50

### Input
```python
fbc = FullBodyConstraintSet(
    skeleton=soma30,
    frame_indices=torch.tensor([50]),
    global_joints_positions=torch.tensor([[[x0, y0, z0], [x1, y1, z1], ...]]),  # [1, 27, 3]
    global_joints_rots=torch.tensor([[[R0], [R1], ...]]),  # [1, 27, 3, 3]
)
```

### Constraint Processing
```python
# During FullBodyConstraintSet initialization:
smooth_root_2d = global_joints_positions[:, skeleton.root_idx, [0, 2]]  # from pelvis position
root_y_pos = global_joints_positions[:, skeleton.root_idx, 1]
global_root_heading = compute_global_heading(global_joints_positions, skeleton)

# update_constraints() is called:
# 1. Adds all (frame, joint) pairs for positions:
data_dict["global_joints_positions"].append(positions.reshape(-1, 3))  # [27, 3]
index_dict["global_joints_positions"].append(create_pairs([50], [0..26]))  # [[50,0], [50,1], ..., [50,26]]

# 2. Also adds root trajectory:
data_dict["smooth_root_2d"].append(smooth_root_2d)
index_dict["smooth_root_2d"].append(torch.tensor([50]))

# 3. And root height:
data_dict["root_y_pos"].append(root_y_pos)
index_dict["root_y_pos"].append(torch.tensor([50]))

# 4. And heading:
data_dict["global_root_heading"].append(global_root_heading)
index_dict["global_root_heading"].append(torch.tensor([50]))
```

### Imputation Output
After `create_conditions()`:
```python
observed_motion = torch.zeros(T, 333)  # per-sequence max length T
motion_mask = torch.zeros(T, 333, dtype=bool)

# Frame 50 gets filled:
observed_motion[50, [0, 2]] = [smooth_root_x, smooth_root_z]  # root 2D
observed_motion[50, 1] = root_y
observed_motion[50, [3, 4]] = [cos(ψ), sin(ψ)]  # heading
observed_motion[50, 5:86] = local_joints_positions.flatten()  # all 27 joints × 3
# Note: global_rot_data NOT filled here (not used in practice)

motion_mask[50, [0, 1, 2, 3, 4]] = True
motion_mask[50, 5:86] = True
```

All 86 dims of frame 50 are now constrained; all other frames remain unmasked.

---

## 8. Key Differences from HyMotion M2M

| Aspect | KIMODO | HyMotion M2M |
|---|---|---|
| **Coordinate frame** | Global (world-space) rotations | Local (SMPL parent-relative) rotations |
| **Root handling** | Smooth root (explicitly smoothed) | abs_rel transl (6D: 3 absolute + 3 relative) |
| **Constraint application** | Direct imputation at every diffusion step | VACE conditioning (reactive/inactive split, no imputation) |
| **Mask semantics** | Joint-frame grid (T×J), 6D groups | Dim-frame grid (T×138), per-dim control |
| **Constraint dimensions** | 5 types: root_2d, root_y, heading, rot, pos | Implicit: any (T, 138) binary mask |
| **Position vs rotation** | Both global positions AND global rotations | Only relative/abs rotations + transl (no separate positions) |
| **Foot contacts** | Explicitly modeled (4D output) | Not modeled |
| **Multiple constraints per step** | Yes (e.g., full-body = pos + rot + root + heading) | Yes (flexible per-dim mask) |
| **IK requirement** | No (global positions imputed directly) | Yes (local constraints cannot directly control world positions) |

---

## 9. Supported Tasks & Their Mask Patterns

| Task | Constraint Type(s) | Masked Dims | Example |
|---|---|---|---|
| Text-to-Motion | None | [] | No imputation |
| Motion In-betweening | FullBody on keyframes | [0:86] at keyframes | Start/end frames |
| End-effector IK | EndEffector @ specific frames | [0:5] (root) + [5:86] (ee joints) | Hand/foot tracking |
| 2D Waypoint following | Root2D on sparse frames | [0, 2] at waypoints | Positional path |
| Full-body editing | FullBody on entire sequence | [0:86] per frame | Motion retargeting |
| Multi-prompt blend | FullBody on transition zone | [0:86] at boundaries | Segment stitching |

---

## Summary

KIMODO's constraint system is elegantly simple:
1. **5 constraint types** covering different semantic aspects (trajectory, heading, joint pos/rot)
2. **Direct imputation** during diffusion: GT values forcibly replace noisy predictions at constrained dimensions
3. **Global coordinate frame** enables World-space IK-free control
4. **Binary mask** tells the model which dimensions are constraints vs. to-be-denoised
5. **Smooth root** representation stabilizes trajectory tracking vs. noisy pelvis

The key innovation is treating all constraints uniformly as dimension-wise imputation, rather than as separate auxiliary losses or special model branches.
