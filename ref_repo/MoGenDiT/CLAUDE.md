# MoGenDiT — 参考工作分析

## 基本信息

- **项目**：MoGenDiT（原名 DiffNet / MoreDiff）— 基于 DiT 的运动修复扩散模型
- **作者**：chengxuzuo（内部项目）
- **原始代码**：`/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/`
- **本地副本**：`ref_repo/MoGenDiT/`（原始 + 2026-03-24 补丁已合并）
- **HF-Trainer 集成**：`hftrainer/pipelines/motion/mogendit_pipeline.py`

---

## 核心定位

MoGenDiT 是一个**运动修复/去噪**模型，不是纯生成模型。它接收一段有缺陷的 motion（来自其他模型的生成结果、mocap 噪声等），输出修复后的 motion。

| | MoGenDiT | HyMotion M2M |
|---|---|---|
| **任务** | 运动**修复**（去噪、位移重生成） | 运动**生成/补全**（从 mask 生成） |
| **输入** | 有缺陷的完整 motion + mask | 部分已知 motion + mask |
| **输出** | 修复后的 motion | 补全后的完整 motion |
| **文本条件** | ❌ 无 | ✅ Qwen3 + CLIP |
| **使用场景** | 后处理 pipeline（在 M2M 输出之后） | 主生成 pipeline |

---

## ⭐ 训练方式（最核心的差异）

### 训练范式总览

MoGenDiT 的训练是**自监督**的，同时学习两个任务：

| | 50% batch：标准扩散去噪 | 50% batch：运动修复 |
|---|---|---|
| **输入** | 部分帧已知（keyframe mask）+ 其余帧加扩散噪声 | **合成退化**的 motion + 扩散噪声 |
| **目标** | 预测完整的干净 motion x₀ | 预测**原始干净** motion（不是退化版本） |
| **mask** | 6 种 mask 模式混合采样 | 几乎全零（仅 1-10 帧参考帧为已知） |
| **本质** | Inpainting-style 扩散 | 退化修复 + 扩散去噪 |

**关键**：loss target 始终是**原始干净 motion**。即使 50% batch 的输入是退化后的 motion，loss 也对标原始干净版本。模型同时学会去除扩散噪声和修复运动缺陷。

### 训练数据流（单次迭代）

```
1. 加载 batch: (motion [B,224,201], length [B])
   ↓
2. 生成 keyframe_mask [B,224,201]（6 种 mask 模式混合）
   ↓
3. 50% batch → 运动退化处理：
   a. mask 清零（几乎全 uncond）
   b. 保留前 1-10 帧为 keyframe（mask=1）
   c. 对 motion 施加合成退化（8 种退化类型随机分段应用）
   d. keyframe 帧恢复为干净值
   ↓
4. 扩散前向过程（mask-aware）：
   - 已知帧（mask=1）：不加噪，保持干净/退化值
   - 未知帧（mask=0）：加标准扩散噪声
   ↓
5. 模型输入 = concat([x_t, mask], dim=-1)   → 2 × motion_dim
   ↓
6. 模型预测 pred_x₀（直接预测去噪后的完整 motion）
   ↓
7. Loss = L1(pred_x₀, clean_x₀)  ← target 始终是原始干净 motion
```

### Mask 模式（6 种）

| 模式 | 概率 | 描述 |
|------|------|------|
| `random_frame` | 20% | 随机选 5-10% 帧作为 keyframe |
| `random_phrase` | 20% | 随机连续段（2-20 帧）作为 keyframe |
| `random_start_end` | 20% | 首尾各保留 1-20%，中间 50-90% gap |
| `block_trans` | 10% | 只 mask translation（仅重新生成位移） |
| `joint_only` | 10% | 只保留 joint position（mask rotation） |
| `uncond` | 20% | 全零 mask（纯无条件生成） |

### 运动退化（Motion Degradation）— 核心创新

代码：`motion_process/motion_representation.py` `motion_degradation_batch()`

序列被随机分段（每段 10-30 帧），每段独立应用一种退化：

| 退化类型 | 概率 | 模拟的真实缺陷 |
|---------|------|-------------|
| **identity** | ~33% | 无退化（干净段） |
| **joint_orientation_pops** | ~9.5% | 关节朝向突变（marker 混淆）：随机 Euler 角扰动，最大 90° |
| **joint_rotation_pops** | ~9.5% | 关节旋转突变（沿 FK chain 传播） |
| **pose_twist** | ~9.5% | 持续姿态扭曲（遮挡导致的恒定偏移） |
| **candy_wrapper_twist** | ~9.5% | 糖果纸扭转（IK 歧义：绕骨轴旋转，位置不变但旋转错误）— 仅影响肩/髋 |
| **frozen_frame** | ~9.5% | 帧冻结（tracking 丢失时重复上一帧） |
| **d_translation_drift** | ~9.5% | 位移漂移（线性 + 高斯噪声在 delta-translation 上累积） |
| **d_translation_distortion** | ~9.5% | 位移畸变（深度估计错误导致的缩放/旋转扭曲） |

**退化后，keyframe 帧恢复为干净值**：`motion[keyframe_mask==1] = motion_clean[keyframe_mask==1]`

### 扩散前向过程（Mask-Aware + 退化叠加）

代码：`trainer/my_trainer.py` L164-182、`EasyDiffusion/base_diffusion.py` L148-169

```python
# trainer/my_trainer.py L164-182
x0 = batch.clone()                               # x0 = 干净 motion
if motion_degradation:
    x0[degrade_idx] = motion_degradation_batch(   # 50% batch：x0 = 退化后 motion
        motion=batch[degrade_idx], ...)

x_t, noise = diffusion.q_sample(                  # 标准扩散加噪（mask-aware）
    x0=x0, t=t, obs_mask=keyframe_mask, ...)

# EasyDiffusion/base_diffusion.py L158-166
x_noise = sqrt_ac * x0 + sqrt_1mac * noise       # 标准高斯噪声，无额外扭曲
noise_mask[obs_mask >= 1] = False                 # 已知帧不加噪
x_t = x0.clone()
x_t[noise_mask] = x_noise[noise_mask]             # 只对未知帧加噪

# queue 里的 loss target：
data_queue.put((x_wrapped, t, batch, ...))        # batch = 原始干净 motion
```

**精确行为**：

| batch 类型 | x0（扩散起点） | x_t（模型输入的 noisy motion） | loss target |
|-----------|-------------|---------------------------|------------|
| 非退化（50%） | 干净 motion | `√ᾱ · clean + √(1-ᾱ) · noise`（mask=0 帧）；`clean`（mask=1 帧，不加噪） | **原始干净 motion** |
| 退化（50%） | **退化后** motion | `√ᾱ · degraded + √(1-ᾱ) · noise`（mask=0 帧）；`degraded`（mask=1 参考帧，不加噪，但退化帧上 mask 已清零所以仍加噪） | **原始干净 motion** |

退化 batch 的关键细节：
1. keyframe_mask 先清零（L169），再把前 1-10 帧设为 mask=1（L171-172）
2. `motion_degradation_batch` 对 motion 施加退化，但 mask=1 的参考帧**恢复为干净值**
3. q_sample 中 mask=1 的参考帧不加噪 → x_t 中参考帧 = 干净值
4. 其余帧 mask=0 → 从退化 motion 出发加噪：`√ᾱ · degraded + √(1-ᾱ) · noise`
5. loss target = `batch` = 原始干净 motion → **模型必须同时去除扩散噪声 + 修复退化缺陷**

噪声类型：标准高斯，`noise_remap_mode="identity"`（默认），无额外变换。

### 推理时的 Imputation

每个去噪步中：
```python
# 强制恢复已知区域的值
x_wrap["x_t"][obs_mask] = x_original[obs_mask]
```

三种 imputation 模式：
- `all`：每步都恢复（包括最后一步）— 最严格
- `skip_last`：每步恢复，最后一步除外 — 默认
- `none`：从不恢复 — 完全自由生成

### 与 HyMotion M2M 训练方式的根本差异

| | MoGenDiT | HyMotion M2M |
|---|---|---|
| **训练范式** | 自监督修复（退化 → 恢复原始） | 条件生成（mask → 补全） |
| **50% batch** | **合成退化** + 扩散去噪 | 仅 mask + flow matching 去噪 |
| **退化类型** | 8 种真实 mocap 缺陷模拟 | ❌ 无退化，只有 mask |
| **Mask 含义** | 1=已知（不加噪），0=未知（加噪+需修复） | 1=需生成，0=已知 |
| **扩散过程** | Mask-aware（只对未知区域加噪） | 全序列均匀加噪（mask 信息通过 VACE concat 传入） |
| **模型输入** | `concat([x_t, mask])` = 2 × dim | `concat([x_t, inactive, reactive, mask])` = 4 × dim |
| **预测目标** | x₀（完整干净 motion） | velocity v 或 x₁ |
| **loss target** | 原始干净 motion（即使输入是退化版本） | flow matching target |
| **模型既学去噪也学修复** | ✅ 同时 | ❌ 只学 mask 补全 |

**最关键的差异**：MoGenDiT 的训练让模型同时学会了"从噪声恢复"和"修复真实 motion 缺陷"两个能力。HyMotion M2M 只学了前者（从 noise/mask 生成），没有修复缺陷的能力。

### 加噪/Mask 策略的精确差异

这是两个系统最核心的设计差异，必须精确理解。

**MoGenDiT 的加噪流程**：
```
1. mask=1 的帧（已知）→ x_t 保持原值，不加噪
   模型在 x_t 中直接看到干净值

2. mask=0 的帧（待生成）→ x_t = √ᾱ·x0 + √(1-ᾱ)·ε
   标准 DDPM 加噪

3. input = concat([x_t, mask])
   模型知道 x_t 中哪些是干净的（mask=1），哪些是 noisy 的（mask=0）
```

**HyMotion M2M 的加噪流程**：
```
1. 所有帧均匀加噪 → x_t = (1-t)*noise + t*clean
   已知帧也被加了噪声！x_t 中看不到任何干净值

2. 已知帧信息通过 VACE 传入：
   inactive = src_motion * (1 - mask)  →  已知帧的干净值（独立通道）
   reactive = 全零
   src_mask = mask

3. input = concat([x_t, inactive, reactive, src_mask])
   模型要从 inactive 通道"读取"已知信息，然后"写回"到输出
```

**核心区别**：

| | MoGenDiT | HyMotion M2M |
|---|---|---|
| **已知帧在 x_t 中的值** | **干净**（不加噪） | **noisy**（和生成帧一样加噪） |
| **已知帧信息传递路径** | 直接在 x_t 中 + mask 标记 | 间接通过 VACE inactive 通道 |
| **推理 imputation 的一致性** | ✅ 训练时已知帧=干净，推理时恢复=干净，**分布一致** | ⚠️ 训练时已知帧=noisy，推理时恢复=干净，**分布不一致** |

**这解释了为什么**：
- MoGenDiT 可以在推理时每步 impute 已知帧，且效果好——因为训练时已知帧就是干净的
- M2M 如果推理时做 imputation，可能出问题——因为训练时已知帧是 noisy 的，突然变干净会 OOD
- M2M 要做 imputation，理想方案是训练时也改成 mask-aware 加噪（已知帧不加噪），但这需要改 flow matching 插值逻辑

### 已知帧精确保留：Imputation 变体实验设计

MoGenDiT 的推理 imputation（每步恢复已知帧）保证输出中已知帧**和输入完全一致**（零误差）。
这对 completion 和 edit 任务都有用——用户明确不想改的部分应该完全不变。

以下变体需要分别实验对比：

| 变体 | 训练时加噪 | 推理时 imputation | 训练/推理一致性 | 预期效果 |
|------|----------|-----------------|---------------|---------|
| **V0 (current)** | 全序列均匀加噪 | 不做 imputation | ✅ 一致 | [P]-MPJPE > 0，已知帧不精确 |
| **V1 (P2 exact_match)** | 全序列均匀加噪 | ODE 完成后才覆盖 | ⚠️ 不完全一致（但最后一步影响小） | [P]-MPJPE = 0，边界可能 jitter |
| **V2 (C1 每步 impute)** | 全序列均匀加噪 | 每步恢复已知帧 | ❌ 不一致（训练时 noisy，推理时干净） | 不确定，需要实验 |
| **V3 (C1 skip_last)** | 全序列均匀加噪 | 每步恢复，最后一步除外 | ❌ 不一致但最后一步可调和 | MoGenDiT 默认，可能最平衡 |
| **V4 (mask-aware 训练+impute)** | **已知帧不加噪** | 每步恢复已知帧 | ✅ 一致 | 理论最优，但需要改 flow matching |
| **V5 (mask-aware 训练+skip_last)** | **已知帧不加噪** | 每步恢复，最后一步除外 | ✅ 基本一致 | V4 的平滑版 |

**建议执行顺序**：V1（零成本）→ V3（改推理代码）→ V4/V5（改训练逻辑）

---

## 一、动作表示

### MoGenDiT：OccamMotionRep（201-dim 有效，263-dim 填充）

代码：`motion_process/motion_representation.py`

| 分量 | 维度 | 说明 |
|------|------|------|
| `pose` | 132 | 22 关节 × 6D rotation（**column-major** rot6d） |
| `joint` | 66 | 22 关节 × 3D 局部位置（缩放 /2） |
| `stationary` | 22 | 22 关节 × 1D 接触状态标志 |
| padding | 43 | 零填充到 263 |
| **有效维度** | **220** | `data_dim = 22×6 + 22×3 + 22×1 = 220` |

> 注：模型 input dim 为 263（含 padding），但实际有效信息只有 220 维。

### HyMotion M2M：135-dim

| 分量 | 维度 | 说明 |
|------|------|------|
| `translation` | 3 | absolute translation |
| `rotation` | 132 | 22 关节 × 6D rotation（**row-major** rot6d） |
| **合计** | **135** | |

### 关键差异

| | MoGenDiT | HyMotion M2M |
|---|---|---|
| **Translation** | 3d absolute（在 joint 中隐式包含，通过 root joint） | 3d absolute（显式独立维度） |
| **Rotation 6D 约定** | **Column-major** `[R00,R10,R20,R01,R11,R21]` | **Row-major** `[R00,R01,R10,R11,R20,R21]` |
| **Joint position** | ✅ 22×3=66 dims（局部，缩放 /2） | ❌ 无 |
| **Stationary/Contact** | ✅ 22×1=22 dims（逐关节接触标志） | ❌ 无 |
| **Velocity** | ❌ 不在表示中（在 loss 中计算） | ❌ 无 |

**⚠️ Rotation 6D 约定不兼容**：两者的 rot6d 不可直接混用，需要 `[0,2,4,1,3,5]` 重排转换。

### Normalization

| | MoGenDiT | HyMotion M2M |
|---|---|---|
| **方式** | **Egocentric 对齐**（运行时计算） | **Per-dim mean/std**（预统计文件） |
| **水平位置** | 第 0 帧 pelvis = 原点 | 按 mean 减均值 |
| **朝向** | 旋转到第 0 帧正前方 = +Z | 不做朝向对齐 |
| **高度** | 前 60 帧最低关节 = 地面 = 0 | 按 mean 减均值 |
| **外部文件** | 不需要 | 需要 `Mean.npy` / `Std.npy` |

```python
# MoGenDiT egocentric normalization 核心流程
# motion_process/motion_representation.py L800-840
def normalization(motion, height_reset=True):
    pose, joint, trans = decode(motion)
    global_joint = joint + trans.unsqueeze(1)
    # 1. 水平：pelvis[0] = 原点
    trans[:, [0,2]] -= trans[0:1, [0,2]]
    # 2. 朝向：旋转整段 motion 使第 0 帧面向 +Z
    R_ego = get_ego_gv(R_root[0]).T
    # ... 应用 R_ego 到 pose 和 joint ...
    # 3. 高度：前 60 帧最低点 = 地面
    if height_reset:
        init_h = global_joint[:60, :, 1].min()
        trans[:, 1] -= init_h
    return encode(pose, joint, trans)
```

---

## 二、模型架构（MoreDiff DiT）

代码：`model/more_diff.py`、`model/my_model.py`

### 三个规模

| | MoreDiff-0.03B | MoreDiff-0.1B（推荐） | MoreDiff-0.3B |
|---|---|---|---|
| Depth | 8 | 12 | 18 |
| Hidden dim | 512 | 768 | 1024 |
| Heads | 8 | 12 | 16 |
| 参数量 | ~30M | ~100M | ~300M |

### 架构特性

| 组件 | 实现 | 代码位置 |
|------|------|---------|
| **Input projection** | `Linear(motion_dim × 2, d_model)` — 输入 = concat([x_t, mask]) | `model/more_diff.py` `motion_2_token` |
| **Output projection** | `Linear(d_model, motion_dim)` | `token_2_motion` |
| **Conditioning** | **AdaLN-Zero**：timestep embedding → 6 个调制参数（scale/shift/gate × MSA + FFN） | `model/more_diff.py` L438 `AdaLN` |
| **位置编码** | **RoPE**（Rotary Position Embedding），freq_base=10000，max_len=5000 | 每个 attention head 独立应用 |
| **注意力** | **滑动窗口**，window_size=90 帧（@30fps ≈ 3 秒上下文） | `get_window_mask()` L83-95 |
| **FFN** | d_model → 2×d_model → d_model，SiLU 激活 | `DiTBlock` |
| **Dropout** | 0.1（可配置） | |

### 与 HyMotion M2M（HunyuanMotion MMDiT）的对比

| | MoGenDiT | HyMotion M2M |
|---|---|---|
| **架构类型** | 标准 DiT + AdaLN-Zero | MMDiT（6 双流 + 12 单流） |
| **参数量** | 0.1B（推荐） | 0.46B |
| **输入格式** | `concat([x_t, mask])` = 2 × motion_dim | `concat([x_t, inactive, reactive, mask])` = 4 × motion_dim (VACE) |
| **Conditioning** | AdaLN（仅 timestep） | AdaLN + 双流 text cross-attention |
| **位置编码** | RoPE（旋转式，per-head） | RoPE |
| **注意力窗口** | 90 帧 | 60 帧（narrowband） |
| **Motion 条件** | 无显式条件（输入就是 degraded motion + mask） | VACE（4 通道 concat） |
| **Text 条件** | ❌ | ✅ Qwen3-8B + CLIP-L |

---

## 三、扩散框架

代码：`EasyDiffusion/base_diffusion.py`

### 核心参数

| | MoGenDiT | HyMotion M2M |
|---|---|---|
| **框架** | **DDPM**（GaussianDiffusion） | **Flow Matching**（Rectified Flow） |
| **训练 timesteps** | 1000（离散） | 连续 t ∈ [0, 1] |
| **Noise schedule** | **Cosine** β schedule（s=0.008） | 无（线性插值 `x_t=(1-t)x0+tx1`） |
| **预测目标** | **x₀**（直接预测去噪后的 motion）— `ModelMeanType.START_X` | **velocity** `v = x1 - x0` 或 **x₁** |
| **推理采样** | DDPM 或 DDIM（支持自定义时间步） | Euler ODE（50 步） |
| **Fast sampling** | ✅ 10 步 DDIM：`[999, 750, 500, 250, 100, 50, 25, 10, 5, 0]` | 50 步 Euler（固定） |

### 训练前向过程

```python
# EasyDiffusion/base_diffusion.py — q_sample
x_t = sqrt(alpha_cumprod_t) * x0 + sqrt(1 - alpha_cumprod_t) * noise

# Mask-aware：只对 unmasked 区域加噪
noise_mask = ~obs_mask & length_mask
x_t[noise_mask] = x_noise[noise_mask]
```

**关键设计**：MoGenDiT 的前向过程是 **mask-aware** 的——已知区域（mask=1）不加噪，只对待修复区域加噪。这和 KIMODO 的 imputation 思路一致。

### 推理反向过程（DDIM）

```python
# DDIM sampling（eta=0 时完全确定性）
pred_x0 = model(x_wrap, t)                    # 模型直接预测 x0
mean = sqrt(a_prev) * pred_x0 + sqrt(1-a_prev-σ²) * (x_t - sqrt(a_t)*pred_x0) / sqrt(1-a_t)
x_{t-1} = mean + σ * noise
```

---

## 四、训练 Loss

代码：`trainer/my_trainer.py`

### Loss 组成

```
L_total = L_pose + L_joint + L_trans + L_velocity [+ L_kinematic]
```

| Loss 项 | 公式 | 作用 |
|---------|------|------|
| **L_pose** | `weighted_masked_loss(pred_pose, gt_pose)` | Rotation 6D 重建 |
| **L_joint** | `weighted_masked_loss(pred_joint, gt_joint)` | Joint position 重建 |
| **L_trans** | `weighted_masked_loss(pred_trans, gt_trans)` | Stationary/contact 重建 |
| **L_velocity** | `masked_loss(pred_vel, gt_vel)` — `vel = global_joint[1:] - global_joint[:-1]` | 帧间平滑 |
| **L_kinematic**（可选） | FK(pred_rotation) vs pred_joint_position 一致性 | Rotation↔Position 一致性 |

### 与 HyMotion M2M Loss 对比

| | MoGenDiT | HyMotion M2M |
|---|---|---|
| **主 loss** | MSE/L1 on x0（直接预测 motion） | SmoothL1 on velocity/x1 |
| **Velocity loss** | ✅ 帧差分 `joint[1:]-joint[:-1]` | ❌ 无 |
| **FK loss** | ✅ 可选 `kinematic_loss_batch()`，训练时启用 | 代码已实现但**未使用**（`keypoints3d_weight=0`） |
| **Joint position loss** | ✅ 直接监督 66-dim joint positions | ❌ 表示中无 joint position |
| **Translation 加权** | 不单独加权 | 5× 加权（`trans_dim_weight=5.0`） |
| **Contact/Stationary loss** | ✅ 22-dim stationary 监督 | ❌ 无 |
| **Geometric loss**（可选） | ✅ 刚体约束 + FK + drift | ❌ 无 |

### Kinematic Loss 详情

代码：`motion_representation.py` `kinematic_loss_batch()`

```python
def kinematic_loss_batch(self, pose_6d, joint, length, l1_weight, l2_weight):
    # 1. 从 rotation 做 FK → 预测的 joint positions
    fk_joint = forward_kinematics(pose_6d)

    # 2. FK 一致性：FK 得到的 position 应该等于表示中的 position
    L_fk = MSE(fk_joint, joint)

    # 3. Velocity 一致性：相邻帧位移应该平滑
    vel_fk = fk_joint[1:] - fk_joint[:-1]
    vel_joint = joint[1:] - joint[:-1]
    L_vel = MSE(vel_fk, vel_joint)

    return L_fk + L_vel
```

### Geometric Loss 详情

代码：`trainer/geometric_loss.py`

| 项 | 公式 | 作用 |
|---|---|---|
| Rigid body | `\|bone_length_pred - bone_length_gt\|²` | 骨骼长度恒定 |
| FK consistency | `\|FK(rotation) - joint_position\|²` | 旋转和位置一致 |
| Drift | `\|Δposition - velocity × Δt\|²` | 速度和位移一致 |

---

## 五、修复模式

代码：`motion_process/motion_refiner.py`

### 三种模式

| 模式 | 做法 | 推理步数 | eta | 适用场景 |
|------|------|---------|-----|---------|
| `denoise` | 加少量噪声 → 去噪 | 10 步 DDPM | 1.0（随机） | 轻微噪声/抖动修复 |
| `ada_denoise` | 自适应噪声量 → 去噪 | 自适应 | 可配置 | 自动判断缺陷程度 |
| `trans_regen` | 只重新生成 translation（保持 pose 不变） | 10 步 DDIM | 0.0（确定性） | root 位移异常（脚滑、穿地） |

### Windowed Processing（长序列处理）

```python
def refine(motion, window_size=224, prev_padding=1):
    # 1. 先统一 normalize（补丁修复：不再每窗口独立 normalize）
    motion = normalization(motion, height_reset=True)

    # 2. 滑窗处理
    for each window [begin, end]:
        # 窗口至少 120 帧（补丁修复：旧版 30 帧）
        end = max(end, begin + prev_padding + 120)

        # 距离监控：如果 root 位移 > 4m，截断窗口
        if dist > cutoff_dist:
            end = cutoff_point

        # 修复当前窗口
        window_motion = refine_one_window(motion[begin:end])

    # 3. 物理模拟后处理（补丁新增）
    if flat_ground_trans:
        motion = FlatGroundSimulator.correct_translation(motion)
```

### Imputation 模式（DDIM loop 中）

| 模式 | 行为 | 用途 |
|------|------|------|
| `skip_last` | 每步恢复 mask 区域的原始值，**最后一步除外** | 默认，平衡约束精度和生成自由度 |
| `all` | 每步都恢复（包括最后一步） | 最严格的约束遵守 |
| `none` | 从不恢复 | 完全自由重新生成 |

---

## 六、物理模拟后处理（FlatGroundSimulator）

代码：`animo/simulator.py`（补丁升级版）

### 流程

```
修复后 motion
  → decode 为 pose + joint + trans
  → 计算全局关节位置和速度
  → FlatGroundSimulator 逐帧模拟：
      1. 接地检测（任意关节，基于速度 + 关节底部高度）
      2. FK 位移计算（以最稳定接地关节为参考）
      3. QP 优化融合（FK 位移 vs 期望位移）
      4. 浮空状态机（概率平滑切换）
  → 输出修正后的 translation
  → encode 回 motion tensor
```

### 接地检测（补丁升级）

| | 旧版 | 新版（补丁） |
|---|---|---|
| 检测关节 | 仅左右脚 (joint 10, 11) | **任意 22 关节** |
| 高度判断 | 关节中心 Y | 关节**底部** Y（中心 - 半径） |
| 速度阈值 | 0.1 m/s | 0.15 m/s |
| 接地条件 | 速度 < 阈值 | 速度 < 阈值 **且** 底部高度 ∈ [-0.1m, 0.1m] |

每个 SMPL 关节的物理半径（`skeleton/smpl_body.py` `JOINT_RADII`）：
- Pelvis/Hip/Spine/Head：0.10m
- Clavicle：0.08m
- Knee/Neck/Shoulder：0.05m
- Ankle/Elbow/Wrist：0.02m
- Foot (toe)：0.01m

### QP 优化

```python
# 融合 FK 位移和期望位移
d_trans = w_qp * d_trans_fk + (1 - w_qp) * d_trans_desired
# w_qp = 0.5（固定），eps = 1e-2
# QP solver: quadprog，warm-start from des_qdot
```

---

## 七、EMA 和 Checkpoint

| | MoGenDiT | HyMotion M2M |
|---|---|---|
| **EMA decay** | 0.999 | 无（消融实验 T1 拟加 0.995） |
| **EMA 起始步** | 2000 步后 | — |
| **Checkpoint 格式** | `model_XXXX.pth` + `ema_model_XXXX.pth` | Accelerate `save_state()` |
| **推理用 EMA** | ✅ 推荐 | — |

---

## 八、HyMotion M2M 可借鉴的技术点

### ✅ 训练方式（最高优先级）

| # | 技术点 | MoGenDiT 实现 | 对 M2M 的价值 | 难度 |
|---|--------|-------------|-------------|------|
| **B0a** | **Motion Degradation 训练** | 50% batch 施加合成退化（8 种类型），loss target 为原始干净 motion | M2M 当前只学 mask 补全，缺乏修复缺陷的能力。加入 degradation 训练后，模型能同时学会"补全"和"修复" | P1 |
| **B0b** | **Mask-Aware 加噪** | 已知帧不加噪（`obs_mask=1` 区域保持干净值） | 让训练和推理 imputation 的分布一致。这是 V4/V5 变体的前提 | P1 |
| **B0c** | **推理 Imputation** | 每个去噪步强制恢复已知帧值（skip_last 模式） | 保证已知帧零误差。变体 V1-V5 见上方实验设计 | P0-P1 |
| **B0d** | **block_trans / joint_only mask** | 只 mask translation 或只保留 joint position 的 mask 模式 | M2M 已有 M1-M6 策略，但这两种新模式值得加入 | P0 |

### ✅ Loss 设计（高优先级）

| # | 技术点 | MoGenDiT 实现位置 | 对 M2M 的价值 | 难度 |
|---|--------|-----------------|-------------|------|
| **B1** | **FK consistency loss** | `motion_representation.py` `kinematic_loss_batch()` | M2M 已有接口但未启用。对应消融实验 **L1** | P0 |
| **B2** | **Velocity smoothness loss** | `trainer/my_trainer.py`：帧差分 L1 | 减少 jitter，对应消融实验 **L4** | P0 |
| **B3** | **Geometric loss（刚体约束）** | `trainer/geometric_loss.py`：骨骼长度恒定 + drift 惩罚 | 防止骨骼长度漂移 | P1 |

### ✅ 后处理和推理（中优先级）

| # | 技术点 | MoGenDiT 实现位置 | 对 M2M 的价值 | 难度 |
|---|--------|-----------------|-------------|------|
| **B4** | **FlatGroundSimulator 后处理** | `animo/simulator.py` | 物理模拟修正脚滑/穿地/悬浮 | P1 |
| **B5** | **高度重置** | `normalization(height_reset=True)` | 前 60 帧最低点 = 地面 | P0 |
| **B6** | **Trans-only 重生成** | `trans_regen` 模式 | VACE mask 可实现：mask translation dims=1，rotation dims=0 | P0 |
| **B7** | **滑窗长序列处理** | `refiner.refine()` window_size=224 | M2M 推理长序列时可借鉴 | P1 |

### ✅ 需要表示扩展（低优先级）

| # | 技术点 | MoGenDiT 实现位置 | 对 M2M 的价值 | 难度 |
|---|--------|-----------------|-------------|------|
| **B8** | **Joint position 在表示中** | OccamMotionRep 的 66-dim joint position | 验证了 joint position 纳入表示是可行的 | P2 |
| **B9** | **Stationary/Contact 在表示中** | OccamMotionRep 的 22-dim stationary | 逐关节接触状态作为显式特征 | P2 |
| **B10** | **Egocentric 对齐作为预处理** | `normalization()` 中旋转到第 0 帧朝向 | 减轻模型学习全局朝向的负担 | P1 |

### ⚠️ 集成注意事项

| 风险 | 说明 | 解决方案 |
|------|------|---------|
| **Rotation 6D 约定** | MoGenDiT column-major ≠ M2M row-major | 接口处做 `[0,2,4,1,3,5]` 重排 |
| **Motion dim** | MoGenDiT 263 ≠ M2M 135 | 需要 135↔201 转换（补全 joint position 用 FK） |
| **Normalization** | Egocentric ≠ mean/std | 在接口处显式转换 |

---

## 九、完整技术规格对比

| 维度 | MoGenDiT | HyMotion M2M |
|------|----------|-------------|
| **任务** | 运动修复（去噪/重生成） | 运动补全/生成 |
| **Motion dim** | 220（padded 263） | 135 |
| **表示内容** | pose(132) + joint(66) + stationary(22) | transl(3) + rotation(132) |
| **Rotation 约定** | Column-major rot6d | Row-major rot6d |
| **Normalization** | Egocentric（运行时对齐） | Per-dim mean/std（预统计） |
| **模型架构** | DiT + AdaLN-Zero（0.03B/0.1B/0.3B） | MMDiT 双流+单流（0.46B） |
| **Input format** | `concat([x_t, mask])` = 2 × dim | `concat([x_t, inactive, reactive, mask])` = 4 × dim (VACE) |
| **位置编码** | RoPE (per-head) | RoPE |
| **注意力窗口** | 90 帧 | 60 帧 |
| **Conditioning** | AdaLN（仅 timestep） | AdaLN + text cross-attention |
| **Text 条件** | ❌ | ✅ Qwen3-8B + CLIP-L |
| **扩散框架** | DDPM（1000 步，cosine schedule） | Flow Matching（连续 t∈[0,1]） |
| **预测目标** | **x₀**（直接预测去噪 motion） | **velocity** v=x1-x0 或 x₁ |
| **推理步数** | 10 步 DDIM（fast sampling） | 50 步 Euler ODE |
| **Loss** | pose + joint + velocity + trans [+ kinematic + geometric] | SmoothL1(velocity/x1) + trans_weight |
| **FK loss** | ✅ 训练时启用 | 代码存在但未使用 |
| **Velocity loss** | ✅ 帧差分 MSE | ❌ |
| **Contact modeling** | ✅ 22-dim stationary + 物理模拟 | ❌ |
| **EMA** | ✅ decay=0.999, start=2000 | ❌（消融实验 T1 拟加） |
| **物理后处理** | ✅ FlatGroundSimulator | ❌ |
| **长序列** | 滑窗（window=224, padding=1） | 截断到 max_frames |

---

## 十、补丁更新报告（2026-03-24）

### 补丁文件

| 文件 | 对应路径 | 改动 |
|------|---------|------|
| `motion_refiner.py` | `motion_process/` | 分段衔接 bug 修复 + 物理模拟后处理 |
| `motion_representation.py` | `motion_process/` | 高度重置逻辑修正 |
| `data_loader.py` | `trainer/` | 训练时启用高度重置 |
| `animo/` | `animo/` | 平地运动学模拟升级 |

### 关键改动

**1. 分段衔接 Bug 修复**：旧版每个窗口独立 normalize → 新版统一 normalize 一次。`prev_padding` 从 20→1 帧。

**2. 高度重置修正**：地面估计从只用第 0 帧 → 用前 60 帧最低点。`pre_stitch` 中被 `pass` 禁用的代码重新启用。

**3. 物理模拟后处理**：新增 `flat_ground_trans=True`，在修复末尾用 `FlatGroundSimulator` 修正 root translation。

**4. 通用接地检测**：从只看脚 → 任意关节可接地（per-joint 半径）；FK 位移参考关节优化；QP warm-start。
