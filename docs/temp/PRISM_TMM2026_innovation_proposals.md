# PRISM TMM 2026 创新模块提案：详细调研与修改建议

> 生成日期: 2026-05-12
> 目标: 解决 PRISM 论文"效果好但缺少具有创新性的模块"的核心问题

---

## 一、问题诊断

### 1.1 ECCV 审稿人一致指出的根本问题

三位审稿人的 weakness 高度一致:
- **Reviewer cL6x**: "incremental novelty" — per-joint tokenization 来自 MoGenTS, noise-free conditioning 来自 Diffusion Forcing
- **Reviewer qAQ1**: "both contributions exist in prior work" — 缺乏独立的技术创新点
- **Reviewer FG3s**: "limited contribution" — 工程组合而非方法创新

### 1.2 TMM 版本现状

TMM 版本通过 "honest novelty decomposition" 策略改善了定位,但**仍未添加任何新的创新模块**。这对 TMM (CCF-B 期刊) 可能勉强够,但并不保险——尤其是审稿人可能来自同一pool。

### 1.3 核心矛盾

PRISM 的架构是:
```
SMPL Motion → Joint-Factorized VAE (from MoGenTS) → 2D Latent Grid → Flow-Matching DiT (from Wan/ViMoGen) → Per-Token Timestep (from Diffusion Forcing) → Self-Forcing (from Self-Forcing)
```

每一个模块都有明确的 prior art。**需要至少一个具有技术深度的新模块**,使论文从 "careful engineering combination" 升级为 "has its own technical contribution"。

---

## 二、调研发现：近期 SOTA 方法的创新点分析

| 方法 | 创新点 | 与PRISM的关系 |
|------|--------|--------------|
| **ANT** (2025) | 频率感知的自适应去噪阶段 (低频结构→高频细节)，Dynamic CFG | 可借鉴频域思想 |
| **Free-T2M** (2025) | DCT 低频一致性 loss + 语义一致性 loss，阶段感知训练 | 频域 loss 可直接适配 |
| **LMR** (Think Before You Move, 2025) | 双粒度 tokenizer (reasoning latent + execution latent)，两阶段 think-then-act | 层次化生成思路 |
| **POMP** (CVPR 2025) | 运动学-动力学双模块 + 相位流形桥接 | 物理一致性思路 |
| **FlashMo/MotionSiT** (2025) | SO(3) 李群关节旋转扩散，频率感知 token 稀疏化 | 旋转空间几何结构 |
| **UniMoGen** (2025) | 运动学感知注意力掩码 (joint-ancestor attention) | 结构化注意力 |
| **MOGO** (2025) | 残差量化层次因果 Transformer | 层次化 token |
| **Part-Joint Attention** (2025) | 动态图卷积 + 部位-关节注意力 | body-part grouping |
| **HY-Motion** (2024) | 窗口注意力 + 非对称 masking + DPO/GRPO | 训练范式 |
| **CoDA** (2025) | 分体扩散 + 梯度流协调 (body/left-hand/right-hand) | 分体去噪 |
| **RoPAR** (2025) | per-joint 置信度驱动自适应去噪 | per-joint 自适应噪声 |

---

## 三、创新模块提案

基于调研,我提出 **3个候选创新模块** (按推荐优先级排序),以及 **2个辅助改进**。建议选择 **1-2个核心模块** 实施。

---

### 提案 A: Kinematic-Aware Structured Attention (KASA) — 运动学感知的结构化注意力 ⭐⭐⭐ (强烈推荐)

#### 动机

PRISM 当前的 DiT 使用标准 full self-attention: 所有 joint tokens 之间平等交互。但人体运动有明确的**运动学层次结构** (kinematic tree): 肩膀带动肘部、肘部带动手腕、髋部带动膝盖再带动脚踝。这种 parent-child 依赖关系在标准 attention 中完全被忽略——模型必须从数据中隐式学习这些耦合,浪费了大量容量。

这一点是 PRISM 独有的机会: **正因为 latent space 是 joint-factorized 的 2D grid,我们才能在 attention 层面引入运动学结构先验**。这在 monolithic 1D latent 中根本无法实现。

#### 核心设计

```
Standard DiT Attention:
  All K=23 joint tokens ↔ All K=23 joint tokens (full attention)

KASA (Kinematic-Aware Structured Attention):
  Level 1 — Kinematic Group Attention:
    将 23 个 joints 按运动学链分为 5 个 body part groups:
    - Trunk: root, pelvis, spine1, spine2, spine3, neck, head (7 joints)
    - Left Arm: L_collar, L_shoulder, L_elbow, L_wrist (4 joints)
    - Right Arm: R_collar, R_shoulder, R_elbow, R_wrist (4 joints)
    - Left Leg: L_hip, L_knee, L_ankle, L_foot (4 joints)
    - Right Leg: R_hip, R_knee, R_ankle, R_foot (4 joints)

  Level 2 — Cross-Group Routing Attention:
    仅在 group 代表 token (group mean/CLS) 之间做 attention,
    然后将信息 route 回各 group 内部

  交替使用:
    - 偶数层: intra-group attention (每个 body part 内部交互, 建模运动学链)
    - 奇数层: inter-group routing (body part 之间协调, 建模如手臂-腿的协调摆动)
```

#### 为什么这是新的

- **UniMoGen** 用 ancestor-only attention mask,但它是为 skeleton-agnostic (跨物种) 设计的,且在 generation 层面未验证
- **Part-Joint Attention** 用于 motion prediction (预测),不是 generation (生成),且没有层次 routing 机制
- **PRISM 的 KASA** 特别之处在于:
  1. **双层 attention**: 既有 intra-group (建模运动学链的局部耦合) 又有 inter-group routing (建模 body part 间的全局协调),而非简单的 mask
  2. **与 per-joint 2D latent grid 的天然配合**: 这是 joint-factorized latent 的 "专属" 创新——因为 latent 本身就是 per-joint 的,才能做 kinematic-aware attention
  3. **计算效率**: Full attention 复杂度 O(K²),KASA 是 O(G·k² + G²),当 K=23, G=5, k≈5 时有 ~3× 加速

#### 架构集成

```python
# 在现有 DiT block 中替换 spatial (joint-axis) attention:
class KASABlock(nn.Module):
    def __init__(self, dim, groups=KINEMATIC_GROUPS):
        self.intra_group_attn = nn.ModuleList([
            MultiHeadAttention(dim) for _ in groups
        ])
        self.group_router = MultiHeadAttention(dim)  # 5×5 cross-group
        self.group_proj = nn.Linear(dim, dim)  # route back

    def forward(self, x, t_embed):
        # x: [B, T', K, D] — joint-factorized latent grid
        if self.layer_idx % 2 == 0:
            # Intra-group: each body part attends internally
            out = []
            for g, group_joints in enumerate(self.groups):
                x_g = x[:, :, group_joints, :]  # [B, T', k_g, D]
                out.append(self.intra_group_attn[g](x_g))
            return concat_groups(out)
        else:
            # Inter-group routing
            group_reps = [x[:, :, g, :].mean(dim=2) for g in self.groups]
            group_reps = stack(group_reps)  # [B, T', G, D]
            routed = self.group_router(group_reps)
            # Scatter back to individual joints
            return scatter_to_joints(routed, x)
```

#### 预期效果

- **改善远端关节质量** (手腕、脚踝): 运动学链先验使旋转误差在链上的传播更可控
- **减少 inter-limb 不协调**: 明确的 trunk-limb 信息传递比全局 attention 更精准
- **提升长序列稳定性**: 结构化 attention 限制了误差跨 body-part 的传播
- **计算效率**: 减少 attention FLOPs 约 2-3×

#### 对审稿人的说服力

- 这是一个 **genuinely novel module**: 不是简单的 mask/group attention,而是 kinematic-tree-aware 的双层 routing 机制
- 它 **只有在 joint-factorized latent 下才有意义**: 强化了 "per-joint latent + structured attention = co-design" 的叙事
- 它有 **明确的消融空间**: full attention vs. group-only vs. KASA → 证明运动学先验的价值
- **审稿人 cL6x 的 W2** 提到 "VAE design doesn't interact with generation" — KASA 直接反驳: VAE 的 per-joint 结构使 DiT 能用 kinematic attention

---

### 提案 B: Frequency-Aware Joint-Adaptive Flow Matching (FAJFM) — 频率感知的关节自适应流匹配 ⭐⭐⭐ (强烈推荐)

#### 动机

当前 PRISM 的 flow matching 对所有 joint tokens 使用**相同的噪声调度**。但不同关节的运动特性差异巨大:
- **Root/pelvis**: 低频为主 (大尺度轨迹变化,平滑)
- **Spine joints**: 中频 (上身摆动)
- **Extremities (wrists, ankles, head)**: 高频为主 (快速摆动、触地反弹等)

生物力学研究也证实:近端关节 (proximal) 通常控制低频的姿态规划,远端关节 (distal) 负责高频的精细执行。**One-size-fits-all 的噪声调度忽略了这种频谱异质性**,导致:
- 远端关节在低噪声时仍欠拟合 (高频细节丢失)
- 近端关节在高噪声时过度扰动 (低频结构被破坏)

#### 核心设计

**Per-Joint Adaptive Noise Schedule**: 每个 joint token 不再共享全局时间步 t,而是根据其**运动学位置** (在 kinematic tree 上的深度) 使用自适应的噪声水平:

```
t_j = t · α_j + β_j

where:
  t: global flow-matching timestep (shared)
  α_j: per-joint noise sensitivity (learnable or depth-based)
  β_j: per-joint noise offset
  depth_j: distance from root in kinematic tree
```

具体来说:
1. **深度自适应调度 (Depth-Adaptive Schedule)**:
   - Root (depth=0): 使用 t_root = t · 0.8 (更少噪声,保护轨迹结构)
   - Mid-chain (depth=1-2): 使用 t_mid = t · 1.0 (标准调度)
   - Distal (depth=3-4, e.g., wrist/ankle): 使用 t_distal = t · 1.2 (更多噪声,鼓励学习高频)

2. **频率引导的训练 Loss**:
   - 对 velocity prediction $v_θ$ 施加 **DCT 频域加权**:
   ```
   L_freq = Σ_j w_j · ||DCT(v_θ^j) - DCT(v_gt^j)||²
   where w_j 对远端关节的高频分量加权更大
   ```

3. **可学习的关节重要性权重**:
   - 在 DiT 的 timestep embedding 之后,添加一个小型 MLP 预测每个 joint 的"噪声灵敏度系数" α_j
   - 这使模型能**自适应地**学习每个关节在不同去噪阶段的最优噪声水平

#### 为什么这是新的

- **ANT** 做的是时间维度的频率自适应 (early steps = low freq, late steps = high freq),但 **所有 joints 共享** 同一调度 → 没有 per-joint 区分
- **Free-T2M** 做的是 DCT loss,但在 monolithic latent 上,且仅是 loss 层面,没有噪声调度层面的改变
- **RoPAR** 做了 per-joint confidence-based masking,但用于 motion prediction 而非 generation
- **PRISM 的 FAJFM** 独特之处在于:
  1. 将 **空间 (per-joint)** 和 **时间 (denoising step)** 两个维度的频率特性统一建模
  2. **运动学深度驱动**: 噪声调度与 kinematic tree depth 直接关联,有物理解释性
  3. 只有 **joint-factorized latent** 才能做 per-joint noise scheduling — 这再次强化了 "structured latent space 的独特价值" 叙事
  4. 这实际上是对 PRISM 已有的 "per-token timestep conditioning" 的**自然推广**: 从 "binary (clean vs. noisy)" 推广到 "continuous per-joint noise levels"

#### 架构集成

```python
# 扩展现有的 per-token timestep embedding
class JointAdaptiveTimestep(nn.Module):
    def __init__(self, dim, num_joints=23):
        # Kinematic depth prior
        self.register_buffer('joint_depth',
            torch.tensor([0, 0, 1, 2, 3, 4, 1, 2, 3, 4, ...]))  # SMPL tree depth
        self.depth_to_alpha = nn.Sequential(
            nn.Embedding(5, dim),  # depth 0-4
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )  # outputs α_j ∈ [0, 1] → rescale to [0.7, 1.3]
        self.beta = nn.Parameter(torch.zeros(num_joints))

    def forward(self, t, joint_idx):
        # t: [B, T'] global timestep
        alpha = self.depth_to_alpha(self.joint_depth[joint_idx])
        alpha = 0.7 + 0.6 * alpha  # [0.7, 1.3]
        t_joint = t * alpha + self.beta[joint_idx]
        return t_joint.clamp(0, 1)
```

#### 预期效果

- **远端关节质量提升**: 适当增加远端噪声 → 更好的高频学习 → 手腕/脚踝动作更精细
- **轨迹稳定性**: 适当减少 root 噪声 → 更稳定的全局运动规划
- **MBench 物理指标改善**: per-joint 自适应有望减少 jitter、foot-slide
- **FID/R-Precision 同时改善**: 更精确的噪声分配 → 更高效的生成学习

#### 对审稿人的说服力

- **与 per-token timestep 的深度关联**: 将 binary per-token timestep 推广为 continuous per-joint adaptive schedule,这是一个自然但 non-trivial 的推广
- **物理可解释性**: kinematic depth 直接对应运动频率特性,有生物力学依据
- **只有在 joint-factorized latent 上才能实现**: 再次论证 "latent design 是被忽略的关键维度"
- **回应审稿人 qAQ1 的 "novelty concerns"**: 这不再是简单组合现有方法,而是利用结构化 latent 的特性提出的新噪声调度范式

---

### 提案 C: Kinematic Chain Consistency Regularization (KCCR) — 运动学链一致性正则化 ⭐⭐ (推荐)

#### 动机

PRISM 的 VAE 有 FK loss 监督,但在 **DiT 生成端** 没有任何运动学结构约束。生成的 per-joint latents 可能在 decode 后出现运动学不一致: 父关节和子关节的旋转组合产生不自然的姿态。

现有方法 (如 POMP) 通过物理模拟来保证一致性,但代价太高。我们可以利用 joint-factorized latent 的结构,在 **生成过程中** (每个 Euler step) 施加轻量级的运动学链一致性约束。

#### 核心设计

**Generation-Time Kinematic Consistency Loss**: 在 flow matching 的 training loss 中,除了标准的 velocity MSE,添加一个**运动学链一致性项**:

```
L_total = L_flow + λ_kc · L_kinematic_chain

L_kinematic_chain = Σ_{(parent,child) ∈ tree}
    || FK(θ̂_parent, θ̂_child) - FK(θ_parent, θ_child) ||²
```

其中 FK 是 **可微的前向运动学** (利用 SMPL 的运动学树),直接在 DiT 预测的 velocity field 上计算。

更具体:
1. **One-step denoised estimate**: 用当前预测的 v_θ 做一步 Euler 更新,得到估计的 clean latent ẑ₀
2. **VAE decode ẑ₀ → rotation parameters**
3. **FK chain**: 计算运动学链上相邻关节对的 joint position consistency
4. **反传梯度**: 这个 loss 反传到 DiT 的参数

#### 为什么这是新的

- PRISM 的 FK loss 仅在 **VAE** 层面 → 确保 reconstruction 的运动学一致性
- 但 **generation** 层面没有这种约束 → 当 DiT 生成的 latent 分布偏离 VAE 训练分布时,decode 可能产生运动学不一致
- KCCR 将 FK 约束从 VAE 端扩展到 **DiT generation 端**,形成端到端的运动学一致性
- 不同于 POMP 的物理模拟 (需要场景/接触信息),KCCR 是纯运动学的、轻量级的

#### 架构集成

```python
# 在 DiT 训练 loss 中添加
class KinematicChainConsistency(nn.Module):
    def __init__(self, vae_decoder, smpl_model):
        self.vae_decoder = vae_decoder  # frozen
        self.smpl = smpl_model  # frozen, for FK

    def forward(self, v_pred, z_t, t, z_0_gt):
        # One-step estimate of clean latent
        z_0_hat = z_t - t * v_pred  # Euler estimate
        # Decode to rotation space
        rot_hat = self.vae_decoder(z_0_hat)  # [B, T, 23, 6]
        rot_gt = self.vae_decoder(z_0_gt)
        # FK: compute joint positions
        pos_hat = self.smpl.forward_kinematics(rot_hat)  # [B, T, 23, 3]
        pos_gt = self.smpl.forward_kinematics(rot_gt)
        # Chain consistency: parent-child position error
        return F.mse_loss(pos_hat, pos_gt)
```

#### 预期效果

- **减少 generation 端的运动学不一致** (尤其是远端关节位置误差)
- **MBench physics metrics 改善**: jitter, foot-float 等
- **缩小 "20× rFID 不能翻译为 20× generation FID" 的 gap**
- **MPJPE 进一步降低**

#### 限制

- 训练时需要通过 frozen VAE decoder 做 forward pass → 增加约 20% 训练时间
- 需要权重调节 λ_kc 避免过度约束 DiT 的创造力

---

## 四、辅助改进提案

### 辅助 D: Per-Joint Importance Weighting in Loss — 关节重要性加权损失

在 DiT 的 flow matching loss 中,不同 joint token 使用不同的 loss 权重:

```
L = Σ_j w_j · || v_θ^j - v_gt^j ||²
```

其中 w_j 基于:
- **运动学深度**: 远端关节权重更大 (FK 误差放大效应)
- **运动幅度统计**: 高方差关节权重更大
- **可学习**: 用辅助网络预测

这不是一个独立的大创新,但可以作为 KASA 或 FAJFM 的自然组成部分,增加技术深度。

### 辅助 E: Temporal-Frequency Decomposed Loss — 时频分解损失

将 Free-T2M 的 DCT 低频一致性 loss 适配到 PRISM 的 per-joint latent:

```
L_tfd = Σ_j [ λ_low · ||DCT_low(v_θ^j) - DCT_low(v_gt^j)||²
           + λ_high · ||DCT_high(v_θ^j) - DCT_high(v_gt^j)||² ]
```

- 早期去噪步 (t 大): 加重 λ_low (保护全局轨迹结构)
- 晚期去噪步 (t 小): 加重 λ_high (精细化关节运动细节)
- 与 FAJFM 配合使用效果最佳

---

## 五、推荐实施方案

### 方案 1: KASA + FAJFM (最强,两个新模块) ⭐⭐⭐

**叙事**: PRISM 的 joint-factorized latent space 不仅提升 reconstruction 质量,更**解锁了两种结构感知的生成技术**:
1. KASA: 利用 per-joint latent 的空间结构,引入运动学感知的分层注意力
2. FAJFM: 利用 per-joint latent 的语义对应,引入关节自适应的噪声调度

**核心论点升级**: "We demonstrate that a structured latent space is not merely beneficial for reconstruction, but is a **prerequisite** for kinematic-aware generation — enabling both structured attention (KASA) and joint-adaptive noise scheduling (FAJFM) that are impossible in monolithic latent spaces."

**Contribution 重写**:
1. Joint-factorized latent space (systematic extension of MoGenTS)
2. **Kinematic-Aware Structured Attention** (KASA) — novel, enabled by joint-factorized latent ← 新增
3. **Frequency-Aware Joint-Adaptive Flow Matching** (FAJFM) — novel, extends per-token timestep ← 新增
4. Per-token timestep conditioning + Self-Forcing for streaming
5. SOTA on HumanML3D, MotionHub, BABEL + user study

### 方案 2: KASA 单独 (性价比最高,一个核心新模块) ⭐⭐⭐

**叙事**: Joint-factorized latent 的价值不止于 reconstruction — 它使 DiT 能感知人体的运动学结构。KASA 将运动学树先验注入 attention,与 per-joint latent 形成 co-design。

**好处**: 实施相对简单 (只改 attention 层),消融干净,物理直觉强。

### 方案 3: FAJFM 单独 (与 per-token timestep 最连贯) ⭐⭐

**叙事**: 将 PRISM 的 per-token timestep 从 binary (clean/noisy) 推广为 continuous per-joint adaptive schedule,利用运动学深度作为频率先验。

**好处**: 与现有 per-token timestep 机制高度融合,实施改动最小。

---

## 六、消融实验设计 (无论选哪个方案)

### 对于 KASA:
| 变体 | 描述 |
|------|------|
| Full attention (baseline) | 当前 PRISM |
| Intra-group only | 仅 body part 内部 attention |
| Intra + inter (KASA) | 完整方案 |
| Random grouping | 验证运动学分组比随机分组更好 |

### 对于 FAJFM:
| 变体 | 描述 |
|------|------|
| Shared timestep (baseline) | 当前 PRISM |
| Fixed depth-based α | 固定的深度→噪声映射 |
| Learnable α (FAJFM) | 可学习的关节自适应调度 |
| + DCT freq loss | 叠加频域损失 |

### 对于 KCCR:
| 变体 | 描述 |
|------|------|
| VAE FK only (baseline) | 当前 PRISM |
| + Gen-time FK (KCCR) | 添加生成端 FK 一致性 |

---

## 七、回应 ECCV 审稿人 Weakness 的映射

| 审稿人 Weakness | 提案如何回应 |
|----------------|-------------|
| **cL6x-W1**: Per-joint from MoGenTS, inpainting from Diffusion Forcing | KASA/FAJFM 是**只有在 per-joint latent 上才能做的新方法**,证明 per-joint 不只是 MoGenTS 的 copy |
| **cL6x-W2**: VAE design doesn't interact with generation | KASA 直接利用 VAE 的 per-joint 结构来改进 DiT 的 attention; FAJFM 利用 joint 的运动学属性做自适应调度 |
| **qAQ1-W1**: Both contributions exist in prior work | KASA 和 FAJFM 是全新的技术贡献,不存在于任何 prior work |
| **qAQ1-W3**: Missing stepwise VAE ablation | 6个消融维度 (见第六节) |
| **FG3s-W3**: Limited contribution | 从 2 个贡献升级为 4 个,其中 2 个是全新的 |
| **FG3s-W4**: 20× rFID doesn't translate | KCCR 将 FK 约束从 VAE 延伸到 generation 端,缩小 gap |

---

## 八、实施可行性评估

| 模块 | 代码改动量 | 训练成本增加 | 可行性 |
|------|-----------|-------------|--------|
| **KASA** | ~200 行 (attention 层替换) | ~10% (attention 计算量可能减少) | ✅ 高 |
| **FAJFM** | ~100 行 (timestep embedding 扩展) | ~5% (仅 embedding 计算) | ✅ 高 |
| **KCCR** | ~150 行 (loss 添加) | ~20% (需 VAE forward) | ✅ 中高 |
| **辅助 D** | ~30 行 (loss weight) | ~0% | ✅ 极高 |
| **辅助 E** | ~50 行 (DCT loss) | ~5% | ✅ 高 |

所有模块都可以在现有 PRISM 框架上**增量实施**,不需要重新设计架构。

---

## 九、原始建议 (需修订)

> ⚠️ **以下原始建议基于可以重新训练的假设,但用户明确要求"尽量采取能够使用当前预训练模型的方案"。**
> PRISM 预训练模型 (1.4B DiT, 300K steps on 80 V100s) 成本极高,全量重训不现实。
>
> 请参看 **第十节: 预训练模型兼容的修订方案** 了解修订后的推荐。

**原方案 1 (KASA + FAJFM): ❌ KASA 不兼容预训练模型**
- KASA 改变了 attention 结构 (full attention → group attention + routing),所有 pretrained attention weights 失效
- 需要完全从头训练,成本不可接受

**原方案 2 (仅 KASA): ❌ 同上**

**原方案 3 (仅 FAJFM): ⭕ 部分兼容,需分层讨论 — 见第十节**

---

## 十、预训练模型兼容的修订方案 ⭐⭐⭐⭐⭐ (核心章节)

### 10.0 兼容性分析总结

通过详细阅读 PRISM 源码 (transformer_prism.py, embedding.py, prism_backend.py, prism_trainer.py, bundle.py),对各候选创新模块与预训练 DiT 模型的兼容性做出如下判定:

| 模块 | 兼容性 | 原因 | 推荐度 |
|------|--------|------|--------|
| **KASA (结构化注意力)** | ❌ 不兼容 | Transformer 所有层使用 `WanTransformerBlockWithMask` 的 full self-attention,改为 group attention 会使所有 QKV 权重失效 | ❌ 放弃 |
| **FAJFM-Inference (推理时关节自适应时间步)** | ✅ 完全兼容 | 仅修改 denoising loop 中 `temp_ts` 的标量值;模型的 sinusoidal→MLP→projection pipeline 接收任意标量输入,无需改权重 | ⭐⭐⭐⭐⭐ |
| **FAJFM-FT (可学习关节时间步参数)** | ✅ 兼容 (轻量微调) | 在 `condition_embedder` 前插入 ~200 参数的 learnable α_j/β_j 模块,冻结 DiT backbone, 仅微调 5K-10K steps | ⭐⭐⭐⭐ |
| **KCCR-Guidance (推理时 FK 梯度引导)** | ⚠️ 部分兼容 | 需移除 `@torch.no_grad()`,添加 FK loss backprop 到 latents (不修改模型权重),增加推理时间 ~50% | ⭐⭐⭐ |
| **Loss-FT (DCT 频域 + 关节权重 loss)** | ✅ 兼容 (轻量微调) | 仅修改 `prism_trainer.py` 的 loss 函数,模型架构不变,微调 5K-10K steps | ⭐⭐⭐ |

### 兼容性关键发现 — 时间步嵌入管线分析

PRISM 的时间步处理管线 (embedding.py `WanTimeTextEmbedding`):

```
原始时间步标量 t (每个 token 独立)
  ↓ Timesteps(freq_dim=256)        # 正弦编码: scalar → [freq_dim]
  ↓ TimestepEmbedding(freq_dim → dim)  # MLP: 线性→SiLU→线性
  ↓ SiLU activation
  ↓ Linear(dim → 6*dim)           # 投影: → 6 个 AdaLN 调制参数
  ↓ AdaLN modulation per block     # 作用于每一层的 hidden states
```

**关键**: 这条管线的输入是任意标量值 `t`。无论 `t = 500` 还是 `t = 500 * 1.2 + 0.1 = 600.1`,正弦编码和 MLP 都能处理。PRISM 已原生支持 per-token timestep (训练时 condition frames 的 t=0 与生成 frames 的 t>0 共存)。因此,修改每个 joint token 的 timestep 标量值**完全兼容**预训练权重。

具体注入点在 `prism_backend.py` 第 287 行:
```python
temp_ts = (first_frame_mask[0][0] * t).flatten()  # ← 这里是标量,可以乘以 per-joint α 系数
```

---

### 10.1 修订方案 A: Kinematic-Adaptive Flow Scheduling (KAFS) — 运动学自适应流调度

> 对应原提案 B (FAJFM) 的预训练兼容版本,分为推理版和微调版两个层级。

#### 10.1.1 KAFS-Inference: 零成本推理增强 (Zero-Shot, 无需任何训练)

**原理**: 在 Euler ODE 积分的每一步,根据每个关节在 SMPL 运动学树上的深度,对其 timestep 值进行缩放:

```
t_j = t · α_j

where:
  t: 全局 flow-matching timestep (scheduler 给出的, 如 999, 950, ...)
  α_j: 关节 j 的噪声灵敏度系数, 由运动学深度决定
```

**SMPL-22 关节运动学深度与 α_j 映射**:

| 深度 | 关节 | α_j (推荐初始值) | 直觉 |
|------|------|-----------------|------|
| 0 | Pelvis (root) | 0.85 | 根部: 低频主导, 减少噪声→保护全局轨迹 |
| 1 | L_Hip, R_Hip, Spine1 | 0.90 | 近端: 结构性关节, 略减噪声 |
| 2 | L_Knee, R_Knee, Spine2 | 0.95 | 中间: 接近标准调度 |
| 3 | L_Ankle, R_Ankle, Spine3, Neck, L_Collar, R_Collar | 1.00 | 标准调度 |
| 4 | L_Foot, R_Foot, Head, L_Shoulder, R_Shoulder | 1.05 | 远端: 高频细节, 略增噪声 |
| 5 | L_Elbow, R_Elbow | 1.10 | 远端: 手臂摆动高频 |
| 6 | L_Wrist, R_Wrist | 1.15 | 末梢: 最高频, 最大噪声 |

> 注: α_j > 1 意味着该关节获得更大的有效噪声水平,鼓励模型在高噪声阶段更充分地修正高频成分;
> α_j < 1 意味着该关节获得更小的有效噪声水平,让低频结构在去噪过程中更快收敛/更稳定。

**精确代码注入点** (prism_backend.py, denoising loop):

```python
# === 当前代码 (line 287) ===
temp_ts = (first_frame_mask[0][0] * t).flatten()

# === 修改为 KAFS-Inference ===
# SMPL-22 运动学深度 → α 系数 (23 个 joint token: 1 transl + 22 joints)
# 注意: PRISM 的 joint tokens 包含 translation token (index 0),
# translation 不属于旋转关节,使用 α=0.85 (类似 root)
KAFS_ALPHA = torch.tensor([
    0.85,  # transl token (类同 root)
    0.85,  # Pelvis (depth=0)
    0.90, 0.90,  # L_Hip, R_Hip (depth=1)
    0.90,  # Spine1 (depth=1)
    0.95, 0.95,  # L_Knee, R_Knee (depth=2)
    0.95,  # Spine2 (depth=2)
    1.00, 1.00,  # L_Ankle, R_Ankle (depth=3)
    1.00,  # Spine3 (depth=3)
    1.05, 1.05,  # L_Foot, R_Foot (depth=4)
    1.00,  # Neck (depth=3)
    1.00, 1.00,  # L_Collar, R_Collar (depth=3)
    1.05,  # Head (depth=4)
    1.05, 1.05,  # L_Shoulder, R_Shoulder (depth=4)
    1.10, 1.10,  # L_Elbow, R_Elbow (depth=5)
    1.15, 1.15,  # L_Wrist, R_Wrist (depth=6)
], device=latents.device, dtype=latents.dtype)

# 需要根据 patch_size 和 latent 布局将 23 个 alpha 映射到 flatten 后的 token 序列
# 如果 patch_size=(1,1), 则 latent_joints = 23, alpha 直接 tile T 次
alpha_map = KAFS_ALPHA.unsqueeze(0).expand(latent_frames, -1).flatten()  # [T'*J]
temp_ts = (first_frame_mask[0][0] * t * alpha_map.to(t.device)).flatten()
timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
```

**预期效果**:
- Root/pelvis 的 effective timestep 被缩小 15% → 全局轨迹更稳定、减少 drift
- Wrist/ankle 的 effective timestep 被放大 10-15% → 高频细节更丰富、减少手脚僵硬
- **零训练成本**: 直接在推理代码中添加 ~15 行,不碰模型权重

**风险与缓解**:
- 过大的 α 偏移 (如 α_j=0.5 或 α_j=2.0) 会超出模型训练分布 → 初始值保守 (0.85~1.15 范围)
- 可能需要对不同 scheduler (如 Euler vs DDIM) 的 timestep 数值范围做归一化
- 建议做 α range 的 grid search: {0.90~1.10}, {0.85~1.15}, {0.80~1.20}

#### ~~10.1.2 KAFS-FT: 可学习关节自适应参数~~ ❌ 已移除 (需要 5K-10K steps 微调)

> **移除原因**: 时间限制,无法进行任何重新训练。KAFS-Inference (§10.1.1) 已提供零训练成本的完整方案。如有后续时间,可在论文接收后作为 camera-ready 补充实验。

#### 10.1.2 论文叙事

**KAFS 的核心论点**: PRISM 的 per-token timestep 机制 (源自 Diffusion Forcing 的 binary clean/noisy 设定) 在 joint-factorized latent space 中获得了全新的语义——每个 joint token 的 timestep 对应该关节的**局部去噪进度**。KAFS 将这一机制从 binary (condition frame t=0 vs. generation frame t>0) 推广为 **continuous per-joint adaptive scheduling**:

- Binary per-token timestep: `t_j ∈ {0, t}` (Diffusion Forcing)
- KAFS: `t_j = t · α_j + β_j` where α_j, β_j 由运动学深度驱动

这一推广只有在 **joint-factorized latent space** 中才有意义——因为 monolithic 1D latent 中没有关节语义,无法定义 per-joint 的 α_j。**KAFS 证明了 per-joint latent 不仅改善了 reconstruction 质量 (VAE 层面),更解锁了 generation 端的关节自适应去噪控制 (DiT 层面)——这是 monolithic latent 无法实现的。**

**与 prior work 的区分**:
- ANT: 时间维度的频率自适应 (low freq → high freq across denoising steps),所有 token 共享
- RoPAR: per-joint confidence-based masking for motion prediction, 不涉及 diffusion timestep
- Diffusion Forcing: per-token timestep 但仅 binary, 无运动学结构先验
- **KAFS**: 首次将运动学结构先验注入 per-token flow-matching timestep scheduling

---

### 10.2 修订方案 B: FK-Guided Denoising (FKGD) — 前向运动学引导去噪

> 对应原提案 C (KCCR) 的推理时兼容版本。

**核心思路**: 在 Euler ODE 积分的每一步,利用 SMPL 前向运动学 (FK) 计算 one-step denoised estimate 的关节位置一致性 loss,将梯度反传到 **latents** (不修改模型权重),实现运动学一致性引导。

这与 classifier-free guidance 的思路类似,但用的是**运动学 FK 损失**而非分类器:

```
Algorithm: FK-Guided Euler Step

Input: latents z_t, timestep t, model θ (frozen)
1. z_t.requires_grad_(True)                        # 开启梯度
2. v_pred = DiT_θ(z_t, t)                           # 模型预测 velocity
3. z_0_hat = z_t + (1-t)/t * (z_t - v_pred)         # one-step denoised estimate (Tweedie-like)
   (或简化: z_0_hat = z_t - t * v_pred)
4. motion_hat = VAE_decode(z_0_hat)                  # decode → [B, T, 23, 6] rot6d
5. positions_hat = FK(motion_hat)                    # SMPL FK → [B, T, 22, 3] joint positions
6. L_fk = FK_consistency_loss(positions_hat)         # 运动学一致性损失 (见下)
7. grad = torch.autograd.grad(L_fk, z_t)[0]
8. z_t_guided = z_t - λ_fk * grad                   # 梯度引导
9. z_t.requires_grad_(False)
10. z_{t-1} = scheduler.step(v_pred, t, z_t_guided)  # 正常 Euler step
```

**FK 一致性损失设计**:

```python
def fk_consistency_loss(positions, velocities=None):
    """
    positions: [B, T, J, 3] — FK 计算的关节世界坐标
    """
    # 1. 骨骼长度一致性: 相邻关节间距离应在帧间保持恒定
    bone_lengths = []
    for parent, child in SMPL_KINEMATIC_EDGES:
        bl = (positions[:, :, child] - positions[:, :, parent]).norm(dim=-1)  # [B, T]
        bone_lengths.append(bl)
    bone_lengths = torch.stack(bone_lengths, dim=-1)  # [B, T, num_bones]
    bone_var = bone_lengths.var(dim=1)  # 帧间方差,应为 0
    L_bone = bone_var.mean()

    # 2. 关节加速度平滑性 (anti-jitter)
    if positions.shape[1] >= 3:
        accel = positions[:, 2:] - 2 * positions[:, 1:-1] + positions[:, :-2]
        L_smooth = accel.norm(dim=-1).mean()
    else:
        L_smooth = 0.0

    # 3. 脚部触地约束 (anti-skating)
    foot_indices = [7, 8, 10, 11]  # L_Ankle, R_Ankle, L_Foot, R_Foot
    foot_y = positions[:, :, foot_indices, 1]  # Y 坐标
    foot_contact = (foot_y < 0.05).float()  # 触地帧
    if foot_contact.sum() > 0:
        foot_vel = (positions[:, 1:, foot_indices] - positions[:, :-1, foot_indices]).norm(dim=-1)
        L_foot = (foot_vel[:, :] * foot_contact[:, 1:]).mean()
    else:
        L_foot = 0.0

    return L_bone + 0.5 * L_smooth + 0.3 * L_foot
```

**实现要点**:
1. **移除 `@torch.no_grad()`**: prism_backend.py 的 denoising loop 和 `__call__` 方法有 `@torch.no_grad()` 装饰,需移除以支持梯度
2. **VAE 和 model 都 freeze**: 梯度只回传到 `z_t`,不更新任何模型参数 (`model.eval()`, `vae.eval()`)
3. **计算成本**: 每步多 1 次 VAE decode + 1 次 FK + 1 次反传,推理时间增加约 50-100%
4. **可选: 仅在特定步骤应用** (如最后 10 步或每隔 5 步),减少计算开销

**预期效果**:
- 减少运动学不一致 (骨骼长度变化、穿模)
- 改善物理指标: jitter, foot-skating, bone-length-error
- 对定量指标 (FID, R-Precision) 影响较小 (FK guidance 主要改善物理合理性)

**风险**:
- 梯度引导强度 λ_fk 需要仔细调节: 太大会扰乱 ODE 轨迹,太小效果不明显
- 推理速度降低: 在实时/streaming 场景中可能不可接受
- 论文叙事上不如 KAFS 强 — FK guidance 是推理时 trick,不是训练创新

---

### ~~10.3 修订方案 C: Frequency-Aware Loss Fine-Tuning (FALF)~~ ❌ 已移除 (需要 5K-10K steps 微调)

> **移除原因**: 时间限制,无法进行任何重新训练。FALF 需要 5K-10K steps 的微调训练,不符合 zero-training-cost 约束。如有后续时间,可在论文接收后作为补充实验。
>
> **注意**: FALF 中的 translation/rotation 分开计算均值的思想已独立实现到当前 `PrismTrainer.train_step()` 中 (通过 `translation_loss_weight` 参数),这一改动不需要重新训练即可应用于未来的训练 run。

---

### 10.4 修订后的推荐实施方案 (仅 inference-time 方法)

> ⚠️ **约束**: 没有时间进行任何重新训练。以下所有方案均为 zero-training-cost,直接使用现有预训练模型。

#### 方案 1-Revised: KAFS-Inference + (可选) FKGD ⭐⭐⭐⭐⭐ (首选)

**核心叙事**: PRISM 的 joint-factorized latent space 不仅提升 reconstruction 质量,更**解锁了关节自适应的去噪控制** — 这在 monolithic latent 中不可能实现。

1. **KAFS-Inference** (第 10.1.1 节): 核心创新模块
   - 推理时运动学深度驱动的 per-joint timestep scaling (零成本)
   - 通过 α_j 的 grid search 获得最优超参
2. **FKGD** (第 10.2 节): 可选增强
   - 推理时 FK consistency guidance (如果物理指标需要进一步提升)

**Contribution 列表重写**:

1. **Joint-factorized causal VAE**: Per-joint 2D latent grid with streaming capability (systematic extension)
2. **Kinematic-Adaptive Flow Scheduling (KAFS)**: Per-joint timestep modulation driven by kinematic tree depth — the first method to inject anatomical structure priors into per-token flow matching scheduling ← **核心新贡献**
3. Per-token timestep + Self-Forcing for streaming generation
4. SOTA on HumanML3D, MotionHub, BABEL + user study + ablation

**总成本**:
- 代码改动: ~15 行 (KAFS-Inference)
- 训练成本: **零** — 直接使用现有 checkpoint
- 推理成本增加: 几乎为零 (KAFS 仅多一次 element-wise 乘法)

#### 方案 2-Revised: 仅 KAFS-Inference (最低成本) ⭐⭐⭐⭐

如果时间极其紧迫:
- KAFS-Inference: 零训练,直接在推理代码中添加 ~15 行
- 通过 α_j 的 grid search 获得最优超参
- 论文聚焦 "per-joint latent enables per-joint adaptive scheduling" 的叙事
- 消融: full-model baseline → KAFS fixed α → random α (control) → uniform α (control)

**最大优势**: 完全不需要微调,所有现有 checkpoint 直接使用

---

### 10.5 完整消融实验设计 (方案 1-Revised, 仅 inference-time)

> ⚠️ 所有消融变体均为 inference-time,zero-training-cost。

| # | 变体 | KAFS | FKGD | 训练成本 |
|---|------|------|------|---------|
| 1 | Baseline (PRISM) | ❌ | ❌ | 0 |
| 2 | + KAFS-Inference (fixed α, depth-driven) | ✅ fixed | ❌ | 0 |
| 3 | + KAFS-Inference + FKGD | ✅ fixed | ✅ | 0 |
| 4 | Random α (control) | ✅ random | ❌ | 0 |
| 5 | Uniform α=1.2 (control) | ✅ uniform | ❌ | 0 |

**消融目标**:
- #2 vs #1: 验证运动学深度先验的价值 (zero-cost)
- #3 vs #2: 验证 FK guidance 在 KAFS 基础上的额外增益
- #4 vs #2: 验证运动学分组比随机分组更好 (关键! 反驳 "arbitrary per-joint scaling" 的质疑)
- #5 vs #2: 验证差异化 α 比 uniform scaling 更好

**评估指标**:
| 类别 | 指标 |
|------|------|
| 生成质量 | FID, R-Precision (Top-1/2/3), MM-Dist |
| 物理合理性 | Jitter, Foot-sliding, Bone-length-error, MPJPE |
| 关节级分析 | Per-joint MPJPE (root vs mid vs distal), Per-joint jitter |
| 频率分析 | Low-freq error (DCT <2Hz), High-freq error (DCT >8Hz) |
| 流式生成 | Boundary smoothness (segment junction quality) |

---

### 10.6 回应 ECCV 审稿人 Weakness 的更新映射

| 审稿人 Weakness | KAFS 如何回应 | 补充回应 |
|----------------|--------------|---------|
| **cL6x-W1**: Per-joint from MoGenTS, inpainting from DF | KAFS 是**只有在 per-joint latent 上才能做的新调度方法**,证明 per-joint 不只是 MoGenTS 的 copy — 它解锁了新的生成控制维度 | FKGD 进一步利用了关节结构 |
| **cL6x-W2**: VAE design doesn't interact with generation | KAFS 直接利用 VAE 的 per-joint 结构来控制 DiT 的 per-token timestep scheduling。VAE 的关节分解 → DiT 的运动学自适应去噪 = 端到端的 co-design | FK guidance 从 VAE 反传到 DiT |
| **qAQ1-W1**: Both contributions exist in prior work | KAFS 是全新贡献: 首次在 flow matching 中引入运动学结构先验的 per-joint timestep 调度 | 5 个消融实验 |
| **qAQ1-W3**: Missing stepwise ablation | 5 个消融维度 (见 10.5), 含 per-joint 分析和频率分析 | Random/uniform control 验证 |
| **FG3s-W3**: Limited contribution | 从 2 个贡献升级为 3 个,其中 KAFS 是全新的核心技术贡献 | TMM 篇幅允许详细展开 |
| **FG3s-W4**: 20× rFID doesn't translate | FKGD 直接改善 generation-side 的物理指标 | Per-joint 指标分析 |

---

## 十一、实施路线图

### Phase 1: KAFS-Inference 验证 (1-2天)

1. 在 `prism_backend.py` 的 denoising loop 中添加 KAFS α_j mapping (~15 行)
2. 在 HumanML3D test set 上运行 baseline vs. KAFS-Inference
3. 做 α range grid search: {0.90~1.10}, {0.85~1.15}, {0.80~1.20}
4. 收集 FID, R-Precision, per-joint MPJPE, jitter 指标
5. 如果 KAFS-Inference 有明显提升 → 进入 Phase 2; 否则调整 α 策略

### ~~Phase 2: KAFS-FT + FALF 微调~~ ❌ 已移除

> 需要训练,不符合 zero-training-cost 约束。

### Phase 2 (原 Phase 3): 消融与论文 (2-3天)

1. 运行 5 个 zero-cost 消融实验 (见 §10.5)
2. 收集所有指标,制作 ablation tables
3. 撰写 KAFS 方法章节 + 实验分析
4. 更新论文 contribution 列表

### Phase 3 (可选): FKGD 实现 (2天)

1. 移除 `@torch.no_grad()`, 添加 FK guidance 代码
2. 调节 λ_fk 梯度引导强度
3. 评估推理速度和质量 trade-off

**总时间**: 约 5-7 天 (Phase 1-2 为核心,Phase 3 为可选)

> ⚠️ **约束**: 所有方案均使用当前预训练模型,零训练成本。

---

## 参考文献

- ANT: https://arxiv.org/abs/2506.02452
- Free-T2M: https://arxiv.org/abs/2501.18232
- LMR (Think Before You Move): https://arxiv.org/abs/2512.24100
- POMP: CVPR 2025
- FlashMo/MotionSiT: SO(3) rotation diffusion
- UniMoGen: skeleton-agnostic motion generation
- Part-Joint Attention: https://link.springer.com/article/10.1038/s41598-025-18520-x
- RoPAR: per-joint confidence denoising
- CoDA: coordinated part-wise diffusion
- MoGenTS: NeurIPS 2024, per-joint tokenization
- Diffusion Forcing: NeurIPS 2024, per-token noise levels
- Self-Forcing: autoregressive self-conditioning
