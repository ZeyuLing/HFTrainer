# ref_repo — 参考工作索引

本目录收录与 HyMotion M2M 相关的参考工作（论文 + 开源代码），用于技术对比和借鉴。

**消融实验清单**：[m2m_ablation_experiments.md](m2m_ablation_experiments.md)（25 个实验，基于 KIMODO / UMO 对比设计）

---

## 工作列表

| 工作 | 机构 | 时间 | 是否开源 | 核心任务 | 分析文档 |
|------|------|------|---------|---------|---------|
| **KIMODO** | NVIDIA | 2026-3-16 | ✅ 已开源 | Text-to-Motion, Keyframe, End-effector, Trajectory, Multi-prompt | [KIMODO/CLAUDE.md](KIMODO/CLAUDE.md) |
| **UMO** | Brown / MIT / Meta / MPI / HKU | 2026-3-16 | ❌ 承诺开源，暂未发布 | T2M, Temporal Inpainting, Instruction Editing, Trajectory, Obstacle Avoidance, Reaction Generation | [UMO/CLAUDE.md](UMO/CLAUDE.md) |
| **MoGenDiT** | 内部（chengxuzuo） | 2026-3-24 补丁 | 内部代码 | 扩散修复（去噪、位移重生成、分段衔接） | [MoGenDiT/CLAUDE.md](MoGenDiT/CLAUDE.md) |
| **SOAR** | NUS / Alibaba / Microsoft | 2026-04 | ❌ 暂未开源 | Diffusion Post-Training：Exposure Bias Correction via On-Policy Rollout + Dense Self-Correction | [SOAR/CLAUDE.md](SOAR/CLAUDE.md) |
| **StableMotion** | SFU / Lightspeed Studios / NRC Canada | SIGGRAPH Asia 2025（2026-04 归档） | ✅ 已开源 | Motion Cleanup / Detect-and-Fix on Unpaired Corrupted Data | [StableMotion/CLAUDE.md](StableMotion/CLAUDE.md) |

---

## 阅读指引

每个工作的 `CLAUDE.md` 包含：

1. **基本信息**：论文标题、作者、时间、代码链接
2. **论文核心内容**：问题定位、主要创新点、网络架构、动作表示、训练细节、支持任务
3. **对比分析**：与 HyMotion M2M（我方工作）的详细对比表格，以及核心设计理念差异的深度分析
4. **可借鉴的点**：对我方工作有实际参考价值的技术点

---

## 快速摘要

### KIMODO（NVIDIA，开源）

**核心贡献**：
- 两阶段 Transformer Denoiser（root model + body model，interleaved 训练）
- Global joint rotation 表示（世界坐标系 6D rotation，无需 FK chain 即可直接 imputation）
- Smooth root（对 pelvis 水平分量平滑，减少足部滑动）
- Imputation-based conditioning（`x̃_t = m ⊙ x_tgt + (1-m) ⊙ x_t`，concat binary mask）
- 700 小时 optical mocap（Bones Rigplay），生产级别质量

**与我方差异**：DDPM vs Flow Matching；Global rotation vs Local rotation（SMPL）；Imputation vs VACE；无 per-joint mask（joint-level，不是 dim-level）；两阶段 vs 单模型

**参考价值**：Global rotation 对世界坐标系约束有优势；Separated CFG；Foot contact 显式建模；两阶段训练 curriculum

---

### UMO（Brown/MIT/Meta，暂未开源）

**核心贡献**：
- 三种帧级元操作嵌入 `[preserve]`/`[generate]`/`[edit]`，组合描述任意任务
- Temporal Fusion：element-wise add to input embedding，仅 0.207M 额外参数
- 几何约束全部序列化为结构化文本，无需专用 spatial conditioning 模块
- 单统一模型覆盖 6+ 种任务，多任务 joint training 反而提升各任务性能

**与我方差异**：帧级（whole-body）vs 逐维度（T×138）；[edit] 语义 vs 二值 mask（缺 edit 概念）；temporal fusion vs VACE；缺 part-level control（UMO 自身 limitation）

**参考价值**：[edit] 操作概念扩展 M2M 的语义表达能力；几何约束 via 结构化文本；E_ctx 初始化复用预训练权重

---

### SOAR（NUS/Alibaba/Microsoft，暂未开源）

**核心贡献**：
- 针对 diffusion post-training 的 exposure bias 问题：SFT 在 GT forward process 状态上训练，但推理时 x_t 来自模型自身预测（off-trajectory）
- **On-policy rollout + re-noise + dense per-timestep correction**：对当前模型做 1 步 stop-gradient ODE rollout 得到 off-trajectory state，re-noise 后监督模型向 clean target 回溯
- 不需要 reward model、preference labels 或 negative samples — 完全 self-supervised
- SD3.5-Medium 上 GenEval 0.70→0.78，超越更大的 SD3.5-Large (0.71)

**与我方差异**：T2I vs Motion；Flow Matching 框架完全相同；SOAR 是 post-training 方案，不改训练数据或标注

**参考价值**：**直接适用于 M2M v2 post-training** — M2M 使用同样的 rectified flow 框架，存在明显的 exposure bias（50 步 ODE 误差累积、边界跳变）。与 `_man` 互补：_man 解决 known regions 分布匹配，SOAR 解决 generated regions 的 exposure bias。**不需要任何额外数据标注**。

---

### StableMotion（SFU / Lightspeed Studios，开源）

**核心贡献**：
- **Quality indicator 作为动作特征的额外通道**：把"当前帧是否损坏"的二值 label 作为表示向量的最后一维，和 body 特征一起参与扩散
- **两模式联合训练**：同一 batch 按 fraction 切分 — Detection mode（给 body，预测 label）+ Inpainting mode（给 label，重生成损坏帧 body），用 cosine schedule 的随机 mask ratio
- **Detect-then-Fix 推理**：先 MC 平均预测 label → ±1 帧膨胀 → 构建 inpainting mask → 重新扩散
- **SITS（Soft-Inpaint Time Schedule）**：逐帧自适应起始 timestep `ceil(sin((label+0.5)·π/2)·T)`，干净帧少步修正、脏帧大步重生成
- **Ensemble Cleanup**：best-of-N 候选 × 模型自身 re-detection 评分 argmin
- **可微 foot-lock classifier guidance**
- **Unpaired corrupted data paradigm**：训练无需 clean 参考数据，只需 quality 检测器做弱监督

**与我方差异**：任务是 cleanup（M2M 是 generation / completion）；DDPM (1000 train + DDIM infer) vs Flow Matching；Global SMPL RIFKE + 1-dim label vs 135-dim 无 label；帧级 label vs 帧×维度 mask；需要两阶段 × MC × ensemble 推理 vs 一次 50 步 ODE

**参考价值**：
- **P0**：quality channel 可直接并入 M2M 表示（缓解 `train_hymotion_400h.json` 中 ~85K 低质量样本稀释训练信号的问题，不用丢数据）；detect 阶段输出的 per-frame label 可以直接作为 MoGenDIT adaptive repair 的 mask（弥补当前 `ada_denoise` 未用 adaptive mask 的缺陷）
- **P1**：SITS 可移植到 M2M post-train；ensemble best-of-N 与 SOAR 正交，推理时额外一档质量提升
- **P2**：unpaired training 范式可直接对接 `motion_annot_web/quality_check_rules/` 的 P0-P2 checker，做 self-distillation 式循环迭代 cleanup 模型

---

## 三个模型实现 Motion Completion 的核心差异

三个模型都能做 motion in-between、keyframe infilling 等 completion 任务，但**实现方式有本质不同**。
以 "给定首尾帧，补全中间过渡" 这个最典型的 MIB 任务为例说明。

---

### 总览

| | KIMODO | UMO | HyMotion M2M（我方） |
|---|---|---|---|
| **一句话概括** | T2M 模型 + 推理时 imputation 注入约束 | T2M 模型 + 轻量 adapter 注入 context | 原生 completion 模型，训练时就学 mask 模式 |
| **训练时见过 completion 任务吗？** | Phase 2 见过（imputation + mask） | 见过（P/G/E 标记 + source motion） | 见过（VACE conditioning + M1-M6 mask 策略） |
| **约束如何进入模型** | 每个去噪步**硬替换** noisy x 的对应维度 | source motion 通过 **element-wise add** 注入 input embedding | source motion 通过 **channel concat** 作为额外输入通道 |
| **模型架构改动** | 输入维度 ×2（motion+mask） | 加 E_ctx MLP（0.207M），backbone 不变 | input_encoder 适配 4× motion_dim（VACE） |

---

### KIMODO：Imputation（推理时硬替换）

**核心思想**：模型本体是一个 T2M diffusion model。Motion completion 通过**推理时在每个去噪步硬替换约束维度**实现。

**训练**：
- Phase 1（500k steps）：纯 T2M，模型不知道约束的存在
- Phase 2（500k steps）：随机采样约束（keyframe position 等），impute 到输入，训练模型适应 "部分维度被替换" 的输入分布
- 输入 = `concat([x_imputed, mask], dim=-1)`，维度 333→666

**推理时 MIB 流程**（给定首尾帧 position）：
```
每个去噪步 t:
  1. x_t = 当前 noisy motion (333-dim)
  2. x_t[首帧, position_dims] = GT_首帧_position   ← 硬替换
     x_t[尾帧, position_dims] = GT_尾帧_position   ← 硬替换
  3. mask[首帧, position_dims] = 1
     mask[尾帧, position_dims] = 1
  4. model_input = concat([x_t, mask])   (666-dim)
  5. model 预测 x_0
  6. scheduler 计算 x_{t-1}
  7. 回到步骤 1，x_{t-1} 又会被硬替换
```

**约束生效原理**：
- Position 维度每步被强制覆盖 → 精确锁定
- **Rotation 维度不锁定** → 完全由模型去噪生成
- 模型靠 FK loss（训练时 γ=10）学到的 rotation↔position 统计相关性来"猜"合理的 rotation
- ⚠️ 最终渲染用 rotation 做 FK，和约束的 position **有几 cm 误差**

**特点**：
- 优势：T2M 质量不受影响（Phase 1 就是纯 T2M）；约束类型灵活（position/rotation/trajectory/heading 都能 impute）
- 劣势：约束是外部强加的，模型没有"理解"约束意图；position-only 约束时 rotation 可能不一致

---

### UMO：Temporal Fusion（context 注入 input embedding）

**核心思想**：在预训练好的 T2M 模型基础上，加一个极轻量的 adapter（E_ctx），把 source motion 作为 context 通过 element-wise add 注入 input embedding。

**训练**：
- 冻结 backbone（HY-Motion-Lite MMDiT），只训练 E_ctx MLP（0.207M）+ 3 个 meta-op embeddings
- 每帧标记为 Preserve/Generate/Edit 之一
- source motion + meta-op embedding 通过 E_ctx 编码后，与 backbone 的 input embedding 相加：
  ```
  x'_t = E_in(x_t) + E_ctx(s̃)    其中 s̃_i = source_motion_i + Emb(τ_i)
  ```
- 多任务联合训练（MIB、prediction、editing、reaction 等同时训练）

**推理时 MIB 流程**（给定首尾帧的完整 motion）：
```
准备阶段：
  source = [首帧_motion, 0, 0, ..., 0, 尾帧_motion]   (T, 201)
  τ =      [P,           G, G, ..., G, P          ]   (T,)
  s̃ = source + Emb(τ)

每个去噪步 t:
  1. x_t = 当前 noisy motion (201-dim)
  2. input_emb = E_in(x_t) + E_ctx(s̃)   ← context 通过加法注入
  3. model 正常去噪（backbone 看到的是 modified input embedding）
  4. scheduler 计算 x_{t-1}
  注意：没有任何硬替换操作
```

**约束生效原理**：
- Source motion 通过 E_ctx 编码后加到 input embedding → 模型在 latent space 里"知道"首尾帧是什么
- **完全由模型自主决定如何遵守约束** → 没有硬替换，约束是"建议"而非"强制"
- Preserve 帧的精度靠训练时的 loss 约束（[P]-MPJPE 指标），不是推理时保证
- ⚠️ P 帧的输出和原始 source 有轻微偏差（[P]-MPJPE ≈ 0.95mm，不是零）

**特点**：
- 优势：极轻量（0.207M 参数）；backbone 不变，T2M 能力完整保留；多任务统一；[edit] 语义强大
- 劣势：**帧级粒度**，无法做 per-joint 控制（论文 Limitation）；P 帧不精确；约束是软的，依赖模型学习

---

### HyMotion M2M：VACE（约束作为额外输入通道）

**核心思想**：把 source motion 和 mask 显式编码为额外输入通道，与 noisy motion 一起 concat 送入模型。模型从训练一开始就在各种 mask 模式下学习 completion。

**训练**：
- **不分阶段**，从第一步就同时训练 T2M 和 completion
- 6 种 mask 策略混合采样（M1 random cell 25%, M2 random block 15%, M3 temporal 25%, M4 joint 15%, M5 full mask 5%, M6 keyframe 15%）
- 输入 = `concat([x_t, inactive, reactive, src_mask], dim=-1)` = 4 × motion_dim
  ```
  inactive = src_motion * (1 - src_mask)   ← 已知部分的 motion 值
  reactive = src_motion * src_mask          ← 待生成区域的 motion 值（split reactive）
  src_mask                                  ← 二值 mask (T, 135)
  ```
- input_encoder 适配 4×135=540 dim 输入

**推理时 MIB 流程**（给定首尾帧的完整 rotation）：
```
准备阶段：
  src_motion = [首帧_motion, 0, 0, ..., 0, 尾帧_motion]   (T, 135)
  src_mask   = [0,           1, 1, ..., 1, 0          ]   (T, 135)
  inactive   = src_motion * (1 - src_mask)    ← 首尾帧 motion 值
  reactive   = src_motion * src_mask          ← 全零（中间帧无 source）

每个去噪步 t:
  1. x_t = 当前 noisy motion (135-dim)
  2. model_input = concat([x_t, inactive, reactive, src_mask])   (540-dim)
  3. model 正常去噪
  4. scheduler 计算 x_{t-1}
  注意：没有硬替换，约束通过 conditioning 通道传入
```

**约束生效原理**：
- src_mask 和 inactive/reactive 让模型**从 input 层就知道**哪些位置有什么值、哪些需要生成
- 模型在训练时大量见过各种 mask pattern → 学会了如何利用这些信息
- **和 UMO 一样是软约束** → mask=0 的帧不保证精确匹配 source（但可以加后处理 P2 exact_match 强制覆盖）
- 粒度是**逐帧逐维度** (T×135) → 支持 per-joint 控制

**特点**：
- 优势：最细粒度（per-dim mask）；原生支持任意 mask 组合（temporal/joint/keyframe/full）；训练和推理分布一致
- 劣势：input_encoder 参数多（4× motion_dim）；T2M 质量可能被 completion 任务稀释（M5 只占 5%）；当前只能约束 rotation，无法直接约束 xyz position

---

### 关键差异对比

| 维度 | KIMODO | UMO | HyMotion M2M |
|------|--------|-----|-------------|
| **约束注入方式** | 硬替换（imputation） | 软注入（element-wise add） | 软注入（channel concat） |
| **约束精度** | position 精确，rotation 有偏差 | 整体有轻微偏差（[P]-MPJPE≈0.95mm） | 整体有偏差（可加后处理修正） |
| **约束空间** | position + rotation（333-dim 都可 impute） | rotation only（201-dim，帧级） | rotation only（135-dim，维度级） |
| **xyz position 控制** | ✅ 直接 impute position 维度 | ❌ 无 position 维度 | ❌ 无 position 维度 |
| **part-level 控制** | ✅ 关节级（impute 对应关节 dims） | ❌ 帧级（论文 limitation） | ✅ 维度级（mask 任意 dims） |
| **约束类型灵活性** | 高（5 种 constraint type） | 中（P/G/E 三种 meta-op） | 高（任意 binary mask） |
| **训练时见 completion** | Phase 2 见过 | 见过（多任务联合训练） | 从头就见（M1-M6 混合） |
| **T2M 质量保护** | Phase 1 纯 T2M → 质量有保障 | backbone 冻结 → 质量完整保留 | M5 只占 5% → 可能稀释 |
| **架构侵入性** | 输入 ×2（+mask 通道） | 加 E_ctx MLP（0.207M） | input_encoder 4× 扩大 |

---

### xyz Position 控制能力总结

| | KIMODO | UMO | HyMotion M2M |
|---|---|---|---|
| **能否直接约束 xyz？** | ✅ 可以（position 在 333-dim 表示中） | ❌ 不可以（表示中无 position） | ❌ 不可以（表示中无 position） |
| **xyz 约束精度** | 软约束（position 精确，但渲染用 rotation FK 后有几 cm 误差） | — | — |
| **要做到 xyz 控制需要什么** | 已支持 | 1) 加 position 到表示 2) 加 FK loss 3) 改 meta-op 支持 position | 1) 加 position 到表示 (R2) 2) 加 FK loss (L1) 3) VACE mask 扩展到 position dims |

---

## 与我方工作（HyMotion M2M）的宏观对比

| 维度 | KIMODO | UMO | HyMotion M2M（我方） |
|------|--------|-----|---------------------|
| **Backbone** | 自研 Transformer Encoder（2-stage，16L×8H×1024，282M） | HY-Motion-Lite MMDiT（460M） | HunyuanMotion MMDiT（0.46B/1.5B） |
| **动作表示** | 全局 6D rotation + smooth root + local joint pos + velocity + foot contact（333 dims，27 joints） | 201-dim（global transl + root 6D + 21 local rot + 22 local pos），SMPL | 135-dim（abs transl 3D + 22 rot_6d），SMPL，**无 local joint positions** |
| **生成范式** | DDPM（1000 steps train，DDIM 100 steps infer） | Flow Matching（rectified flow，50-step Euler ODE） | Flow Matching（velocity 或 x1/JiT，50-step Euler ODE） |
| **Conditioning** | Imputation（直接覆盖 noisy motion + binary mask concat） | Temporal Fusion（element-wise add，0.207M） | VACE（channel-wise concat，4×motion_dim） |
| **Condition 粒度** | 帧×关节（joint-level，6D 一组） | 帧级 whole-body（[P]/[G]/[E]） | 帧×dim（T×135，最细粒度） |
| **Part-level 控制** | ✅（关节级 imputation） | ❌（论文 limitation） | ✅（逐维度 mask） |
| **编辑任务** | ❌ | ✅（[edit] + instruction text） | 部分支持（M4 joint editing） |
| **几何约束** | ✅（2D waypoint/path imputation） | ✅（结构化文本，feed-forward） | ❌（待实现） |
| **Reaction/多人** | ❌ | ✅（dual-identity） | ❌ |
| **足部接触** | ✅（显式建模 + post-process foot lock） | ❌ | ❌ |
| **数据规模** | 700h 高质量 optical mocap | HumanML3D + MotionFix 等公开数据集 | MotionHub（多来源） |
| **开源状态** | ✅ | 承诺但未发布 | ❌（内部） |
