# HyMotion M2M 消融实验清单（借鉴 KIMODO / UMO）

> **目标**：逐个验证 KIMODO / UMO 中值得借鉴的设计是否能提升我们的效果。
> 每个实验只引入一个来自 baseline 的改进，测量对生成质量的影响。
> KIMODO 代码位于 `ref_repo/KIMODO/kimodo/kimodo/`（下文简称 `K/`）
>
> **索引**：本文档位于 `ref_repo/m2m_ablation_experiments.md`，由 `CLAUDE.md` 的"KIMODO / UMO 功能对比"一节引用。
> 各 baseline 的详细分析见 `ref_repo/KIMODO/CLAUDE.md` 和 `ref_repo/UMO/CLAUDE.md`。

---

## Baseline 定义

**Baseline-M2M**：`hymotion_m2m_completion_uncond_fm_046b.py`
```
pred_type=velocity, loss=smooth_l1, trans_dim_weight=5.0, keypoints3d_weight=0
mean_std=135-dim, motion=135d(3 abs_transl + 22×6 local_rot6d)
mask=M1:25% M2:15% M3:25% M4:15% M5:5% M6:15%
text=OFF, EMA=OFF, 后处理=无
从 HY-Motion-1.0-Lite (0.46B, T2M) 初始化。
```

**Baseline-Caption**：`hymotion_m2m_completion_caption_fm_046b.py`
```
同 Baseline-M2M，但 text=ON (pre-extracted Qwen3+CLIP), cond_mask_prob=0.3
```

---

## 实验基础模型与训练设置

### 基础 Checkpoint

所有消融实验从 **HY-Motion-1.0-Lite (0.46B)** 的预训练权重启动（除非另有说明）。这是一个在大规模数据上训练好的 T2M 基础模型，从它出发可以快速观察各项改进的边际效果。

| 模型 | Checkpoint | 说明 |
|------|-----------|------|
| **HY-Motion-1.0-Lite** | `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` | **所有消融实验的基础模型**。0.46B 参数，motion_dim=201，T2M 预训练。加载到我方 M2M 模型时 input_encoder/final_layer shape mismatch 会随机初始化 |
| `Baseline-M2M epoch_162` | `work_dirs/hymotion_m2m_completion_uncond_fm_046b/checkpoint-epoch_162` | 当前最新 uncond baseline checkpoint，用于后处理实验 (P1a, P2, P3) |
| `Baseline-Caption epoch_97` | `work_dirs/hymotion_m2m_completion_caption_fm_046b/checkpoint-epoch_97` | 当前最新 caption baseline checkpoint，用于文本实验 (F1) |

### 训练时长

每个消融实验从 HY-Motion-1.0-Lite 续训 **20 epoch**，足够观察收敛趋势和指标变化。

| 场景 | 续训 epoch 数 | 理由 |
|------|-------------|------|
| **所有训练实验** | **20 epoch** | 足够观察 loss 趋势和指标差异 |
| **后处理/推理实验**（P1a, P2, P3 等） | **0 epoch** | 不需要训练，直接在现有 checkpoint 上推理评估 |

### 硬件配置

| 配置项 | 值 |
|--------|---|
| GPU | 16 × A100/A800 80GB（2 nodes × 8 GPUs） |
| Mixed Precision | fp32（motion 精度要求高，不用 bf16） |
| Batch Size | 32 per GPU × 16 GPUs = 512 effective |
| Gradient Accumulation | 1 |

### 特殊情况

- **T2 curriculum**：Phase1 从 HY-Motion-1.0-Lite 训练 20 epoch（M5=1.0），Phase2 续训 20 epoch（M1-M6）
- **M1 t2m_only**：从 HY-Motion-1.0-Lite 训练 20 epoch（纯 T2M），用于与 Baseline 对比 T2M 质量
- **M2 baseline_mix**：从 HY-Motion-1.0-Lite 训练 20 epoch（标准 M1-M6），作为控制组

---

## 评估指标

### 指标定义

| 指标 | 定义 | 单位 | 评估方式 |
|------|------|------|---------|
| **MPJPE** | FK 后全身关节位置误差（pred vs GT），所有生成帧 | mm | FK → L2 distance |
| **[P]-MPJPE** | 保留帧（mask=0）的关节位置误差 | mm | 同上，只算 mask=0 帧 |
| **Foot Skating** | 接地时脚部 xz 速度（height < 0.05m 时） | cm/s | `K/metrics/foot_skate.py` `FootSkateFromHeight` |
| **Jitter** | 关节加速度二阶差分 `‖p[t+1]-2p[t]+p[t-1]‖` | mm/frame² | FK → 二阶差分 |
| **Ground Penetration** | 脚部 y 坐标低于地面的平均深度 | mm | min(toe_y, 0) |
| **Quality Pass Rate** | `MotionQualityChecker` 综合通过率 | % | 16 个 checker 全部通过 |

### 评估任务（evaluation tasks）

每个训练好的模型在以下 4 种 completion 任务上评估：

| 任务 | mask 设置 | 评估帧 | 说明 |
|------|----------|--------|------|
| **In-between** | 首尾各 30 帧 mask=0，中间 mask=1 | 中间帧 | M3 temporal contiguous |
| **Prediction** | 前 90 帧 mask=0，后续 mask=1 | 后续帧 | M3 prediction |
| **Joint Edit** | 全帧上半身 mask=1，下半身 mask=0 | 上半身帧 | M4 joint contiguous |
| **Full Gen** | 全部 mask=1 | 全部帧 | M5 unconditional |

测试集：从 hymotion_400h 的 val split 随机抽取 200 条。

### 每个实验的评估指标

| ID | 主要评估指标 | 次要指标 |
|----|------------|---------|
| L1 fk_loss | **MPJPE** (应直接改善) | Jitter |
| L3a/b trans_w | **MPJPE (transl component)** | Foot Skating |
| L4 velocity_loss | **Jitter** (应改善) | MPJPE |
| T1 ema | MPJPE | Quality Pass Rate |
| T2 curriculum | MPJPE, 收敛速度 | Quality Pass Rate |
| M1 t2m_only | **Quality Pass Rate** (T2M 质量) | MPJPE (M2M zero-shot) |
| M2 baseline_mix | **MPJPE** (M2M 质量) | Quality Pass Rate |
| M3 t2m_heavy | Quality Pass Rate + MPJPE | 全部 |
| P1a foot_lock | **Foot Skating** | Ground Penetration |
| P2 exact_match | **[P]-MPJPE** (应降至 0) | 边界 Jitter |
| P3 ground_fix | **Ground Penetration** (应降至 0) | Foot Skating |
| F1 caption_cfg | Quality Pass Rate | MPJPE |

---

## 删除/保留分析

> 以下分析基于对 KIMODO 开源代码（**注意：KIMODO 开源仓库只包含推理代码，不含训练代码**，训练细节来自论文）和 UMO 论文的仔细验证。

### 已删除的实验及理由

| 原 ID | 原实验名 | 删除理由 |
|--------|---------|---------|
| R1 | global_rotation | **工程量过大，收益不确定**。KIMODO 使用 global rotation 的核心优势是支持 world-space position imputation，但我方 M2M 使用 VACE conditioning（非 imputation），global rotation 的优势无法体现。需要重写整个 FK chain、数据加载、Mean/Std 统计，且预训练权重完全不可复用（input/output dim 全变）。 |
| R2 | add_joint_pos | **维度变化大，与 KIMODO 实现差异大**。KIMODO 的 joint_pos 是相对 smooth root 的 offset（需要先实现 smooth root），而非简单的 FK position。直接加 66d FK position 会导致 motion_dim 从 135→201，input encoder 从 540→804，预训练权重完全不可复用。单独做这个实验不如做 L1 (FK loss) 更直接有效。 |
| R3b | add_velocity_repr | **与 R3a/L4 (velocity loss) 高度重叠**。KIMODO 将 velocity 作为表示的一部分（81 dims），但同时也在 loss 中用 velocity。加 velocity 到表示需要改 motion_dim（135→201），成本高。velocity smoothness loss (L4) 能以更低成本达到类似效果。 |
| R4 | smooth_root | **ADMM 平滑器实现复杂度过高**。KIMODO 的 smooth root 使用 500-iteration ADMM multigrid solver（`K/motion_rep/smooth_root.py`），需要在数据预处理阶段对每个样本运行优化，大幅增加数据加载时间。且 smooth root 的核心优势是配合 imputation-based conditioning 做轨迹约束（我方不用 imputation）。 |
| R5 | heading_encode | **贡献边际化**。仅增加 2 dims（cos ψ, sin ψ），对模型容量几乎无影响。KIMODO 需要 heading 是因为它用 global coordinate frame 且不做 canonicalization；我方使用 absolute translation 已经隐式编码了朝向信息。 |
| R6 | kimodo_full_repr | **需要同时实现 R1-R5 全部改动**，且需要重写数据加载 pipeline、重算 Mean/Std（273 维）。这是一个系统级重构而非消融实验。如果需要验证 KIMODO 表示的整体效果，应该直接 fork KIMODO 代码在我方数据上训练，而非改造我方 pipeline。 |
| C1-temporal_fusion | temporal_fusion | **UMO 未开源，无法验证实现细节**。temporal fusion（element-wise add）的实现看似简单，但 E_ctx 初始化为 E_in 的复制这一关键细节需要精确实现（UMO backbone 是 HY-Motion-Lite 460M，我方用同架构但 input_dim 不同）。且 temporal fusion 本质上是把 VACE 换成另一种 conditioning，不是增量改进。 |
| C2 | three_class_mask | **mask 粒度与 UMO 不兼容**。UMO 的 [P]/[G]/[E] 是 frame-level（整帧操作），我方是 dim-level (T×135)。引入 3-class mask 需要定义 dim-level 的 [edit] 语义（什么时候某个 dim 是 "edit" 而非 "generate"？），这是一个开放性设计问题，不适合作为消融实验。 |
| C1-imputation | imputation (KIMODO 风格硬替换) | **架构侵入性过高，不属于消融实验范畴**。将 VACE 换成 imputation 需要修改 input_encoder（540→270 dim），修改训练循环（每个 denoising step 增加硬替换逻辑），修改推理 pipeline。这实质上是两种不同的 conditioning 架构对比，而非在现有架构上做增量改进。且 KIMODO 开源代码不含训练代码，imputation 的训练细节（constraint sampling 策略、loss 计算范围等）无法从代码验证。 |
| L2 | foot_contact_loss | **需要新增 prediction head + 修改模型输出维度**。KIMODO 在 333-dim 表示中已包含 4-dim foot contact 作为输出，但我方模型输出是 135-dim 纯 rotation+translation。新增 foot contact prediction 需要：1) 在 final_layer 输出增加 4 dims 2) 新增 BCE loss head 3) 在 FK 中计算 foot joint positions 作为 GT。这些改动触及模型架构，不属于 config-only 消融。将来可以作为独立改进单独实现。 |
| P1b | foot_lock_contact | **依赖 L2 (foot contact) 训练结果**。需要先完成 L2 训练出 4D foot contact prediction，然后才能做 contact-based foot lock。L2 已删除，此实验也不再可行。 |
| F2 | text_trajectory | **需要结构化文本解析 + 专用数据**。UMO 将轨迹序列化为 JSON 格式文本（`{type:"line", start:(x1,z1), end:(x2,z2)}`），需要制作带有轨迹标注的训练数据，成本极高。且 UMO 的 Traj.Err=18.78cm 远不如优化方法（2.93cm），说明纯文本编码精度有限。 |

### 保留的实验及代码验证结果

> **重要说明**：KIMODO 开源代码仅包含推理/demo 代码，**不含训练代码**。
> 因此 EMA、loss weights、curriculum 等训练细节只能从论文验证，无法在代码中确认。
> 推理侧代码（imputation、CFG、post-processing）已在代码中确认。
>
> **KIMODO 论文 Sec 4.3 训练细节摘要**（已从论文原文核实）：
> - **Loss 公式** Eq.(1)：7 个分量，smooth L1 loss，权重 γ1=γ3=γ5=10, γ2=2, γ4=3, γ6=4, γ7=5
> - **优化器**：Adam-atan2，lr=2e-5，batch=2048（16×A100-SXM4-80GB）
> - **EMA**：decay=0.995，every 10 steps，全程使用（两个 phase 都有）
> - **Curriculum**：Phase 1 (500k steps) 纯 T2M + 10% text dropout；Phase 2 (500k steps) mix text+constraints，移除 dropout
> - **序列长度**：最大 10 sec (300 frames @ 30fps)
> - **Two-stage denoiser**：两个 transformer encoder（root + body），各 16 层 × 8 heads × 1024 dim，总计 282M
> - **额外 register tokens**：49 个全零 extra tokens，增强表征能力
>
> **UMO 论文训练细节摘要**（已从论文原文核实）：
> - **Backbone**：HY-Motion-Lite (460M)，**冻结**，只训练 E_ctx (0.207M) + 3 个 meta-op embeddings
> - **训练**：4×B200, batch=256, lr=5e-5，Unified 100k steps / Expert 6k steps per task
> - **推理**：50-step Euler ODE，CFG scale=2.0
> - **关键结果**：Unified FID=9.46 >> Expert FID=17.04（T2M on HumanML3D, Table 4）

以下每个实验都经过验证：

| ID | 验证来源 | 验证发现 |
|----|---------|---------|
| L1 | `K/skeleton/kinematics.py:15-115` FK 实现；**论文 Sec 4.3 Eq.(1)** | KIMODO 论文 Eq.(1) 明确列出 7 个 loss 分量，其中 γ7=5 对应 FK(ĵ^a)−j^p（FK consistency loss），γ1=γ3=10 对应 position 项。这是所有分量中最高优先级的约束。开源 FK 代码 `K/skeleton/kinematics.py` 确认了完整 FK chain 实现。**注意**：训练 loss 代码未开源，γ 值来自论文 Eq.(1) 原文。 |
| L3a/b | **KIMODO 论文 Sec 4.3 Eq.(1)** | KIMODO 有 7 个 loss 分量权重：γ1=γ3=γ5=10, γ2=2, γ4=3, γ6=4, γ7=5。其中 γ1 对应 root position (r^p)，γ3 对应 body joint position (j^p)，γ7=5 对应 FK consistency FK(j^a)−j^p。trans_dim_weight 类比 γ1/γ3 的 position 加权效果。**注意**：权重值来自论文 Eq.(1)，训练代码未开源。 |
| L4 | `K/motion_rep/feature_utils.py` + `kimodo_motionrep.py:88`；**论文 Eq.(1) γ4=3** | KIMODO 将 velocity 显式计算（帧差分）并加入表示（81 dims）。论文 loss γ_vel (γ4)=3（非之前认为的 2）。我方可以在 loss 中加 velocity smoothness 项而不改表示。 |
| T1 | **KIMODO 论文 Sec 4.3** | 论文原文："Exponential Moving Average (EMA) is applied every 10 steps throughout training with a decay of 0.995 to maintain an average of the denoiser parameters, which is then used at test time." — **精确匹配**我们的设置。 |
| T2 | **KIMODO 论文 Sec 4.3** | 论文原文："For the first 500k steps (phase 1), the model is trained purely on the text-to-motion task with no constraints given as input. For the second 500k steps (phase 2), the model is trained on a mix of text and kinematic constraints." Phase 2 中 10% 无约束，25% 混合两种约束。Dropout=0.1 仅 Phase 1，Phase 2 移除（避免 drop 掉已 impute 的约束）。 |
| P1a | `K/postprocess.py:181-346` `post_process_motion()` | KIMODO 后处理实际调用 C++ `motion_correction.motion_postprocess.correct_motion()` 实现 IK 修正。参数包括 `contact_threshold=0.5`, `root_margin=0.04m`。**注意**：C++ IK solver 实现复杂（`MotionCorrection/` 目录），我方 P1a 使用纯 Python 启发式简化版。 |
| P2 | `K/postprocess.py` `extract_input_motion_from_constraints()` | KIMODO 后处理中通过 `hip_translations_input` / `rotations_input` 提取约束帧的 GT 值，传入 C++ motion_postprocess。约束帧的原始值确实被保留。我方 P2 用简单 `output[mask==0] = source[mask==0]` 覆盖即可。 |
| P3 | `K/postprocess.py:112-178` `create_working_rig_from_skeleton()` | KIMODO 在构建 working rig 时设置 `above_ground_offset=0.007m`（非 SOMA 骨骼）或 `0.02m`（SOMA），确保 toe 最低点在地面以上。我方 P3 用简单 root_y 上移实现同等效果。 |
| F1 | `K/model/cfg.py` separated CFG **代码确认** | separated CFG 公式 `out = out_uncond + w_text*(out_text - out_uncond) + w_constr*(out_constr - out_uncond)` 在 `ClassifierFreeGuidedModel.forward()` line 94-129 中完整实现，包含三路 forward pass（text_only, constraint_only, uncond）。**代码确认**。UMO 论文使用 CFG scale=2.0。 |
| M1/M2/M3 | **UMO 论文 Table 4** (HumanML3D T2M) | UMO 多任务联合训练不仅不降低 T2M 质量，反而**显著提升**：Unified FID=9.46 vs Expert FID=17.04（好 44%）。精确数据：Unified R@1=0.774, R@3=0.933, MM-D=15.22 vs Expert R@1=0.763, R@3=0.931, MM-D=15.49。UMO 冻结 HY-Motion-Lite backbone 只训练 0.207M adapter（temporal fusion），Unified 模型 100k steps vs Expert 单任务 6k steps。我方全参数训练可能退化程度不同。**UMO 训练配置**：4×B200, batch=256, lr=5e-5, 50-step Euler ODE, CFG=2.0。 |

---

## 完整实验配置

### 训练类实验

| ID | 实验名 | 类别 | 唯一改动（相对 Baseline-M2M） | 来源 | 难度 | Config 文件 |
|----|--------|------|---------------------------|------|------|------------|
| **L1** | fk_loss | Loss | keypoints3d_weight: 0 → 0.1 | KIMODO 论文 γ_pos=10 | P0 | `ablation_l1_fk_loss.py` |
| **L3a** | trans_w1 | Loss | trans_dim_weight: 5 → 1 | KIMODO 对照 | P0 | `ablation_l3a_trans_w1.py` |
| **L3b** | trans_w10 | Loss | trans_dim_weight: 5 → 10 | KIMODO 论文 γ_pos=10 | P0 | `ablation_l3b_trans_w10.py` |
| **L4** | velocity_loss | Loss | +velocity smoothness loss (帧差分 L1) | KIMODO 论文 γ_vel=2 | P0 | `ablation_l4_velocity_loss.py` |
| **T1** | ema | 训练 | +EMA(decay=0.995, interval=10 steps) | KIMODO 论文 Sec 4.1 | P0 | `ablation_t1_ema.py` |
| **T2** | curriculum | 训练 | Phase1: 20ep M5=1.0; Phase2: 20ep M1-M6 | KIMODO 论文 Sec 4.1 | P0 | `ablation_t2_curriculum_p1.py` + `_p2.py` |
| **M1** | t2m_only | 混合 | M5=1.0（纯 T2M，无 completion） | 对照实验 | P0 | `ablation_m1_t2m_only.py` |
| **M2** | baseline_mix | 混合 | 标准 M1-M6（= Baseline-M2M），作为控制组 | 对照实验 | P0 | = `Baseline-M2M` |
| **M3** | t2m_heavy_mix | 混合 | M5=50% + 其余各 10% | KIMODO+UMO 启发 | P0 | `ablation_m3_t2m_heavy.py` |
| **F1** | caption_cfg | 文本 | text conditioning ON, cond_mask_prob=0.3 | KIMODO separated CFG (代码确认) + UMO CFG=2.0 | P0 | = `Baseline-Caption` |

### 推理/后处理类实验（不需要训练）

| ID | 实验名 | 类别 | 唯一改动 | 基于 Checkpoint | 来源 | 难度 |
|----|--------|------|---------|----------------|------|------|
| **P1a** | foot_lock_heuristic | 后处理 | +height 启发式 foot lock（height<0.05m 时锁定 xz） | Baseline-M2M epoch_162 | KIMODO `postprocess.py` (C++ IK 简化版) | P1 |
| **P2** | exact_match | 后处理 | mask=0 帧强制用原始 GT 值覆盖模型输出 | Baseline-M2M epoch_162 | KIMODO `extract_input_motion_from_constraints()` | P0 |
| **P3** | ground_fix | 后处理 | 脚部 y<0 时整体上移 root（offset=0.007m） | Baseline-M2M epoch_162 | KIMODO `create_working_rig_from_skeleton()` above_ground_offset | P0 |

---

## T2M vs T2M+M2M 混合训练对比（重点实验）

> **核心问题**：M2M 多任务训练（M1-M6 mask 策略混合）是否会降低 T2M 生成质量？

### 实验设计

| 实验 | mask 策略 | T2M 占比 | M2M 占比 | 目标 |
|------|----------|---------|---------|------|
| **M1** (t2m_only) | M5=100% | 100% | 0% | T2M 上界：纯 T2M 训练能达到的最佳质量 |
| **M2** (baseline_mix) | M1:25% M2:15% M3:25% M4:15% M5:5% M6:15% | 5% | 95% | M2M 上界：当前默认 M2M 训练 |
| **M3** (t2m_heavy) | M5:50% + 其余各 10% | 50% | 50% | 均衡：T2M 和 M2M 各占一半 |

### 评估方法

**T2M 质量评估**（M5 full mask 推理，无 motion condition）：
- Quality Pass Rate（MotionQualityChecker 16 个 checker 通过率）
- Foot Skating
- Jitter
- Ground Penetration

**M2M 质量评估**（In-between 任务，首尾帧 mask=0）：
- MPJPE（生成帧 vs GT）
- [P]-MPJPE（保留帧偏差）
- Foot Skating
- Jitter

### 预期结果

| 实验 | T2M 质量 | M2M 质量 | 预期趋势 |
|------|---------|---------|---------|
| M1 (100% T2M) | **最高** | 最低（zero-shot） | T2M 基准线 |
| M2 (5% T2M) | 可能较低 | **最高** | M2M 基准线，T2M 可能退化 |
| M3 (50% T2M) | 较高 | 中等 | 平衡点 |

**关键对比**：
1. M1 vs M2 的 T2M 质量差异 = T2M 退化程度
2. M2 vs M3 的 M2M 质量差异 = 增加 T2M 占比对 M2M 的影响
3. 如果 M1 ≈ M3 的 T2M 质量，说明 50% T2M 足以保持质量
4. 如果 M2 >> M3 的 M2M 质量，说明 M2M 需要高占比才能学好

**UMO 参考**：UMO 论文 Table 4 显示 Unified 模型在 HumanML3D T2M 任务上 FID=9.46 **远优于** Expert 模型（FID=17.04），
说明多任务训练不一定降低 T2M 质量，甚至有显著 synergy 效应。但注意 UMO 冻结 backbone 只训练 0.207M adapter，
且训练步数不同（Unified 100k steps vs Expert 6k steps），与我方全参数训练场景不完全可比。

---

## 实验详细说明

### L1: FK Loss（Forward Kinematics Constraint Loss）

**动机**：KIMODO 的 loss 中 FK 项权重最高（γ_pos=10），说明 3D joint position 监督对生成质量至关重要。我方当前只有 rotation-space loss，缺乏 3D 空间约束。

**实现**：
- M2MLoss 已支持 `keypoints3d_weight`，只需在 config 中从 0 改为 0.1
- FK 计算通过 `bundle.geometry` 模块的 `rot6d_to_keypoints3d()` 实现
- Loss = SmoothL1(local_kp3d_pred, local_kp3d_gt)，其中 local = 相对 root joint

**Config 改动**：
```python
model = dict(
    losses_cfg=dict(keypoints3d_weight=0.1),
    body_model_path='ref_repo/MoGenDiT/motion_process/body_model/smplh',
)
```

### L2: Foot Contact Loss（已删除）

> **已移至"已删除实验"列表**。需要新增 prediction head 和修改模型输出维度，不属于 config-only 消融。

### L4: Velocity Smoothness Loss

**动机**：KIMODO 将 velocity 作为表示的一部分（81 dims）并在 loss 中加权（γ_vel=2）。velocity loss 可以约束相邻帧之间的变化平滑度，减少 jitter。我方可以在不改表示的前提下，在 loss 中加入 velocity 惩罚项。

**实现**：
```python
# 在 train_step 中
velocity_pred = pred_motion[:, 1:] - pred_motion[:, :-1]  # (B, T-1, 135)
velocity_gt = gt_motion[:, 1:] - gt_motion[:, :-1]
velocity_loss = SmoothL1(velocity_pred, velocity_gt)
```

**Config 改动**：
```python
model = dict(losses_cfg=dict(motion_smoothness_weight=0.5))
```

### T1: EMA (Exponential Moving Average)

**动机**：KIMODO 论文 Sec 4.1 提到使用 EMA（decay=0.995）。EMA 能平滑训练过程中的参数波动，通常能提高生成质量的稳定性。注意：KIMODO 训练代码未开源，update_interval 等细节不可验证，我们设置 update_interval=10 为合理默认值。

**实现**：通过 `EMAHook` 实现，只需在 config 中启用：
```python
default_hooks = dict(
    ema=dict(type='EMAHook', decay=0.995, update_interval=10),
)
```

### T2: Curriculum Training

**动机**：KIMODO 使用两阶段 curriculum：Phase 1（500k steps）纯 T2M，Phase 2（500k steps）加入 constraint conditioning。这样模型先学会生成高质量 motion，再学习如何遵守约束。

**实现**：
- Phase 1：M5=1.0（全部 mask=1，纯 unconditional generation），20 epoch
- Phase 2：切换到标准 M1-M6 混合策略，续训 20 epoch

### C1: Imputation（已删除）

> **已移至"已删除实验"列表**。将 VACE 换成 imputation 属于架构级变更，不属于消融实验。

### F1: Caption + CFG

**动机**：验证文本条件对 M2M 质量的影响。KIMODO 使用 10% text dropout + separated CFG（w_text=2.0, w_constr=2.0）；UMO 使用 CFG scale=2.0。

**实现**：直接使用已有的 `Baseline-Caption`（`hymotion_m2m_completion_caption_fm_046b.py`），该 config 已设置 cond_mask_prob=0.3。

### P1a: Foot Lock Heuristic

**动机**：KIMODO 后处理通过 C++ IK solver（`motion_correction` 包）实现 foot contact 检测 + IK 修正，大幅减少 foot skating。我方实现简化版：纯 Python 启发式 foot lock（检测低高度帧，锁定脚部 xz 位移），不需要 IK solver。

**KIMODO 实际实现**（`K/postprocess.py`）：
- `post_process_motion()` 调用 C++ `motion_correction.motion_postprocess.correct_motion()`
- 参数：`contact_threshold=0.5`, `root_margin=0.04m`
- C++ IK solver 完成全链条修正（foot lock + root correction + IK）

**我方简化实现**（纯 Python，无 IK）：
```python
# 1. FK 得到 foot joint positions
# 2. 检测 contact: height < 0.05m AND xz_velocity < 0.15 m/s
# 3. Contact 帧的 foot xz position 锁定为前一帧值
# 4. IK 反算 joint rotations（可选简化为直接修 translation）
```

### P2: Exact Match（约束帧精确覆盖）

**动机**：VACE 是 soft conditioning，mask=0 的帧输出不一定精确等于 source motion（[P]-MPJPE > 0）。KIMODO 后处理中直接用 GT 值覆盖约束帧。这是一个零成本的精度提升。

**实现**：
```python
output_motion[mask == 0] = source_motion[mask == 0]  # 精确覆盖
```

### P3: Ground Fix（地面穿透修正）

**动机**：生成结果可能出现脚部穿透地面（y < 0）。KIMODO 使用 above_ground_offset=0.007m 确保脚部始终在地面以上。

**实现**：
```python
# FK 得到 toe positions
min_toe_y = min(L_toe_y, R_toe_y)
if min_toe_y < 0:
    root_y += (-min_toe_y + 0.007)  # 整体上移 root
```

---

## 按类别汇总

| 类别 | 数量 | 实验 ID |
|------|------|---------|
| 监督 Loss | 4 | L1, L3a, L3b, L4 |
| 训练策略 | 2 | T1, T2 |
| 多任务混合 | 3+baseline | M1, M2 (=baseline), M3 |
| 后处理 | 3 | P1a, P2, P3 |
| 文本条件 | 1 | F1 (=Baseline-Caption) |
| **合计** | **13+baseline** | |

## 按难度汇总

| 难度 | 实验 | 数量 |
|------|------|------|
| **P0: 只改 config/推理** | L1, L3a, L3b, L4, T1, T2, M1, M3, P2, P3, F1 | **11** |
| **P1: 改推理代码/数据** | P1a | **1** |
| **P2: 改架构/大改动** | (无) | **0** |

## 已完成的代码改动

### 1. M2MLoss 新增 motion_smoothness_weight（for L4）

文件：`hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py`

- 新增 `motion_smoothness_weight` 参数（默认 0.0），控制帧差分 velocity smoothness loss
- 在 `forward()` 末尾，当 `motion_smoothness_weight > 0` 且 `pred_x1/gt_x1` 可用时，计算：
  ```python
  pred_vel = pred_x1[:, 1:] - pred_x1[:, :-1]  # 帧差分
  gt_vel = gt_x1[:, 1:] - gt_x1[:, :-1]
  smooth_loss = SmoothL1(pred_vel, gt_vel)  # 只在相邻有效帧上计算
  ```

### 2. M2MLoss 新增 trans_dim_weight（for L3a/L3b）

文件：`hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py`

- 新增 `trans_dim_weight` 参数（base config 默认 5.0），对 velocity/x1 loss 的前 3 维（translation）施加额外权重
- 补偿 translation 维度 (3/135) 在均匀 mean reduction 中被稀释的问题

### 3. Trainer 支持 FK loss（for L1）+ smoothness loss（for L4）

文件：`hftrainer/trainers/motion/hymotion_m2m_trainer.py`

- `train_step()` 在 `velocity` 和 `x1` 两种 pred_type 下，当 `keypoints3d_weight > 0` 时，
  从 predicted x1 和 GT x1 计算 FK 3D keypoints，传入 M2MLoss
- 当 `motion_smoothness_weight > 0` 或 `keypoints3d_weight > 0` 时，在 velocity 模式下
  额外计算 `pred_x1 = x_t + (1-t) * pred_vel` 用于 FK/smoothness loss
- 新增 `_compute_fk_keypoints()` 方法：denormalize → 拆分 transl/rot6d → 调用 `bundle.body_model` FK → 返回 (B, L, J, 3) keypoints

### 4. Config 文件（10 个）

目录：`configs/hymotion_m2m/ablation/`

| Config | 关键改动 |
|--------|---------|
| `ablation_m2_baseline.py` | 控制组，标准 M1-M6，20 epochs |
| `ablation_m1_t2m_only.py` | M5=100%（纯 T2M） |
| `ablation_m3_t2m_heavy.py` | M5=50% + 其余各 10% |
| `ablation_l1_fk_loss.py` | keypoints3d_weight=0.1, body_model_path=ref_repo/MoGenDiT/...smplh |
| `ablation_l3a_trans_w1.py` | trans_dim_weight=1.0 |
| `ablation_l3b_trans_w10.py` | trans_dim_weight=10.0 |
| `ablation_l4_velocity_loss.py` | motion_smoothness_weight=0.5 |
| `ablation_t1_ema.py` | EMAHook(decay=0.995, update_interval=10) |
| `ablation_t2_curriculum_p1.py` | M5=100%（Phase 1） |
| `ablation_t2_curriculum_p2.py` | load_from Phase 1 ckpt, 标准 M1-M6 |

> 注意：L2 (foot_contact) 和 C1 (imputation) 已从实验列表删除，无对应 config。

### 5. 评估脚本和启动脚本

| 文件 | 用途 |
|------|------|
| `scripts/eval_m2m_ablation.py` | 评估脚本：4 种 completion 任务 × 5 指标 |
| `scripts/launch_ablation_experiments.sh` | 批量启动/评估脚本 |

## 优先级排序（建议执行顺序）

### Round 1: Config-Only 实验（可立即启动，16 GPU × 20 epoch）

| 优先级 | ID | 理由 |
|--------|---|------|
| 1 | **M1** (t2m_only) | T2M baseline，所有混合训练实验的对照 |
| 2 | **M3** (t2m_heavy) | T2M+M2M 均衡训练，验证 T2M 退化 |
| 3 | **L1** (fk_loss) | KIMODO 最重要的 loss 项，P0 难度 |
| 4 | **L4** (velocity_loss) | 减少 jitter 的直接手段 |
| 5 | **T1** (ema) | 零成本稳定性提升 |
| 6 | **L3a** (trans_w1) | 对照实验 |
| 7 | **L3b** (trans_w10) | 接近 KIMODO 配置 |
| 8 | **T2** (curriculum Phase1) | curriculum 训练第一阶段 |

### Round 2: 推理/后处理实验（不需要训练，可立即评估）

| 优先级 | ID | 理由 |
|--------|---|------|
| 1 | **P2** (exact_match) | 零成本精度提升 |
| 2 | **P3** (ground_fix) | 零成本穿透修正 |
| 3 | **P1a** (foot_lock) | 需要实现 foot contact detection |

### Round 3: 需要代码改动的实验

| 优先级 | ID | 理由 |
|--------|---|------|
| 1 | **F1** (caption) | 已有 config，直接用 Baseline-Caption 即可 |

---

## 启动指南

### Taiji 集群任务状态（2026-03-25 已提交）

所有 9 个 Round 1 训练实验已提交到 Taiji 集群（chongqing A100，每实验 2 nodes × 8 GPUs = 16 GPUs）：

| Taiji task_flag | 实验 ID | Config | 状态 |
|----------------|---------|--------|------|
| `ablation_m2m_m2_baseline` | M2 | `ablation_m2_baseline.py` | SUBMITTED |
| `ablation_m2m_m1_t2m_only` | M1 | `ablation_m1_t2m_only.py` | SUBMITTED |
| `ablation_m2m_m3_t2m_heavy` | M3 | `ablation_m3_t2m_heavy.py` | SUBMITTED |
| `ablation_m2m_l1_fk_loss` | L1 | `ablation_l1_fk_loss.py` | SUBMITTED |
| `ablation_m2m_l3a_trans_w1` | L3a | `ablation_l3a_trans_w1.py` | SUBMITTED |
| `ablation_m2m_l3b_trans_w10` | L3b | `ablation_l3b_trans_w10.py` | SUBMITTED |
| `ablation_m2m_l4_velocity_loss` | L4 | `ablation_l4_velocity_loss.py` | SUBMITTED |
| `ablation_m2m_t1_ema` | T1 | `ablation_t1_ema.py` | SUBMITTED |
| `ablation_m2m_t2_curriculum_p1` | T2-P1 | `ablation_t2_curriculum_p1.py` | SUBMITTED |

监控命令：
```bash
taiji_client task_running_list                           # 查看所有运行中的任务
taiji_client logs --tf ablation_m2m_m2_baseline          # 查看特定任务日志
taiji_client instance_detail --tf ablation_m2m_m2_baseline  # 查看实例详情
```

### 环境要求

- 当前代码运行环境（T4 节点）**不能直接启动训练**，需要在 Taiji 集群上执行
- Taiji 平台会自动设置 `NODE_LIST`, `NODE_NUM`, `CHIEF_IP`, `INDEX` 等环境变量
- 训练脚本 `tools/taiji_dist_train.sh` 会自动检测这些变量并配置分布式参数

### Round 1: 启动 9 个训练实验

在 Taiji 集群上，每个实验分配 16 GPU（2 nodes × 8 GPUs）：

```bash
# 方法 1: 批量启动（在 Taiji 提交节点上）
bash scripts/launch_ablation_experiments.sh 1

# 方法 2: 逐个提交 Taiji 作业（推荐，更可控）
# 每个实验单独提交，示例（共 9 个训练实验 + T2 Phase2 = 10 次提交）：
bash tools/taiji_dist_train.sh configs/hymotion_m2m/ablation/ablation_m2_baseline.py
bash tools/taiji_dist_train.sh configs/hymotion_m2m/ablation/ablation_m1_t2m_only.py
bash tools/taiji_dist_train.sh configs/hymotion_m2m/ablation/ablation_m3_t2m_heavy.py
bash tools/taiji_dist_train.sh configs/hymotion_m2m/ablation/ablation_l1_fk_loss.py
bash tools/taiji_dist_train.sh configs/hymotion_m2m/ablation/ablation_l3a_trans_w1.py
bash tools/taiji_dist_train.sh configs/hymotion_m2m/ablation/ablation_l3b_trans_w10.py
bash tools/taiji_dist_train.sh configs/hymotion_m2m/ablation/ablation_l4_velocity_loss.py
bash tools/taiji_dist_train.sh configs/hymotion_m2m/ablation/ablation_t1_ema.py
bash tools/taiji_dist_train.sh configs/hymotion_m2m/ablation/ablation_t2_curriculum_p1.py
```

### Round 1.5: Curriculum Phase 2（依赖 Phase 1 完成）

Phase 1 (`ablation_t2_curriculum_p1`) 训练完成后，启动 Phase 2：

```bash
bash tools/taiji_dist_train.sh configs/hymotion_m2m/ablation/ablation_t2_curriculum_p2.py
```

Phase 2 config 自动从 `work_dirs/ablation_t2_curriculum_p1/checkpoint-epoch_20` 加载 Phase 1 权重。

### 监控训练进度

```bash
# 查看 loss 日志
tail -f work_dirs/ablation_*/*/train.log

# 查看 checkpoint
ls work_dirs/ablation_*/checkpoint-epoch_*

# 查看所有实验状态
for d in work_dirs/ablation_*; do
    name=$(basename $d)
    latest=$(ls -d $d/checkpoint-epoch_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    echo "$name: ${latest:-NO_CHECKPOINT}"
done
```

### Round 2: 评估所有完成的实验

```bash
bash scripts/launch_ablation_experiments.sh 2
```

### Round 2 手动评估（单个实验）

```bash
python scripts/eval_m2m_ablation.py \
    --config configs/hymotion_m2m/ablation/ablation_m2_baseline.py \
    --checkpoint work_dirs/ablation_m2_baseline/checkpoint-epoch_20 \
    --num-samples 200 \
    --num-steps 50 \
    --output work_dirs/ablation_m2_baseline/eval_results.json
```

---

## 评估结果

> 以下表格将在实验完成后填入。

### 训练实验结果

| ID | 实验名 | train loss (final) | In-between MPJPE (mm) | In-between [P]-MPJPE (mm) | Prediction MPJPE (mm) | Joint Edit MPJPE (mm) | Full Gen Quality Pass Rate (%) | Foot Skating (cm/s) | Jitter (mm/f²) | Ground Penetration (mm) |
|----|--------|----|----|----|----|----|----|----|----|----|
| **M2** | baseline (Baseline-M2M) | — | — | — | — | — | — | — | — | — |
| **M1** | t2m_only | — | — | — | — | — | — | — | — | — |
| **M3** | t2m_heavy_mix | — | — | — | — | — | — | — | — | — |
| **L1** | fk_loss | — | — | — | — | — | — | — | — | — |
| **L3a** | trans_w1 | — | — | — | — | — | — | — | — | — |
| **L3b** | trans_w10 | — | — | — | — | — | — | — | — | — |
| **L4** | velocity_loss | — | — | — | — | — | — | — | — | — |
| **T1** | ema | — | — | — | — | — | — | — | — | — |
| **T2** | curriculum | — | — | — | — | — | — | — | — | — |
| **F1** | caption_cfg | — | — | — | — | — | — | — | — | — |

### 后处理实验结果（基于 Baseline-M2M epoch_162）

| ID | 实验名 | In-between MPJPE (mm) | [P]-MPJPE (mm) | Foot Skating (cm/s) | Jitter (mm/f²) | Ground Penetration (mm) |
|----|--------|----|----|----|----|-----|
| — | No post-process | — | — | — | — | — |
| **P2** | exact_match | — | — | — | — | — |
| **P3** | ground_fix | — | — | — | — | — |
| **P2+P3** | exact_match + ground_fix | — | — | — | — | — |
| **P1a** | foot_lock | — | — | — | — | — |
| **P1a+P2+P3** | all post-process | — | — | — | — | — |

### T2M vs M2M 混合训练对比（重点结果）

| 实验 | T2M 占比 | T2M Quality Pass Rate (%) | T2M Foot Skating (cm/s) | T2M Jitter (mm/f²) | M2M In-between MPJPE (mm) | M2M [P]-MPJPE (mm) | 结论 |
|------|---------|---|---|---|---|---|------|
| **M1** (t2m_only) | 100% | — | — | — | — | — | T2M 上界 |
| **M2** (baseline_mix) | 5% | — | — | — | — | — | M2M 上界 |
| **M3** (t2m_heavy) | 50% | — | — | — | — | — | 均衡 |
