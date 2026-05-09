# HyMotion M2M v3: Unified Text-Motion Conditioning via Condition-Routed Flow Matching

> Status: PROPOSAL (2026-05-08)
> Priority: P0 — Caption model is fundamentally broken
> Debug machines: lzy_debug_machine_1, lzy_debug_machine_2

---

## 1. Problem Statement

### 1.1 现状量化

| Model | Task | jitter_pos (mean) | foot_skating | ee_error |
|-------|------|-----------|-------|----------|
| **uncond_local** (epoch 2700) | E1 (T2M, no text) | ~213 | 0.023 | — |
| **uncond_local** | E4 (end-effector) | ~213 | 0.026 | 0.21 |
| **caption_local_phase2** (epoch 2890) | E1 (T2M, with text) | **990** | 0.153 | — |
| **caption_local_phase2** | E4 (end-effector + text) | **1778** | 0.077 | 0.43 |

**结论**：caption 模型的 jitter_pos 是 uncond 的 **4.6x-8.3x**，输出几乎不可用。不仅无法理解 text，motion quality 本身也严重退化。

### 1.2 已排除的原因

- [x] null_vtxt_feat 全零 bug（2026-03-27 已修复，Phase 2 从修复后的 Phase 1 resume）
- [x] text embedding 缓存错误（debug_report 确认 cache hit 100%，embedding 形状正确）
- [x] CFG guidance scale OOD（已从 5.0 降到 2.0，仍然差）

### 1.3 根因分析

**核心矛盾**：MAN (Mask-Aware Noise) 训练中，x_t[known] = clean_motion 提供了一个极强的 shortcut，模型直接从 x_t 读取已知区域信息，不再需要 text 条件来推断语义。这导致 text cross-attention 在 Phase 2 训练过程中逐渐退化（attention entropy collapse）。

具体机制：
```
Phase 1 (pure T2M, mask=1 everywhere):
  x_t = (1-t)*noise + t*x1 → all noisy, model MUST read text to know what to generate
  text_attention = STRONG ✅

Phase 2 (mixed, many samples have large known region):
  x_t[known] = clean_motion → model reads motion directly from x_t
  x_t[generate] = noisy → model should use text, but...
  由于 x_t[known] 已经提供了 motion context, 模型可以从 known regions 推断 generate regions
  text_attention → ATROPHY ❌

Inference (T2M, all mask=1):
  x_t = pure noise → no motion info in x_t
  model tries to use text → attention has atrophied → garbage output ❌
```

**验证方法**：
1. Phase 1 checkpoint 的 E1 应该比 Phase 2 好（text attention 未退化）
2. Phase 2 如果只在 mask=1 的纯生成样本上 evaluate，应该也差（整个网络的 text pathway 退化）
3. uncond 模型没有这个问题（不存在 text pathway 需要维护）

### 1.4 为什么 KIMODO 没有这个问题

KIMODO 使用 **两阶段训练**（Phase 1: 500K steps 纯 T2M → Phase 2: 500K steps 混合），但 KIMODO 的 imputation 机制不同：
- KIMODO 每步 denoise 时 impute GT 到 x_t，模型看到的 x_t 是 "一部分干净 + 一部分 noisy"
- 关键区别：KIMODO 的模型在 Phase 1 已经完全学会了 text-to-motion，Phase 2 只加了约束遵循能力
- KIMODO Phase 2 的 text_cfg 和 Phase 1 一致，text attention 不会退化
- 我们的 Phase 2 改变了整个 x_t 的分布（从全 noisy → 部分 clean），迫使模型重新学习

---

## 2. 方案设计：Condition-Routed Flow Matching (CRFM)

### 2.1 核心思想

受 Seedance 2.0 的 Dual-Branch DiT、UMO 的 meta-operation token、KIMODO 的分阶段训练启发，提出 **Condition-Routed Flow Matching (CRFM)**：

**关键洞察**：text condition 和 motion condition 的角色根本不同：
- **Text condition** = "semantic intent"（想让模型做什么动作）
- **Motion condition** = "spatial constraint"（哪些部分已经确定了）
- 两者应该通过 **不同路径** 进入模型，且有 **显式的交互机制** 防止互相覆盖

### 2.2 架构设计

```
                  ┌─────────────────────────────────────────────────────┐
                  │              Motion DiT (MMDiT backbone)             │
                  │                                                      │
  text ──────────►│  Cross-Attention (text tokens, FROZEN from Phase1)  │
  (Qwen3+CLIP)   │         ↓                                           │
                  │  ┌──────────────────────────────────────────────┐   │
                  │  │          Self-Attention                       │   │
  x_t + ρ_emb    │  │  (motion tokens + condition routing token)    │   │
  ─────────────►  │  └──────────────────────────────────────────────┘   │
                  │         ↓                                           │
  motion cond ───►│  Condition Injection via AdaLN-Cond                │
  (VACE context)  │         ↓                                           │
                  │  Output: predicted velocity v_θ                     │
                  └─────────────────────────────────────────────────────┘
```

#### 2.2.1 三个核心组件

**A. Condition Density Embedding (CDE) — 条件密度嵌入**

```python
# 计算当前样本的 mask density（生成比例）
mask_density = src_mask.mean()  # scalar in [0, 1]
# 0.0 = all known (identity)
# 1.0 = all generate (pure T2M)

# 用 sinusoidal embedding 编码（类似 timestep embedding）
cde = sinusoidal_embed(mask_density, dim=1024)  # (B, 1024)

# 注入方式：和 timestep embedding 一起通过 AdaLN 调制模型
combined_condition = timestep_embed + alpha * cde
# alpha 是可学习的 scale factor，初始化为 0
```

**作用**：显式告诉模型"当前有多少信息需要从 text 获取"。density=1.0 时模型知道要完全依赖 text；density=0.1 时知道大部分信息来自 motion condition。

**新颖性**：现有工作（KIMODO、MoGenDiT、UMO）都没有显式编码 mask density。VACE 只通过 mask channel 隐式传递这个信息。显式编码让模型更容易学习 text/motion 的注意力分配。

**B. Frozen Text Cross-Attention + Trainable Motion Self-Attention**

```python
class ConditionRoutedMMDiTBlock(nn.Module):
    def __init__(self, ...):
        # Text cross-attention: FROZEN from Phase 1 (or T2M pretrained)
        self.text_cross_attn = FrozenMultiHeadAttention(...)
        
        # Motion self-attention: TRAINABLE
        self.motion_self_attn = MultiHeadAttention(...)
        
        # Routing gate: learns to balance text vs motion info
        self.route_gate = nn.Sequential(
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.Linear(256, 2),  # [text_weight, motion_weight]
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x, text_tokens, cde):
        # Text pathway (frozen, always active)
        text_out = self.text_cross_attn(query=x, kv=text_tokens)
        
        # Motion self-attention (trainable, processes x_t + VACE)
        motion_out = self.motion_self_attn(x)
        
        # Adaptive routing based on condition density
        weights = self.route_gate(cde)  # (B, 2)
        text_w = weights[:, 0:1].unsqueeze(-1)    # (B, 1, 1)
        motion_w = weights[:, 1:2].unsqueeze(-1)  # (B, 1, 1)
        
        # Combined output
        out = text_w * text_out + motion_w * motion_out + x  # residual
        return out
```

**作用**：
- Frozen text attention 保证 text 理解能力不退化（任何训练阶段都保持 Phase 1 / T2M pretrained 水平）
- Trainable motion self-attention 学习 completion/editing 能力
- Route gate 根据 CDE 动态分配权重：pure T2M 时 text_w 大，strong condition 时 motion_w 大

**新颖性**：
- 不同于 UMO 的 adapter-add（完全冻结 backbone）
- 不同于 KIMODO 的两阶段（phase 之间有 catastrophic forgetting 风险）
- 我们 **选择性冻结 text pathway，同时训练 motion pathway + routing**

**C. Text-Motion Consistency Regularization (TMCR)**

```python
def tmcr_loss(pred_with_text, pred_without_text, mask_density, threshold=0.3):
    """
    当 mask_density > threshold（motion condition 强）时，
    确保 pred_with_text 和 pred_without_text 有显著差异。
    这防止模型忽略 text condition。
    """
    # Only apply when motion condition is strong
    apply_mask = (mask_density > threshold).float()  # (B,)
    
    # Measure difference in generated regions only
    diff = (pred_with_text - pred_without_text).abs().mean(dim=(-1, -2))  # (B,)
    
    # Encourage minimum difference (text should matter)
    min_diff_target = 0.01  # minimum expected difference
    loss = F.relu(min_diff_target - diff) * apply_mask
    
    return loss.mean()
```

**作用**：即使 motion condition 很强，也强制模型让 text 对输出产生影响。这是显式的 anti-forgetting 机制。

**新颖性**：类似于 adversarial text conditioning，但通过正则化而非对抗训练实现。比简单的 CFG dropout (cond_mask_prob) 更直接。

### 2.3 训练策略

#### Phase 0: 初始化（不训练）
- 从 T2M pretrained checkpoint (HY-Motion-1.0-Lite) 加载
- 冻结 text cross-attention layers
- 随机初始化 CDE embedding layer、route_gate
- 随机初始化 VACE input projection (因为 input_dim 变了)

#### Phase 1: Pure T2M Warmup (10-20 epochs)
- 全部 mask=1，纯文本到动作生成
- 目的：让 CDE 学会 density=1.0 时把 text_w 推高
- route_gate 初始化为 text_w=0.7, motion_w=0.3
- 此阶段只训练 route_gate + CDE + motion self-attn 的一小部分

#### Phase 2: Mixed Training (主训练)
- 混合 T2M + completion + editing，使用 v3 sampler
- 所有组件都训练（text cross-attn 始终冻结）
- TMCR loss 开始生效（每 4 步计算一次，避免 overhead）
- mask_aware_noise=True

#### 关键超参

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `text_frozen_layers` | all 18 layers | 彻底防止 text atrophy |
| `cde_dim` | 1024 | 匹配 feat_dim |
| `cde_alpha_init` | 0.0 | 渐进式引入 CDE 信号 |
| `route_gate_init` | text=0.7, motion=0.3 | 偏向 text（保守） |
| `tmcr_weight` | 0.01 | 不需要太强，只防止完全忽略 |
| `tmcr_threshold` | 0.3 | mask_density > 0.3 时施加 |
| `tmcr_interval` | 4 | 每 4 步计算一次（需要额外 forward） |
| `cond_mask_prob` | 0.15 | CFG dropout (比现在的 0.1 稍高) |
| `batch_size` | 16 | 因为 TMCR 需要额外 forward，降低 bs |

### 2.4 推理策略

```python
# Dual-CFG inference (text + motion condition separation)
def inference_step(model, x_t, text_cond, motion_cond, mask, t,
                   text_cfg_scale=3.0, motion_cfg_scale=1.0):
    """
    text_cfg_scale: controls text conditioning strength
    motion_cfg_scale: controls motion conditioning strength (usually 1.0)
    """
    # Full conditional prediction
    v_full = model(x_t, text_cond, motion_cond, mask, t)
    
    # Text-null prediction (motion condition preserved, text dropped)
    v_no_text = model(x_t, null_text, motion_cond, mask, t)
    
    # Apply text CFG only to generated regions
    v_guided = v_no_text + text_cfg_scale * (v_full - v_no_text) * mask
    
    # Known regions: just use v_full (text doesn't matter for known)
    v_final = v_guided * mask + v_full * (1 - mask)
    
    return v_final
```

**关键改进**：text CFG 只应用于 generated regions（mask=1），不影响 known regions。这避免了 CFG 破坏已知区域的连续性。

### 2.5 与现有方案的对比

| 方面 | Current (Phase 1→2) | KIMODO | UMO | **CRFM (ours)** |
|------|---------------------|--------|-----|-----------------|
| Text-motion 交互 | 隐式竞争 | 分阶段隔离 | Adapter add (冻结) | **显式路由 + 正则** |
| Text atrophy 防护 | 无 | 靠分阶段 | 天然（冻结） | **TMCR + frozen text attn** |
| 条件密度感知 | 无 | 无 | Meta-op token (P/G/E) | **CDE (连续编码)** |
| Motion condition | VACE + MAN | Imputation | Adapter | **VACE + MAN + routing** |
| 训练效率 | 1x | 2x (两阶段) | 0.5x (只训 adapter) | **~1.3x** (TMCR overhead) |
| 发表潜力 | — | CVPR 2025 | CVPR 2025 | **CVPR/ICLR** |

---

## 3. 实施计划 (分阶段)

### Stage 0: 快速验证假设 (1-2 days, debug machine)

**目标**：验证 "text atrophy" 假设是否成立。

**实验 0.1**: Phase 1 vs Phase 2 的 text attention entropy 对比
```bash
# 在 debug machine 上，加载两个 checkpoint，跑 10 个 T2M 样本
# 记录每层 text cross-attention 的 entropy
python3 tools/probe_text_attention.py \
    --ckpt_phase1 work_dirs/hymotion_m2m_v2_caption_local_phase1/checkpoint-epoch_47/model.safetensors \
    --ckpt_phase2 work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_2890/model.safetensors \
    --data_file data/annotation/eval_e1_t2m_rewritten.json \
    --num_samples 10
```

**预期结果**：Phase 2 的 text attention entropy 显著低于 Phase 1（attention 退化为 uniform/collapsed）。

**实验 0.2**: Phase 1 checkpoint 直接跑 E1 T2M 评估
```bash
# Phase 1 没有 MAN，直接用标准 pipeline 推理
python3 tools/eval_m2m_v2_all_tasks.py \
    --config configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase1.py \
    --ckpt work_dirs/hymotion_m2m_v2_caption_local_phase1/checkpoint-epoch_47/model.safetensors \
    --tasks E1 --num-samples 20 --save-npz --use-rewritten
```

**预期结果**：Phase 1 的 E1 jitter_pos 应该远低于 Phase 2 的 990（如果假设成立）。

### Stage 1: Minimal Fix — Frozen Text Attention (3-5 days)

最小改动验证核心思路是否有效。

**改动清单**：
1. `bundle.py`: 添加 `freeze_text_attention` 选项
2. `hymotion_m2m_trainer.py`: Phase 2 训练时冻结 text cross-attention 参数
3. 新 config: `hymotion_m2m_v2_caption_local_phase2_frozen_text.py`

```python
# bundle.py 新增
def freeze_text_attention_layers(self):
    """Freeze all text cross-attention layers in the transformer."""
    for name, param in self.motion_transformer.named_parameters():
        if 'cross_attn' in name or 'ctxt' in name:
            param.requires_grad_(False)

# trainer.py __init__ 新增
if getattr(self.bundle, 'freeze_text_layers', False):
    self.bundle.freeze_text_attention_layers()
```

**训练计划**：
```bash
# 从 Phase 1 epoch_47 resume，开始 Phase 2 但冻结 text attention
python3 tools/taiji_submit.py m2m_v2_caption_local_phase2_frozen \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2_frozen_text.py \
    --host_num 2
```

**评估**：100 epoch 后跑 E1 和 E4，如果 jitter < 500 即为成功。

### Stage 2: Full CRFM Implementation (7-10 days)

**文件清单**：

| File | Change | Lines |
|------|--------|-------|
| `hftrainer/models/motion/hymotion_m2m/network/condition_routing.py` | **NEW** — CDE + RouteGate modules | ~150 |
| `hftrainer/models/motion/hymotion_m2m/network/mmdit.py` | 修改 — 注入 CDE 到 AdaLN, 添加 route_gate | ~60 |
| `hftrainer/models/motion/hymotion_m2m/bundle.py` | 修改 — 添加 CDE 初始化, freeze_text 逻辑 | ~40 |
| `hftrainer/trainers/motion/hymotion_m2m_trainer.py` | 修改 — 添加 TMCR loss 计算 | ~50 |
| `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 修改 — 添加 dual-CFG 推理逻辑 | ~30 |
| `configs/hymotion_m2m_v3/` | **NEW** — v3 系列 config | ~4 files |
| `tests/unit/test_condition_routing.py` | **NEW** — 单元测试 | ~80 |

**实现优先级**：

1. CDE (condition density embedding) — 最简单，独立模块
2. Frozen text attention — 修改 bundle init
3. Route gate — 在 mmdit block 中加入
4. TMCR loss — trainer 中额外 forward pass
5. Dual-CFG inference — pipeline 修改

### Stage 3: Training & Evaluation (2-3 weeks)

**Taiji 任务计划**：

| Task | Config | GPU | Phase | 预计时间 |
|------|--------|-----|-------|---------|
| `m2m_v3_cde_only` | CDE embedding only | 2x8 V100 | Phase 1→2 | 3 days |
| `m2m_v3_frozen_text` | frozen text + CDE | 2x8 V100 | Phase 2 | 3 days |
| `m2m_v3_full_crfm` | full CRFM | 4x8 V100 | Phase 1→2 | 7 days |
| `m2m_v3_ablation_no_tmcr` | CRFM w/o TMCR | 2x8 V100 | Phase 2 | 3 days |

---

## 4. 备选方案 (如果 CRFM 不够 novel)

### 4.1 Alternative A: Flow-Matching Inpainting with Learned Noise Schedule

**灵感来源**：RF-Inversion (2024) + Flow Matching

核心想法：不是对所有区域用 uniform noise schedule，而是：
- Known regions: 从 t=0 开始就 clean (MAN, 已有)
- Generated regions: 使用 **learned per-token noise schedule** τ(t, density, text_relevance)
- 噪声调度学习：让模型自己决定在什么 t 开始"写入" text 信息

```python
# Learned noise schedule: different tokens see different effective timestep
tau = noise_schedule_net(mask_density, text_embedding)  # (B, L) in [0, 1]
effective_t = t * tau  # per-token effective timestep
x_t = (1 - effective_t) * noise + effective_t * x1
```

**新颖性**：Per-token adaptive noise schedule 在 flow matching for motion 中未见报道。可以类比为 "attention-based noise annealing"。

### 4.2 Alternative B: Dual-Stream Condition DiT (DS-CDiT)

**灵感来源**：Seedance 2.0 的 Dual-Branch DiT + HunyuanMotion 的 dual-stream

核心想法：将现有 MMDiT 的 dual-stream 重新定义：
- Stream 1 (text-stream): 处理 text tokens + 生成区域的 motion tokens
- Stream 2 (motion-stream): 处理已知区域的 motion tokens + 生成区域的 motion tokens
- 两个 stream 在 single-stream blocks 中合并

```
Text Stream:    [text_tokens, gen_motion_tokens] → self-attn → cross-attn with stream 2
Motion Stream:  [known_motion, gen_motion_tokens] → self-attn → cross-attn with stream 1
                              ↓ merge in single-stream blocks
Output:         [gen_motion_output]
```

**优势**：text 和 motion condition 物理分离在不同 stream，不可能互相覆盖。

**劣势**：需要大幅修改 MMDiT 架构，无法从 T2M pretrained 直接 resume。

### 4.3 Alternative C: Progressive Condition Integration (PCI)

**灵感来源**：Curriculum Learning + Progressive GAN

核心想法：在单次 ODE 积分过程中，逐步引入不同条件：
- t ∈ [0, 0.3]: 只用 text condition（建立语义结构）
- t ∈ [0.3, 0.7]: 引入 motion condition（细化空间约束）
- t ∈ [0.7, 1.0]: 两者联合（最终精修）

```python
def progressive_velocity(model, x_t, t, text, motion_cond, mask):
    if t < 0.3:
        # Early phase: text only
        return model(x_t, text, null_motion_cond, full_mask, t)
    elif t < 0.7:
        # Middle phase: gradually introduce motion condition
        blend = (t - 0.3) / 0.4
        cond = blend * motion_cond + (1 - blend) * null_motion_cond
        return model(x_t, text, cond, mask, t)
    else:
        # Late phase: full conditioning
        return model(x_t, text, motion_cond, mask, t)
```

**新颖性**：Progressive condition injection 在 ODE 积分中的应用。类似于 diffusion 中的 time-dependent conditioning，但在 flow matching 框架下形式更自然。

**优势**：训练时不需要额外 loss，只需在训练时模拟这种 progressive conditioning。

---

## 5. 论文构思

### 5.1 Title Options

1. "Condition-Routed Flow Matching for Universal Human Motion Synthesis"
2. "UniMotion: Unifying Text and Spatial Conditioning in Human Motion Generation"
3. "Flow-Match-Complete: Text-Guided Universal Motion Completion via Condition Routing"

### 5.2 核心贡献

1. **问题定义**：首次系统分析 text-motion condition competition 问题，揭示 mask-aware flow matching 中的 text atrophy 现象
2. **方法**：提出 CRFM 框架——通过 condition density embedding + frozen text pathway + adaptive routing gate 实现 text/motion 条件的和谐共存
3. **任务统一**：单模型首次同时覆盖：
   - Pure T2M (text-to-motion)
   - Arbitrary motion completion (inbetween, prediction, joint completion, keyframe interpolation)
   - Text-conditioned motion completion (在约束基础上遵循文本语义)
   - Motion editing (VACE reactive channel)
   - Motion repair (completion + editing)
4. **SOTA**：在 HumanML3D / KIT-ML / 我们的 400H 数据集上超越 KIMODO, UMO, MoGenDiT

### 5.3 实验设计

| Experiment | Metric | Baseline |
|-----------|--------|----------|
| E1: Pure T2M | FID, diversity, R-precision, motion quality (jitter, skating) | MDM, MoMask, HY-Motion, KIMODO |
| E2-E7: Motion Completion | MPJPE, boundary smoothness, jitter | MoGenDiT, KIMODO |
| E10: Part-level control | MPJPE_masked, part accuracy | KIMODO |
| E4: End-effector | ee_error, trajectory ADE | KIMODO |
| NEW: Text-conditioned completion | R-precision + MPJPE_cond | None (new task) |
| Ablation: w/o CDE | all | — |
| Ablation: w/o frozen text | all | — |
| Ablation: w/o TMCR | all | — |
| Ablation: w/o route gate | all | — |

---

## 6. Debug 计划 (lzy_debug_machine_1/2)

### 6.1 环境确认

```bash
# Machine 1: 确认 GPU 状态
ssh lzy_debug_machine_1
nvidia-smi
# 预期: 4x V100 32GB

# Machine 2: 同上
ssh lzy_debug_machine_2
nvidia-smi
```

### 6.2 快速 smoke test 路径

```bash
# Step 1: 验证当前训练框架可正常跑 caption config
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
python3 tools/train.py configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py

# Step 2: 验证 caption pipeline 可推理
python3 -c "
from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline
print('Pipeline import OK')
"

# Step 3: 实现 CDE 模块 → 跑 1 step
python3 tools/train.py configs/hymotion_m2m_v3/smoke_crfm.py
```

### 6.3 Debug 里程碑

| Day | Milestone | Criterion |
|-----|-----------|-----------|
| D1 | 假设验证完成 | Phase 1 E1 jitter < Phase 2 |
| D2 | CDE 模块实现 + 单元测试通过 | forward/backward shape 正确 |
| D3 | Frozen text attention 实现 | Phase 2 训练 loss 正常下降 |
| D4 | Route gate 实现 | smoke test 1 step 通过 |
| D5 | TMCR loss 实现 | 训练 100 step 无 NaN |
| D7 | Stage 1 验证完成 | frozen_text + Phase 2 的 E1 jitter < 500 |
| D10 | Full CRFM smoke test | 所有组件联调，1 epoch 完成 |
| D14 | 提交 Taiji 大规模训练 | 4x8 V100 任务 running |
| D21 | 初步评估结果 | E1 + E4 + E10 指标 |

### 6.4 Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Frozen text attention 导致 T2M 质量下降 | High | 验证 Phase 1 ckpt 冻结后是否还能正常 T2M |
| CDE 引入新的 instability | Medium | alpha 初始化为 0，渐进增大 |
| TMCR 额外 forward 显存不够 | Medium | 降低 batch size 到 12；或只每 N 步算一次 |
| Route gate 退化为 constant | Low | 监控 gate 输出的 variance |
| 改 mmdit 后无法加载 pretrained | High | 新层 zero-init，旧层严格对齐 |

---

## 7. 具体代码修改计划

### 7.1 condition_routing.py (NEW)

```python
"""Condition Routing modules for CRFM v3."""

import math
import torch
import torch.nn as nn
from torch import Tensor


class ConditionDensityEmbedding(nn.Module):
    """Encode mask density as a continuous embedding via sinusoidal + MLP.
    
    Similar to timestep embedding but encodes the fraction of tokens
    that are to be generated (mask_density in [0, 1]).
    """
    
    def __init__(self, dim: int = 1024, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer('freqs', freqs)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # Initialize to near-zero so CDE has minimal effect at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, density: Tensor) -> Tensor:
        """
        Args:
            density: (B,) float in [0, 1], mask density per sample
        Returns:
            (B, dim) embedding
        """
        args = density.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


class ConditionRouteGate(nn.Module):
    """Adaptive gate that balances text vs motion attention contributions.
    
    Conditioned on CDE, outputs soft weights for text and motion pathways.
    """
    
    def __init__(self, cde_dim: int = 1024, init_text_bias: float = 0.5):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(cde_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2),
        )
        # Initialize bias so that text_weight starts at init_text_bias
        # sigmoid(bias) = init_text_bias → bias = log(p/(1-p))
        with torch.no_grad():
            logit = math.log(init_text_bias / (1 - init_text_bias + 1e-8))
            self.gate[-1].bias.copy_(torch.tensor([logit, -logit]))
    
    def forward(self, cde: Tensor) -> Tensor:
        """
        Args:
            cde: (B, cde_dim) condition density embedding
        Returns:
            (B, 2) weights [text_w, motion_w], softmax-normalized
        """
        return torch.softmax(self.gate(cde), dim=-1)
```

### 7.2 mmdit.py 修改 (概要)

```python
# 在 HunyuanMotionMMDiT.__init__ 中添加:
self.cde = ConditionDensityEmbedding(dim=feat_dim)
self.route_gates = nn.ModuleList([
    ConditionRouteGate(cde_dim=feat_dim) for _ in range(num_layers)
])

# 在 forward 中:
# 1. 计算 mask density
mask_density = x_mask.float().mean(dim=-1)  # 从 src_mask 提取
cde_emb = self.cde(mask_density)

# 2. timestep embedding 中融合 CDE
t_emb = self.timestep_encoder(timesteps) + cde_emb

# 3. 每层添加 route gate (在 dual-stream blocks 中)
for i, block in enumerate(self.dual_stream_blocks):
    gate_weights = self.route_gates[i](cde_emb)
    x = block(x, ctxt, t_emb, gate_weights=gate_weights)
```

### 7.3 Trainer 修改 (TMCR loss)

```python
# 在 HyMotionM2MTrainer.train_step 中添加 (仅 CRFM 模式):
if self.use_tmcr and global_step % self.tmcr_interval == 0:
    # Extra forward with null text (no gradient for text encoder)
    with torch.no_grad():
        null_vtxt = self.bundle.null_vtxt_feat.expand(B, -1, -1)
        null_ctxt = self.bundle.null_ctxt_input.expand(B, self.max_text_len, -1)
    
    # Forward with null text
    x_input_null = torch.cat([x_t, vace_context], dim=-1)
    pred_null = self.bundle.predict_flow(
        x_input=x_input_null, ctxt_input=null_ctxt, 
        vtxt_input=null_vtxt, timesteps=timesteps,
        x_mask_temporal=tgt_padding_mask, ctxt_mask_temporal=ctxt_mask_temporal,
    )
    
    # TMCR: enforce text influence in generated regions
    mask_density = src_mask.mean(dim=(-1, -2))  # (B,)
    diff = ((pred - pred_null.detach()) * src_mask).abs().mean(dim=(-1, -2))
    tmcr_loss = F.relu(0.01 - diff) * (mask_density < 0.7).float()
    losses['tmcr'] = tmcr_loss.mean() * self.tmcr_weight
```

---

## 8. 时间线

| Week | 目标 | 产出 |
|------|------|------|
| W1 (5/8-5/14) | 假设验证 + Stage 1 minimal fix | 数据证明 text atrophy; frozen_text 方案跑通 |
| W2 (5/15-5/21) | Full CRFM 实现 + smoke test | 所有模块代码完成，单机 1 epoch 跑通 |
| W3 (5/22-5/28) | Taiji 大规模训练 | 4x8 V100 任务提交，loss 正常下降 |
| W4 (5/29-6/4) | 初步评估 + 调参 | E1/E4/E10 指标达标 (jitter < 300) |
| W5-6 (6/5-6/18) | 完整评估 + 论文撰写 | 全 15 任务评估完成，论文 draft |

---

## 9. Success Criteria

| Metric | Target | Current (caption_phase2) | Stretch |
|--------|--------|--------------------------|---------|
| E1 jitter_pos (T2M) | < 400 | 990 | < 250 |
| E1 foot_skating | < 0.05 | 0.153 | < 0.03 |
| E4 jitter_pos (caption + motion cond) | < 400 | 1778 | < 250 |
| E4 ee_error (end-effector) | < 0.25 | 0.43 | < 0.15 |
| E10 part accuracy | 与 uncond 持平 | — | — |
| E1 R-precision (text alignment) | > 0.5 (HumanML3D) | 未测 | > 0.6 |

**Hard requirement**：caption 模型在所有 motion completion 任务上的质量不能低于 uncond 模型（当前是 4-8x 差距）。

---

## 10. 与现有工作的关系

本方案 **不替代** 以下已有工作，而是建立在其基础上：
- MAN (mask-aware noise): 保留，作为 motion conditioning 的基础
- SOAR post-training: 兼容，CRFM 训练完成后可继续做 SOAR
- v3 universal mask sampler: 保留，Phase 2 使用
- 198-dim representation: 保留
- KIMODO-style aux loss: 保留

本方案 **替代** 以下工作：
- current Phase 1→2 curriculum (改为 CRFM-aware curriculum)
- current `mask_text_cond` 的简单 Bernoulli dropout (改为 TMCR)
- current `cond_mask_prob=0.1` 的 CFG (改为 dual-CFG)

---

## Appendix A: Related Work 速查

| Paper | Venue | Key Idea | 与我们的关系 |
|-------|-------|----------|-------------|
| VACE (2503.07598) | arXiv 2025 | Video All-in-One via inactive/reactive channels | 我们的 conditioning 基础 |
| KIMODO (CVPR 2025) | CVPR 2025 | Two-phase: T2M → imputation | 对比方案，证明分阶段有 text atrophy 风险 |
| UMO (CVPR 2025) | CVPR 2025 | Frozen backbone + lightweight adapter | 启发 frozen text pathway |
| Seedance 2.0 | Industry 2026 | Dual-branch DiT + MM-RoPE | 启发 condition routing |
| SOAR (2604.12617) | arXiv 2026 | Self-correction for flow matching exposure bias | 正交互补方案 |
| RF-Inversion | NeurIPS 2024 | Flow matching inversion for editing | 启发 progressive conditioning |
| Chat Image 2 / GPT-Image-2 | Industry 2026 | Multi-round editing with conditioning | 启发 dual-CFG |
| HY-Motion 1.0 | arXiv 2025 | Scaled DiT for motion | 我们的 pretrained backbone |
| PackDiT | arXiv 2025 | Joint text-motion generation | 对比方案 |
