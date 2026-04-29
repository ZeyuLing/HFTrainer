# SOAR: Self-Correction for Optimal Alignment and Refinement in Diffusion Models

## Basic Info

| Field | Value |
|-------|-------|
| Paper | [arXiv 2604.12617](https://arxiv.org/abs/2604.12617) |
| Authors | You Qin, Linqing Wang, Hao Fei, Roger Zimmermann, Liefeng Bo, Qinglin Lu, Chunyu Wang |
| Date | 2026-04 |
| Open Source | **No** (截至 2026-04-17 尚无公开代码) |
| Local PDF | `ref_repo/SOAR/SOAR_paper.pdf` |
| Backbone | SD3.5-Medium (Rectified Flow / DiT) |
| Task | Text-to-Image post-training (通用框架，可扩展到任意 flow/diffusion 生成) |

---

## Problem: Exposure Bias in Diffusion Post-Training

SFT 和 RL 之间存在根本性 gap：

| Stage | On-trajectory? | Signal | Issue |
|-------|---------------|--------|-------|
| **SFT** | Yes (GT forward process states) | Dense (per-step MSE) | 推理时模型用自身预测（off-trajectory），分布 mismatch |
| **RL** | Yes (model rollout) | Sparse (terminal reward) | Credit assignment 困难，reward hacking 风险 |
| **SOAR** | Partially (1-step rollout) | Dense (per-step correction) | None — 兼得两者优势 |

**Exposure bias 核心**：SFT 训练时 x_t 来自 GT forward process（`x_t = (1-t)*noise + t*x_clean`），但推理时 x_t 来自模型自身 ODE 积分的累积预测。一旦早期步产生误差，后续步进入从未优化过的 OOD 区域，误差复合放大。

---

## SOAR Algorithm

### Core Idea

**On-policy rollout + re-noise + dense correction supervision**：
1. 用当前模型做一步 stop-gradient ODE rollout，得到 off-trajectory state
2. 对 off-trajectory state 重新加噪（re-noise），在多个噪声水平上采样
3. 在每个采样点上，监督模型向原始 clean target 回溯（correction target）

### Pseudocode (Algorithm 1)

```python
for each training batch (z0_clean, caption):
    # --- Base loss (standard SFT) ---
    z1 = randn_like(z0_clean)                    # Gaussian noise
    t0 ~ U[0, 1]
    sigma_t0 = t0                                 # rectified flow: sigma = t
    z_t0 = (1 - sigma_t0) * z0_clean + sigma_t0 * z1   # on-trajectory state
    v_on = model(z_t0, caption, t0)               # model prediction
    v_gt = z1 - z0_clean                          # GT velocity
    L_base = w(sigma_t0) * ||v_on - v_gt||^2

    # --- SOAR correction loss ---
    # Step 1: single stop-gradient ODE rollout
    with stop_gradient:
        v_cfg = v_uncond + w_cfg * (v_cond - v_uncond)  # CFG velocity
    t1 = max(t0 - 1/K, 0)                        # one step towards clean (K=total steps)
    z_hat_t1 = z_t0 + (sigma_t1 - sigma_t0) * v_cfg    # off-trajectory state

    # Step 2: re-noise (sample N auxiliary points)
    L_corr = 0
    for n in range(N):
        sigma_t_prime ~ U[sigma_t1, 1]            # auxiliary noise level
        alpha = (sigma_t_prime - sigma_t1) / (1 - sigma_t1)
        z_t_prime = (1 - alpha) * z_hat_t1 + alpha * z1  # re-noised (SAME z1!)

        # Step 3: correction target — steer back to z0_clean
        v_corr = (z_t_prime - z0_clean) / sigma_t_prime

        # Step 4: model forward on off-trajectory point
        v_off = model(z_t_prime, caption, t_prime)
        L_corr += w(sigma_t_prime) * ||v_off - v_corr||^2

    # Combined loss
    L_SOAR = (L_base + lambda * L_corr) / (B + lambda * P)
    update(model, L_SOAR)
```

### Key Design Choices

1. **Shared z1**：base loss 和 correction loss 使用**同一个** z1，保持 re-noised states 在原始 transport ray 附近。实验证明比 fresh noise 好。

2. **Stop-gradient rollout**：rollout 的 velocity 不回传梯度，只用于生成 off-trajectory state。防止梯度穿过 rollout 步骤。

3. **Correction target = (z_t' - z0) / sigma_t'**：让 off-trajectory state 也指向同一个 clean target z0，保证 goal consistency。

4. **Dense supervision**：N 个 auxiliary 点 × 每个都有明确的 per-timestep correction target。不需要 reward model。

---

## Experimental Results (SD3.5-Medium)

### Broad-Data (10K steps, 286K image-caption pairs)

| Metric | Base (SFT) | SOAR | Improvement |
|--------|-----------|------|-------------|
| GenEval | 0.70 | **0.78** | +11% |
| OCR | 0.64 | **0.67** | +5% |
| PickScore | — | +0.15 | |
| HPSv2.1 | — | +0.005 | |
| Aesthetic | — | +0.11 | |

**SOAR on SD3.5-Medium surpasses larger SD3.5-Large on GenEval (0.78 vs 0.71)**。

### vs RL (GRPO)

| | SFT | GRPO (RL) | SOAR |
|---|-----|-----------|------|
| High-Aesthetic | 5.74 | 5.87 | **5.94** |
| ClipScore | 0.297 | 0.296 | **0.300** |
| Reward hacking | No | Yes (ClipScore -2.8%) | **No** |
| Compute | Low | High | **Low** |

---

## 适用性分析：SOAR → HyMotion M2M v2

### 1. Framework Compatibility ✅

M2M 使用 **flow matching（rectified flow）**，与 SD3.5-Medium 完全同框架：

| | SD3.5-Medium (SOAR) | HyMotion M2M |
|---|---|---|
| 生成范式 | Rectified flow | Flow matching (rectified flow) |
| 噪声调度 | `x_t = (1-t)*noise + t*clean` | `x_t = (1-t)*noise + t*clean` |
| 预测目标 | velocity `v = noise - clean` | velocity `v = x1 - x0` |
| ODE solver | Euler | midpoint / Euler |
| Architecture | DiT | MMDiT (dual+single stream) |

**SOAR 的 rectified flow 公式可直接移植到 M2M**。

### 2. M2M 存在明显的 Exposure Bias

M2M 的 exposure bias 比图像生成可能**更严重**：

| 因素 | 影响 |
|------|------|
| **50 步 ODE 积分** | 比 SD3.5 的几步采样更多步骤，误差累积更多 |
| **时序数据** | 前帧误差影响后帧的时序一致性 |
| **VACE conditioning** | 生成区域的误差通过 self-attention 影响模型对 context 的理解 |
| **Post-hoc blend** | 标准模型推理后的硬拼接 → 边界跳变的根因之一 |
| **_man imputation** | 仅解决 known regions 的分布匹配，generated regions 仍有 exposure bias |

**关键洞察**：`_man` 变体通过 mask-aware noise + imputation 让 known regions 的 x_t 分布匹配（训练时 clean、推理时也 replace 为 clean），但 **generated regions 仍然有标准 exposure bias**。SOAR 可以专门针对 generated regions 做 correction。

### 3. SOAR for M2M: 实现方案

```python
# --- Adapted SOAR for M2M _man variant ---

# Base loss (current M2M training, unchanged)
x0 = randn_like(tgt_motion)
x1 = tgt_motion  # normalized clean motion
t ~ U[0,1]
x_t = (1-t)*x0 + t*x1

# Mask-aware noise (existing _man)
if mask_aware_noise:
    keep_mask = 1 - src_mask
    x_t = x_t * src_mask + x1 * keep_mask  # known=clean

vace_ctx = prepare_vace_input(src_motion, src_mask)
x_input = cat([x_t, vace_ctx])
v_on = model(x_input, text, t)
L_base = SmoothL1(v_on, x1 - x0)  # or generation_mask weighted

# SOAR correction (NEW)
with torch.no_grad():
    v_rollout = model(x_input, text, t)  # stop-gradient velocity
    if cfg_scale > 1:
        v_rollout = v_uncond + cfg * (v_cond - v_uncond)

dt = -1.0 / K  # K = num_steps (50)
t1 = max(t + dt, 0)  # note: flow matching convention t=0 noise, t=1 clean
sigma_diff = t1 - t   # negative (towards noise)
x_hat = x_t + sigma_diff * v_rollout  # off-trajectory state

# Mask-aware: only rollout generated regions
if mask_aware_noise:
    x_hat = torch.where(keep_mask, x1, x_hat)  # keep known regions clean

L_corr = 0
for n in range(N):
    t_prime ~ U[t1, 1]
    alpha = (t_prime - t1) / (1 - t1)
    z_re = (1-alpha) * x_hat + alpha * x0  # re-noise with SAME x0

    # Mask-aware: keep known regions clean at new t
    if mask_aware_noise:
        z_re = torch.where(keep_mask, x1, z_re)

    # Correction target: steer towards x1 (clean)
    # For flow matching: v_corr should make z_re flow to x1
    v_corr = (z_re - x1) / t_prime  # analogous to SOAR Eq. for rectified flow
    # Note: in FM convention, v = dx/dt = x1 - x0, and x_t flows from noise to clean.
    # The correction target ensures: z_re + v_corr * (1 - t_prime) ≈ x1

    x_re_input = cat([z_re, vace_ctx])
    v_off = model(x_re_input, text, t_prime)
    L_corr += SmoothL1(v_off, v_corr, generation_mask=src_mask)

L_total = L_base + lambda * L_corr
```

### 4. 不需要额外数据标注 ✅

**SOAR 完全不需要任何额外标注。** 这是它相对 RL 的核心优势：

| 需要什么 | SOAR | RL (GRPO/DPO) | 当前 M2M SFT |
|---------|------|---------------|-------------|
| 配对数据 (motion, mask, text) | ✅ 已有 | ✅ 已有 | ✅ 已有 |
| Reward model | ❌ 不需要 | ✅ 需要 | ❌ 不需要 |
| Preference labels | ❌ 不需要 | ✅ 需要 | ❌ 不需要 |
| Negative samples | ❌ 不需要 | 部分需要 | ❌ 不需要 |
| Quality annotations | ❌ 不需要 | ❌ 不需要 | ❌ 不需要 |

**SOAR 的 correction target 完全来自 clean target 本身**（`v_corr = (z_re - z0)/sigma`），不依赖任何外部信号。

现有的训练数据（`train_hymotion_400h.json` 549K 条，或 quality-filtered 456K 条）直接可用，无需任何修改。

### 5. Expected Benefits for M2M

| Benefit | Mechanism | Severity in M2M |
|---------|-----------|-----------------|
| **减少边界跳变** | 模型学会从 off-trajectory 状态回溯 → 推理误差不累积 | 🔴 高 — 当前主要痛点 |
| **提升时序连贯性** | Dense per-timestep correction → 每步都有正确方向 | 🟡 中 |
| **改善长序列质量** | 50 步 ODE 的后期步骤在 on-policy 分布上训练过 | 🟡 中 |
| **不损害已有质量** | L_base 保留，L_corr 是额外监督 | ✅ Safe |
| **与 _man 互补** | _man 解决 known regions；SOAR 解决 generated regions | ✅ Complementary |

### 6. Implementation Considerations

#### 6.1 VACE Context 处理

Off-trajectory rollout 改变了 x_t，但 **VACE context（inactive/reactive/mask）不变**。这是正确的：
- VACE 是 conditioning input，描述"哪些区域已知 + 值是什么"
- x_t 是 denoising 的 latent trajectory
- 两者独立变化是合理的

#### 6.2 Mask-Aware Noise 交互

对于 `_man` 变体，known regions 在 x_t 中 = clean。SOAR rollout 后：
- known regions：应保持 clean（rollout 不应改变 → 用 mask select）
- generated regions：变为 off-trajectory → re-noise → correction

实现时需要确保 mask-aware 逻辑在 SOAR 的每个步骤中正确应用。

#### 6.3 CFG 交互

- **Uncond 模型**（`uncond_fm_man_046b`）：rollout 直接用 v_on，无 CFG。简化实现。
- **Caption 模型**（`caption_fm_man_046b`）：rollout 用 CFG velocity（和推理一致）。需要额外 unconditional forward pass。

#### 6.4 Compute Overhead

| 配置 | 每步 forward passes | 相对 SFT 开销 |
|------|-------------------|-|
| SFT (current) | 1 | 1x |
| SOAR N=1 | 1 (base) + 1 (rollout, no-grad) + 1 (correction) = 3 | ~2x (rollout is no-grad) |
| SOAR N=2 | 1 + 1 + 2 = 4 | ~2.5x |
| SOAR N=1 + CFG | 1 + 2 + 1 = 4 | ~2.5x |

**建议**：post-training 阶段用 SOAR，不需要从头训练。在现有最优 checkpoint（`uncond_fm_man_046b` epoch 1000）上做 5K-10K 步 SOAR post-training。

#### 6.5 Post-Training vs From-Scratch

SOAR 论文明确定位为 **post-training**：先 SFT 训练好基础模型，再做 SOAR correction。这与 M2M 的工作流完美匹配：

```
Phase 1: Current SFT training (已完成, uncond_fm_man_046b, 1000 epochs)
Phase 2: SOAR post-training (NEW, 5K-10K steps on best checkpoint)
```

### 7. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Off-trajectory states 太 noisy（模型还不够好时 rollout 无意义） | Low — 我们用 epoch 1000 checkpoint | 只在 well-trained checkpoint 上做 post-training |
| Lambda 过大导致 base loss 退化 | Medium | 从 lambda=0.1 开始，monitor base/corr loss 比例 |
| Motion-specific 问题（rot6d 空间 vs pixel 空间） | Low | Correction target 在 normalized motion 空间操作，和 base loss 一致 |
| 过拟合（数据量 456K，post-training 只需几 K 步） | Low | 短 schedule + LR warmup |

---

## 总结

| Question | Answer |
|----------|--------|
| **能否适用于 M2M v2？** | **✅ 是，高度适用。** Flow matching 框架完全匹配，M2M 存在明确的 exposure bias，SOAR 是当前最合适的 correction 方案。 |
| **需要额外数据标注吗？** | **❌ 不需要。** SOAR 完全 self-supervised，correction target 来自 clean target 本身。现有训练数据直接可用。 |
| **实现复杂度？** | **中等。** 主要改动在 trainer，增加 rollout + re-noise + correction loss。约 100-150 行新代码。Pipeline 不需要改。 |
| **期望收益？** | **边界跳变减少、时序连贯性提升、长序列质量改善。** 与 _man 互补（_man 解决 known regions，SOAR 解决 generated regions）。 |
| **推荐方案？** | 在 `uncond_fm_man_046b` (epoch 1000) 上做 5K-10K 步 SOAR post-training，lambda=0.1, N=1, LR=2e-5。 |
