# E9 D_strict_mask_d2_b3 抖动/抽搐分析 (2026-04-22)

## 问题
用户反馈 D_strict_mask_d2_b3 QC pass 不错，但生成的动作**抖动**、**抽搐**，"可能被 LQ 保留帧拉扯"。

## 诊断数据

Sample 00001 (uncond_local × D_strict_mask_d2_b3):
- R_Wrist jitter (mean acc): **0.0103 m/frame²** (约 9.3 m/s² @30fps)
- L_Wrist jitter: 0.0085 m/frame²
- L_Elbow jitter: 0.0062 m/frame²
- Max acc: **0.132 m/frame²** ≈ 120 m/s²（单帧抖动过 10g 加速度）

上肢远端关节（wrist/elbow）抖动最严重 → 与用户"抽搐"观察一致。

## 根因

### Pipeline 设定
- mask: strict adaptive mask (joint-aggregate + dilate + min_blob filter)
- `replacement_guidance='skip_last'`（CLI 默认）
- `clean_motion=LQ`（_run_inference_pass 设置）
- editing_mode=False（completion 模式）

### 每 ODE step 的行为
```python
# from hymotion_m2m_pipeline.py
x[keep_mask] = LQ[keep_mask]    # imputation replacement
# 其它区域从 noise 跑 flow
```

### 分布不匹配（OOD）

v2 model 用 MAN 训练（`mask_aware_noise=True`），训练时 flow step:
```python
x_t = (1-t) * noise + t * x1        # x1 = clean GT
x_t[keep] = x1[keep]                 # ← 训练时 known 是 CLEAN GT
```

E9 推理时：
```python
x_t[keep] = clean_motion[keep]       # ← clean_motion = LQ (CORRUPTED)
```

**关键区别**：
- Training: 已知帧 = clean motion manifold
- Inference: 已知帧 = **corrupted/jittery motion**
- Generated 区域的模型在训练中从未见过"邻居 known 帧带着 jitter"的情况
- 模型试图协调 generated 部分与 jittery known 部分 → 产生怪异/抽搐输出

### 为什么 wrist/elbow 最糟
1. **LQ 自身在 wrist 有抖动**（E9 datalist 的 defect_type 经常是 jitter/wrist_twist）
2. 即使 strict mask 过滤了部分 wrist 缺陷位置，**仍有很多 LQ wrist 帧被保留**（adaptive mask recall 不完美）
3. 保留的 jittery LQ wrist 帧 → model 在 generated wrist 帧被迫 continue the jitter → **fragment** amplification

### 为什么 boundary 拉扯严重
`_compute_strict_adaptive_mask` 的 min_blob=3 filter 产生的 surviving blob 可能是 3-frame spans，被大量 LQ 帧包围。**每个 blob 的首末帧**都是尖锐的 known/generated boundary → 模型在 boundary 上需要无缝衔接 jittery 的 LQ → 产生 jerky 过渡。

## 可能的修复方向

### Option 1: Editing mode (reactive=LQ)
让模型知道已知帧是 **corrupted LQ** 而非 clean：
- `_editing_mode=True`
- reactive channel 携带 LQ 值（而不是 0）
- 模型训练时在 editing mode 见过 corruption（15% editing_prob + 5 corruptors）
- 应该对 LQ corruption 更鲁棒

但 editing mode 的问题是：reactive=LQ 告诉模型"这里是 LQ + 需要修复"，但 completion 的 imputation 等于强制 x_t[keep]=LQ，可能冗余。需要对比测试。

### Option 2: Pre-smooth LQ before imputation
在把 LQ 作为 `clean_motion` 之前，先对 LQ 做 **Gaussian 时间滤波**（平滑 high-freq jitter），减少模型"被拉扯"的幅度。代价：保留帧可能丢失一些 GT 细节。

### Option 3: Reduce blob sharpness (平滑 boundary)
改 mask 从 binary 变成 **soft mask**（在 blob boundary 附近用 0.5、0.25 过渡）。结合 `replacement_guidance='flow_interp'` 模式（已有但未使用）。

### Option 4: MoGenDIT-style motion degradation 训练
根本解法：v2 model 只有 15% editing + 5 corruptors，不如 MoGenDIT 的 50% batch + 8 corruptors。扩展训练 corruption 覆盖率和类型。

## 推荐

先尝试 **Option 1 (editing mode)** + **Option 3 (soft mask / flow_interp)**，分别 ablation，观察抖动是否降低。Option 4 是长期改进，不在本轮范围。
