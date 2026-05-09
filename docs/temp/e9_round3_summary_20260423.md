# E9 Round 3 修复总结 (2026-04-23)

## 用户反馈 4 个问题 + 处理

### 1. Strict mask 中 masked 区域从何处起步？
**回答**：默认从**纯噪声 z** 起步（pipeline.py:322 `y0 = torch.where(keep_mask, x_clean, z)`），每步 `skip_last` 把 keep 区域强制设为 x_clean (=LQ)。
**新增选项**：`_sdedit_tau > 0` 让 masked 区域从 `(1-τ)*z + τ*LQ` 起步，anchored to LQ。
  - `D_strict_mask_d2_b3_sdedit03` (τ=0.3)
  - `D_strict_mask_d2_b3_sdedit05` (τ=0.5)

### 2. StableMotion 结果错误 ✅ 已修复
**根因**：StableMotion feats decoder 返回 `trans = [trajectory, pelvis_joint_z]`，但 M2M 的 trans 是 SMPL **root translation**（比 pelvis joint 高 ~0.22m = `-bone_offsets[0].y`）。同时 canonicalization（init_rotZ、trajectory[0]）也没有 revert。
**Fix**：在 `scripts/run_stablemotion_e9.py` 的 decanon 阶段：
  - Rotate back by +rotZ0_angle (Z-axis)
  - Add traj0_xy back to XY
  - Add ground_shift + pelvis_offset (≈ +0.22m) to gravity axis
**验证**：LQ pelvis Y [0.68, 0.93] vs HQ [0.66, 0.93]，keep frames diff 0.028m。

### 3. D_ada_denoise_t010 HQ 完全不 follow LQ ✅ 已诊断+修复
**根因**：Stage 1 已正确做 SDEdit τ=0.5。**但 Stage 3 restore pipeline.sdedit_tau = prev_tau = 0**，所以 Stage 3 的 mask=1 区域从纯噪声起步，生成的"修复"和 LQ 无关（虽然 QC pass rate 高因为生成了一个不同但干净的动作）。
**Fix**：加 `_ada_stage3_sdedit_tau` kwarg 控制 Stage 3 的 SDEdit τ。新增：
  - `D_ada_denoise_t010_s3tau03` (τ=0.3)
  - `D_ada_denoise_t010_s3tau05` (τ=0.5) — QC 掉到 37.7% 但 HQ 真正 follow LQ

### 4. Smooth 做了什么？
`_gaussian_temporal_smooth` 只对 `clean_motion`（LQ，用于 imputation 参考值）做时间维 Gaussian 卷积。σ=1 帧 ≈ 5 Hz cutoff @ 30fps。**只作用于 keep 区域**（mask=0），generated 区域原样 pass through。目的：减少 MAN training OOD 导致的 boundary 拉扯。**不是对 generated 区域去抖动**。

### 5. Smooth1/Smooth2 基本相同 ✅ 已清理
实测 smooth1 (σ=1): jitter=1467, qc=64.2% / smooth2 (σ=2): jitter=1525, qc=64.2%
smooth1 略优，smooth2 已 deprecate 并删除。

### 6. 局部跳变溯源 ✅ 已修复
**根因**：binary mask 在 0↔1 transition 处有硬边界，pipeline 每步 imputation `x[keep] = LQ` 让 boundary 两侧速度场不连续，产生加速度峰值 ≈ 0.13 m/frame²（实测 frame 107-108 所有关节同时 spike）。
**Fix**：加 `_boundary_smooth_radius` / `_boundary_smooth_sigma` kwargs 做 post-process Gaussian blending：在每个 mask transition 的 ±radius 帧窗内，用 smoothed_output 和 output 以 tent weight 加权混合。
**结果**：
  - D_strict_mask_d2_b3 (baseline): jitter=1504, QC=60.9%
  - D_strict_mask_d2_b3_smooth1: jitter=1467, QC=64.2%
  - **D_strict_mask_d2_b3_bsmooth** (radius=3, σ=2): **jitter=1350, QC=69.3%** ← 最佳
  - D_strict_mask_d2_b3_bsmooth_tight (radius=2, σ=1): jitter=1356, QC=69.3%

### 7. QC Pass Rate 语义确认 ✅
`motion_quality_checker.py:307`: `is_valid = len(failed_checks) == 0`，即一条数据要通过**所有 20 个 checker** 才 `is_valid=True`。然后 aggregate 求 mean 作为 QC Pass Rate。**用户担心的 "以 checker 为单位" 没发生** — 指标语义正确。

## E9 当前 Dashboard 最终状态（10 个 runs）

| run | setting | jitter | QC pass |
|---|---|---|---|
| 2764 | uncond_local D_ada_denoise_t010 (原) | 546 | **82%** ⚠️ HQ 不 follow LQ |
| **2780** | uncond_local D_strict_mask_d2_b3_bsmooth ⭐ | **1350** | **69%** ✅ follow LQ |
| 2781 | uncond_local bsmooth_tight | 1356 | 69% |
| 2767 | uncond_local smooth1 | 1467 | 64% |
| 2766 | uncond_local d2_b3 baseline | 1504 | 61% |
| 2779 | uncond_local sdedit05 | 1966 | 25% |
| 2777 | uncond_local s3tau05 | 1974 | 38% |
| 2778 | uncond_local sdedit03 | 2169 | 17% |
| 2776 | uncond_local s3tau03 | 2320 | 28% |
| 2775 | stablemotion | 3374 | 13% |

**用户需要对比**：
- 重视 QC + 不介意 HQ 与 LQ 差异大 → **D_ada_denoise_t010 原版**（run 2764）
- 重视 follow LQ + 质量平衡 → **D_strict_mask_d2_b3_bsmooth**（run 2780，推荐默认）
- 极致 follow LQ（改动最少） → **D_strict_mask_d2_b3_sdedit05**（run 2779）

## 下一步建议

1. 将 `D_strict_mask_d2_b3_bsmooth` 作为 E9 默认推荐 setting
2. 调研 flow_interp 模式（replacement_guidance='flow_interp'）是否能根治 boundary 跳变
3. Post-train 引入 motion degradation 训练，让模型真正学会 repair 而非 regenerate