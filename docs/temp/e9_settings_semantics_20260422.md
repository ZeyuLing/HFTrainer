# E9 Motion Repair — 7 Setting 语义说明 (2026-04-22)

## 任务
输入 low-quality (LQ) motion（有 jitter/foot_sliding/joint_jump 等缺陷），输出修复后 HQ motion。

## Setting 速查表

| Setting | 策略 | 与 LQ 相似度 | 适用 |
|---------|-----|------------|------|
| A_adaptive_inpaint | MoGenDIT adaptive mask + SDEdit τ=0.5 | **高**（不变区域锁 LQ） | 保守修复，信任 adaptive mask |
| B_post_replace | 全生成 + post-hoc 按帧混合 | **低**（defect 帧完全重写） | ⚠️ 问题帧整帧替换，像新生成 |
| D_ada_denoise_t010 | 两阶段：全生成→检测变化→skip_last 重生成 | **低**（Stage 1 完全无视 LQ） | ⚠️ "change" 阈值决定保留率 |
| D_strict_mask_d2_b3 | 紧化 MoGenDIT mask + skip_last | **高**（不变区域锁 LQ） | 强信任 adaptive mask |
| D_qc_mask_d1 | QC invalid_mask (dilate=1) | **高** | 信任 quality checker 标记 |
| D_qc_mask_d2 | QC invalid_mask (dilate=2, 推荐) | **高** | 平衡 |
| D_qc_mask_d3 | QC invalid_mask (dilate=3, 激进) | 中 | 扩大修复范围 |

---

## 详细说明

### A_adaptive_inpaint（原推荐）
```
mask = MoGenDIT adaptive_mask (T, 198)   # 0=keep, 1=generate
src_motion = LQ * (1-mask)  # 保留帧置 LQ 值，缺陷帧置零
推理：从 x_1 = τ·noise + (1-τ)·LQ (τ=0.5) 开始去噪（SDEdit 风格）
      replacement='none'
输出 = pipeline 生成结果（unmasked 区域被 VACE inactive 通道锁住在 LQ）
```
**特点**：不变区域精确等于 LQ，只修复 adaptive mask 标的位置。
**优点**：视觉上和 LQ 接近，改动最小。
**缺点**：依赖 MoGenDIT adaptive mask 的准确性。

### B_post_replace（用户吐槽的）
```
Stage 1 推理：mask=all-1 (完全生成), sdedit=0 (纯噪声出发)
            → 得到 C_full (和 LQ 完全无关的新动作)
Stage 2 post-hoc blend:
    adaptive_mask = 读缓存
    frame_defect[t] = (adaptive_mask[t] 任意维度为 1) ? 1 : 0    ← 整帧 01
    dilate + 三角平滑 → w_C[t] ∈ [0, 1]
    output[t] = C_full[t] * w_C[t] + LQ[t] * (1 - w_C[t])
```
**问题所在**：
- Stage 1 的 C_full 完全不参考 LQ（从纯噪声出发 + mask=all-1）
- 只要某帧 adaptive mask 有**任何**关节问题，该整帧 w_C=1 → **整帧 pose 被 C_full 替换**
- 实测：E9 大多数 LQ 有 foot_sliding/jitter 全程问题 → 大多数帧 defect=True → 几乎整段被 C_full 覆盖 → 看起来和 LQ "完全不一样"

**为什么 QC pass rate 还不错**（uncond_local 83%）：因为 C_full 是**新生成的合理动作**，虽然和 LQ 不像，但 QC checker 能通过。**对 metric 好，对可视化体验差。**

### D_ada_denoise_t010（MoGenDIT 原版）
```
Stage 1: mask=all-1, replacement='none', sdedit=0
       → 纯生成得到 stage1_output
Stage 2: change = |LQ_normalized - stage1_output|  # (T, 135)
       per_joint_change = change.reshape(T, 22, 6).max(axis=-1)  # (T, 22)
       clean[t, j] = per_joint_change[t, j] <= 0.10  ← MoGenDIT 默认阈值
       new_mask[t, j] = NOT clean[t, j]  # True = generate
Stage 3: 用 new_mask 重跑推理，replacement='skip_last', clean_motion=LQ
       → 最终输出
```
**问题所在**：
- Stage 1 同样**纯噪声生成**，完全不参考 LQ（和 B 相同缺陷）
- Change 阈值 0.10 是在 **normalized 空间**，相当于 0.1×std ≈ 0.2 rad（rotation）、~20cm（translation）
- 常见 LQ 缺陷是"缓慢抖动 + 脚滑" → change 全 < 0.10 → mask 全 0 → Stage 3 几乎不生成任何东西，输出 ≈ LQ
- 或者 LQ + Stage 1 差异大（因为 Stage 1 是随机生成）→ mask 全 1 → Stage 3 又是纯生成 → **和 LQ 无关**
- 阈值对 motion scale 不鲁棒

**可能的改进方向**：
- Stage 1 改为 SDEdit 从 LQ 出发（τ=0.5 左右加噪）
- 或者 Stage 1 直接用 MoGenDIT adaptive mask（已经指出了缺陷位置）

### D_qc_mask_d1 / d2 / d3（2026-04-22 新增）
```
qc_defect_mask = quality_checker 对 LQ 跑过后标记的 (T, 22) invalid_mask
                 (OR 所有失败 checker 的 per-frame per-joint 结果)
kinematic_dilate (SMPL-22 parent/child) + temporal_dilate=d
root_taint (pelvis 也会污染全身)
mask = expand to (T, 135)
推理：replacement='skip_last', clean_motion=LQ, mask
```
**特点**：信任 QC checker 的 **invalid_mask**（而非 MoGenDIT 的 change-based mask）。QC checker 能发现 persistent 缺陷（脖子弯 180°等 change=0 但明显错的问题）。Unmasked 区域强保留 LQ → **修复后和 LQ 在非缺陷区一致**。

### D_strict_mask_d2_b3
```
raw_mask = MoGenDIT adaptive_mask
joint-aggregate + dilate + min_blob filter → strict_mask
推理：replacement='skip_last', clean_motion=LQ, strict_mask
```
**特点**：紧化 MoGenDIT mask 后做 imputation（类似 A 但更干净）。

---

## 用户吐槽原因总结

> "B_post_replace, D_ada_denoise_t010 完全不符合原始输入的动作内容"

**确实如此**。这两个 setting 的 Stage 1 都是**纯噪声生成**，完全不参考 LQ：
- **B**：`mask=all-1 + sdedit_tau=0` → 从纯噪声出发，生成一个**新** motion
- **D_ada_denoise**：`replacement='none' + sdedit=0` → 同样从纯噪声出发

然后 post-hoc 用 mask 把 LQ 混回去，但：
- B 的 mask 是整帧 0/1 → defect 帧完全被 C_full 替换
- D 的阈值判定不稳 → 要么几乎全生成要么几乎不生成

**推荐 setting**：
- 看修复**视觉**效果：`A_adaptive_inpaint`、`D_qc_mask_d2`、`D_strict_mask_d2_b3`
- 看 **QC pass 指标**：`B_post_replace` (83%), `D_ada_denoise_t010` (71%)

这两者分离的根本原因：**当前模型在纯生成任务上比 repair 任务表现好**（训练数据里 T2M/completion 多，repair pattern 少），所以"从头生成"比"局部修补"更容易通过 QC。但对"修复"语义不对 —— 用户期望的是**保留 LQ 内容 + 局部修复缺陷**，不是**生成一个看起来 OK 的无关动作**。
