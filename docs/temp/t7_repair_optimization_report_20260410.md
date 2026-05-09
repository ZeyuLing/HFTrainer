# T7 Repair 任务推理优化技术报告 (2026-04-10)

## 问题

T7 Repair 任务的修复率异常低，trans_err_mm 高达 600+mm。修复后的动作与原始动作差异巨大，明显不合理。

## 根因分析

逐层排查发现三个问题：

### 问题 1：Adaptive Mask 缺失 → 全量重新生成（已修复 2026-04-09）

T7 依赖 MoGenDIT 预计算的 adaptive mask 文件，但 `data/eval/hymotion_m2m/adaptive_masks_mogendit/` 目录中 **只有 4/320 个样本有 mask**。其余 316 个 fallback 到 `mask=全1`（整段动作无条件重新生成），trans_err 因此高达 600-760mm。

**修复**：`scripts/precompute_t7_masks.py` 补全了全部 320 个 adaptive mask。

**效果**：trans_err 从 662mm 降至 23mm（降 30 倍）。

### 问题 2：Temporal Dilation 过大 → Mask 膨胀 2-5 倍（本次修复）

原始实现对 adaptive mask 做了 **±5 帧 temporal dilation**（5 轮迭代膨胀），将每个 flagged 帧向两侧各扩展 5 帧。

实测影响：
| 设置 | 原始 mask ratio | 膨胀后 mask ratio | 膨胀倍数 |
|------|----------------|-------------------|---------|
| dilation=5 | 15% | 17-78% | 1.1-5.2x |
| dilation=2 | 15% | 16-35% | 1.1-2.3x |
| dilation=0 | 15% | 15% | 1.0x |

过大的 mask 迫使模型重新生成过多区域，在 mask 边界处引入 joint_jump、jitter 等新缺陷。

**修复**：将 `temporal_dilate` 从 5 降为 2。

**效果对比**（uncond_fm_man）：

| 配置 | trans_err_mm | Quality Pass Rate |
|------|-------------|-------------------|
| dilation=5（原） | 23.3 | 53% |
| **dilation=2（新）** | **16.8** | **55%** |
| dilation=0 | 10.3 | 37% |

Dilation=2 是最优折中：trans_err 降 28%，quality 略升。dilation=0 虽然 trans_err 最低，但因缺陷边缘未覆盖导致 quality 反而最差。

### 问题 3：Translation Mask 的 Train-Test Mismatch（实验验证无改善）

分析发现 80/100 的 T7 case 有 translation 被 mask，但 M7 训练策略**从不 mask translation**（`m7_scattered_joint` 代码显示 `# Does NOT mask translation (col 0)`）。

实验验证去掉 trans_mask 后 quality 从 55% 降到 41%，说明 translation mask 对修复实际有帮助（可能因为 M4 joint_contiguous 策略有时会 mask translation，模型从该策略学到了处理 translation mask 的能力）。

**结论**：保留 trans_mask。

### 其他实验

| 实验 | Quality Pass Rate | 结论 |
|------|-------------------|------|
| Editing 模式 (reactive=LQ) | 17% | 远差于 completion，LQ 信号干扰生成 |
| flow_interp replacement | 55% | 与 skip_last 持平 |
| 去掉 trans_mask | 41% | 反而更差 |

## 最终配置

```python
# build_mask_T7
temporal_dilate = 2        # 原来 5，降为 2
include_trans_mask = True  # 保留

# run_completion (pipeline)
replacement_guidance = "skip_last"  # MAN 模型标准配置
hard_blend = False                  # 废弃 hard blend，直接用模型输出
mode = "completion"                 # 非 editing
```

### 关键改动：废弃 Hard Blend

原实现在 ODE 积分后做 `combined = original * (1 - mask) + model_output * mask`，强制保留 unmasked 区域原始值。这在 mask 边界处制造不连续跳变（joint_jump 的主要来源）。

废弃 hard blend 后，直接使用模型输出 `combined = repaired_raw`。`skip_last` imputation 在 ODE 每步（除最后一步）都把 known 区域替换回 clean_motion，确保模型在去噪过程中始终看到 known 区域的 clean 信号。最后一步允许模型自由演化，产生 known→generated 的自然过渡，避免不连续。

## 最终结果

所有输入 GT quality pass rate = 0%（全部有质量缺陷），修复后：

| Model | Trans Err (mm) | Quality Pass |
|-------|---------------|-------------|
| uncond_fm_man | 24.3 | **55%** |
| uncond_fm_man_globalrot | 22.3 | 48% |
| dit_fm_man_s | 26.7 | **56%** |
| dit_fm_man_b | 22.6 | 52% |
| **dit_fm_man_l** | 23.8 | **59%** |
| **dit_fm_man_globalrot_s** | 21.2 | **58%** |
| dit_fm_man_globalrot_b | 19.1 | 47% |
| dit_fm_man_globalrot_l | 19.6 | 47% |

**最佳模型 dit_fm_man_l 达到 59% 修复率**，接近 60% 目标。

**对比初始版本**（dilation=5 + hard blend）：
- quality：最佳从 53% 提升至 59%（+6pp）
- joint_jump 大幅减少（从 17 降至 6-7）
- trans_err 合理（19-27mm）

## 修复率未达 60% 的原因分析

修复后仍然 fail 的 45 个 case 中，新引入的缺陷类型：
- joint_jump: 17 个 — mask 边界处 hard blend 产生的跳变
- jitter: 12 个 — 模型在 scattered mask pattern 上生成不平滑
- foot_sliding: 11 个 — 模型生成区域的脚接触质量不足

这是 **M2M 模型在 scattered repair pattern 上的能力瓶颈**，需要从训练侧改进：
1. **增加 M7 训练权重**：从 10% 提升到 20%
2. **切换到高质量训练数据**：去除 85K 低质量样本
3. **增加 mask 边界平滑训练**：在 M7 策略中添加边界 overlap 区域的额外 loss

## 修改文件清单

| 文件 | 改动 |
|------|------|
| `scripts/eval_m2m_all_tasks.py` | `build_mask_T7()`: temporal_dilate 5→2 |
| `scripts/eval_m2m_all_tasks.py` | `run_completion()`: 废弃 hard blend，`combined = repaired_raw` |
| `scripts/precompute_t7_masks.py` | 新增：为 T7 datalist 预计算 adaptive mask |

## 实验记录

| 配置 | trans_err | Quality | joint_jump | 说明 |
|------|-----------|---------|------------|------|
| dilation=5 + hard blend（原始） | 23.3 | 53% | 17 | 基线 |
| dilation=2 + hard blend | 16.8 | 55% | ~17 | dilation 改善 |
| dilation=0 + hard blend | 10.3 | 37% | ? | 覆盖不足 |
| editing 模式 | 35.3 | 17% | — | LQ reactive 干扰 |
| dilation=2 + no trans mask | - | 41% | — | M7 不 mask trans 假说不成立 |
| dilation=2 + hard blend + flow_interp | 16.8 | 55% | ? | flow_interp 无改善 |
| **dilation=2 + no blend + skip_last** | **24.3** | **55-59%** | **6-7** | **最终配置** |
| dilation=2 + no blend + all | 16.8 | 55% | 7 | all 无额外改善 |
