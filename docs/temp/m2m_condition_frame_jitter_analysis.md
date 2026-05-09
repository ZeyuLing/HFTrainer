# M2M Condition Frame Jitter Analysis

## 现象

HYMotion M2M 推理结果在 condition frame（条件帧）附近存在明显抖动/跳变：
- 边界帧 acc 峰值为平均值的 **5-12 倍**
- E7 (first_frame) 首帧 acc ratio = 12.56x（非常严重）
- E2 (in-between) 首尾边界 acc ratio = 5.3x / 8.8x
- E3 (keyframe) 多 keyframe 处累积跳变

## 根因分析

`scripts/eval_m2m_all_tasks.py:781` 使用 `replacement_guidance='skip_last'` 模式。

该模式的 per-step 实现（`hymotion_m2m_pipeline.py:292`）：
```python
elif rep_mode == 'all' or (rep_mode == 'skip_last' and not is_last_step):
    x = torch.where(keep_mask, x_clean, x)
```

**硬替换**（hard replacement）：每个 ODE step 用 clean motion 强制覆盖 known 区域。这导致两个问题：

1. **训练-推理分布不匹配**：训练时 `x_t[known] = x1`（纯 clean）；而推理时的 x 是 flow matching integration 的结果，在 t < 1 时应该是 `x_t = (1-t)*z0 + t*x1` 的 noisy 版本。硬替换为纯 x1 偏离了训练分布。

2. **边界处速度不连续**：keep frames 被硬替换为精确 clean 值，generated frames 通过 ODE 积分到了一个 flow-match 最优解，两者在边界处存在 O(1) 级别的差异，造成可见跳变。

`skip_last` 只是不在最后一步替换，实际上问题在**倒数第二步**的替换已经固化了跳变。

## 可选解决方案

### 方案 A：切换到 `flow_interp` 模式（推荐）

代码中已经实现：
```python
if rep_mode == 'flow_interp' and not is_last_step:
    t_next = t[i + 1]
    x_interp = (1 - t_next) * z0 + t_next * x_clean
    x = torch.where(keep_mask, x_interp, x)
```

在每步用 `flow_interp(z0, x_clean, t_next)` 替代 known 区域。这保持了训练分布（x_t 的正确 t-依赖形态），显著降低边界跳变。

**修改**: `scripts/eval_m2m_all_tasks.py:781`
```python
replacement = "flow_interp"  # was: "skip_last"
```

**代价**: 需要重新跑所有 eval 任务（~几小时）。

### 方案 B：推理后处理（零训练代价，仅修可视化）

在 eval dashboard 的 `load_npz_positions()` 后对 motion_135 的 **条件边界帧附近做 5-10 帧的加权平滑**。这是纯前端/后端可视化优化，不改变数据本身，但能消除视觉抖动。

```python
# 对 boundary frames 的 ±3 帧窗口做 Savitzky-Golay 或 Gaussian smoothing
# 权重保证 condition frame 不变（w=1），gen frame 边界权重渐变
```

**代价**: 指标会失真（MPJPE 等会改变），不能用于论文数据。仅用于 demo 展示。

### 方案 C：使用 `position_constraint` 而非 replacement（只适用于 E4/E6）

Pipeline 支持 position constraint 通过 IK 投影。对 end-effector 任务应该用这个代替 hard replace。当前 eval 脚本没有使用，可以评估效果。

## 推荐

用**方案 A**：一行代码改动 + 重跑 eval。预期效果：
- boundary acc ratio 从 5-12x 降到 1.5-2x
- 视觉上 condition frame 附近连续

如果不能重跑 eval，则**方案 B** 作为临时视觉修复。

## 相关文件

- 推理 pipeline: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py:287-293`
- Eval 脚本: `scripts/eval_m2m_all_tasks.py:781`
- 可视化计算: `motion_annot_web/eval_dashboard/utils.py:load_npz_positions()`
