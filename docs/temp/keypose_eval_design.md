# Keypose Guidance Evaluation — 技术方案

## 概述

Keypose Guidance 评估系统测试以下场景：给定一段源动作（src motion）和从目标动作中提取的 1-2 个关键帧姿态（target keyposes），模型能否生成一段新动作，使其既保持源动作的整体风格，又精确经过指定的关键帧姿态。

## 数据

- **源数据**：PeacekeeperElite_part4 的 before/after motion pairs（155 对）
- **格式**：135-dim rot6d（3 translation + 132 body rot6d = 22 joints × 6）
- **来源**：`data/PeacekeeperElite_MB/PeacekeeperElite_part4_{before,after}_MB/`

## Pipeline

```
                                    ┌─────────────────────┐
                                    │  Target Motion       │
                                    │  (after/手工修复)     │
                                    └────────┬────────────┘
                                             │ 提取 1-2 个 keypose
                                             ▼
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Src Motion   │───▶│  M2M / MoGenDIT│───▶│  Raw Output      │───▶│  Postprocess   │
│  (before/原始)│    │  SDEdit/Denoise│    │  (模型几乎≈src)   │    │  (核心修正逻辑) │
└──────────────┘    └───────────────┘    └──────────────────┘    └────────┬───────┘
                                                                         │
                                                                         ▼
                                                                 ┌──────────────┐
                                                                 │  Final Output │
                                                                 └──────────────┘
```

## Keypose 选择（`select_keyposes`）

从 target motion 中自动选择 1-2 个关键帧：

1. 计算每帧 before vs after 的 body pose 差异（dims 3:135）
2. `peak_ratio = max_diff / mean_diff`：
   - `> 3.0` 且序列足够长 → k=2（差异集中在局部，有两个明显 peak）
   - 否则 → k=1（全局均匀变化或短序列）
3. 贪心选 top-K peak，间距 ≥ min_gap=10 帧

**实际分布**：155 pairs 中 153 个 k=1，2 个 k=2。

## 后处理方案（`postprocess_output`）

> **核心约束**：只使用 src motion + keypose 帧的 target pose，不使用 after motion 的其他帧信息。

模型（M2M SDEdit / MoGenDIT denoise）在 step=10 下几乎不改变输入，输出 ≈ src motion。真正的修正由后处理完成。

### 输入

- `output_motion`：模型输出（实际贡献有限）
- `before_motion`：src motion（用户提供）
- `after_motion[ki]`：仅使用 keypose 帧的 target pose（用户提供的 1-2 帧）

### 修正逻辑

**correction delta** = `after_motion[ki, 3:] - before_motion[ki, 3:]`

这是一个固定的 pose 差向量（包含 root orientation + body joints），描述了"在这个姿态上需要改什么"。

**权重计算**：对 src motion 的每一帧，基于其 body pose（dims 9:135）与 keypose 的距离计算权重：

```python
dist = ||before_motion[f, 9:135] - before_motion[ki, 9:135]||
max_dist = max(corr_norm * 1.5, percentile_40(all_dists))
weight = 0.5 * (1 + cos(π * dist / max_dist))  if dist < max_dist else 0
```

- 姿态越接近 keypose → 权重越高 → 修正越强
- 循环动作中，每次经过类似姿态都会被修正（不需要显式检测周期）

**时域平滑**：相邻帧权重差限制 ≤ 0.05/frame，防止突变。

**应用**：

```python
result[f, 3:] = before_motion[f, 3:] + weight[f] * correction_delta
result[f, 0:3] = before_motion[f, 0:3]  # translation 永远保持 src
```

### 静止动作特殊处理

判断条件：>90% 帧的 body velocity < 0.03 且 max_vel < 0.1

静止动作中所有帧姿态几乎相同，直接对所有 dist < max_dist 的帧应用 full correction。

### 关键设计决策

| 决策 | 理由 |
|------|------|
| 使用 correction delta 而非 absolute target pose | 保留每帧自身的 root orientation 和动态特征 |
| Translation 不修改 | MoGenDIT heading denormalization 会污染 translation |
| Root orientation (3:9) 包含在 correction 中 | before/after 间连续变化，不会跳变 |
| 基于 pose distance 的连续权重 | 比离散的 argrelmin 等价帧更平滑 |
| Temporal smoothing ±0.05/frame | 防止权重突变但允许足够的修正持续时间 |
| max_dist = max(corr×1.5, p40) | 确保循环动作的周期帧被覆盖 |

## 模型变体

| 模型 | Config | Rotation | 说明 |
|------|--------|----------|------|
| M2M-MAN (local) | `uncond_fm_man_046b` | local_rot | Flow Matching + Mask-Aware Noise |
| M2M-MAN (global) | `uncond_fm_man_globalrot_046b` | global_rot | 同上，global rotation |
| MoGenDIT 0.1B | external | local_rot | 外部 diffusion repair model |

由于 postprocess 主导输出，各模型 + 各参数（sde strength / denoise steps）的指标几乎一致。

## 评估指标

| 指标 | 单位 | 计算方式 |
|------|------|----------|
| KF MPJPE | rot6d L2 | `||output[ki] - after[ki]||`，包含 translation + root + body |
| Global MPJPE | rot6d L2 | 所有帧的 `||output - after||` 均值 |
| Src MPJPE | rot6d L2 | `||output - before||` 均值（修正幅度） |
| Boundary Smooth | rot6d L2 | keypose 附近的加速度不连续度 |

注：MPJPE 此处不是毫米单位，是 135 维 rot6d 空间的 L2 距离。

## Viewer

Web 可视化工具：`motion_annot_web/keypose_eval/`

- 3D SMPL mesh 对比：原动作 / 修复结果 / 参考结果 / 目标 Keypose
- 时间轴标记：黄色 = target keypose，橙色 = 等价帧（weight > 0.5）
- 播放时 output mesh 变色：经过 keypose 帧变黄，等价帧变橙
- 支持 URL 分享、WebM 录制、"看差异" 模式
- 下拉框切换模型变体，侧栏浏览文件列表

## 文件

| 文件 | 功能 |
|------|------|
| `scripts/eval_keyframe_pose_guidance.py` | 评估主脚本（keypose 选择 + 模型推理 + 后处理 + 指标） |
| `motion_annot_web/keypose_eval/app.py` | Viewer 后端 Flask API |
| `motion_annot_web/keypose_eval/templates/index.html` | Viewer 前端 |
| `output/eval_keyframe_pose/` | 评估结果（NPZ 文件 + summary） |
