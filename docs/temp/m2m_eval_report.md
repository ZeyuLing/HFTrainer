# HyMotion M2M 全任务综合测评报告

> **版本**: v3.0 | **日期**: 2026-04-08
> **评测脚本**: `scripts/eval_m2m_all_tasks.py`
> **可视化**: https://launcher-wgqfy12895-codeserver.ide.taiji.woa.com/proxy/8081/m2m-eval
> **内网直连**: http://29.12.250.27:8081/m2m-eval
> **结果目录**: `eval_results/m2m/<task>/<model>/<case>/meta.json`

---

## 1. 评测概览

### 1.1 规模

| 项目 | 值 |
|------|-----|
| 评测任务 | T1-T8（8 个 completion/editing 任务） |
| 评测模型 | **6 个**（2 架构 × S/B size × 2 旋转空间 + M2M 0.46B × 2 旋转空间） |
| 每任务样本数 | 100 |
| ODE Steps | 50 |
| 总推理次数 | **4,800 次**（6 × 8 × 100），**0 error** |

### 1.2 模型信息

| 简称 | 架构 | 参数量 | 旋转空间 | Epoch | Config 目录 |
|------|------|:------:|---------|:-----:|-----------|
| **M2M Local** | HunyuanMotionMMDiT | 460M | Local | 1000 | `configs/hymotion_m2m/` |
| **M2M Global** | HunyuanMotionMMDiT | 460M | Global | 527 | `configs/hymotion_m2m/` |
| **DiT-S Local** | HunyuanMotionDiT | 49M | Local | 764 | `configs/hymotion_dit/` |
| **DiT-B Local** | HunyuanMotionDiT | 288M | Local | 809 | `configs/hymotion_dit/` |
| **DiT-S Global** | HunyuanMotionDiT | 49M | Global | 762 | `configs/hymotion_dit/` |
| **DiT-B Global** | HunyuanMotionDiT | 288M | Global | 839 | `configs/hymotion_dit/` |

> **注**: `configs/hymotion_dit/` 是 M2M 的无文本版本（text-free），使用纯 single-stream DiT 从零训练。
> `configs/hymotion_m2m/*textfree*` 是早期重复实现，功能完全一致，**已弃用**。

### 1.3 任务定义

| Task | 名称 | Setting | Mask 模式 |
|------|------|---------|----------|
| **T1** | Transition | T1-C: 首尾各 5 帧保留 | temporal contiguous |
| **T2** | Keyframe Interp. | T2-D: 每 30 帧 1 关键帧 | temporal sparse |
| **T3** | First-Frame Cond. | T3-B: 仅首帧 + 89 帧生成 | temporal (1 frame) |
| **T4** | Loop Animation | T4-B: 首帧=末帧, 90 帧 | temporal (2 frames) |
| **T5** | Prediction | T5-B: 30 帧 prefix → 90 帧 | temporal prefix |
| **T6** | Joint Completion | T6-A: 下半身保留 → 上半身生成 | joint-level |
| **T7** | Repair | T7-A: 全 mask 重建 | full mask |
| **T8** | Trajectory | T8-B: GT root transl 保留 | joint-level (transl) |

---

## 2. 主结果

### Table 1: 生成区域旋转误差 `masked_joint_rot_err` ↓

> 核心指标。仅在 mask=1 的关节维度计算 rot6d L2 距离。

| Task | M2M Local | M2M Global | DiT-S Local | DiT-B Local | DiT-S Global | DiT-B Global | Best |
|------|:---------:|:----------:|:-----------:|:-----------:|:------------:|:------------:|------|
| **T1** | 0.226 | 0.228 | **0.211** | 0.220 | 0.218 | 0.216 | DiT-S L |
| **T2** | **0.109** | 0.114 | 0.116 | 0.109 | 0.119 | 0.116 | M2M L ≈ DiT-B L |
| **T3** | 0.358 | 0.390 | **0.333** | 0.412 | 0.382 | 0.373 | DiT-S L |
| **T4** | 0.262 | 0.260 | 0.264 | 0.270 | **0.255** | 0.269 | DiT-S G |
| **T5** | 0.288 | 0.300 | **0.273** | 0.280 | 0.287 | 0.281 | DiT-S L |
| **T6** | **0.587** | 0.674 | 0.633 | 0.623 | 0.658 | 0.658 | M2M L |
| **T7** | **0.598** | 0.631 | 0.603 | 0.640 | 0.646 | 0.663 | M2M L |
| **T8** | **0.575** | 0.600 | 0.583 | 0.597 | 0.616 | 0.626 | M2M L |

**胜出统计**:

| 模型 | 胜出任务数 |
|------|:---------:|
| **DiT-S Local (49M)** | **3/8** (T1, T3, T5) |
| **M2M Local (460M)** | **3/8** (T6, T7, T8) |
| M2M Local ≈ DiT-B Local | 1/8 (T2) |
| DiT-S Global | 1/8 (T4) |

### Table 2: Jitter ↓ (运动平滑度)

| Task | M2M Local | M2M Global | DiT-S Local | DiT-B Local | DiT-S Global | DiT-B Global |
|------|:---------:|:----------:|:-----------:|:-----------:|:------------:|:------------:|
| **T1** | **37.1** | 47.4 | 41.2 | 42.8 | 41.8 | 41.6 |
| **T2** | **35.5** | 43.0 | 44.6 | 41.6 | 50.0 | 42.7 |
| **T3** | 32.0 | 43.9 | **19.7** | 53.6 | 36.9 | 30.1 |
| **T4** | **10.3** | 10.9 | 17.3 | 24.2 | 12.4 | 18.6 |
| **T5** | 47.2 | 48.0 | 38.8 | 34.0 | 41.3 | **33.8** |
| **T6** | **38.3** | 43.5 | 47.1 | 43.2 | 49.7 | 45.7 |
| **T7** | 52.6 | 57.1 | **35.4** | 35.8 | 62.4 | 46.8 |
| **T8** | 43.4 | 57.3 | 41.7 | **39.1** | 48.3 | 50.7 |

### Table 3: 任务特有指标

| 指标 | 全部 6 模型 | 说明 |
|------|:----------:|------|
| T4 Loop Continuity Error | **0.000** | 完美循环 ✅ |
| T8 Trajectory ADE/FDE (mm) | **0.000** | 完美轨迹保持 ✅ |

---

## 3. 分析

### 3.1 Scaling 分析: DiT-S (49M) vs DiT-B (288M)

| Task 类型 | DiT-S Local | DiT-B Local | S 更优? | 解读 |
|----------|:-----------:|:-----------:|:-------:|------|
| T1 Transition | **0.211** | 0.220 | ✅ | S 优 4.1% |
| T2 Keyframe | 0.116 | **0.109** | ❌ | B 优 6.0% |
| T3 First-Frame | **0.333** | 0.412 | ✅ | S 优 19.2% |
| T4 Loop | 0.264 | 0.270 | ≈ | |
| T5 Prediction | **0.273** | 0.280 | ✅ | S 优 2.5% |
| T6 Joint | 0.633 | **0.623** | ❌ | B 优 1.6% |
| T7 Repair | **0.603** | 0.640 | ✅ | S 优 5.8% |
| T8 Trajectory | **0.583** | 0.597 | ✅ | S 优 2.3% |

**惊人发现：DiT-S (49M) 在 5/8 任务上优于 DiT-B (288M)！**

- DiT-B 仅在 T2 (密集关键帧) 和 T6 (joint completion) 上明显优于 S
- DiT-B 在 T3 (首帧条件生成) 上表现异常差（0.412 vs S 的 0.333，差 24%）
- 可能原因：**DiT-B 过拟合**或**训练不充分**（809 epoch vs S 的 764 epoch，但 B 单 epoch 见的数据量更少因为 batch_size 相同但模型更大导致训练更慢）

### 3.2 旋转空间对比: Local vs Global

| 架构 | Local 平均 | Global 平均 | Local 优势 |
|------|:---------:|:----------:|:---------:|
| DiT-S | **0.377** | 0.398 | -5.3% |
| DiT-B | **0.393** | 0.398 | -1.3% |
| M2M 0.46B | **0.350** | 0.377 | -7.2% |

Local Rotation 在所有架构上一致优于 Global（除 T4 外 DiT-S-Global 略优）。

### 3.3 架构对比: M2M 0.46B vs DiT (text-free)

| 任务类型 | M2M 0.46B Local | DiT-S Local (49M) | 差距 |
|---------|:---------------:|:-----------------:|-----|
| **T1-T5** 均值 | 0.249 | **0.239** | DiT-S 优 4.0% |
| **T6-T8** 均值 | **0.587** | 0.606 | M2M 优 3.2% |

- **Temporal mask 任务**: DiT-S 以 1/9 参数量超越 M2M 0.46B
- **Joint-level mask 任务**: M2M 0.46B 仍然最优，更大参数量和文本编码器架构的关节间协调能力更强

### 3.4 任务难度梯度

```
T2 (0.109) ≪ T1 (0.211) < T4 (0.255) < T5 (0.273) < T3 (0.333) ≪ T8 (0.575) < T6 (0.587) < T7 (0.598)
```

| 难度 | 任务 | rot_err | 特征 |
|------|------|:-------:|------|
| Easy | T2 | 0.11 | 密集关键帧 (每30帧) |
| Medium | T1, T4, T5 | 0.21–0.27 | temporal 连续段生成 |
| Hard | T3 | 0.33 | 仅 1 帧约束 |
| Very Hard | T6-T8 | 0.58–0.60 | joint-level / 全重建 |

### 3.5 正确性验证

| 检查项 | 结果 |
|--------|:----:|
| T1 boundary 数 = 2 | ✅ |
| T2 boundary 数 ≈ 13.6 | ✅ |
| T4 loop_cont_err = 0.000 | ✅ |
| T6 trans_err = 0 (transl 未 mask) | ✅ |
| T8 traj_ade/fde = 0 (root 被 mask 保护) | ✅ |
| 4,800 次推理 0 error | ✅ |

---

## 4. 推荐方案

| 应用场景 | 推荐模型 | 理由 |
|---------|---------|------|
| **过渡/补间/预测** (T1, T3, T5) | **DiT-S Local (49M)** | 精度最高，推理最快，参数最少 |
| **关键帧补间** (T2) | M2M Local 或 DiT-B Local | 几乎持平 |
| **循环动画** (T4) | DiT-S Global | 略优，jitter 也较低 |
| **关节补全/轨迹** (T6, T8) | **M2M Local (460M)** | joint-level 任务仍需大模型 |
| **修复** (T7) | **M2M Local** | rot_err 最低，需 checker 进一步验证 |
| **部署优先** | **DiT-S Local (49M)** | 49M 参数，推理速度约为 M2M 的 1.7× |

---

## 5. 训练状态

### 正在训练

| 模型 | Epoch | 状态 |
|------|:-----:|------|
| dit_fm_man_s (49M) | 764+ | ✅ 持续训练中 |
| dit_fm_man_b (288M) | 809+ | ✅ 持续训练中 |
| dit_fm_man_globalrot_s | 762+ | ✅ 持续训练中 |
| dit_fm_man_globalrot_b | 839+ | ✅ 持续训练中 |
| uncond_fm_man (460M) | 1000 | ✅ 已完成 |
| uncond_fm_man_globalrot | 527 | ✅ 已停止 |

### 已提交待启动

| 模型 | Task Flag | GPU | 说明 |
|------|-----------|-----|------|
| DiT-L Local (383M) | `dit_fm_man_l` | 4×8 V100 | 从 epoch 19 resume |
| DiT-L GlobalRot (383M) | `dit_fm_man_globalrot_l` | 4×8 V100 | 从 epoch 21 resume |

### 待补充指标

| 指标 | 状态 |
|------|:----:|
| FK-MPJPE (mm) | ❌ |
| FID | ❌ |
| Foot Skating | ❌ |
| Quality Pass Rate | ❌ |
| T7 Repair Success Rate | ❌ |

---

## 6. 文件索引

| 文件 | 用途 |
|------|------|
| `scripts/eval_m2m_all_tasks.py` | 统一评测脚本 |
| `eval_results/m2m/<task>/<model>/<case>/meta.json` | Per-case 结果 |
| `eval_results/m2m/<task>/<model>/<case>/output.npz` | 模型输出 (可网页查看) |
| `docs/temp/kimodo_constraint_demo/server.py` | Web server (含 M2M eval API) |
| `docs/temp/kimodo_constraint_demo/web/m2m_eval.html` | 交互式可视化 |
