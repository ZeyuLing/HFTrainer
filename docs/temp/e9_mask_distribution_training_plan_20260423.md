# E9 Mask Distribution Mismatch — 训练侧根治方案

**日期**: 2026-04-23
**背景**: E9 Motion Repair 中 `m2m_strict_bsmooth_combo` 在 case 00165 等样本上出现脖子突然抬头/转头的 phantom motion。Ablation 证实根因不是 post-processing，而是**推理时 mask 分布与训练 mask 分布 mismatch**，导致模型在 strict_mask 产物（"某 joint 长连续 mask + 其他 joint 高密度 anchor"）上处于 OOD 区域。
**目标**: 让训练 mask 分布覆盖推理 mask 分布，根治 jitter 问题。

---

## 1. 问题诊断回顾

### 1.1 Ablation 关键结果（case 199 head rotation delta max）

| 实验 | 设置 | head max rot Δ | 含义 |
|------|------|----------------|------|
| A | 纯 T2M（全 joint 全时刻生成）| 0.001 | baseline，生成是平滑的 |
| B | Mask 仅 head（其他 joint 全 anchor 成 clean GT）| **0.260**（81 帧 >6°）| 触发 jitter |
| C | 所有 joint 只 anchor 3 帧 | 0.713（4 帧 >6°）| 稀疏 anchor 不是主因 |
| D | Mask head + 其他 joint 全 anchor（仅 head 留 3 帧）| 0.591（83 帧 >6°）| 确认"密集 anchor 包围 free joint"触发 |

→ **"单 joint 被长连续 mask + 周围 joints 被密集 anchor"** 是触发 pattern。

### 1.2 为什么训练有 M1 random_cell 还会 OOD

"随机 Bernoulli 采样"不等于"任意 mask pattern 都被覆盖"：

- 训练集采样分布 = Bernoulli product，集中在 **typical set**（joint 间近似独立 + 边际 mask 率接近全局 p）
- E9 adaptive strict_mask 产物 = **typical set 外的尾部**（joint 间强相关 + 单 joint 连续长块 + 密度异质）
- 概率估算（T=200, J=22, p=0.3）：
  - "Head 全 mask + 其他 21 joint 全 keep" 的训练出现概率 ≈ 10^(-650)
  - "Head 连续 80 帧 mask + 其他 joint mask 率≈0.05" ≈ 10^(-200)
- 训练 5×10^5 样本对这些 pattern 的有效覆盖 ≈ 0

**M7 scattered_joint** 虽然设计模拟 checker mask，但参数是 1-8 frame dilation + 1-3 joints per spot，**差推理产物（30-100 连续帧 + kinematic chain 相关）一个数量级**。

---

## 2. 候选方案（按可行性递增）

### 方案 1: Self-distillation / Inference-aware mask training
- **做法**: 训练时在 clean motion 上注入 synthetic LQ artifact → 用当前 checkpoint 的 checker pipeline 产出 adaptive mask → 作为训练 mask
- **优点**: 训练分布严格包含推理分布
- **缺点**: 需要 synthetic LQ injector + 把 checker pipeline 搬进 dataloader；每 batch +30% CPU 时间
- **工程量**: 大

### 方案 2: 直接从真实 LQ data 标注 mask
- **做法**: 利用 `motion_annot_web/m2m_database` 里 93K 低质量样本 + 预跑的 checker output mask
- **致命缺陷**: 缺 HQ ground truth（LQ motion 没有对应 clean 版本）
- **结论**: 单独不可行，必须配合方案 3

### 方案 3: Synthetic LQ injection（principled，推荐中期方案）
- **做法**: 在 clean HQ motion 上反向注入 checker 定义的 artifact，然后用 adaptive detector 得 mask
  ```
  x_hq_clean → inject_artifact(jitter/foot_sliding/joint_jump/...) → x_lq
  mask = run_adaptive_detector(x_lq)   # 与推理同分布
  训练目标: (x_lq, mask) → x_hq_clean
  ```
- **优点**:
  - HQ GT 天然有（就是原 clean motion）
  - Mask 分布 = detector 产物 = 推理分布
  - 每种 artifact 可独立控制比例
- **缺点**: 需要实现所有 15+ checker 的反向过程；synthetic ≠ real LQ
- **复用点**: `motion_annot_web/quality_check_rules/` 已有 checker 阈值/逻辑
- **工程量**: 中-大（1-2 周）

### 方案 4: M8 mock_adaptive 采样策略（最轻量，首选短期方案）
在 `hftrainer/datasets/motion/motionhub/transforms/universal_mask.py` 加一个新策略模拟 E9 adaptive detector 产物：

```python
def m8_mock_adaptive(T: int, grid: np.ndarray, rng):
    """M8: Mock E9 adaptive detector output.

    Simulates strict_mask pattern: 1-3 'corrupted' joints each with
    long contiguous block (20-120 frames), kinematic chain neighbors
    co-masked with 50% probability, mild temporal dilation d=2.
    """
    # Step 1: 1-3 个被污染的 joints
    n_corrupt = rng.randint(1, 4)
    corrupt_joints = rng.choice(range(1, NUM_JOINT_GROUPS), size=n_corrupt, replace=False)

    # Step 2: 每个 corrupt joint 一个长连续块
    for j in corrupt_joints:
        block_len = rng.randint(20, min(121, T))
        t_start = rng.randint(0, max(1, T - block_len))
        grid[t_start:t_start+block_len, j] = 1.0

        # Step 3: Kinematic chain dilation
        for nb in KINEMATIC_NEIGHBORS[j]:
            if rng.random() < 0.5:
                grid[t_start:t_start+block_len, nb] = 1.0

    # Step 4: Temporal dilation (d=2) - 与推理一致
    # (实现略，可复用 M7 的 dilate helper 或新写)
```

- 采样权重调整: `M1:15, M2:10, M3:20, M4:12, M5:3, M6:12, M7:8, M8:20`
- **优点**: 零额外数据依赖；1-2 小时实现；可直接在现有 training script 启用
- **缺点**: 依然是 heuristic，和真实 detector 产物有 gap；M8 参数需要对齐 E9 实际 detector 统计

### 方案 5: Mask-conditional consistency loss
同一 clean `x1` 采两个 mask `m_A`（M1）和 `m_B`（M8），要求 `x_hat_A[m_A==0]` 和 `x_hat_B[m_B==0]` 在交集上一致。
- 理论正确但训练开销翻倍，超参 λ 难调
- **优先级低**，除非 1-4 都不够

---

## 3. 推荐执行路线

### Phase 1 (短期，1-2 天) — 验证假设
实现 **方案 4 (M8 mock_adaptive)**，训练 1 个 finetune job：
- Base: 最新 HyMotion M2M v2 checkpoint
- 改动: 只替换 `PrepareM2MUniversalMask` 的 strategy weights，加入 M8
- 训练: 5-10K steps finetune
- **验证指标**: E9 case 00165 head rotation jitter（max/frames>6°）
  - 若从 0.591/83 → <0.1/<10，mask OOD 假设被证实
  - 若几乎无变化，问题在更深的 motion prior / loss 层面

### Phase 2 (中期，1-2 周) — 根治
基于 Phase 1 结果：
- 若有效 → 实现 **方案 3 (Synthetic LQ injection)**，把所有 checker 的反向过程补齐，做一次完整 retrain
- 若无效 → 转向 loss/representation 层（kinematic consistency loss / velocity-space loss）

### Phase 3 (长期) — Online adaptation
实现 **方案 1 (self-distillation)**，让训练过程持续 refresh detector output 作为 mask 分布锚点。

---

## 4. 需要的文件改动清单

### Phase 1 改动（最小集）
- `hftrainer/datasets/motion/motionhub/transforms/universal_mask.py`:
  - 新增 `m8_mock_adaptive` 函数
  - 定义 `KINEMATIC_NEIGHBORS` 邻接表（与 E9 strict_mask 使用的同一套 SMPL kinematic tree）
  - 注册到 `_STRATEGY_FN`
  - 更新 `DEFAULT_STRATEGY_WEIGHTS`
- `configs/hymotion_m2m_v2/` 下新建 finetune config:
  - 继承现有 v2 config
  - 加载 latest checkpoint
  - Steps ≈ 10K，lr 折半
- 评测复用 `tools/eval_m2m_v2_all_tasks.py`，关注 E9 task 的 `m2m_strict_bsmooth_combo` setting

### Phase 3 改动（self-distillation 时）
- 需要 `hftrainer/datasets/motion/motionhub/transforms/artifact_inject.py`（新）
- 需要把 checker pipeline 的 batch 版本搬进 dataloader（避免 per-sample Python overhead）

---

## 5. 评测协议

### 主要对比指标
- E9 task 所有 case 的 rotation jitter（max Δrot、frames>6° 占比）
- `m2m_strict_bsmooth_combo` setting 下 QC pass rate（单 checker + 整体）
- 对照 baseline：当前 v2 checkpoint 直接推理

### 重点 case
- **00165**: head phantom motion（已确认的 failure case）
- 其他 E9 task 中出现过 jitter 的 case（从 dashboard 标注中挑选 top-5）
- 从 `motion_annot_web/eval_dashboard` 可筛选 per-case fail checker

### Regression 测试
- 必须验证 M8 不伤害其他任务（T2M, M2M 常规 completion）
- 跑完整 `eval_m2m_v2_all_tasks.py --save-npz --use-rewritten`

---

## 6. 风险与未解问题

1. **M8 参数的"真实性"**: mock pattern 和 real detector 产物之间的分布差距需要量化（KL / Wasserstein over mask stats）。后续可做个小工具：比较 M1-M8 采样 mask 的统计直方图 vs E9 推理时 mask 的直方图。

2. **Phase 1 若无效**说明什么:
   - Mask 不是主因 → 更可能是 `x_t[keep] = x1[keep]` 的时序不一致性 + skip_last replacement_guidance
   - 需要重新审视 pipeline 的 SDEdit 机制
   - 或者 motion prior 本身在 rare pose transition 上不稳定

3. **E9 detector 本身是否应该改**: 当前 strict_mask 的 kinematic dilation + blob filter 直接把问题放大。推理侧可以先验证：
   - 关闭 kinematic spatial dilation 只保留 temporal dilation，case 00165 是否缓解
   - 如果缓解明显，短期可以先在 detector 侧加 cap（限制单 joint 连续 mask 长度），双管齐下

---

## 7. 交接 checklist

- [ ] Phase 1: 实现 M8 策略 + 更新 weights
- [ ] Phase 1: 创建 finetune config，提交 Taiji 任务
- [ ] Phase 1: 跑 eval on E9，对比 case 00165 及其他 top failure cases
- [ ] Phase 1: 产出 "M8 是否有效" 结论文档
- [ ] Phase 2: 根据 Phase 1 结果决定方案 3 vs loss 层改动
- [ ] 辅助: 做 mask 分布对比工具（M1-M8 vs E9 real detector）

---

## 关联文档

- 问题现象: `docs/temp/e9_d_strict_mask_jitter_20260422.md`
- E9 dashboard 重构: `docs/temp/e9_settings_semantics_20260422.md`
- Adaptive mask 设计: `hftrainer/models/motion/CLAUDE.md` §Repair Pipeline Comparison
- Universal mask 训练: `hftrainer/datasets/motion/motionhub/transforms/universal_mask.py`
