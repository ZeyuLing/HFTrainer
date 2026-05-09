# E9 后处理方案评估报告

**日期**: 2026-04-23
**问题**: 后处理能否解决 E9 `m2m_strict` 的 jitter / phantom motion 问题？
**数据源**: `motion_annot_web/eval_dashboard/eval_dashboard.db`, task_id='E9', 43 samples

---

## 1. 评测指标

- **QC pass rate**: 整体 checker 通过率（所有 checker 都 valid 才算 pass）
- **qc_jitter / qc_joint_jump / qc_foot_sliding**: 单 checker 通过率（pass=1.0）
- **jitter_pos**: position-space jitter (越低越好)
- **foot_skating_ratio**: 脚滑帧占比

---

## 2. 实测数据（全部 E9 task, 43 samples）

| Model (setting: default) | QC Pass | jitter% | jump% | slide% | jitter_pos | skate% |
|--------------------------|--------:|--------:|------:|-------:|-----------:|-------:|
| m2m_strict (无后处理)       | **60.9%** | 92.1% | 95.3% | 88.4% | 1504 | 21.6% |
| m2m_strict_bsmooth        | 63.7% | 92.6% | 95.8% | 89.8% | 1467 | 20.2% |
| m2m_strict_smooth1        | 64.2% | 93.5% | 95.8% | 89.8% | 1467 | 21.6% |
| m2m_strict_bsmooth_accelK3| 67.9% | 93.0% | 97.2% | 89.8% | 1385 | 21.0% |
| m2m_strict_bsmooth_savgol5| 72.6% | **100.0%** | 99.1% | 87.9% | 795 | 19.4% |
| **m2m_strict_bsmooth_combo** | **74.9%** | 99.5% | **100.0%** | 89.3% | **603** | **18.9%** |
| stablemotion (baseline)   | 22.8% | 0% | 26.5% | 31.6% | 1744 | 26.6% |

---

## 3. 关键结论

### ✅ Post-processing 确实有效

从 baseline → best post-proc：
- **QC pass**: 60.9% → **74.9%**（+14 pp，相对 +23%）
- **jitter_pos**: 1504 → **603**（-60%）
- **qc_jitter 通过率**: 92.1% → **99.5%**（几乎无抖）
- **qc_joint_jump 通过率**: 95.3% → **100%**（完全无跳变）
- **foot_skating_ratio**: 21.6% → 18.9%（-12.5%）

`bsmooth_combo`（Savitzky-Golay + bone smoothing + accel spike removal）是当前最优后处理组合。

### ❌ 仍未解决的问题

1. **foot_sliding 没改善**：88.4% → 89.3%（几乎持平）
   - 后处理改 rotation 不能治脚滑（需要 contact-aware 约束或 retarget）
2. **QC pass 还差 25 pp 才到 100%**
   - 剩余 failure 主要来自 foot_sliding + 少量 candy_wrapper / neck / rotation_velocity
3. **Phantom motion（如 00165 脖子突然抬头）**：根因是 rotation 生成 OOD，不是抖。
   - `qc_neck` = 98.1%（后处理前后不变），说明 neck checker 漏检了 00165 的"幅度合理但位置突兀"的 phantom motion
   - jitter 指标降下来了，但**"结构性错误的大幅度动作"** savgol 平滑不掉（它是平滑的大幅度错误）

### 判定：Post-processing 部分解决问题，但触及天花板

| 问题类型 | 后处理能否解决 | 证据 |
|---------|------------|-----|
| 高频 jitter / joint jump | ✅ **已解决** | qc_jitter 99.5%, qc_jump 100% |
| jitter_pos 数值 | ✅ **大幅改善** | 1504 → 603 (-60%) |
| Foot sliding | ❌ **不解决** | 88.4% → 89.3%, delta=0.9 pp |
| Phantom motion (平滑但大幅度错误) | ❌ **原理上不解决** | qc_neck 恒=98.1%，case 00165 仍有问题 |
| 整体 QC pass 到 90%+ | ❌ **剩余 gap 在结构错误** | 74.9% 是后处理天花板 |

---

## 4. 为什么 Post-processing 有天花板

Post-processing 基于**时序平滑**假设：相邻帧的 rotation/position 应该连续变化。所以：

- **能滤掉**: 单帧 spike、高频抖动、不连续 jump
- **滤不掉**: 平滑的错误（rotation 慢慢从正确轨迹偏离到错误姿态），或者"结构性错误"（如 phantom head rotation，曲线本身是光滑的只是位置错了）

E9 `case 00165` 的 neck 问题是后者 — savgol 能让 rotation 曲线更光滑，但不能让它回到"LQ 本来的 neck 姿态"附近。

同理 foot_sliding: 错误的是 pelvis translation 与脚 position 的相对关系，局部平滑不能修复。

---

## 5. 结论与建议

### 短期（可立即落地）
- **当前 best setting**: `m2m_strict_bsmooth_combo`，QC pass 74.9%
- 相比 raw output 已经是巨大进步（+14 pp, jitter -60%）
- **继续做 post-processing tuning 的 ROI 已经很低** — 高频问题基本被拍平了

### 中期（需要训练侧干预）
为突破 75% 天花板，必须解决：
1. **Phantom motion**（根因：mask 分布 OOD）
   - 方案见 `docs/temp/e9_mask_distribution_training_plan_20260423.md`
2. **Foot sliding**（根因：contact 约束缺失）
   - 需要在 loss 或 post-processing 层加 foot contact detection + retarget

### 决策建议
- **Accept 当前 post-processing + 保留 combo 设置**
- **Phase 1 (M8 mock_adaptive finetune) 仍值得做** — 主攻 phantom motion 这个后处理天花板外的问题
- 若 Phase 1 后 QC pass 突破 80%，训练方案成立；若无效，重新评估 pipeline 的 SDEdit 机制

---

## 6. 数据出处

查询：
```sql
SELECT m.name, a.metric_name, a.mean
FROM eval_runs r
JOIN models m ON r.model_id=m.id
JOIN agg_metrics a ON a.eval_run_id=r.id
WHERE r.task_id='E9' AND a.metric_name IN (
  'qc_pass', 'qc_jitter', 'qc_joint_jump', 'qc_foot_sliding',
  'qc_neck', 'qc_candy_wrapper', 'qc_rotation_velocity',
  'jitter_pos', 'jitter_135', 'foot_skating_ratio', 'foot_avg_skate'
);
```
DB: `motion_annot_web/eval_dashboard/eval_dashboard.db`
Runs: 2766 (strict), 2767 (smooth1), 2836 (accelK3), 2838 (savgol5), 2857 (stablemotion), 2872 (combo), 2873 (bsmooth)
