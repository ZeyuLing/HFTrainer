# MoMask 评测可信度分析与重启方案 (2026-05-08)

## TL;DR
1. **当前所有 MoMask 数字均不可信**——native 评测 FID 2.67 vs paper 0.045（差 60×）；MotionStreamer evaluator FID 539 vs paper 12.232（差 40×）；MotionCLIP evaluator 同样无法复现。
2. **根因不是单一 bug，而是多重不兼容**：测试集重建误差 + 263→272 表征不可逆 + IK 约定差异 + 半角 yaw bug（已修一处）。
3. **唯一干净路径**：(a) 下载 HumanML3D 官方 263-dim 数据（不要再从 humanml3d_272 反推）；(b) 用 `joints2smpl` 把 263 输出投到 SMPL 参数；(c) 走 `representation_272` 前向得到忠实的 272；(d) 同一份 SMPL params 也喂 MotionCLIP，三方对比。
4. **更稳的 alternative**：在 `humanml3d_272` 上从头 retrain MoMask（MotionStreamer 论文就是这么做的）。约 1-3 天 A100。

---

## 1. 证据链：为什么现有数字不可信

### 1.1 Native MoMask evaluator：FID 2.67 vs paper 0.045

```
                | FID    | R@1   | R@2   | R@3   | MM-D  | Div
MoMask paper GT | 0.002  | 0.511 | 0.703 | 0.797 | 2.974 | 9.503
MoMask paper    | 0.045  | 0.521 | 0.713 | 0.807 | 2.958 | 9.620
我的 native GT  | 0.000  | 0.432 | 0.617 | 0.731 | 3.546 | 8.227
我的 native pred| 2.672  | 0.443 | 0.638 | 0.739 | 3.319 | 9.976
```

GT-only 行已经显著偏离 paper：R@1 0.432 vs 0.511，MM-D 3.546 vs 2.974。说明**测试集本身就不对**。

**根因**：我用 `tools/build_h3d263_test_from_h3d272.py` 从 MotionStreamer 的 humanml3d_272（30 fps）反推 263：
- 30 fps → 20 fps 线性插值（位置层面，引入低通滤波）
- `process_file` 里调用 `uniform_skeleton(positions, tgt_offsets)`：把每段 motion 的骨长重缩放到 000021 的骨长（**改变身体几何**）
- 把首帧朝向归 Z+、首帧 root xz 归 0、地面对齐

整个 pipeline 跟 HumanML3D 官方发布的 `new_joint_vecs/*.npy` 不是同一份数据。MoMask 是在官方 263 上训练的，用我重建的版本评测自然偏。

### 1.2 MotionStreamer evaluator：FID 539 vs paper 12.232

`tools/diag_h3d263_to_272_roundtrip.py` 的逐通道对比（已修 yaw 半角后）：

```
block               | rms     | orig std | 相对误差
root_xz_vel  [0:2]  | 0.022   | 0.013    | 170%
heading_d_6d [2:8]  | 0.025   | 0.471    | 5%
joints_pos   [8:74] | 0.20    | 0.485    | 41%
joints_vel  [74:140]| 0.013   | 0.008    | 160%
joints_rot [140:272]| 0.68    | 0.50     | 136%  ← 比信号本身还大
```

`joints_rot` 块**误差比信号还大**。深挖后这是**根本不可调和的表征不兼容**：

- MoMask 263 的 `data[..., 67:193]`（126 维）：HumanML3D `Skeleton.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)` 提取的 **parent-relative 局部旋转**，6D 是 cont6d **列主序** `[col0, col1]`
- MotionStreamer 272 的 `[140:272]`：从 SMPL axis-angle `pose_body[:, :66]` → quaternion → matrix → 6D **行主序** `[row0, row1]`
- 即使把列/行主序换对，**两种 IK 提取的局部旋转本身就不同**：
  - HumanML3D 的 IK 对 root 用 `face_joint_indx=[2,1,17,16]` 计算 forward 方向（注意 `smooth_forward=True` 还会跨帧平滑），把 pitch/roll 全部塞进 spine 局部旋转
  - SMPL axis-angle 是直接的 SMPL forward kinematics 参数，root 包含完整 yaw+pitch+roll
  - 同一个 motion 在两种约定下的 21 个 non-root 局部旋转也**不相等**

⇒ 把 MoMask 263 的 cont6d 当作 SMPL 局部旋转直接塞进 272 的 `[140:272]`，**永远不可能**重现 paper 的 FID。

### 1.3 MotionCLIP evaluator 之前 R@1 0.156

类似根因：MotionCLIP 在我们自己的 135-dim SMPL-22 表征上训练。把 MoMask 263 输出（HumanML3D-IK 风格）转成 SMPL 表征本身就是有损过程，且我们当时的转换管道也存在类似问题。

---

## 2. 已修复 / 已验证的 bug

| Bug | 位置 | 状态 |
|-----|------|------|
| Yaw 半角约定（`recover_root_rot_pos` 用 `cumsum(rot_vel)` 表示 `R_y(2α)` 而非 `R_y(α)`，我之前直接当 α 用） | `tools/convert_momask263_to_h3d272.py:decode_263_to_pose` | 已修 (commit pending) |
| `joints_rot` 块仍然 rms 0.68 | 同上 | **不可修**，根因是 IK 约定差异 |
| `joints_pos` 块 rms 0.20 | 同上 | 部分可修：root pitch/roll 在 263 里物理丢失，需要从 joint 位置 IK 反推 SMPL params 才能恢复 |

`tools/diag_converter_only.py` 里把上述 ID-pipeline 整个 bypass 掉做的纯 converter 自洽测试也显示 `joints_pos` rms 0.97（合成的 root_yaw 约定和 263 实际约定刚好相反，是诊断脚本本身的 bug，不影响主结论）。

---

## 3. 可执行的修复方案

### 方案 A: 用 HumanML3D **官方** 263-dim 数据 + joints2smpl 投影 (推荐)

1. **下载官方数据**：从 HumanML3D 仓库 (`https://github.com/EricGuo5513/HumanML3D`) clone 后按 README 跑 prepare 脚本，得到原始 `new_joint_vecs/`、`Mean.npy`、`Std.npy`、`test.txt`、`texts/*.txt`。约 30 GB+。
2. **MoMask native eval（重新版）**：直接在官方 263 数据上跑 `gen_t2m.py` 推理 + 官方 `eval_t2m_trans_res.py` 评测。这样 GT-only 应当复现 paper 的 R@1≈0.511。
3. **joints2smpl 投影**：clone `https://github.com/wangsen1312/joints2smpl`（VPoser 优化版）。每条 263 motion → recover_from_ric → (T, 22, 3) joints → 优化求 SMPL `(β, θ_axis_angle, trans)`。
4. **SMPL → 272**：`smpl_85_face_z_transform.py` + `representation_272.py` 走 MotionStreamer 的官方前向。这样 `[140:272]` 的 IK 约定彻底对齐。
5. **评测三方**：(a) MotionStreamer evaluator on 272；(b) MotionCLIP evaluator on SMPL-22；(c) MoMask native on 263。同一份 motion 三个数字都给出来，互相校验。

代价：joints2smpl 优化每条 motion 2-5 分钟，~4000 条测试样本 → A100 单卡 8-15 小时。

### 方案 B: 在 `humanml3d_272` 上从头 retrain MoMask

这正是 MotionStreamer 论文 Table 1 的做法（"All baselines are trained from scratch following their original implementations"）。

1. 把 MoMask 的 RVQ + MaskTransformer + ResidualTransformer 改成接 272-dim 输入
2. 用 `humanml3d_272` 训练
3. 用 MotionStreamer evaluator 评测

代价：~1-3 天 A100。能精确复现 paper FID 12.232 / R@1 0.621。

### 方案 C: 接受现状，更新 paper 里的表述（**不推荐**，但是最快）

把 Tab. 1 的 MoMask 行直接置空（已经做了），在 sec:t2m 写：
> Cross-evaluator comparison with MoMask is omitted because MoMask operates in a 263-dim representation that is fundamentally incompatible with both our and MotionStreamer's evaluators. Faithful comparison requires retraining MoMask on a SMPL-compatible representation, which is beyond the scope of this work.

代价：0。但这等于承认我们没在公平 baseline 下展示自己的方法。

---

## 4. 推荐选哪条

| | 方案 A (joints2smpl) | 方案 B (retrain) | 方案 C (放弃) |
|---|---|---|---|
| 工作量 | 中（10-20 工时 + 8-15h GPU） | 大（1-3 天 GPU + 调参） | 小 |
| 复现 paper FID 12.232 | 接近，可能 ±20% | 精确 | N/A |
| 给 Blender 用 | ✅ 直接出 SMPL params | ❌ 仍然只有 272，需要再投 SMPL | ❌ |
| 写 paper 时心安理得 | ✅ | ✅ | ❌ |
| 可推广到其他 263 baseline (T2M-GPT, AttT2M 等) | ✅ 同一管道复用 | ❌ 每个 baseline 都要 retrain | N/A |

**强烈推荐方案 A**。它一次解决 user 提出的两个问题：(1) Blender 可用的 SMPL 输出；(2) 跨 evaluator 对齐。

---

## 5. 下一步具体 action items

如果选方案 A：
1. [ ] 在 lzy_debug_machine_1/2 安装 joints2smpl 依赖（VPoser、SMPL/SMPL-X 模型）
2. [ ] 下载 HumanML3D 官方数据 (~30 GB) 到 `data/HumanML3D/`
3. [ ] 跑一次 MoMask native eval（官方测试集）→ 验证 GT R@1 ≈ 0.511
4. [ ] 写 `tools/joints2smpl_pipeline.py`，对 MoMask 的 263 prediction 做 joints → SMPL → 272
5. [ ] 跑三方 evaluator，更新 Tab. 1 + plan.md
6. [ ] 同样的 pipeline 跑 MotionCLIP，三方数据出齐

如果选方案 B：
1. [ ] 把 MoMask 改成接 272 输入，搬到 hf_trainer 框架
2. [ ] 在 `humanml3d_272` 训练 RVQ → MaskTransformer → ResidualTransformer
3. [ ] 直接走 MotionStreamer evaluator
4. [ ] paper Tab. 1 写"MoMask\* (retrained on 272)"

如果选方案 C：
1. [ ] 写好 caption + 章节解释
2. [ ] 同时做 paper 内的策略调整（focus 我们和 MotionStreamer / VerMo / 其他 272-native baseline 的对比）
