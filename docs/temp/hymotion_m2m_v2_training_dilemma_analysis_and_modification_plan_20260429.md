# HYMotion M2M v2 训练困境系统分析与修改计划

日期: 2026-04-29

本文基于当前 `hf_trainer` 工作树中的 HYMotion M2M v2 config、trainer、loss、mask sampler、pipeline、评测脚本和本地 `work_dirs` 训练日志。当前 4 条主线训练仍在继续，且已有较晚 checkpoint：

- `caption_local_phase2`: 本地可见至 `checkpoint-epoch_1810`
- `caption_global_phase2`: 本地可见至 `checkpoint-epoch_1420`
- `uncond_local_046b`: 本地可见至 `checkpoint-epoch_1670`
- `uncond_global_046b`: 本地可见至 `checkpoint-epoch_1540`

建议默认不打断现有训练，把下面修改做成下一批 ablation / finetune。

## 1. 当前困境的结构性判断

### 1.1 网络输入不是“几乎全是 condition”，但 condition anchor 过强

当前 v2 采用 198-dim 表示：

- `[0:3]`: translation
- `[3:135]`: 22 joints rot6d
- `[135:198]`: 21 joints position channel, XZ relative to pelvis, Y absolute

模型输入为 `[x_t, reactive, src_mask]`，即 594 维。`src_mask=1` 表示生成区，`src_mask=0` 表示已知 condition。

`mask_aware_noise=True` 时，trainer 会做：

```python
x_t = x_t * src_mask + x1 * (1 - src_mask)
```

所以已知区在每个 timestep 都是 clean GT。VACE 又输入 `reactive=src_motion*src_mask` 和 mask 本身。抽样当前 mask 分布：

| 配置 | pure-gen | mask mean | any-known frame | full-known frame | full-gen frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| caption phase2 v3 K0=0.16 | 17.4% | 88.3% | 36.0% | 8.8% | 64.0% |
| v3 default K0=0.10 | 11.0% | 87.8% | 39.0% | 9.1% | 61.0% |
| caption v2 tier2 pure=0.40 | 13.7% | 95.7% | 42.2% | 1.1% | 57.8% |

结论：cell-level 大部分仍是 generate，不是 condition 过密；真正问题是 sample-level 经常出现少量但非常可靠的 clean anchor，例如 prefix、keyframe、trajectory、end-effector、foot-ground。模型可以通过 anchor 推断 motion，caption 的训练责任被稀释。

### 1.2 loss 仍然被单一 velocity 标量主导

当前训练日志显示 loss 主要由 `loss_velocity` 主导，KIMODO aux 和 smoothness 通常只有几个千分点：

- `loss_velocity`: 常见 `0.02-0.06`
- `loss_smoothness`: 常见 `0.0001-0.001`
- `loss_aux_joint_pos`: 常见 `0.001-0.006`
- `loss_aux_joint_vel`: 常见 `0.0002-0.001`
- `loss_aux_fk_consistency`: 常见 `0.0002-0.001`

`M2MLoss` 里 velocity 是一个全维度标量，只对 translation 前 3 维乘 `trans_dim_weight=5.0`。这会把 translation、root rotation、body rotation、position channel、hand/foot 误差混在一起。脚滑、手部接触、自交互都可能被平均掉。

### 1.3 smoothness 不是 foot-skating loss

当前 `loss_smoothness` 是 denoised `x1` 的一阶时间差匹配：

```python
pred_motion_vel = pred_x1[:, 1:] - pred_x1[:, :-1]
gt_motion_vel = gt_x1[:, 1:] - gt_x1[:, :-1]
```

它能减少 jitter、稳定局部速度，但不理解脚是否接地，也不惩罚“脚平滑地滑”。因此继续单纯提高 `motion_smoothness_weight` 不会根治滑步。

### 1.4 KIMODO aux 是正确方向，但缺显式 contact

当前新增的 `KimodoStyleAuxLoss` 已经提供：

- `aux_joint_pos`: FK global joint position vs GT
- `aux_joint_vel`: FK global joint velocity vs GT
- `aux_fk_consistency`: predicted pos channel vs FK-derived pos channel

这比旧的 root-relative FK consistency 更接近物理问题。但它仍然没有显式 contact label，不会直接表达“接触期间脚的 XZ velocity 应为 0、Y 应贴地、不能穿地”。

### 1.5 mask v3 明显优于 v2，但还不能覆盖任意 condition

v3 Rank-K sampler 已经把 eval mask coverage 从 v2 的 13/32 提升到 23/32 effective settings。但 E4 sparse end-effector、E14 transition、E2 mid60 等仍是低概率区域。E9 repair 的 strict/adaptive mask 还会产生“单关节长连续生成 + 其他关节密集 anchor”的 OOD pattern。

所以 mask 需要从“通用随机 prior”升级为：

```text
universal random + eval hard cases + failure replay + detector/adaptive masks
```

### 1.6 精确自交互不是当前 position mask 能解决的问题

当前 position constraint 更适合“某个 joint 到某个世界点”。但“手放在腿上”“双手合拢”是 relational contact：

- hand-hand: 两只手之间的距离、相对速度、手掌朝向
- hand-leg: 手到 thigh/shin surface/capsule 的距离、法向、非穿透

这需要 pairwise contact 表示、训练数据挖掘、pairwise loss，以及 inference-time relational projection。

## 2. 修改目标与验收指标

目标不是一次性重做架构，而是用最小侵入改动解决三个最影响体验的问题：

1. **滑步下降**：E6 / locomotion / T2M 中 `foot_skating_ratio`、`foot_avg_skate` 明显下降。
2. **caption 不被 condition 淹没**：caption-only 和 weak-condition 场景的 text-motion alignment 提升，不因 mixed training 退化。
3. **mask 泛化增强**：E4/E9/E14 等 hard mask 的 MPJPE、jitter、phantom motion 下降。

建议验收线：

- foot metrics: `foot_skating_ratio` 相对当前 best checkpoint 下降 20% 以上，且 `foot_penetration` 不上升。
- E9 repair: strict/adaptive mask 下 phantom rotation/jitter failure case 数下降 30% 以上。
- E4/E6: condition hit rate 或 end-effector error 不退化，`ee_hit_rate_5cm` 不下降超过 2 个百分点。
- caption: pure T2M 文本语义人工抽检和自动指标不退化；优先看 caption local/global phase2 的 T2M subset。

## 3. 修改计划

### Phase 0: 保留现有训练，先补可观测性

不改变主 loss 数值，先把问题拆出来看清楚。

#### 3.1 拆分 velocity loss 日志

在 `M2MLoss` 中增加分组统计，但默认不改变总 loss：

- `velocity_trans`: `[0:3]`
- `velocity_root_rot`: `[3:9]`
- `velocity_body_rot`: `[9:135]`
- `velocity_pos`: `[135:198]`
- `velocity_pos_foot`: foot/ankle 对应 position channel
- `velocity_pos_hand`: wrist/hand 对应 position channel

默认行为：

- `loss_velocity` 的数值保持与当前一致。
- 新增项只用于 logging，不参与总 loss，命名为 `stat_velocity_*` 或以 `loss_` 前缀但权重为 0 时不加入总和。

注意：当前 trainer 会 `sum(losses.values())`，所以如果返回日志项必须确保不会被计入总 loss。推荐返回两个 dict：

- `losses`: 参与反传
- `metrics`: 仅日志

或者在 trainer 里识别 `stat_` 前缀，不加入总 loss。

#### 3.2 增加 mask 分布日志

每 N step 记录 batch-level mask stats：

- `mask_mean`
- `pure_gen_ratio`
- `any_known_frame_ratio`
- `full_known_frame_ratio`
- `trans_mask_mean`
- `rot_mask_mean`
- `pos_mask_mean`
- `edit_mode_ratio`

用途：确认不同 config / phase / sampler 的真实采样分布，避免只看配置推断。

### Phase 1: 直接解决滑步

#### 3.3 新增 contact-aware foot loss

在 KIMODO aux loss 后追加显式 foot contact loss，独立成 `FootContactAuxLoss` 或并入 `KimodoStyleAuxLoss`。

输入：

- `pred_x1_norm`
- `gt_x1_norm`
- `mean/std`
- `bone_offsets`
- `rotation_space`
- `data_mask_temporal`
- `timesteps`

从 FK 得到 pred/GT world positions。使用脚相关 joint：

- ankles: `7, 8`
- feet/toes: `10, 11`

GT contact 检测：

```python
gt_foot_y = gt_world[..., foot_ids, 1]
gt_foot_xz_vel = gt_world[:, 1:, foot_ids, [0, 2]] - gt_world[:, :-1, foot_ids, [0, 2]]
contact = (gt_foot_y[:-1] < ground_y + 0.05) & (||gt_foot_xz_vel|| < 0.01)
```

Loss terms：

- `aux_foot_sticky`: contact frame 上 `||pred_foot_xz[t+1] - pred_foot_xz[t]||`
- `aux_foot_height`: contact frame 上 `|pred_foot_y - ground_y|`
- `aux_foot_penetration`: `relu(ground_y - pred_foot_y)`
- optional `aux_foot_contact_vel_match`: contact frame 上 pred/GT foot velocity SmoothL1

默认权重建议：

- `foot_sticky_weight=20.0`
- `foot_height_weight=5.0`
- `foot_penetration_weight=10.0`
- `foot_vel_match_weight=5.0`
- `warmup_steps=2000`
- `timestep_squared_weighting=True`

权重理由：当前 aux weighted loss 常在 `1e-3` 量级，foot loss 应该先做到 `loss_velocity` 的 5%-15%，不要一开始超过主 loss。

#### 3.4 不再把 smoothness 当主修复手段

保持 `motion_smoothness_weight=0.5`，不要先提高到 1.0。若 foot loss 后 jitter 上升，再单独 ablate `smoothness=0.75/1.0`。

### Phase 2: 平衡 caption 与 condition frame

#### 3.5 增加 VACE condition dropout / scale jitter

目标：让模型不能总是依赖 clean condition anchor。

在 trainer 构建 `vace_context` 前或 `prepare_vace_input` 内增加训练时随机扰动：

- `vace_drop_prob=0.10`: 整个样本丢掉 condition，只保留 caption 和 mask。
- `known_cell_drop_prob=0.05`: 对 known cells 随机改成 generate，使 condition 更稀疏。
- `condition_value_noise_std=0.02`: 对 known clean values 加小噪声，只在训练时。
- `vace_scale_range=(0.5, 1.0)`: reactive/value 分支随机缩放，mask 分支保持二值。

默认只对 caption configs 启用；uncond configs 可只启用 `known_cell_drop_prob`，不启用 `vace_drop_prob`。

#### 3.6 提高 caption phase2 的 pure generation 比例

当前 caption phase2 v3 使用：

```python
k_weights=(0.16, 0.513, 0.233, 0.065, 0.029)
```

下一批 caption ablation 建议：

```python
k_weights=(0.25, 0.458, 0.208, 0.058, 0.026)
```

即 pure-gen 从约 16% 提到 25%，其余 K=1..4 按比例缩放。若 T2M 仍弱，再试 30%：

```python
k_weights=(0.30, 0.428, 0.194, 0.054, 0.023)
```

`cond_mask_prob` 先保持 0.1，不优先加到 0.25。原因：问题核心是 VACE anchor 过可靠，不是 text dropout 太低；盲目提高 text dropout 会减少有文本监督的样本。

### Phase 3: mask 泛化和 failure replay

#### 3.7 增加 hard-case mask mixture

在 v3 sampler 外层增加一个 `hard_case_prob`，默认 0.20，只在下一批 finetune config 开启。hard cases 包含：

- E4 sparse end-effector: wrist/ankle/foot sparse periodic position-only masks
- E6 foot contact: foot/ankle position-only contact frames
- E9 adaptive mock: 单关节或 kinetic chain 长连续 generate，其他区域 dense known
- E14 transition: left/right segment transition windows
- E2 mid60: 中段 completion

推荐实现方式：

```text
PrepareM2Mv2Condition
  if rng.random() < hard_case_prob:
      mask = sample_hard_case_mask(...)
  else:
      mask = sample_condition_v3(...)
```

这比继续调 v3 global weights 更直接，因为这些 failure 是尾部 pattern，不适合用单一 prior 兼顾。

#### 3.8 加 failure replay mask 数据

从评测脚本或 dashboard 输出中保存真实失败 mask：

- E9 strict/adaptive repair masks
- E4 failed sparse EE masks
- E14 boundary masks

新建轻量 replay 文件格式：

```json
{
  "task": "E9",
  "setting": "strict_bsmooth_combo",
  "mask_path": "...npz",
  "motion_length": 180,
  "motion_dim": 198
}
```

训练时以 `replay_mask_prob=0.05-0.10` 抽样 replay masks，并按当前 clip length resize/crop。初期只用于 finetune，不进主训练默认。

### Phase 4: 精确自交互

#### 3.9 先做评测与数据挖掘，再做训练接口

短期不要直接改网络结构。先离线挖 GT 中的 relational contact：

- hand-hand contact: left/right wrist or palm 距离小于 5cm，持续超过 10 frames，相对速度低。
- hand-leg contact: wrist 到 thigh/shin segment/capsule 距离小于 5cm，持续超过 10 frames，相对速度低。

输出 contact annotations：

```json
{
  "frame_start": 40,
  "frame_end": 95,
  "type": "hand_leg",
  "source_joint": "right_wrist",
  "target_segment": ["left_hip", "left_knee"],
  "offset": [dx, dy, dz],
  "normal": [nx, ny, nz]
}
```

#### 3.10 新增 relational contact loss

训练 loss：

- `contact_distance`: source joint 到 target joint/segment/capsule 的距离
- `contact_relative_velocity`: contact 期间 source 与 target 的相对速度
- `contact_non_penetration`: hand 不进入 leg capsule 内部过深

inference：

- 扩展 `PositionConstraint` 为 `RelativePositionConstraint` 和 `PairDistanceConstraint`。
- 先支持 hand-hand，再支持 hand-leg capsule。

这部分作为第二阶段长期能力，不阻塞 foot/mask/caption 修复。

## 4. 实验矩阵

### 4.1 先跑最小 ablation

以当前 latest checkpoint 为 base，优先 local/global 各选一个代表：

| 实验 | Base | 改动 | 目的 |
| --- | --- | --- | --- |
| A0 | 当前 latest | 只加 logging | 获得 velocity/mask 分组统计 |
| A1 | caption local phase2 | foot contact loss | 看 T2M + E6/locomotion 滑步 |
| A2 | uncond local | foot contact loss | 排除 text 因素，看纯 M2M 修复能力 |
| B1 | caption local phase2 | pure-gen 25% + VACE dropout | 看 caption balance |
| C1 | uncond local | hard-case masks 20% | 看 E9/E4/E14 mask 泛化 |
| D1 | caption local phase2 | A1+B1+C1 | 综合候选 |

每个 finetune 先跑 5k-10k steps 或等价短 epoch，不直接跑 80 天完整长训。

### 4.2 成功后再扩展

若 local 有效：

- 复制到 global rotation。
- 对 caption global phase2 做相同综合配置。
- 对 uncond global 做 foot + hard-case mask。

若 local 无效：

- 先检查 split velocity / foot loss 量级是否太小。
- 若 foot loss 有效但 caption 退化，再降低 VACE dropout 或 pure-gen 回到 20%。
- 若 mask hard-case 有效但常规 E1/E2 退化，降低 `hard_case_prob` 到 0.10。

## 5. 推荐代码落点

### 5.1 loss 与日志

- `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py`
  - 增加 velocity split metrics。
  - 不改变默认 `loss_velocity` 数值。

- `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py`
  - 增加 foot contact loss，或新建 `foot_contact_aux_loss.py` 并由 trainer 调用。

- `hftrainer/trainers/motion/hymotion_m2m_trainer.py`
  - 支持 non-loss metrics logging，不把 `stat_*` 加入总 loss。
  - 增加 mask stats logging。
  - 调用 foot contact aux。

### 5.2 condition balance

- `hftrainer/models/motion/hymotion_m2m/bundle.py`
  - 增加训练时 VACE dropout/scale/noise 配置，或者只提供 helper，由 trainer 控制。

- `configs/hymotion_m2m_v2/*phase2*.py`
  - 新建 ablation config，不直接覆盖当前在跑 config。
  - caption phase2: `k_weights` 提到 pure-gen 25%。
  - 增加 `vace_drop_prob / known_cell_drop_prob / condition_value_noise_std / vace_scale_range`。

### 5.3 mask hard cases

- `hftrainer/datasets/motion/motionhub/transforms/condition_sampler_v3.py`
  - 增加 hard-case sampler 函数，或新建 `condition_sampler_hard_cases.py`。

- `hftrainer/datasets/motion/motionhub/transforms/prepare_m2m_v2.py`
  - 增加 `hard_case_prob` 和 `replay_mask_prob`。

### 5.4 self-interaction

- 新建离线挖掘脚本：`tools/mine_self_contact_annotations.py`
- 新建 pair constraint / loss 后再接入 trainer 和 pipeline。

## 6. 风险控制

1. **不要改正在跑的 config**：新建 ablation config，避免污染已有 work_dir。
2. **loss 权重从小开始**：foot loss 先控制在总 loss 的 5%-15%。
3. **metrics 先行**：没有 split velocity / mask stats 前，不再盲调大权重。
4. **caption dropout 不先提高**：先削弱 VACE anchor，再观察 text alignment。
5. **hard masks 不进默认主线**：先 finetune 验证，不直接替换 v3。

## 7. 立即执行 checklist

- [ ] 新增 velocity split metrics，不改变训练 loss。
- [ ] 新增 batch mask stats logging。
- [ ] 实现 foot contact aux loss。
- [ ] 新建 `caption_local_phase2_foot_contact_ablation.py`。
- [ ] 新建 `uncond_local_foot_contact_ablation.py`。
- [ ] 实现 VACE condition dropout/scale jitter。
- [ ] 新建 `caption_local_phase2_caption_balance_ablation.py`，pure-gen 25%。
- [ ] 实现 hard-case mask sampler。
- [ ] 新建 `uncond_local_hard_mask_ablation.py`。
- [ ] 跑 E6/E9/E4/T2M subset 评测，形成对比表。

## 8. 默认决策

若无人再指定偏好，下一步建议按以下顺序实施：

1. **先做 Phase 0 logging**：代价最低，立刻提升诊断能力。
2. **再做 Phase 1 foot contact loss**：直接对应当前最严重的滑步。
3. **并行做 Phase 2 caption balance ablation**：不和 foot loss 强耦合。
4. **随后做 Phase 3 hard-case masks**：解决 E4/E9/E14 泛化。
5. **最后做 Phase 4 self-interaction**：需要新评测和新约束表示，工程量最大。

这条路线不会浪费当前 4 个长训任务；它们继续作为 baseline / teacher / checkpoint source，新实验只在其最新 checkpoint 上短程 finetune 验证。
