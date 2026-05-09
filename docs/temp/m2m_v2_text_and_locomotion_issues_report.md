# HyMotion M2M v2 训练问题技术方案报告

**报告日期**：2026-04-21
**问题对象**：HyMotion M2M v2 (0.46B, 198-dim)
**数据来源**：`motion_annot_web/eval_dashboard/` 评估可视化
**作者**：投针于 M2M v2 训练审计与消融实验的一次系统梳理

---

## 0. TL;DR（结论先行）

经过对 `configs/hymotion_m2m_v2/`、`hftrainer/trainers/motion/hymotion_m2m_trainer.py`、`hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py`、`hftrainer/datasets/motion/motionhub/transforms/condition_sampler_v2.py` 与 `prepare_m2m_v2.py` 的源码审计，本报告识别出两类问题的根因，并给出可执行的解决方案。

**Problem 1 — 文本语义退化**：主因是 VACE 的强已知条件（`reactive`+`mask`）+ `cond_mask_prob=0.1` 偏低 + pure-generation 样本仅占 16% + generation_mask 在 velocity/x1 loss 里把梯度集中在生成区（而生成区天然有相邻 known 参考），使得模型获得"读 VACE 即可"的捷径，text condition 在训练中不必被使用。

**Problem 2 — 动作/locomotion 质量**：主因是全链路缺 3D 空间监督（`keypoints3d_weight=0.0`、`translation_weight=0.0`），FK 一致性 loss 是 root-relative（对全局脚位置盲视）、权重 0.1 且 warmup 2000 步过弱，没有 foot contact / foot skating loss，Tier 1 `trans_keep=0.2` + Tier 2 `trajectory` 仅 mask XZ 造成平移条件过弱，198-dim 的 `XZ_rel + Y_abs` 混合坐标使 rotation↔translation 不自洽，整体缺少 root trajectory consistency 和 limb-length 保持。

**解决方案**分三个时间尺度：
- **短期（config 层可立即验证，无需改代码）**：`cond_mask_prob=0.1→0.25`、`pure_gen=16%→30%`、`trans_keep=0.2→0.5`、打开 `keypoints3d_weight=1.0` 与 `translation_weight=1.0`、`fk_consistency_weight=0.1→0.5`、`fk_consistency_warmup_steps=2000→500`。
- **中期（~1 周内可落地）**：新增 foot-contact/foot-skating loss、global-FK（含 pelvis）loss、limb-length preservation loss、root trajectory smoothness loss；推理 CFG 标定 + 推理时 VACE strength annealing。
- **长期（表示与训练策略重构）**：逻辑分解两分支（T2M head + M2M head 共享 backbone）、VACE noise-stage gating（早期 timestep 降低 VACE 信号）、采用 logit-normal timestep 采样、SOAR 后训（对应已有方案）、rotation-space 默认切 `global`、考虑混入 PhysHOI/UnderPressure 式的物理先验。

**报告存储路径**：`docs/temp/m2m_v2_text_and_locomotion_issues_report.md`。

---

## 1. 问题陈述与复现路径

### 1.1 问题 1：文本语义理解退化

**现象**：
- HyMotion M2M v2 在 T2M（即 pure generation）模式下的语义表现明显弱于 HyMotion T2M 1.0。
- 尤其当条件较多（inbetween / prefix / keyframes / end_effector）时，text 几乎"失效"——模型行为由 known 区 + 空间 cue 决定，忽略文本。

**Dashboard 佐证**：
- `eval_dashboard` 对 E1（T2M 纯文生）等任务的主指标（R-precision/MMDist/CLIP-S/FID）明显劣化。
- 在 E3/E4/E5 等带 spatial cue 的任务中，替换 caption 结果变化甚小——证明文本梯度未有效传到运动输出。

### 1.2 问题 2：动作生成质量（locomotion）

**现象**：
- 短距 locomotion "滑过去"（根节点位移发生但下肢步态循环缺失、脚与地面无接触节奏）。
- 弯腰 / 趴下类动作违反物理执行顺序（缺少屈膝、前倾的因果链）。
- 常见 foot skating、float、pelvis drift。

**Dashboard 佐证**：
- `foot_skate_velocity`、`jitter_pos`、`mpjpe_masked` 在各 locomotion / bend / prone 任务上偏高。
- `loop_position_error`（E10 cyclic）偏高说明根轨迹一致性差。

---

## 2. 当前训练方式审计（Where-the-bug-is）

本节将把"代码事实"与"它导致的问题"逐项对应。所有引用均来自仓库内当前版本的 config/trainer/loss/transform。

### 2.1 VACE 条件结构：已知条件过强

**代码事实**：
- `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`：
  ```python
  input_dim=_motion_dim * 3,   # 594
  vace_condition_mode='no_inactive',
  ```
  模型输入 = `[x_t(198), reactive(198), mask(198)]`。
- `reactive` 通道在 known 区（`mask==1`）里是 clean 的、**已经被归一化后的真值 partial observation**，`mask` 通道也完整标出 known 位置；两者同时让"生成区需要什么"的几何约束信息被直接显式喂进 transformer 输入层。

**问题映射**：
- 这是"text 被忽略"的**首要 shortcut**。生成区的值只要和相邻 known 帧**几何上衔接**，loss（velocity）就会迅速下降；text 给出的全局语义对降 loss 的增量贡献相对很小，导致文本梯度主导权极低。
- 一个直观的反事实检查：pure generation 时 `reactive≡0, mask≡0`，此时必须靠 text —— 这恰好是 T2M-1.0 擅长的场景，也是 M2M v2 最差的场景。

### 2.2 CFG dropout `cond_mask_prob=0.1` 过低

**代码事实**：
- 各 caption 配置：`cond_mask_prob=0.1`（`hymotion_m2m_v2_caption_local_046b.py`、`_global_`、`phase1/2`）。
- Trainer 通过 `bundle.mask_text_cond(..., cond_mask_prob=self.bundle.cond_mask_prob)` 在训练 step 中随机丢文本。

**问题映射**：
- 推理走 CFG 时需要 `uncond` 分支有意义。当前训练里 `uncond` 仅占 10%，uncond 分布学得偏弱 → CFG scale 放大后 cond 分支相对"出不来"太多信号。
- 相比之下 HY-Motion 1.0 T2M 的 `cond_mask_prob` 典型值是 0.2–0.3；M2M v2 当前 0.1 + VACE 强条件，相当于给 text 的梯度留了很窄的学习通道。

### 2.3 Pure generation 样本占比偏低

**代码事实**：
- `PrepareM2Mv2Condition(tier2_prob=0.4)` + Tier2 权重 `pure_gen=0.40`。
- 纯 T2M 样本占比 = `0.4 × 0.4 = 16%`。
- Tier1（60%）为"部分已知 + 部分生成"，必含 VACE 强条件。

**问题映射**：
- 模型只在 16% 的 step 里真正必须"靠 text"；剩余 84% 的 step 里 text 的梯度贡献都被 VACE 稀释。
- 对比：HY-Motion 1.0 T2M 是 100% 的 pure generation；M2M v1/v2 在 completion 训练里是混合比。当前 16% 是两者的下限妥协，文本学不透是必然。

### 2.4 generation_mask 仅作用于 velocity/x1 loss 导致"梯度集中到 known-adjacent 区"

**代码事实**（`hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py`）：
- `velocity_loss` 与 `x1_loss` 都乘以 `generation_mask = 1 - src_mask`（只在 mask=0 生成区回传）；
- `keypoints3d`、`translation`、`motion_smoothness`、`fk_consistency` 均不乘 generation_mask。

**问题映射**：
- 在 completion 场景下，"生成区"通常是被 known 帧夹在中间 / 紧邻的少数帧，生成区的局部几何由相邻 known 几乎决定 → velocity loss 下降极快 → text gradient 的 signal-to-noise 比更低。
- 这个设计本身是合理的（只监督需要生成的区），但和"VACE 强条件"组合时会加剧 shortcut。

### 2.5 3D-space 监督整体关闭

**代码事实**：
- `_base_hymotion_m2m_v2_046b.py`：`keypoints3d_weight=0.0`, `translation_weight=0.0`。
- `m2m_loss.py` 中 `keypoints3d_loss` 还是 root-relative：
  ```python
  local_keypoints3d = pred_keypoints3d[:, :, 1:22] - pred_keypoints3d[:, :, 0:1, :]
  ```
  即便打开权重，这份 loss 也看不到全局脚的位置。

**问题映射**：
- 训练完全靠 rot6d + 198-dim 位置通道的"代理几何"监督；没有 world-frame 的 3D 端点（pelvis、脚、手）直接监督。
- 直接导致：`foot skating`、`root drift`、`bend-prone 物理反了`都得不到对应 loss 抗拒。

### 2.6 FK 一致性 loss 过弱、无 pelvis、warmup 过长

**代码事实**：
- `fk_consistency_weight=0.1`，`fk_consistency_warmup_steps=2000`。
- 实现上只算 `rot6d → fk → joint_pos` 与"198-dim 里的 position 通道"是否一致；position 通道本身是 `XZ 相对 pelvis, Y 绝对, pelvis 已去除`。

**问题映射**：
- 仅对 rotation-pos **两个内部分量**做自洽，**不约束 global pelvis + global foot**。这等同"自己和自己不吵架"，但 FK 后的 full-body 在世界系里可以整体漂。
- 权重 0.1 且等 2000 步才启动 → 模型前 2000 步先学会"用位置通道拟合"，此后 FK 才来纠正。可能导致收敛到"position 通道对、但 rot6d 不一致"的解。

### 2.7 平移条件过弱：Tier 1 `trans_keep=0.2` + Tier 2 `trajectory` 只 mask XZ

**代码事实**（`condition_sampler_v2.py`）：
- Tier 1：`sample_translation(trans_keep=0.2)` —— 只有 20% 概率保留完整 3-D 平移作为 known。
- Tier 2 `trajectory`：仅保留 XZ 根轨迹，释放所有 rotation；换言之模型"知道要去哪"但"没被逼着脚要落在何处"。

**问题映射**：
- 短距 locomotion "滑过去"的直接推手：trajectory 约束给了 XZ，但 foot/y 约束没有 → 模型最便宜的解就是"根按轨迹走、腿不合理地跟"。
- 对 bend/prone 类场景，trans 极少被 pinned → 模型把"重心下降"学成"平移往下"，而非"屈膝+前倾"。

### 2.8 没有 foot-contact / foot-skating loss

**代码事实**：`m2m_loss.py` 无 foot-related 监督；`motion_smoothness_weight=0.5` 仅是 rot6d 位置的速度二阶平滑，与地面接触无关。

**问题映射**：directly 对应"locomotion 质量差、脚滑"。

### 2.9 198-dim `XZ_rel + Y_abs` 的坐标混合与 rot6d 不自洽

**代码事实**（`_base_`）：
```
[0:3]      translation (SMPL trans)
[3:135]    22 joints × 6D rot6d (row-major)
[135:198]  21 joints × 3D position (XZ rel pelvis, Y absolute, pelvis excluded)
```

**问题映射**：
- 同一帧的 rot6d（根相对）+ joint position（XZ 相对 pelvis、Y 绝对）+ trans（绝对）三路语义并不一致，加上 rotation_space=local 时 rot6d 的 child joint 依赖父 joint → 在部分 mask 情景下会出现"已知的 rot 对应一个 position 通道给出的目标点"的**几何不可能**组合。
- FK consistency 权重 0.1 无法矫正这个全局漂移。

### 2.10 Text encoder 大多冻结，仅 `text_refiner` 学习

**代码事实**（`_base_`）：
```python
text_encoder=dict(),
text_refiner_cfg=dict(num_layers=2),
```
Qwen3 + CLIP 使用 `LoadPreExtractedTextEmbedding`（离线提取）→ 仅 2 层 `text_refiner` 做"微调 adapter"。

**问题映射**：
- text 能给 MMDiT 的信号路径本身就窄（2 层 refiner），加上 VACE 强条件 shortcut，整条链路中"text → backbone"的有效信号被进一步压制。

### 2.11 Timestep 采样：uniform（velocity 预测）

**代码事实**（`hymotion_m2m_trainer.py` 注解）：velocity pred_type 下使用 `torch.rand()` 均匀采样。

**问题映射**：
- Rectified Flow 在 `t≈0` 和 `t≈1` 区域是"信息最稀薄"和"细节最敏感"的两端；均匀采样会把大量 step 花在中段。
- SD3、Flux 实证 `logit-normal`（偏中段但尾部更重）更优。M2M v2 仍用 uniform 是一个被忽视的调节点。

### 2.12 无 SOAR-style 后训，存在 exposure bias

**代码事实**：目前 caption local/global 主线配置没有 SOAR 后训；仅在 `soar/` 下有 smoke 和 plan。

**问题映射**：
- 推理时的 `x_t` 来自上一步自预测的 `x_{t-Δ}`，与训练时的"真 clean 加噪"分布失配。该失配在 M2M（有 VACE）场景下会和 VACE shortcut 共振，令 recognition、节奏、脚接触都更差。

### 2.13 batch 小 + mixed_precision='no' 导致 throughput 低 → 训练步数不够

**代码事实**：`batch_size=20`（caption），`mixed_precision='no'`。

**问题映射**：文本条件学到"可用"通常需要 ~200–400K steps；当前 throughput 偏低使 caption 分支长期欠训。

---

## 3. 根因总结矩阵

| 编号 | 事实（Where） | 直接后果 | 影响 |
|------|---------------|----------|------|
| F1 | VACE 输入=`[x_t, reactive, mask]` | text shortcut 被 VACE 取代 | **P1 主因** |
| F2 | `cond_mask_prob=0.1` | uncond 分支欠训、CFG 失效 | P1 |
| F3 | `pure_gen=16%` | text-only 训练样本过少 | P1 |
| F4 | `generation_mask` 仅作用于 v/x1 loss | 生成区梯度被 known-adjacent 主导 | P1 次因 |
| F5 | `keypoints3d_weight=0` / `translation_weight=0` | 无 3D 空间监督 | **P2 主因** |
| F6 | FK loss = root-relative + 0.1 + warmup 2000 | 无全局脚/骨盆监督 | **P2 主因** |
| F7 | `trans_keep=0.2` + Tier2 trajectory 只 XZ | 平移条件弱 → locomotion 滑过去 | **P2 主因** |
| F8 | 无 foot contact / skating loss | 直接导致脚滑 | P2 |
| F9 | 198-dim `XZ_rel + Y_abs` 混合 | rot↔pos↔trans 不自洽 | P2 |
| F10 | text encoder 冻结 + 2 层 refiner | text 信号通路窄 | P1 |
| F11 | timestep uniform（velocity） | 训练-推理分布中段偏重 | P1/P2 |
| F12 | 无 SOAR 后训 | exposure bias | P1/P2 |
| F13 | bs=20、no amp | caption 分支欠训 | P1 |

---

## 4. 解决方案

按可落地时间排序。每一条都给出"改哪里、期望效果、对应 F 编号、验证手段"。

### 4.1 短期（config 层面，无需改代码，2~3 天可跑起来）

| 改动 | From | To | 对应 F | 说明 |
|------|------|----|---------|------|
| `cond_mask_prob` | 0.1 | **0.25** | F2 | 给 uncond 分支 25% 训练；CFG 推理可支持 scale≤7.5 |
| `pure_gen`（tier2 weight） | 0.40 (=16%) | **0.60 (=24%)** 或 `tier2_prob=0.5` 再搭 0.6→30% | F3 | 让 T2M exposure 接近 HY-Motion 1.0 的 ~30% |
| `keypoints3d_weight` | 0.0 | **1.0** | F5 | 打开 3D 关节点监督 |
| `translation_weight` | 0.0 | **1.0** | F5 | 打开根平移监督 |
| `fk_consistency_weight` | 0.1 | **0.5** | F6 | 提权，强化 rot↔pos 自洽 |
| `fk_consistency_warmup_steps` | 2000 | **500** | F6 | 让 FK 尽早参与 |
| `trans_keep`（condition_sampler_v2） | 0.2 | **0.5** | F7 | 提升 Tier1 pin 住平移的概率（需改 transform 参数或加到 PrepareM2Mv2Condition） |
| `motion_smoothness_weight` | 0.5 | **1.0** | 辅助 | 更强的速度平滑，配合 keypoints3d 一起稳定轨迹 |
| `trans_dim_weight` | 5.0 | **10.0** | F7 | 显式抬高 trans 在 velocity loss 的分量，克服 trans 在 198-dim 中占比低带来的稀释 |

**产出新 config**（建议命名）：
- `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_v2_fix1.py`
- `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_v2_fix1.py`

**训练与评估**：
```bash
# 2 节点 16 卡 V100
python3 tools/taiji_submit.py m2m_v2_fix1_local \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_v2_fix1.py \
    --host_num 2

# 评估（必须 --save-npz --use-rewritten）
python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local_v2_fix1 \
    --tasks E1 E3 E5 E7 E9 E10 E11 E12 \
    --max-samples 100 \
    --save-npz --use-rewritten \
    --output-dir eval_runs/m2m_v2_fix1
```

预期：
- E1（T2M）MMDist/CLIP-S/R-prec 显著回升（应至少追平 T2M 1.0 的 60–70%）。
- foot_skate_velocity / loop_position_error / mpjpe_masked 同步下降 10–25%。

### 4.2 中期（需新增 1~2 个 loss/训练组件，1 周内可落地）

#### 4.2.1 Foot-contact & Foot-skating loss
```python
# 基于当前帧 FK 得到的 ankle + toe 位置
foot_positions = fk(pred_rot6d)[:, :, [foot_indices]]  # [B, T, 4, 3]
# 接触候选：Y 坐标低于阈值
contact = (foot_positions[..., 1] < ground_thresh).float()
# 连续接触帧的水平速度 → 要求≈0
foot_xz_vel = foot_positions[:, 1:, :, [0, 2]] - foot_positions[:, :-1, :, [0, 2]]
foot_skate_loss = (contact[:, 1:] * foot_xz_vel.norm(dim=-1)).mean()
```
加入 `m2m_loss.py` 后 `foot_skate_weight=1.0`，直接正面击中 locomotion "滑过去"。

#### 4.2.2 Global FK loss（含 pelvis）
把当前 root-relative 的 `keypoints3d_loss` 改为可选 `global` 模式：不减 pelvis、保留所有 22 个 joint。
```python
global_keypoints3d_loss = smooth_l1(pred_fk_global, gt_fk_global).mean()
```
权重 0.5 左右。配合 translation_weight 一起锚定 world-frame 几何。

#### 4.2.3 Limb-length preservation loss
```python
# bone_indices：[parent_idx, child_idx] 列表（20 条）
bone_len_pred = (pred_joint[..., child, :] - pred_joint[..., parent, :]).norm(-1)
bone_len_gt = (gt_joint[..., child, :] - gt_joint[..., parent, :]).norm(-1)
limb_len_loss = (bone_len_pred - bone_len_gt).abs().mean()
```
防止训练中 rot6d 误差堆叠造成末端漂移。

#### 4.2.4 Root trajectory smoothness loss
```python
root_vel = trans[:, 1:] - trans[:, :-1]
root_acc = root_vel[:, 1:] - root_vel[:, :-1]
root_traj_loss = root_acc.pow(2).mean()
```
直接对 root 做速度/加速度 L2 正则，压制"平移不自洽"。

#### 4.2.5 Logit-normal timestep sampling（velocity）
替换 `torch.rand()` 为：
```python
t = torch.sigmoid(torch.randn(B) * 1.0)
```
对应 SD3 / Flux 的标准实践，对 rectified-flow velocity 预测显著有帮助。

#### 4.2.6 推理 CFG 系数标定 + VACE strength annealing
- 标定：在 eval 脚本里 sweep `cfg_scale∈{1.5,3,5,7.5,10}`，选 `MMDist/CLIP-S` 最佳点作为 dashboard 默认值。
- VACE strength annealing：在推理 denoising 的早期 step（t≈1）把 `reactive/mask` 通道 *scale* 到 0.5；后半段再升回 1.0。可以让 text 在大 t（语义决定阶段）主导粗构，小 t 再由 VACE 定几何。只需推理 pipeline 加一行 scale。

**新 config**（示例）：
```python
model = dict(
    losses_cfg=dict(
        loss_type='smooth_l1',
        velocity_weight=1.0,
        trans_dim_weight=10.0,
        motion_smoothness_weight=1.0,
        keypoints3d_weight=1.0,      # global mode
        keypoints3d_mode='global',    # 新增
        translation_weight=1.0,
        fk_consistency_weight=0.5,
        fk_consistency_warmup_steps=500,
        foot_skate_weight=1.0,        # 新
        limb_len_weight=0.2,          # 新
        root_traj_weight=0.3,         # 新
    ),
    cond_mask_prob=0.25,
)
trainer = dict(
    timestep_sampler='logit_normal',  # 新
)
```

### 4.3 长期（表示/训练策略重构，2~4 周）

#### 4.3.1 T2M ↔ M2M 双头共享 backbone
- 把 `HyMotionM2MBundle` 拆成"共享 backbone + 两个 task head"：
  - **T2M head**：输入 `x_t`，VACE 通道置零 → 纯 T2M。
  - **M2M head**：输入 `[x_t, reactive, mask]` → 完整 VACE。
- 训练按 batch-level 混合（1:1），确保"text → backbone"梯度信号不被 VACE 吃掉。
- 这个方案与 KIMODO Phase 1→2 curriculum 互补：curriculum 是 temporal dilution，dual-head 是 structural isolation。

#### 4.3.2 VACE noise-stage gating（比 4.2.6 更系统）
- 训练层面给 `reactive/mask` 通道乘一个 `f(t)`：
  - `f(t) = sigmoid(k*(t0 - t))` —— 在大 t（噪声多）时 f→0，小 t 时 f→1。
  - 这样训练与推理同分布，并强迫 text 在粗构阶段起作用。

#### 4.3.3 切换到 global rotation 作为主线
- `hftrainer/models/motion/CLAUDE.md` 已有 +41% neighbor predictability 证据。
- 虽然 global 对部分 completion 任务有 consistency 代价，但对 locomotion 的"全局一致"更友好；local 改为 polish/post-edit 专线。

#### 4.3.4 SOAR 后训（已有设计）
- `docs/temp/soar_m2m_v2_post_training_plan.md` 已就绪；应在 caption global fix1 收敛后作为 stage-2 自动接上。
- 目标：解决 exposure bias；M2M 场景下 SOAR 特别重要，因为 VACE shortcut 会放大推理分布失配。

#### 4.3.5 物理先验 / 接触先验（UnderPressure / PhysHOI 路径）
- 用预训练的 foot pressure estimator 提供 "是否该接触"的监督信号；或直接用 PhysHOI-style contact mask（toe+heel + 低 Y）加 soft loss。
- 中期 4.2.1 是其最小化实现；长期可接入外部 estimator 做更严的监督。

#### 4.3.6 训练数据质量过滤
- 根据 `hftrainer/models/motion/CLAUDE.md` §Training Data Quality Issue，当前 base config 用的是 `train_hymotion_400h_hq_20260403.json`（高质量），已改善；但 early epoch 有 549K 混合的历史 stage，应从 pretrained 检查 (`caption_local_046b/epoch_183`) 的数据版本重新训，避免残留。

#### 4.3.7 Text encoder 解冻或加 LoRA
- 为 Qwen3 上加 r=8 的 LoRA，`text_refiner` 之外再打开 language-side 的梯度通路；使"多约束语义"（"先屈膝再前倾"）能被表示进 embedding。

---

## 5. 建议实施路径（Roadmap）

```
Week 1（立刻）:
  [S1] 出 fix1 config（4.1 全部） → 2 节点 16 卡起训 80 epochs
  [S2] 加 foot-skate + global-keypoints3d + root-traj loss（4.2.1/4.2.2/4.2.4）
  [S3] eval_dashboard 跑 E1/E3/E5/E7/E9/E10/E11/E12 + baseline 对比

Week 2:
  [S4] 接 SOAR 后训（stage-2），跟 fix1 对比
  [S5] 切 caption_global 作为主线；跑 limb-len + logit-normal

Week 3:
  [S6] Dual-head 结构改动（4.3.1）；backbone 共享、head 分叉
  [S7] VACE noise-stage gating 推理 + 训练对齐

Week 4:
  [S8] 标定 CFG scale；final eval_dashboard 全量；与 T2M 1.0 / v1 / KIMODO / MoGenDIT 出齐 radar
  [S9] 物理先验补强（4.3.5）作为 stretch goal
```

---

## 6. 消融实验清单（建议必跑）

| Exp | 变量 | 基线 | 预期主指标变化 |
|-----|------|------|--------------|
| A0 | 当前 v2_caption_local_046b（baseline） | — | 作为参照 |
| A1 | +cond_mask_prob 0.1→0.25 | A0 | E1 MMDist ↓ 15% |
| A2 | +pure_gen 16%→30% | A1 | E1 CLIP-S ↑，E3/E5 text 响应更强 |
| A3 | +keypoints3d/translation on | A2 | foot_skate ↓ 10–20% |
| A4 | +foot_skate loss | A3 | foot_skate ↓ 到 baseline 的 50% |
| A5 | +global_fk + limb_len | A4 | mpjpe_masked ↓，bend/prone 任务合理度↑（定性） |
| A6 | +logit-normal timestep | A5 | 全指标小幅改善（~3–5%） |
| A7 | +SOAR 后训 | A6 | 细节与节奏（jitter_pos、temporal） |
| A8 | Dual-head backbone | A7 | E1 追平 T2M 1.0 |
| A9 | Global rotation space | A8 | locomotion 质量进一步稳定 |

每个 exp 跑 30 epochs（fix1 config），E1/E5/E10/E12 四任务 × 100 samples。

---

## 7. 验证指标与接受门槛（Go/No-Go）

| 指标 | baseline（v2 当前） | 目标（短期 fix1） | 目标（长期） |
|------|---------------------|-------------------|--------------|
| E1 MMDist (T2M) | 退化 | 与 T2M 1.0 持平 | ≤ T2M 1.0 |
| E1 CLIP-S | 退化 | ≥ baseline +10% | ≥ T2M 1.0 |
| foot_skate_velocity | 高 | ↓ 30% | ↓ 60% |
| loop_position_error (E10) | 高 | ↓ 20% | ↓ 50% |
| mpjpe_masked | 高 | ↓ 15% | ↓ 30% |
| jitter_pos | 中 | ↓ 10% | ↓ 25% |
| 定性 locomotion（dashboard 3D viewer） | 滑过去 | 可见脚步周期 | 接触节奏正确 |
| 定性 bend/prone | 物理违反 | 初步因果链 | 动作合理 |

---

## 8. 风险与规避

1. **打开 keypoints3d_loss 引入 FK 开销**：FK 是 O(T × 22)，batch_size=20 下可承受；若超时用 `subsample_fk_every=2`。
2. **foot-skate loss 与 MAN 的联动**：Known 区 `x_t=x1` 是 clean 的，FK 后的 foot pos 本来就对；loss 只在生成区取值即可（与 generation_mask 共用）。
3. **`cond_mask_prob` 拉到 0.25 可能让 completion 退化**：在消融 A1 中监控 E7/E11 等 completion-heavy 任务，若退化则改双头方案（4.3.1）而非纯提 prob。
4. **Dual-head 实现复杂度**：可先用 batch-level 的 tag，把一半 sample 的 VACE 通道硬置 0 来验证思路再做结构改动。
5. **VACE annealing 对 repair 任务有副作用**：Repair 时 reactive 是 "corrupted" 值，annealing 可能延迟恢复；对 `edit_repair` task 单开 `f(t)=1`（即不退化）。

---

## 9. 结论

HyMotion M2M v2 的两类问题**不是"模型能力不够"，而是"训练目标函数 + 条件构造让模型学到了捷径"**：
- 文本信号被 VACE 强条件稀释（F1–F4、F10、F13）。
- 世界系几何/物理约束几乎缺失（F5–F9）。

**短期** 通过调权 + 打开 3D loss 即可看到显著回升；**中期** 加 foot-skate/global-FK/limb-len/root-traj 与 logit-normal 即可拉回与 T2M 1.0 同台竞争；**长期** 通过双头结构 + VACE gating + SOAR 可彻底解耦"语义指令"与"几何续写"两个任务的梯度竞争。

**本报告存储路径**：
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/docs/temp/m2m_v2_text_and_locomotion_issues_report.md
```

---

## 附录 A：涉及文件索引

| 文件 | 引用用途 |
|------|---------|
| `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` | 基础 config，所有权重默认值 |
| `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_{local,global}_046b.py` | caption 主线 |
| `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase{1,2}.py` | 现有 curriculum 尝试 |
| `hftrainer/trainers/motion/hymotion_m2m_trainer.py` | `_prepare_and_forward`、MAN、text dropout |
| `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py` | `M2MLoss`，loss 定义 |
| `hftrainer/datasets/motion/motionhub/transforms/condition_sampler_v2.py` | Tier1/Tier2 采样、`sample_translation` |
| `hftrainer/datasets/motion/motionhub/transforms/prepare_m2m_v2.py` | 条件与 corruptor 组合 |
| `hftrainer/models/motion/CLAUDE.md` | 历史 bug、cross-project 对比、global vs local |
| `motion_annot_web/eval_dashboard/CLAUDE.md` | 评估平台、`--save-npz` / `--use-rewritten` 规范 |
| `docs/temp/hymotion_m2m_v2_critical_analysis.md` | 之前已有的 16 条 root cause 调研 |
| `docs/temp/t2m_text_conditioning_bugfix.md` | T2M 文本条件三层 bug 的历史记录 |
| `docs/temp/m2m_v2_training_experiments.md` | curriculum 实验台账 |
| `docs/temp/soar_m2m_v2_post_training_plan.md` | SOAR 后训计划 |

## 附录 B：关键参数快表（当前 vs 建议）

| 参数 | 当前 | 建议（短期） | 建议（中期） |
|------|------|------------|------------|
| `cond_mask_prob` | 0.1 | 0.25 | 0.25 |
| `pure_gen` 占比 | 16% | 24–30% | 30% |
| `tier2_prob` | 0.4 | 0.5 | 0.5 |
| `trans_keep` (Tier1) | 0.2 | 0.5 | 0.5 |
| `keypoints3d_weight` | 0 | 1.0 | 1.0（global） |
| `translation_weight` | 0 | 1.0 | 1.0 |
| `trans_dim_weight` | 5 | 10 | 10 |
| `motion_smoothness_weight` | 0.5 | 1.0 | 1.0 |
| `fk_consistency_weight` | 0.1 | 0.5 | 0.5 |
| `fk_consistency_warmup_steps` | 2000 | 500 | 500 |
| `foot_skate_weight` | — | — | 1.0 |
| `global_fk_weight` | — | — | 0.5 |
| `limb_len_weight` | — | — | 0.2 |
| `root_traj_weight` | — | — | 0.3 |
| `timestep_sampler` | uniform | uniform | logit-normal |
| `rotation_space` (主线) | local | local | global |
| `mixed_precision` | 'no' | 'bf16' (如硬件支持) | 'bf16' |
