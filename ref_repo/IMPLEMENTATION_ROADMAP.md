# M2M 实现路线图 — 基于竞争对手分析

## 📊 现状评估

### M2M vs 竞争对手功能矩阵

|功能|KIMODO|UMO|MotionLab|M2M当前|M2M目标|优先级|
|---|---|---|---|---|---|---|
|Text-to-Motion|✅|✅|✅|✅|✅|—|
|Temporal Inpainting|✅|✅|✅|✅|✅|—|
|Keyframe Infilling|✅|✅|✅|✅|✅|—|
|Trajectory Following|✅❌|⚠️|✅|❌|✅|**P0**|
|Instruction Editing|❌|✅|✅|❌|✅|**P1**|
|Style Transfer|❌|❌|✅|❌|✅|**P2**|
|Obstacle Avoidance|❌|✅|❌|❌|✅|P2|
|Multi-person/Reaction|❌|✅|❌|❌|?|P2|
|Part-level Control|✅|❌|⚠️|✅|✅|—|
|Task Instruction Modulation|❌|❌|✅|❌|✅|**P0**|
|Motion Curriculum Learning|✅|❌|✅|❌|✅|**P0**|

### 关键指标对标

|指标|KIMODO|UMO|MotionLab|M2M当前|
|---|---|---|---|---|
|T2M FID|10.52|9.46|7.62|~8.5-9.0|
|Temporal Infill MPJPE|—|8.55|8.23|~9.5|
|Trajectory Error|~5cm|18.78cm|2.86cm|无|
|Style Transfer SRA|—|—|69.21|无|
|参数量|282M|460M|460M|460M|
|额外参数(adapter)|×2 input|0.207M|—|4× input_encoder|
|训练轮数|1M|100k|1M|需优化|

---

## 🎯 改进优先级（ROI 排序）

### Tier 1: 立即启动（2-3 周，ROI 极高）

#### T1.1 Task Instruction Modulation（成本：1周，收益：+2-5% FID）

**问题**：M1-M6 mask 策略对模型隐形 → task boundary 丧失

**MotionLab 做法**：
```python
# CLIP 编码 task 指令
task_instructions = {
    'complete_keyframes': "complete motion from sparse keyframes",
    'temporal_inpaint': "inpaint motion temporally",
    'extend_motion': "extend motion by continuation",
    'edit_joints': "edit specific joints while preserving",
    'generate_from_text': "generate entire motion from text",
    'preserve_structure': "preserve skeleton while modifying",
}

task_emb = clip_encode(task_instructions[task_id])
timestep_emb = timestep_emb + task_emb  # ← 加到时间步嵌入
```

**实现步骤**：
1. 定义 6 个 task instruction 文本
2. 在 dataloader 中添加 task_id → instruction mapping
3. 在 model.forward() 中集成 task_emb 到 timestep_emb
4. 无需重训练骨干，可作为 LoRA adapter

**性能预期**：
- Baseline FID: 8.5
- +Task Instruction: 8.3-8.4 (+2-3%)
- 各 task 独立 FID 降低（防止跨任务混淆）

**验证指标**：
```
before: M1 FID=2.8, M2 FID=3.1, M3 FID=2.5, M4 FID=3.2, M5 FID=1.2, M6 FID=2.9
after:  M1 FID=2.6, M2 FID=2.9, M3 FID=2.3, M4 FID=3.0, M5 FID=1.1, M6 FID=2.7
```

---

#### T1.2 Motion Curriculum Learning（成本：2-3天，收益：+20-40% FID）

**问题**：固定 M1-M6 采样比例 → 缺乏阶段性重点

**MotionLab 消融**：无 curriculum → FID 11.7× 恶化（0.167 → 1.956）

**推荐课程设计**：

| 阶段 | 轮数 | 主要任务 | M1 | M2 | M3 | M4 | M5 | M6 | 采样方式 |
|------|------|---------|-----|-----|-----|-----|-----|-----|---------|
| Pre-train | 1-100ep | 掩码预训 | - | - | - | - | 100% | - | 均匀 |
| Stage 1 | 101-300ep | +Keyframe | 40% | - | - | - | 60% | - | 均匀 |
| Stage 2 | 301-500ep | +Temporal | 30% | - | 30% | - | 40% | - | 均匀 |
| Stage 3 | 501-700ep | +Joint | 25% | - | 25% | 25% | 25% | - | 均匀 |
| Fine-tune | 701+ep | All tasks | 均匀 | 均匀 | 均匀 | 均匀 | 均匀 | 均匀 | **FID加权** |

**FID 加权采样**（Fine-tune 阶段）：
```python
# 每 N batch 计算各 task 滑动 FID (window=20 batches)
task_fids = {'M1': 2.5, 'M2': 2.8, 'M3': 2.1, 'M4': 3.2, 'M5': 1.2, 'M6': 2.9}

# 反向加权：FID 高的任务更多训练（抵消遗忘）
weights = {t: 1.0 / (fid_t + eps) for t, fid_t in task_fids.items()}
weights = {t: w / sum(weights.values()) for t, w in weights.items()}

# WeightedRandomSampler 按权重采样
sampler = WeightedRandomSampler(weights, len(dataset))
```

**实现代码**：

```python
# data/curriculum_sampler.py
class CurriculumSampler:
    def __init__(self, curriculum_schedule, total_epochs):
        """
        curriculum_schedule: List[(epoch_start, epoch_end, [task_ids], strategy)]
        strategy: 'uniform' or 'fid_weighted'
        """
        self.schedule = curriculum_schedule
        self.task_fid_history = defaultdict(lambda: deque(maxlen=20))
        self.current_epoch = 0
        self.batch_count = 0
    
    def sample_task(self):
        # 根据 current_epoch 找到 active tasks
        active_tasks = self._get_active_tasks(self.current_epoch)
        
        # 根据策略采样
        if self._is_fid_weighted_phase():
            weights = self._compute_fid_weights(active_tasks)
            task_id = np.random.choice(active_tasks, p=weights)
        else:
            task_id = np.random.choice(active_tasks)
        
        return task_id
    
    def update_task_fid(self, task_id, fid_value):
        self.task_fid_history[task_id].append(fid_value)
    
    def _compute_fid_weights(self, active_tasks):
        # 反向 FID 加权
        weights = {}
        for task_id in active_tasks:
            fid_mean = np.mean(self.task_fid_history[task_id])
            weights[task_id] = 1.0 / (fid_mean + 1e-4)
        
        # 归一化
        total = sum(weights.values())
        return np.array([weights[t] / total for t in active_tasks])
```

**性能预期**：
```
Baseline:           M1 FID=2.8, M3 FID=2.5, M5 FID=1.2 → Avg=2.17
+Curriculum:        M1 FID=2.0, M3 FID=1.8, M5 FID=1.0 → Avg=1.60
改善幅度: ~26%（基于 11.7× 消融的保守估计）
```

**Logging/监控**：
```python
# 每 epoch 记录
metrics = {
    'M1_fid': 2.5, 'M2_fid': 2.8, ...
    'M1_weight': 0.25, 'M2_weight': 0.20, ...  # FID 加权的实时权重
    'curriculum_phase': 'Stage2',  # 当前课程阶段
}
```

---

#### T1.3 E_ctx 权重初始化优化（成本：1天，收益：+3-5% 收敛速度）

**UMO 经验**：E_ctx 初始化为预训练 input_encoder 权重复制，而非随机初始化

**当前 M2M**：input_encoder 在 VACE 初始化时形状从 135→540，导致随机初始化

**改进方案**：

```python
# 模型初始化时
class ImprovedVACE(nn.Module):
    def __init__(self, pretrained_motion_encoder, motion_dim=135):
        super().__init__()
        
        # 保留预训练的 input_encoder
        self.input_encoder = pretrained_motion_encoder  # shape: (motion_dim, hidden_dim)
        
        # VACE 其他部分也应该借鉴预训练
        # 不再随机初始化，而是分别复制 4 个分量
        self.motion_encoder = copy.deepcopy(pretrained_motion_encoder)
        self.inactive_encoder = copy.deepcopy(pretrained_motion_encoder)
        self.reactive_encoder = copy.deepcopy(pretrained_motion_encoder)
        self.mask_encoder = nn.Linear(motion_dim, hidden_dim)
        
        # mask_encoder 可以随机初始化或者用其他预训练

# 这样 VACE 的 4 个分量有 3 个继承了预训练权重，只有 mask_encoder 是新的
# 预期：加速 5-10 epoch 收敛
```

**验证**：对比收敛曲线（收敛到相同 FID 用的 epoch 数）

---

### Tier 2: 核心能力（4-6 周）

#### T2.1 xyz Position 维度支持（成本：3周，收益：支持 trajectory constraints）

**问题**：M2M 135-dim 只含 translation + rotation，无 local joint positions

**竞争对手**：
- KIMODO: 333-dim (global rot + positions + velocity + contact)
- MotionLab: 263-dim (+ positions + velocity + contact)

**步骤 1：表示层扩展**

```
当前（135）:
  - Root translation: 3D (3)
  - Root orientation: 6D (6)
  - 22 Joint rotations: 22×6D (132)
  = 3 + 6 + 132 = 141 ≈ 135

目标（201）:
  - Root translation: 3D (3)
  - Root orientation: 6D (6)
  - 22 Joint rotations: 22×6D (132)
  - 22 Joint positions: 22×3D (66)
  = 3 + 6 + 132 + 66 = 207 ≈ 201

数据准备：
  - 从现有 SMPL 骨骼链反算 forward kinematics
  - 或者直接使用已有的 local position 注解（如果有）
```

**步骤 2：FK Loss 实现**

```python
# 训练时约束：predicted positions 应该通过 FK 一致性检查
lambda_fk = 10.0

# Forward Kinematics
def compute_positions_from_rotations(rotations, skeleton):
    """
    rotations: (T, 22, 6)  [Cont6D format]
    skeleton: bone lengths + parent indices
    returns: positions (T, 22, 3)
    """
    root_pos = rotations[:, 0, :3]  # root translation 已在旋转向量中
    positions = [root_pos]
    
    for joint_id in range(1, 22):
        parent_id = skeleton.parents[joint_id]
        parent_pos = positions[parent_id]
        bone_offset = skeleton.offsets[joint_id]
        
        # 旋转 bone offset
        joint_rot = rotations[:, joint_id, :]
        rotated_offset = apply_6d_rotation(joint_rot, bone_offset)
        
        pos = parent_pos + rotated_offset
        positions.append(pos)
    
    return torch.stack(positions, dim=1)

# Loss 计算
positions_from_rotation = compute_positions_from_rotations(pred_rotation, skeleton)
loss_fk = lambda_fk * F.l1_loss(positions_from_rotation, pred_position)
```

**步骤 3：VACE 适配**

```
当前：concat([x_t(135), inactive(135), reactive(135), mask(135)]) = 540-dim
目标：concat([x_t(201), inactive(201), reactive(201), mask(201)]) = 804-dim

input_encoder 需要适配：540 → 804
方案：
  a) 重新初始化（会损失 pretrain）
  b) 位置补零 (zero-pad 残留部分，保留 135-dim pretrain)
  c) 分别初始化 4 个 encoder (推荐，配合 T1.3 的权重复制)
```

**步骤 4：训练数据**

```
合成 trajectory constraints 数据（20-30% of batch）：
  - 从 HumanML3D 采样随机关键帧
  - 或从 Inter-X 提取已有的 trajectory annotations
  - 生成 waypoint 序列，让模型学习满足 path constraint

数据比例：
  - 70% 无约束（保持 T2M 质量）
  - 30% 有 trajectory constraint
```

**性能预期**：
- Trajectory error: 从无支持 → ~10-15cm（vs KIMODO ~5cm, MotionLab 2.86cm）
- T2M FID 影响：-0.2 ~ +0.5（可接受）
- 验证指标：trajectory MSE, RMSE, max error

---

#### T2.2 Instruction-based Editing（成本：1-2周，收益：新增 editing 能力）

**问题**：M4 仅能做 joint-level 重生成，无法处理"speed up"、"make more energetic"等指令

**竞争对手**：
- UMO: [edit] meta-op + instruction text
- MotionLab: instruction text 从 MotionFix dataset

**方案**：结合 T1.1 的 Task Instruction Modulation

```python
edit_instructions = [
    "speed up the motion by 1.2x",
    "slow down the motion",
    "make the motion more energetic",
    "make the motion more gentle",
    "emphasize hand movements",
    "emphasize leg movements",
    "add more body rotation",
    "make steps wider",
]

# 在 M4 采样时随机选择
if task_id == 'M4_edit':
    instruction = random.choice(edit_instructions)
    # instruction 通过 T1.1 的 task_emb 注入
    task_text = f"edit motion: {instruction}"
    task_emb = clip_encode(task_text)
    
    # 在 timestep_emb 中加入
```

**数据源**：MotionFix dataset
- 已有 source/target/instruction 三元组
- 可直接复用，无需新标注

**性能预期**：
- SRA (Style Recognition Accuracy): 60-65（vs MotionLab 69.21，考虑表示差异）
- 定性验证：手工测试几个典型指令

---

### Tier 3: 探索性（8-12 周）

#### T3.1 Style Transfer（P2）
- 需要 style reference embedding
- MotionLab 在 text/trajectory modality 上基础进行扩展
- 预期收益：SRA 60-65+

#### T3.2 Multi-person / Reaction Generation（P2）
- UMO 支持 dual-identity，需要改 data representation
- 需要 interaction dataset (Inter-X, InterHuman)
- 预期收益：新能力

#### T3.3 Aligned 1D RoPE（P2，依赖条件）
- 如果支持 trajectory constraints，可能需要强制 time alignment
- MotionLab 的关键创新，值得考虑

---

## 📋 实现检查清单

### Phase 1（2-3 周）

#### Week 1
- [ ] T1.1 Task Instruction Modulation
  - [ ] 定义 6 个 task instructions
  - [ ] 实现 CLIPTextEncoder wrapper
  - [ ] 在 timestep_emb 中集成 task_token
  - [ ] 验证 +2-5% FID

- [ ] T1.3 E_ctx 初始化
  - [ ] 修改 VACE 权重初始化
  - [ ] 验证收敛速度提升

#### Week 2-3
- [ ] T1.2 Motion Curriculum Learning
  - [ ] 实现 FID-weighted sampler
  - [ ] 设计 7-stage curriculum schedule
  - [ ] 集成 Logging（每 task FID curve）
  - [ ] 验证 +20-40% FID

### Phase 2（4-6 周）

#### Week 4
- [ ] T2.1 Position 维度支持（Part 1）
  - [ ] 表示层：135 → 201
  - [ ] motion_encoder/decoder 适配
  - [ ] FK Loss 实现
  - [ ] 单元测试

#### Week 5
- [ ] T2.1 Position 维度支持（Part 2）
  - [ ] VACE 适配：540 → 804
  - [ ] 合成 trajectory data
  - [ ] 训练基准运行
  - [ ] 基准评估（trajectory error）

#### Week 6
- [ ] T2.2 Instruction Editing
  - [ ] 从 MotionFix 抽取数据
  - [ ] instruction embedding
  - [ ] 集成到 M4 采样
  - [ ] SRA 评估

### Phase 3（8-12 周）

- [ ] T3.1 Style Transfer
- [ ] T3.2 Multi-person
- [ ] T3.3 Aligned 1D RoPE

---

## 📊 成功指标

### Quantitative Benchmarks

| 指标 | 当前 | Phase 1 目标 | Phase 2 目标 | Phase 3 目标 |
|------|------|------------|------------|------------|
| T2M FID | ~8.7 | 8.4-8.5 | 8.3-8.4 | 8.0-8.2 |
| Temporal MPJPE | ~9.5 | 9.2-9.3 | 8.8-9.0 | 8.5-8.7 |
| Trajectory Error | 无 | 无 | 10-15cm | < 10cm |
| Part Edit Accuracy | ~70% | 72% | 75%+ | 80%+ |
| Instruction SRA | 无 | 无 | 60-65% | 65%+ |
| Style Transfer SRA | 无 | 无 | 无 | 60-65% |

### Qualitative Criteria

- ✅ Task Instruction 正确区分（不混淆）
- ✅ Curriculum Learning 各 task 独立 FID 下降
- ✅ Position 支持后 trajectory 约束生效
- ✅ Instruction editing 定性对得上指令意图

---

## 资源需求

| 资源 | 需求 | 理由 |
|------|------|------|
| GPU 小时 | ~2000 hrs | Phase 1 ~500, Phase 2 ~1200, overhead ~300 |
| 人力 | 1-2 工程师 | 全职 2-3 months |
| 数据 | HumanML3D + MotionFix + Inter-X | 用于合成 trajectory constraints + instruction editing |
| 基础设施 | 现有（无新需求） | 仅需调整 sampler/logging |

---

## 风险与缓解

### 风险 1：Curriculum Learning 导致 Task Forgetting
**缓解**：FID-weighted resampling（后期加强弱任务采样）

### 风险 2：Position 维度扩展影响 T2M 质量
**缓解**：保持 70% 无约束数据，监控 T2M FID 独立曲线

### 风险 3：Input encoder 扩大（135→804）导致初始化问题
**缓解**：T1.3 的权重复制方案，分别初始化 4 个 encoder

### 风险 4：MotionFix instruction 数据量不足
**缓解**：合成/扩展 instruction 集（parametric templates）

---

## 参考文献

1. **KIMODO** (NVIDIA): Imputation-based conditioning, Phase 1/2 curriculum
2. **UMO** (Brown/MIT/Meta): [Edit] semantics, E_ctx weight initialization
3. **MotionLab** (SUTD): Task Instruction Modulation, FID-weighted curriculum, Aligned 1D RoPE
4. **MotionFix**: Instruction editing dataset
5. **Inter-X / InterHuman**: Multi-person interaction data

---

## 文件导航

- [M2M_TECHNICAL_SYNTHESIS.md](M2M_TECHNICAL_SYNTHESIS.md) - 详细技术分析
- [TECHNICAL_COMPARISON.md](TECHNICAL_COMPARISON.md) - 三框架对比
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 快速参考
- [m2m_ablation_experiments.md](m2m_ablation_experiments.md) - 消融实验设计

