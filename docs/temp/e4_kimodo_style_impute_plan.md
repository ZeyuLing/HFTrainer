# E4 推理重构方案（KIMODO-style per-step imputation）

## 用户指示（2026-04-22 第3次）
> 不能靠 IK，是在每个 denoise step，将约束关节的 position 通道通过 imputation 进行替换
> （可能需要输入 translation，需要和 KIMODO 的 end effector 控制方式进行比较），
> 期待模型能根据 imputation 输入的 position 通道学到正确的 rotation。

## 对比 KIMODO

| | KIMODO | M2M v2 (本方案) |
|---|---|---|
| Coord frame | Global rotation + global pos | Local rotation + **Scheme D** 位置 (XZ rel pelvis, Y abs) |
| Impute 内容 | smooth_root_2d + root_y + heading + ee_pos + ee_rot | ee_pos_channel (+可选 pelvis_trans) |
| 训练见过 | Phase 2 专门学 impute pattern | T2-5 训练 pattern 已存在 (只 mask EE pos channel) |
| 推理 replace | 每 denoise step 硬替换 | `replacement_guidance='skip_last'` + `clean_motion` |

## T2-5 训练 pattern 已存在（`condition_sampler_v2.py:223-234`）
```python
def sample_tier2_end_effector(T, mask, rng):
    n_ee = rng.randint(1, 5)
    ee_joints = rng.choice(EE_ALL, size=n_ee, replace=False)
    K = min(max(1, rng.geometric(p=0.1)), T)  # 稀疏帧
    frames = sorted(rng.choice(T, size=min(K, T), replace=False))
    for f in frames:
        for j in ee_joints:
            mask[f, 135+(j-1)*3:135+j*3] = 0   # 仅 mask 位置通道 (3 dims)
    # NOTE: pelvis_trans 和 rot6d 都保持 mask=1，完全 generated
```

这正是 E4 需要的。

## 当前 E4 mask（错误）vs 目标 E4 mask（正确）

### 当前（`_build_ee_mask_198` line 194-218）
```python
# cond frame t:
mask[t, 0:3] = 0                         # ❌ 锁 pelvis trans
mask[t, 3+j*6:3+(j+1)*6] = 0             # ❌ 锁 R_Wrist rot6d
mask[t, 135+(j-1)*3:135+j*3] = 0         # ✅ 锁 R_Wrist pos
```
→ 训练中**从未见过**这种组合，OOD → 模型 pose 崩坏，身体下沉。

### 目标（匹配 T2-5）
```python
# cond frame t:
mask[t, 135+(j-1)*3:135+j*3] = 0   # ✅ 仅锁 R_Wrist pos channel (3 dims)
# 其它全 mask=1 → generate
```

## Scheme D 坐标转换

Position channel 格式：
- `X_rel = world_X - pelvis_X` (XZ 相对 pelvis)
- `Y_abs = world_Y`
- `Z_rel = world_Z - pelvis_Z`

### 问题：推理时 pelvis_trans 是 generated
无法精确做 X/Z relative conversion。方案：

**Option 1: 同时 impute pelvis translation (最贴近 KIMODO)**
- mask[t, 0:3] = 0 也锁上
- clean_motion[t, 0:3] = GT pelvis translation
- clean_motion[t, 135+(j-1)*3:...] = GT R_Wrist world pos - GT pelvis pos（正确的 Scheme D）
- **代价**：mask pattern = "trans + ee_pos 都锁"，T2-5 只锁 ee_pos（不锁 trans），还是略 OOD

**Option 2: 只给 Y 约束（Y 是绝对值，不需要 pelvis 参考）**
- mask[t, 135+(j-1)*3+1] = 0（只锁 Y，X/Z 自由）
- clean_motion[t, 同位置] = gt_R_Wrist_Y
- **完全匹配 T2-5**（T2-7 foot grounding 就是这种）
- **代价**：X/Z 约束丢失，手可以在水平面任意位置，只高度对

**Option 3: 给 XYZ 但用 pred 的 pelvis 做 relative (每 step 动态更新 clean_motion)**
- 每 denoise step 从 pred 读当前 pelvis trans，用它计算 X/Z relative
- 实现复杂（需要 pipeline hook）

**推荐：Option 1**（和 KIMODO 一样同时锁 pelvis trans + ee pos，最稳妥）

## 实施步骤

### 1. 修改 `build_end_effector_mask` / `_build_ee_mask_198`
```python
def _build_ee_mask_198(T, joint_names, frame_interval):
    mask = np.ones((T, 198), dtype=np.float32)
    for t in range(0, T, frame_interval):
        mask[t, 0:3] = 0   # pelvis trans (for world anchor)
        for name in joint_names:
            j = JOINT_NAME_TO_IDX[name]
            if j == 0: continue
            # ONLY mask position channel, NOT rot6d
            mask[t, 135+(j-1)*3:135+j*3] = 0
    # 构造 constraint_info 不变
```

### 2. 构造 clean_motion (传给 pipeline)
```python
# In evaluate_sample for E4:
clean_motion_198 = motion_198_norm.clone()   # 从 GT 计算得到的归一化 motion

# 确保 clean_motion 的约束位置是 normalized GT 值
# （GT 的 pos channel 已经是 Scheme D 格式，无需额外转换）
batch['clean_motion'] = clean_motion_198
batch['src_mask'] = mask
```

### 3. 调用推理用 replacement_guidance='skip_last'
```python
outputs = pipeline(
    batch,
    replacement_guidance='skip_last',  # per-step impute in cond regions
    ...
)
```

### 4. 可选 —— post-hoc 不再需要替换 rotation
当前 `evaluate_sample:2275` 做 `output_135[cond_mask] = motion_135[cond_mask]`。
因为 cond_mask 现在**只锁 pelvis_trans + pos_channel**，不涉及 rot6d，所以 rot6d 完全来自模型预测 → 正好我们要的。

## 验证方式
1. 跑 1 个 sample，查 condition 帧 foot_minY：应 ≈ 0（不再 -0.16m）
2. 查 R_Wrist world Y：应接近 GT target（ee_error 应低）
3. 查 rot6d 区间是否平滑（不再每 10 帧跳变）

## 风险
- **训练分布贴近但不 100%匹配**：T2-5 是稀疏帧 + 1-4 个 EE，E4 的 `F_rhand_dense` 是每 5 帧一次（dense），可能 OOD
- **Scheme D 的 Y 用 absolute**：pelvis Y 自由时手的绝对高度应该能 follow，但 X/Z 依赖 pelvis 的精度
- **`_man` vs 标准模型**：`replacement_guidance='skip_last'` 在 `_man` 变体上 train-consistent，但当前 v2 模型（uncond_local / uncond_global / caption_local / caption_global）是否是 `_man`？需确认
