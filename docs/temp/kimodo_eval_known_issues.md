# KIMODO Eval Script Known Issues

`tools/run_kimodo_all_tasks.py` 中存在以下已知问题，影响 condition frame 附近的运动质量。

## P0: FPS Resampling 导致 Constraint Frame 不对齐

**代码位置**: `_run_kimodo_with_constraints()` L672-678, `evaluate_sample()` L779-786

**问题**: KIMODO 模型内部以 20fps 运行，评估数据以 30fps 定义。constraint frame indices 以 30fps 定义但直接传给 20fps 的 KIMODO，导致 constraint 在时间上偏移。推理输出后用 `interp1d` 线性重采样回 30fps，但重采样无法恢复 constraint frame 的精确对齐。

**影响**: Condition frame 附近出现位置跳变，特别是 E2(in-betweening)、E3(keyframe) 中首/尾帧约束不精确。

**修复方向**: 将 constraint frame indices 转换到 KIMODO fps 空间再传入：`kimodo_frame = round(smpl_frame * model_fps / 30)`

## P1: 多层骨架映射精度损失

**映射链**: SMPL-22 rot6d → SMPL-22 FK → global rots → SOMA-30 global rots → SOMA-30 local rots → SOMA-30 FK → SOMA-77 posed_joints → SMPL-22 positions

每一步映射都引入误差：
1. SMPL-22 → SOMA-30: 骨骼长度不同（SOMA30 腿更短约 2.5cm）
2. SOMA-30 → SOMA-77: 手指/面部关节用父关节旋转填充
3. SOMA-77 → SMPL-22: 从 77 关节中提取 22 个对应关节

**影响**: Constraint frame 处，KIMODO 内部满足的是 SOMA-30 空间的约束，但回映射到 SMPL-22 时存在 ~1-3cm 的系统性偏差。在 condition/generation 边界处，这个偏差表现为可见的跳变。

## P1: Ground Alignment 静态偏移

**代码位置**: `smpl22_to_soma30_retarget()` L222-236 (Step 4)

**问题**: Ground offset 从 neutral pose 静态计算（`foot_offset_y = soma_foot_min_y - smplx_foot_min_y`），不随动态姿态变化。当动作包含大幅度弯腿/跳跃时，静态偏移导致脚部浮空 ~6-9cm。

**当前缓解**: 可视化端 (`utils.py`) 已添加 per-sequence ground normalization。

## P2: 缺失 Constraint Frame 验证

constraint builder 函数（`build_constraints_e2` 等）不验证 frame indices 是否在 KIMODO 的实际输出长度范围内。当 fps 转换后帧数不匹配时，超出范围的帧被静默截断。

## P2: E8 Loop 帧数计算

E8 loop 任务中 `T_total` 可能与 KIMODO 实际输出帧数不匹配，导致首尾帧不对齐。

---

## 影响范围

| 任务 | 受影响程度 | 主要表现 |
|------|----------|---------|
| E2 (In-Betweening) | 高 | 首/尾约束帧处跳变 |
| E3 (Keyframe) | 高 | 每个 keyframe 处可见不连续 |
| E4 (End-Effector) | 中 | 末端执行器位置偏差 |
| E5 (Trajectory) | 中 | 轨迹跟随不精确 |
| E7 (First-Frame) | 中 | 首帧约束处跳变 |
| E8 (Loop) | 高 | 首尾帧不对齐 |
| E14-E16 (Transition) | 高 | 过渡边界处明显不连续 |

## 后续计划

重新运行 KIMODO eval 时需修复上述问题。当前已有的评估数据可用于大方向的质量比较，但 condition frame 附近的指标（如 `boundary_accel_jump`、`mpjpe_masked`）不完全可靠。
