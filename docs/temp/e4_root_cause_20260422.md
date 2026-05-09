# E4 黄色/蓝色不对齐 根因分析 (2026-04-22)

## 现象
- 黄色约束环（constraint ring）飘浮在空中
- 蓝色生成 mesh 在地面
- 两者在视觉上明显错位，切换 sample 现象持续

## 根因（实证数据）

E4 样本 Y 坐标分析 (uncond_local / A_rhand_sparse)：

| Sample | GT foot min Y | PRED foot min Y | PRED 问题 |
|--------|---------------|-----------------|-----------|
| 00000  | +0.001 m      | **-0.433 m**    | 脚穿地 43cm |
| 00001  | +0.001 m      | -0.015 m        | 脚穿地 1.5cm |
| 00002  | +0.002 m      | -0.013 m        | 脚穿地 1.3cm |

**即：生成 motion 的脚经常穿透地面（有时 43cm 深）**，而 GT motion 脚底紧贴地面（0.001m）。

## 前端坐标系问题

`canonicalizeGround()` 逻辑（task_detail.html:2524）：
```js
groundOffset = -min(pred_FK_Y)   // 对 sample 0：+0.433m
skeletonGroup.position.y = groundOffset
smplMesh.position.y = groundOffset
```

随后 `recomputeSMPLGroundFromMesh()` 又**单独覆盖** mesh：
```js
smplMesh.position.y = -min(LBS_mesh_Y)  // 用 LBS 蒙皮 min Y
// 不改 skeletonGroup.position.y 和 groundOffset
```

最终渲染位置：
| Element | Y 位置 | Sample 0 具体值 |
|---------|-------|----------------|
| Pred 蓝色 mesh | LBS-based offset → 脚底 Y=0 | mesh 被 LBS 单独修正到地面 |
| Pred 骨架 | pred_FK_Y + groundOffset = 脚底 Y=0 | FK 脚底被抬到 0 |
| **GT 黄色约束** | gt_FK_Y + groundOffset = gt_FK_Y + 0.433 | **gt_FK 的 0.001m → 被抬到 +0.434m** |

**所以黄色约束飘浮 ≈ 43cm = pred 脚下沉的深度**。

## 为什么之前的"解耦 mesh/skeleton"修复无效

2026-04-21 的 fix 解耦了 mesh 和 skeleton，但问题不在 mesh vs skeleton —— 问题在 **pred motion 和 GT motion 在同一 groundOffset 下不匹配**。GT 用 pred 的 groundOffset 必然飘浮。

## 正确修复

### Option A: GT 作为参考坐标系（推荐）
```js
if (TASK_ID === 'E4' && window._e4GtEE) {
    // Use GT's foot min Y as the canonical ground
    const gtMinY = computeGtFootMinY(window._e4GtEE);
    groundOffset = -gtMinY;   // usually ~ -0.001 (near zero)
} else {
    groundOffset = -pred_FK_minY;  // original behavior
}
```

优点：
- 黄色约束沿地面移动（正确）
- 蓝色 pred 骨架若穿地，**真实显示模型缺陷**（诚实的可视化）
- 删除 `recomputeSMPLGroundFromMesh()` 的 E4 override，让 mesh 与 skeleton/constraint 在同一世界坐标系

### Option B: 强行对齐所有元素用 pred ground（原方案）
视觉上"看起来都在地面"但 GT 约束位置错误。用户看到的错觉：pred 很好，GT 飘浮。**误导**。

## 实施

修改 `motion_annot_web/eval_dashboard/templates/task_detail.html`:

1. `canonicalizeGround()` 对 E4 task 时使用 GT-based min Y
2. 移除 E4 的 `recomputeSMPLGroundFromMesh()` 调用（统一坐标系）
3. `drawConstraintMarkers()` 保持 `+groundOffset` 逻辑不变

## 用户视觉体验变化

- **之前**：mesh 在地、constraint 飘浮（看起来是可视化 bug）
- **之后**：constraint 在地、mesh 可能穿地（诚实反映模型 foot-sliding/penetration 问题）

这将让 E4 的 `ee_error` 指标与可视化一致：
- 之前 ee_error 高但视觉上 mesh 与 constraint 近在咫尺（LBS 碰巧对上）
- 之后 ee_error 高的 case 会看到 mesh 的脚/手与 constraint 明显错位
