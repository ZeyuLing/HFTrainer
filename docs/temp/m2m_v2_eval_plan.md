# M2M v2 评测方案

> 日期: 2026-04-14 (更新: 2026-04-14)
> 数据来源: `/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/`
> Datalist 位置: `data/eval/m2m_v2/`
> E1 (T2M) 不在本方案中，沿用已有的 `data/eval/t2m/251125_yiran_subset.json`

---

## 变更记录

- **v3**: 每个任务增加详细的动作类型分布表，包含类别占比、不同动作数、代表动作英文名。分布数据同步写入 datalist JSON 的 `meta.category_distribution_detail`，供 Eval Dashboard 网站直接读取。
- **v2**: E9 改用真实低质量数据（不再使用合成损坏）；所有 datalist 增加 `caption_en` 英文字段；每个任务的动作选择更精细化并文档化选择理由；E6 增加详细的方法论解释

---

## 数据概况

### Private 动捕数据

源目录包含 27 个子目录、**4218 条** mocap 数据（SMPLH 格式，156 维 poses + 3 维 trans），涵盖：

| 类别 | 数量 | 代表动作 |
|------|------|---------|
| sports_ball | 714 | 足球、篮球、排球、乒乓球、网球、羽毛球、棒球 |
| sports_other | 651 | 高尔夫、滑雪、冲浪、杠铃、深蹲、跳远 |
| other | 744 | 杂项（未归类的动作） |
| combat | 435 | 剑术、枪械、盾牌、暗器、格斗 |
| daily_object | 367 | 搬运、切菜、扫地、拖地、叠衣服 |
| daily_stand | 255 | 走路、鞠躬、挥手、握手、拥抱 |
| sitting | 254 | 坐姿动作（划桨、游戏、写字、缝衣） |
| gesture | 223 | OK、比心、点赞、胜利、手势计数 |
| performance | 173 | 唱歌、话剧、说唱 |
| game_hobby | 130 | 跳房子、跳皮筋、VR游戏、麻将、棋 |
| grooming | 118 | 刷牙、梳头、化妆、剪指甲 |
| expression | 102 | 大笑、哭泣、愤怒、惊讶 |
| writing | 37 | 毛笔、钢笔、白板 |
| water_sport | 11 | 皮划艇、划桨板、帆船 |

所有数据通过**跨类别轮询**选取，保证每个 datalist 中各类别分布均匀，每个动作最多选 2 条避免重复。

### 低质量训练数据 (E9专用)

来源: `data/hymotion_m2m_refine_data/data_quality_list/low_quality.json`，共 **85,191 条**低质量动作（由 MotionQualityChecker 自动检测），缺陷分布：

| 缺陷类型 | 数量 | 说明 |
|---------|------|------|
| foot_sliding | 54,178 | 接地脚滑动 |
| candy_wrapper | 10,459 | 手臂反向扭曲 |
| joint_jump | 9,605 | 关节突变 |
| jitter | 8,190 | 稳定段抖动 |
| rotation_velocity | 7,510 | 旋转速度过大 |
| 其他 | ~6,000 | neck, ankle_x, arm_penetration 等 |

---

## 英文 Caption 策略

所有 datalist 的每条样本包含 `caption_en` 字段，已全部翻译为英文（覆盖率 100%）。翻译策略：
- 125 个中文描述性动作名 → 人工翻译为准确英文描述
- 117 个不透明编码名 → 通用描述（motion clip / internal mocap capture）
- ~250 个高频动作关键词 → 字典精确匹配

对于 E11/E13 等需要 text 条件的任务，`caption` 字段直接使用 `caption_en`。

---

## 各任务评测方案

### E2: Motion In-Betweening

| 项 | 值 |
|---|---|
| Datalist | `data/eval/m2m_v2/eval_e2_inbetween.json` |
| 样本数 | 120 |
| 帧数范围 | 60-360 |

**动作选择策略**: 全身明显运动的动作，有清晰的起始/结束姿态。排除 writing 类（动作太微小）。

**动作类型分布**:

| 类别 | 数量 | 占比 | 不同动作数 | 代表动作 (English) |
|------|------|------|-----------|-------------------|
| combat | 10 | 8.3% | 10 | throw a dart forward from a low chest position, throw a dart and hold the follow-through, holster pistol to chest and inspect the side ... |
| daily_object | 9 | 7.5% | 9 | put down an object, throw a paper airplane with a quick arm whip, throw a paper airplane and hold the follow-through pose ... |
| daily_stand | 9 | 7.5% | 9 | handshake, cheer, salute ... |
| expression | 9 | 7.5% | 9 | lean forward provocatively, act out recoiling in fear, raise head proudly ... |
| game_hobby | 9 | 7.5% | 9 | declare pong in mahjong and reveal the set, shuffle playing cards, shake dice ... |
| gesture | 9 | 7.5% | 9 | prayer hands, victory sign, hold up two fingers while presenting second point ... |
| grooming | 9 | 7.5% | 9 | apply makeup, comb hair, button up a shirt ... |
| other | 9 | 7.5% | 9 | reach for a book from a bookshelf while standing, pat shoulder encouragingly, fist pump celebration ... |
| performance | 9 | 7.5% | 9 | jump rope game, hopscotch, singing |
| sitting | 9 | 7.5% | 9 | kayak, sitting |
| sports_ball | 9 | 7.5% | 9 | pass a ball, table tennis, serve a ball ... |
| sports_other | 9 | 7.5% | 9 | golf swing, surfing, skiing ... |
| water_sport | 7 | 5.8% | 2 | sailing |
| kneeling_squat | 4 | 3.3% | 1 | kneel on one knee and kiss hand in greeting |

帧数统计: min=180, max=359, mean=254.2, median=257

**评测设定**:
- **Setting A (经典)**: 保留前 5 帧 + 后 5 帧，生成中间
- **Setting B (长距)**: 仅选 >200 帧的样本，保留前 5 + 后 5
- **Setting C (非对称)**: 保留前 30 + 后 5

**指标**: MPJPE(masked), MPJPE(unmasked), Boundary Accel Jump, Jitter, Bone CV, Foot Skating

---

### E3: Keyframe Interpolation

| 项 | 值 |
|---|---|
| Datalist | `data/eval/m2m_v2/eval_e3_keyframe.json` |
| 样本数 | 120 |
| 帧数范围 | 90-600 |

**动作选择策略**: 较长的连续全身运动序列。需要 >90 帧才能在 Setting A (30帧间隔) 有足够关键帧。

**动作类型分布**:

| 类别 | 数量 | 占比 | 不同动作数 | 代表动作 (English) |
|------|------|------|-----------|-------------------|
| combat | 9 | 7.5% | 9 | throw a dart forward from a low chest position, throw a dart and hold the follow-through, holster pistol to chest and inspect the side ... |
| daily_object | 9 | 7.5% | 9 | put down an object, throw a paper airplane with a quick arm whip, throw a paper airplane and hold the follow-through pose ... |
| daily_stand | 9 | 7.5% | 9 | handshake, cheer, salute ... |
| expression | 9 | 7.5% | 9 | lean forward provocatively, act out recoiling in fear, raise head proudly ... |
| game_hobby | 9 | 7.5% | 9 | declare pong in mahjong and reveal the set, shuffle playing cards, shake dice ... |
| gesture | 9 | 7.5% | 9 | prayer hands, victory sign, hold up two fingers while presenting second point ... |
| grooming | 9 | 7.5% | 9 | apply makeup, comb hair, button up a shirt ... |
| other | 8 | 6.7% | 8 | reach for a book from a bookshelf while standing, pat shoulder encouragingly, fist pump celebration ... |
| performance | 8 | 6.7% | 8 | jump rope game, hopscotch, singing |
| sitting | 8 | 6.7% | 8 | kayak, sitting |
| sports_ball | 8 | 6.7% | 8 | pass a ball, table tennis, serve a ball ... |
| sports_other | 8 | 6.7% | 8 | golf swing, surfing, skiing ... |
| writing | 8 | 6.7% | 6 | writing, fill a fountain pen with an ink converter, whiteboard ... |
| water_sport | 5 | 4.2% | 2 | sailing |
| kneeling_squat | 4 | 3.3% | 1 | kneel on one knee and kiss hand in greeting |

帧数统计: min=180, max=355, mean=250.6, median=244

**评测设定**:
- **Setting A**: 每 30 帧保留一个关键帧 (1s@30fps)
- **Setting B**: 每 60 帧 (2s)
- **Setting C**: 每 15 帧 (0.5s)
- **Setting D**: 随机间距 (10-90帧)

**指标**: MPJPE(masked), MPJPE(unmasked), Jitter, Bone CV, Foot Skating

---

### E4: End-Effector Constraint

| 项 | 值 |
|---|---|
| Datalist | `data/eval/m2m_v2/eval_e4_end_effector.json` |
| 样本数 | 100 |
| 帧数范围 | 60-360 |

**动作选择策略**: 手/脚有明确目标位置的动作。优先: sports_ball, daily_object, combat, gesture, grooming, writing, game_hobby。

**动作类型分布**:

| 类别 | 数量 | 占比 | 不同动作数 | 代表动作 (English) |
|------|------|------|-----------|-------------------|
| combat | 15 | 15.0% | 15 | throw a dart forward from a low chest position, throw a dart and hold the follow-through, holster pistol to chest and inspect the side ... |
| daily_object | 15 | 15.0% | 15 | put down an object, throw a paper airplane with a quick arm whip, throw a paper airplane and hold the follow-through pose ... |
| game_hobby | 15 | 15.0% | 15 | whip a spinning top, declare pong in mahjong and reveal the set, sidearm throw a boomerang ... |
| gesture | 15 | 15.0% | 15 | point at something and tap it a few times, hold up four fingers, victory sign ... |
| grooming | 14 | 14.0% | 12 | pull on a hoodie overhead, comb hair, pull off a hoodie overhead ... |
| sports_ball | 14 | 14.0% | 14 | baseball, soccer, spike a ball ... |
| writing | 12 | 12.0% | 6 | dip brush in ink and adjust the tip, writing, fill a fountain pen with an ink converter ... |

帧数统计: min=181, max=359, mean=254.0, median=251

**评测设定**:
- **Setting A**: 右手腕位置约束，每 10 帧
- **Setting B**: 双脚踝位置约束，每 15 帧
- **Setting C**: 右手+左脚位置约束，每 15 帧
- **Setting D**: Text + 首帧全身 keypose
- **Setting E**: Text + 首末帧全身 keypose

**约束来源**: 从 GT motion 做 FK 得到 world-space joint positions，在指定帧提取作为约束。
**指标**: EE Error Mean/Max, Jitter, Bone CV, MPJPE(masked)

---

### E5: Trajectory Following

| 项 | 值 |
|---|---|
| Datalist | `data/eval/m2m_v2/eval_e5_trajectory.json` |
| 样本数 | 100 |
| 帧数范围 | 60-600 |

**动作选择策略**: **必须**有显著的 root (pelvis) 在 XZ 平面上的移动。排除坐姿、站立手势等原地动作。

**动作类型分布**:

| 类别 | 数量 | 占比 | 不同动作数 | 代表动作 (English) |
|------|------|------|-----------|-------------------|
| combat | 15 | 15.0% | 15 | throw a dart forward from a low chest position, throw a dart and hold the follow-through, holster pistol to chest and inspect the side ... |
| daily_stand | 14 | 14.0% | 14 | cheer, nod, wave hand ... |
| performance | 14 | 14.0% | 14 | jump rope game, hopscotch |
| sports_ball | 14 | 14.0% | 14 | slam dunk, tennis, shoot (soccer) ... |
| sports_other | 14 | 14.0% | 14 | golf swing, surfing, push-up ... |
| other | 12 | 12.0% | 4 | march like a soldier, walk a tightrope, shuffle step side to side ... |
| game_hobby | 11 | 11.0% | 4 | chess/board game |
| sitting | 6 | 6.0% | 1 | sitting |

帧数统计: min=180, max=360, mean=255.2, median=261

**评测设定**:
- **Setting A**: 密集轨迹 (每帧的 root XZ)
- **Setting B**: 稀疏路径点 (每 30 帧)
- **Setting C**: 轨迹 + heading (root rotation)
- **Setting D**: 仅 heading (每 30 帧)

**指标**: Trajectory ADE, Trajectory FDE, Foot Skating, Jitter, Bone CV

---

### E6: Foot Ground Constraint

| 项 | 值 |
|---|---|
| Datalist | `data/eval/m2m_v2/eval_e6_foot_ground.json` |
| 样本数 | 100 |
| 帧数范围 | 60-360 |

**动作选择策略**: 站立/行走/跑步动作，脚部有明显的接地-腾空交替。

**动作类型分布**:

| 类别 | 数量 | 占比 | 不同动作数 | 代表动作 (English) |
|------|------|------|-----------|-------------------|
| combat | 17 | 17.0% | 17 | throw a dart forward from a low chest position, throw a dart and hold the follow-through, holster pistol to chest and inspect the side ... |
| daily_stand | 17 | 17.0% | 17 | cheer, nod, wave hand ... |
| game_hobby | 17 | 17.0% | 17 | chess/board game, play VR rhythm game while standing, declare pong in mahjong and reveal the set ... |
| performance | 17 | 17.0% | 17 | jump rope game, hopscotch, sneer contemptuously ... |
| sports_ball | 16 | 16.0% | 16 | table tennis, soccer, badminton ... |
| sports_other | 16 | 16.0% | 16 | golf swing, kayak, skiing ... |

帧数统计: min=180, max=357, mean=262.5, median=259

**方法论详解**:

Foot Ground Constraint 的核心思想是：**告诉模型"哪些帧脚应该在地面上"，然后检查生成的动作是否遵守了这个约束**。

工作流程:
1. **检测接触帧**: 对 GT motion 做 FK，得到踝关节的世界坐标 `(x, y, z)`。当 ankle Y < 5cm 时，标记为"接触帧"
2. **构建约束 mask**: 在接触帧上，将对应的 ankle 维度设为 mask=0（已知），告诉模型这些帧的脚部状态是固定的
3. **模型生成**: 模型在约束下生成完整动作，需要在接触帧保持脚部贴地
4. **评估**: 检查生成动作在接触帧是否有穿透(penetration)、浮空(float)、滑动(skating)

**两种约束模式**:
- **Rotation mode (135-dim)**: 在接触帧保留 GT 的 ankle rotation (rot6d)。模型需要生成与 GT 旋转一致的脚踝姿态，间接保证脚部贴地。
- **Position mode (198-dim)**: 在接触帧直接约束 ankle 的世界坐标位置。子模式：Y only / XZ only / XYZ

**评测设定**:
- **Setting A_rot**: GT 接触帧，rotation mode (135d)
- **Setting B_rot**: 全帧，rotation mode
- **Setting C_pos_y**: GT 接触帧，position Y only
- **Setting D_pos_xz**: GT 接触帧，position XZ
- **Setting E_pos_xyz**: GT 接触帧，position XYZ

**指标**: Foot Penetration, Foot Float, Foot Skating, Jitter, Bone CV

---

### E7: First-Frame Continuation

| 项 | 值 |
|---|---|
| Datalist | `data/eval/m2m_v2/eval_e7_first_frame.json` |
| 样本数 | 100 |
| 帧数范围 | 60-300 |

**动作选择策略**: 需要多样化的第 0 帧起始姿态。全类别均匀采样。

**动作类型分布**:

| 类别 | 数量 | 占比 | 不同动作数 | 代表动作 (English) |
|------|------|------|-----------|-------------------|
| combat | 8 | 8.0% | 8 | thrust attack, aim a pistol, cleave ... |
| daily_object | 8 | 8.0% | 8 | apply skincare product, kayak, pick up a potted plant from the table with both hands ... |
| daily_stand | 8 | 8.0% | 8 | hug, clap, shrug ... |
| expression | 7 | 7.0% | 7 | laugh, sitting, tired ... |
| game_hobby | 7 | 7.0% | 7 | throw a boomerang, whip a spinning top, chess/board game ... |
| gesture | 7 | 7.0% | 7 | make an upside-down OK gesture, hold up index finger to emphasize the first point, hold up two fingers while presenting second point ... |
| grooming | 7 | 7.0% | 7 | clip toenails, gargle with head tilted back, pull off a hoodie overhead ... |
| other | 7 | 7.0% | 7 | spread arms to keep balance, quickly glance to the side, catch an object ... |
| performance | 7 | 7.0% | 7 | hopscotch, singing, jump rope game |
| sitting | 7 | 7.0% | 7 | kayak, sitting |
| sports_ball | 7 | 7.0% | 7 | juggle a ball, tennis, baseball ... |
| sports_other | 7 | 7.0% | 7 | shot put, surfing, paddleboard ... |
| writing | 6 | 6.0% | 6 | dip brush in ink and adjust the tip, fill a fountain pen with an ink converter, writing ... |
| water_sport | 5 | 5.0% | 2 | sailing |
| kneeling_squat | 2 | 2.0% | 1 | kneel on one knee and kiss hand in greeting |

帧数统计: min=180, max=300, mean=232.7, median=229

**评测设定**:
- 保留第 0 帧（全身 135d mask=0），其余帧 mask=1
- Caption 模型额外提供 `caption_en` 作为 text 条件

**指标**: MPJPE(unmasked), Jitter, Bone CV, Foot Skating

---

### E8: Loop Animation

| 项 | 值 |
|---|---|
| Datalist | `data/eval/m2m_v2/eval_e8_loop.json` |
| 样本数 | 80 |
| 帧数范围 | 60-300 |

**动作选择策略**: **必须**是自然循环/重复的动作。排除一次性动作。

**动作类型分布**:

| 类别 | 数量 | 占比 | 不同动作数 | 代表动作 (English) |
|------|------|------|-----------|-------------------|
| combat | 10 | 12.5% | 9 | reload a pistol held with both hands, run, elbow strike ... |
| daily_stand | 10 | 12.5% | 10 | shake head, wave hand, walk ... |
| other | 10 | 12.5% | 10 | take a photo, tap foot to the beat, grab-and-pat shoulder gesture ... |
| performance | 10 | 12.5% | 10 | jump rope game, hopscotch |
| sitting | 10 | 12.5% | 10 | sitting, kayak |
| sports_ball | 10 | 12.5% | 10 | tennis, soccer, shoot (soccer) ... |
| sports_other | 9 | 11.2% | 9 | golf swing, paddleboard, kayak ... |
| game_hobby | 6 | 7.5% | 2 | draw a tile and slap it on the table, shake dice |
| expression | 5 | 6.2% | 1 | angry |

帧数统计: min=180, max=299, mean=229.7, median=223

**评测设定**:
- **Setting A**: 首=尾帧约束（mask 首尾帧 = 0，其余 = 1）
- **Setting B**: Loop + 密集轨迹
- **Setting C**: Loop + 稀疏路径点

**指标**: Loop Position Error, Loop Velocity Error, Jitter, Bone CV

---

### E9: Motion Repair

| 项 | 值 |
|---|---|
| Datalist | `data/eval/m2m_v2/eval_e9_repair.json` |
| 样本数 | 120 (每种缺陷30条) |
| 数据来源 | `data/hymotion_m2m_refine_data/data_quality_list/low_quality.json` |
| 帧数范围 | 60-600 |

**与之前版本的关键区别**: 不再使用合成损坏，而是使用 MotionQualityChecker 检测出的**真实低质量动作**。

**缺陷类型分布**:

| 类别 | 数量 | 占比 | 不同动作数 | 代表动作 (English) |
|------|------|------|-----------|-------------------|
| defect_jitter | 30 | 25.0% | 29 | internal mocap capture, motion clip, punch |
| defect_joint_jump | 30 | 25.0% | 30 | attack, motion clip, kick ... |
| defect_foot_sliding | 30 | 25.0% | 30 | motion clip, walk, pistol action ... |
| defect_candy_wrapper | 30 | 25.0% | 28 | internal mocap capture, walk, motion clip ... |

帧数统计: min=61, max=590, mean=236.0, median=209

**评测设定**:
- **Setting A (自动检测+修复)**: quality checker 自动检测缺陷区域生成 mask → 模型修复
- **Setting B (Oracle mask)**: 直接使用已知的缺陷帧/关节作为 mask → 模型修复
- **Setting C (全生成 baseline)**: mask=全1，完全重新生成 → 对照组

**评测流程**:
1. 输入缺陷动作 → 模型修复 → 输出修复后动作
2. 对修复后动作重新运行 quality checker → 统计缺陷消除率
3. 对比修复前后的指标改善

**指标**: 缺陷消除率(defect_fix_rate), Jitter, Bone CV, Foot Skating, Foot Penetration

---

### E10: Part-Level Control

| 项 | 值 |
|---|---|
| Datalist | `data/eval/m2m_v2/eval_e10_part_control.json` |
| 样本数 | 100 |
| 帧数范围 | 60-360 |

**动作选择策略**: 上下半身有明显不同运动模式的动作。

**动作类型分布**:

| 类别 | 数量 | 占比 | 不同动作数 | 代表动作 (English) |
|------|------|------|-----------|-------------------|
| combat | 8 | 8.0% | 8 | throw a dart forward from a low chest position, throw a dart and hold the follow-through, holster pistol to chest and inspect the side ... |
| daily_object | 7 | 7.0% | 7 | put down an object, throw a paper airplane with a quick arm whip, throw a paper airplane and hold the follow-through pose ... |
| daily_stand | 7 | 7.0% | 7 | handshake, cheer, salute ... |
| expression | 7 | 7.0% | 7 | lean forward provocatively, act out recoiling in fear, raise head proudly ... |
| game_hobby | 7 | 7.0% | 7 | declare pong in mahjong and reveal the set, shuffle playing cards, shake dice ... |
| gesture | 7 | 7.0% | 7 | prayer hands, victory sign, hold up two fingers while presenting second point ... |
| grooming | 7 | 7.0% | 7 | apply makeup, comb hair, button up a shirt ... |
| other | 7 | 7.0% | 7 | reach for a book from a bookshelf while standing, pat shoulder encouragingly, fist pump celebration ... |
| performance | 7 | 7.0% | 7 | jump rope game, hopscotch, singing |
| sitting | 7 | 7.0% | 7 | kayak, sitting |
| sports_ball | 7 | 7.0% | 7 | pass a ball, table tennis, serve a ball ... |
| sports_other | 7 | 7.0% | 7 | golf swing, surfing, skiing ... |
| writing | 7 | 7.0% | 6 | writing, fill a fountain pen with an ink converter, whiteboard ... |
| water_sport | 5 | 5.0% | 2 | sailing |
| kneeling_squat | 3 | 3.0% | 1 | kneel on one knee and kiss hand in greeting |

帧数统计: min=181, max=348, mean=247.8, median=238

**评测设定**:
- **Setting A**: 保留上半身（spine, neck, head, arms），重生成下半身
- **Setting B**: 保留下半身（hips, knees, ankles, feet），重生成上半身
- **Setting C**: 仅保留 root（pelvis rotation + translation），重生成全部 pose

**指标**: MPJPE(unmasked), Jitter, Bone CV, Foot Skating

---

### E11: Caption Completion

| 项 | 值 |
|---|---|
| Datalist | `data/eval/m2m_v2/eval_e11_caption_completion.json` |
| 样本数 | 100 |
| 帧数范围 | 60-360 |

Caption 使用英文 (`caption_en`)，来自中文动作名的翻译。

**动作类型分布**:

| 类别 | 数量 | 占比 | 不同动作数 | 代表动作 (English) |
|------|------|------|-----------|-------------------|
| combat | 8 | 8.0% | 8 | throw a dart forward from a low chest position, throw a dart and hold the follow-through, holster pistol to chest and inspect the side ... |
| daily_object | 7 | 7.0% | 7 | put down an object, throw a paper airplane with a quick arm whip, throw a paper airplane and hold the follow-through pose ... |
| daily_stand | 7 | 7.0% | 7 | handshake, cheer, salute ... |
| expression | 7 | 7.0% | 7 | lean forward provocatively, act out recoiling in fear, raise head proudly ... |
| game_hobby | 7 | 7.0% | 7 | declare pong in mahjong and reveal the set, shuffle playing cards, shake dice ... |
| gesture | 7 | 7.0% | 7 | prayer hands, victory sign, hold up two fingers while presenting second point ... |
| grooming | 7 | 7.0% | 7 | apply makeup, comb hair, button up a shirt ... |
| other | 7 | 7.0% | 7 | reach for a book from a bookshelf while standing, pat shoulder encouragingly, fist pump celebration ... |
| performance | 7 | 7.0% | 7 | jump rope game, hopscotch, singing |
| sitting | 7 | 7.0% | 7 | kayak, sitting |
| sports_ball | 7 | 7.0% | 7 | pass a ball, table tennis, serve a ball ... |
| sports_other | 7 | 7.0% | 7 | golf swing, surfing, skiing ... |
| writing | 7 | 7.0% | 6 | writing, fill a fountain pen with an ink converter, whiteboard ... |
| water_sport | 5 | 5.0% | 2 | sailing |
| kneeling_squat | 3 | 3.0% | 1 | kneel on one knee and kiss hand in greeting |

帧数统计: min=181, max=348, mean=247.8, median=238

**评测设定**:
- **Setting inbetween**: In-betweening (首末 5 帧) + English caption
- **Setting keyframe**: 每 30 帧关键帧 + English caption

**指标**: MPJPE(masked), Jitter, Bone CV

---

### E13: Multi-Prompt Generation

| 项 | 值 |
|---|---|
| Datalist | `data/eval/m2m_v2/eval_e13_multi_prompt.json` |
| 样本数 | 80 |
| 帧数范围 | 60-240 |

每段使用不同的 English caption 作为 prompt。

**动作类型分布**:

| 类别 | 数量 | 占比 | 不同动作数 | 代表动作 (English) |
|------|------|------|-----------|-------------------|
| combat | 6 | 7.5% | 6 | draw a dart from waist and aim, run, shooting ... |
| daily_object | 6 | 7.5% | 6 | pick up a cup from the table with one hand, throw a paper airplane from a high overhand position, whiteboard ... |
| daily_stand | 6 | 7.5% | 6 | stagger drunkenly, walk, beckon someone to come over ... |
| expression | 6 | 7.5% | 6 | surprised, tired, lean forward provocatively ... |
| game_hobby | 6 | 7.5% | 6 | draw a tile from the wall and examine it, draw a tile without looking and discard it, declare kong in mahjong and draw a replacement tile ... |
| gesture | 6 | 7.5% | 6 | point backward with thumb, prayer hands, hold up two fingers while presenting second point ... |
| grooming | 6 | 7.5% | 6 | shave with an electric razor, wash hair with head tilted back, comb hair ... |
| other | 6 | 7.5% | 6 | spread both hands wide dramatically, struggle to reach down a package from a high shelf, practice forehand swings without a ball ... |
| performance | 6 | 7.5% | 6 | jump rope game, hopscotch, sing a slow ballad ... |
| sitting | 5 | 6.2% | 5 | sitting, kayak |
| sports_ball | 5 | 6.2% | 5 | baseball, soccer, tennis ... |
| sports_other | 5 | 6.2% | 5 | skiing, kayak, paddleboard ... |
| writing | 5 | 6.2% | 5 | writing, whiteboard, dip brush in ink and adjust the tip ... |
| water_sport | 4 | 5.0% | 2 | sailing |
| kneeling_squat | 2 | 2.5% | 1 | kneel on one knee and kiss hand in greeting |

帧数统计: min=180, max=240, mean=205.8, median=202

**评测设定**:
- **Setting A**: 3 个 prompt（从不同样本取 caption_en），5 帧 overlap
- **Setting B**: 5 个 prompt，5 帧 overlap
- **Setting C**: 10 个 prompt，10 帧 overlap

**运行方式**: 自回归滑窗——每段用前一段末尾 N 帧做 in-between 条件，生成下一段。
**指标**: Jitter, Bone CV, Foot Skating, Segment Boundary Smoothness, Total Duration

---

## Datalist 文件结构

### 标准格式 (E2-E8, E10-E13)

```json
{
  "meta": {
    "task_id": "E2",
    "task_name": "Motion In-Betweening",
    "description": "...",
    "total_items": 120,
    "source": "/apdcephfs_cq11/.../Private",
    "category_distribution": {"combat": 10, "daily_object": 9, ...},
    "category_distribution_detail": {
      "combat": {
        "count": 10,
        "percent": 8.3,
        "unique_actions": 10,
        "example_actions": ["上挑肘击", "为手持炮进行折管式装填。", ...],
        "example_captions_en": ["elbow strike", "break-action reload", ...]
      }
    },
    "frame_stats": {"min": 180, "max": 359, "mean": 254.2, "median": 257},
    "min_frames": 60,
    "max_frames": 360
  },
  "data_list": [
    {
      "motion_path": "/apdcephfs_cq11/.../上篮_take_039.npz",
      "action_name": "上篮",
      "caption_en": "layup",
      "category": "sports_ball",
      "num_frames": 269,
      "fps": 30.0,
      "duration_sec": 8.97,
      "source": "dongming_20251224"
    }
  ]
}
```

### E9 修复格式

```json
{
  "meta": {
    "task_id": "E9",
    "task_name": "Motion Repair",
    "description": "...",
    "total_items": 120,
    "source": "data/hymotion_m2m_refine_data/data_quality_list/low_quality.json",
    "defect_distribution": {"jitter": 30, "joint_jump": 30, "foot_sliding": 30, "candy_wrapper": 30},
    "category_distribution_detail": {
      "defect_jitter": {
        "count": 30,
        "percent": 25.0,
        "unique_actions": 29,
        "example_actions": ["..."],
        "example_captions_en": ["internal mocap capture", "motion clip", ...]
      }
    }
  },
  "data_list": [
    {
      "motion_path": "data/hymotion_data/3D/.../motion.npz",
      "action_name": "...",
      "caption_en": "...",
      "category": "defect_jitter",
      "defect_type": "jitter",
      "all_defects": ["jitter"],
      "num_frames": 320,
      "fps": 30.0,
      "duration_sec": 10.67,
      "source": "low_quality_db"
    }
  ]
}
```

每条样本的 NPZ 文件包含: `poses (T, 156)`, `trans (T, 3)`, `betas (1, 16)`, `mocap_framerate`。评测脚本需要将 156 维 SMPLH poses 转换为 22 关节 rot6d (135 维) 后再送入模型。

---

## 重建命令

```bash
python3 tools/build_m2m_v2_eval_data.py
```

随机种子固定 (seed=42)，重复运行结果一致。
