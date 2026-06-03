# PhysFlow 待办：训练语料中的"场景/物体依赖"动作污染问题

状态：**已记录，暂不解决**（2026-06-02）。当前正式训练 `physflow_online_adv_v1` 仍用未过滤全量语料。

## 问题

训练 prompt 来自 HumanML3D（`configs/experiments/physflow_kimodo_g1/physflow_text_train.jsonl`，11,241 条）。
其中相当一部分动作**依赖场景/物体/支撑面**，而这些信息只存在于"看不见的环境"里，动作数据本身只编码关节轨迹。
我们的 frozen judge 在**平地 MuJoCo**里 rollout，没有台阶/椅子/水/横梁，因此这类动作在物理上无法被正确执行或评分。

## 量化（扫 11,241 条 prompt）

| 风险级 | 占比 | 细分 |
|---|---|---|
| HIGH（支撑面/几何缺失，平地上物理不可能） | ~2.7% (302) | crawl 1.0% / sit-on-chair·bench 0.4% / swim·dive 0.4% / stairs·climb 0.3% / ramp·slope 0.2% / balance beam 0.2% / lie down 0.1% / hurdle·vault 0.1% |
| MEDIUM（物体缺失，肢体轨迹仍可跟踪） | ~5.7% (642) | pick up·carry / throw·catch·ball / push·pull / open door |
| HIGH∪MED | ~8.4% (943) | — |
| 宽口径含场景/物体词（含误命中如 "off the ground"、"six steps"） | ~31.6% | 多为误命中 |

典型例：`runs, jumps over something`、`army crawls across the ground`、`walking on a balance beam`、`sitting on a chair`、`a swimming motion while standing`。

## 为什么对本方法有害（不只是噪声）

1. **奖励错误归因**：爬台阶/坐椅子在平地上必然 completion 低/摔倒，但原因是"缺场景"而非"动作生成差"。
2. **反向 reward hacking / 能力退化**：best-of-N 会在同样"需要台阶"的候选里挑**最不像爬台阶（最接近平地走）**的去 SFT → 教 generator 把 "climb stairs" 退化成 "walk flat"。
3. **坏的 tracker 参考**：若进 tracker 池，相当于给站立双足控制器喂"根部凭空升高/脚踩虚步"的不可行参考。
4. 与论文 E3（retarget confound）**不同**：这是"动作语义依赖场景、而仿真无场景"的**可行性 confound**，是独立因子。

## 拟定对策（待实施）

1. prompt 可行性过滤：分 `flat-ground-feasible` vs `scene/object-dependent`（关键词规则 + 小 LLM 分类器）。
   - feasible → 进物理奖励在线闭环；
   - scene-dependent → 仅用于 generator 标准 T2M 文本对齐目标（保住能力），**不参与物理奖励选择**。
2. 产出 `physflow_text_train.feasible.jsonl` + 被剔除清单 + 精确占比。
3. 主结果用过滤版；**未过滤版保留作对照 ablation**（量化"场景污染"对 trackability 的影响）。
4. paper：在 experiments 里显式写成 feasibility/scene-dependence confound + anti-degeneration guard，与 E3 并列。
5. 长期（超出当前范围）：仿真加地形/外部支撑，或给 tracker 加 terrain 条件。

## 关联

- 正式训练：`configs/physflow/physflow_online_adv_v1.py`（未过滤）
- 周期评测：`scripts/embodied/physflow_periodic_eval.py`
- 论文实验：`papers/PhysFlow/sec/sec_4_experiments.tex`（E3 retarget confound 旁应增补此条）
