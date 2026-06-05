# PhysFlow Online Adversarial — Iteration Log

> **完整旧版（478 行，含已废弃 position-aware / rehearsal / overfit 细节）**  
> `docs/temp/physflow_online_adversarial_iteration_log_BACKUP_20260603.md`  
> 临时空白粘贴区：`docs/temp/_scratch.md`

## 2026-06-05 Tracker Benchmark / Guarded Replay 主线

当前判定标准改为：**PhysFlow 训练后的 G1 tracker 必须在统一 tracker-sim benchmark 上超过官方原始 pretrained checkpoint**，不能再与另一个自训 checkpoint 互相比。

已确认的负例：

| checkpoint | LAFAN1-G1 success | AMASS-G1 success | 结论 |
|---|---:|---:|---|
| official `g1-bones-deploy` | 0.800 | 0.879 | 原始 pretrained tracker baseline |
| `physflow_rehearsal_v2` | 0.600 | 0.767 | 相对官方负优化 |

当前正例：

| checkpoint | LAFAN1-G1 success | AMASS-G1 success | AMASS GT mean mm | AMASS jerk | 结论 |
|---|---:|---:|---:|---:|---|
| official `g1-bones-deploy` | 0.800 | 0.878797 | 686.815 | 2460.63 | 原始 pretrained tracker baseline |
| `physflow0605h` | 0.800 | 0.880603 | 602.642 | 2228.48 | AMASS full 上首次同时提升 success / mean GT / jerk / failure rate；LAFAN success 持平 |

LAFAN1-G1 checkpoint sweep（40 motions, 600 sim steps）：

| checkpoint | success | GT error mm | jerk | 结论 |
|---|---:|---:|---:|---|
| official `g1-bones-deploy` | 0.800 | 745.9 | 2320.0 | 论文表主 baseline |
| `warmstart_epoch0` | 0.825 | 769.5 | 2381.9 | 与官方模型权重相同，用于训练 warmstart，不作为方法提升 |
| `warmstart_warmopt` | 0.800 | 724.7 | 2358.9 | 权重同官方，但 optimizer metadata 不适合训练 resume |
| `physflow_rehearsal_v2` | 0.575 | 1294.8 | 3189.5 | 明显负优化 |
| `mix_lr1e6` | 0.800 | 667.0 | 2061.0 | LAFAN 误差下降，但属于旧实验，不能替代新方案结论 |
| `mix_lr5e6` | 0.800 | 739.4 | 2262.8 | 基本持平 |
| `jump_overfit_reset` | 0.175 | 672.2 | 2284.1 | 窄跳跃 overfit，通用跟踪能力崩坏 |

结果路径：`output/lafan1_g1_proto_baseline_eval/lafan1_g1_ckpt_sweep_600step_0605b/summary.md`。

根因复查：

- 旧 `rehearsal / mix / jump` 系列的训练入口实际上把 `native / adversarial / jump` 都指到了小规模 KIMODO rehearsal pool（约 72 条），没有真正的 native replay。
- 这会把官方 tracker 的 AMASS/LAFAN 分布能力冲掉，因此 Tracker after 看起来经常不如 Tracker before。
- 官方 `g1_phuma_train.yaml` 有 67,953 条索引，但本地并没有对应全量原始 motion 文件；目前可直接用于 replay 的是真实存在且已转换的 AMASS-G1 motion shards。

新的 guarded adversarial 训练约束：

| 项 | 当前设置 |
|---|---|
| entrypoint | `scripts/embodied/run_guarded_adversarial_tracker_train.sh` |
| manifest builder | `scripts/embodied/build_weighted_motion_manifest.py` |
| warmstart | `output/physflow_kimodo_g1/checkpoints/g1_released_warmstart_epoch0.ckpt`，模型权重与官方 `g1-bones-deploy` 完全一致 |
| native replay | `output/amass_g1_proto_baseline_eval/debug2_20260604_1904_wxyz_4gpu/motion_shards` |
| adversarial pool | `output/physflow_kimodo_g1/physflow_g1_released_rehearsal_v1_pool` |
| jump pool | `output/physflow_kimodo_g1/jump_hml3d40_noscene_mn1500_reset_env256_40m_noproj/physflow_jump_hml3d40_noscene_mn1500_reset_env256_40m_noproj_run/proto` |
| manifest | `output/guarded_adversarial_tracker/_manifest_w095_adv004_jump001_seed0_containerpath/weighted_motion_manifest.yaml` |
| sampling mass | AMASS native 0.95 / adversarial 0.04 / jump 0.01 |
| LR | actor 1e-6, critic/disc 5e-6（0605d+ smoke） |

下一轮只开 1 台 8 卡 smoke，训练后立即用 LAFAN1-G1 和 AMASS-G1 与 official pretrained 做对照。如果仍负优化，优先改数据与目标：进一步降低 LR、过滤 bad KIMODO rollout、降低 adversarial weight 或改为离线 adversarial fine-tune，而不是继续扩大训练规模。

Guarded smoke 运行记录：

| task | status | 说明 |
|---|---|---|
| `physflow_guarded_adv_w095_lr1e6_0605a` | failed early | manifest 使用 `Path.resolve()` 写成 `/apdcephfs/AILab_DHA/...`，Taiji 容器不可读 |
| `physflow_guarded_adv_w095_lr1e6_0605b` | stopped | container-safe manifest 生效，但 entrypoint 未传 `--ngpu`，Fabric 显示 `devices: 1`，占 8 卡只训 1 卡 |
| `physflow_guarded_adv_w095_lr1e6_0605c` | stopped | 修复 entrypoint：增加 `NGPU` 并传给 `train_agent.py --ngpu 8`；Fabric 已确认 `devices: 8`，但 warmstart 后触发 initial full-eval，Epoch 0 后长时间无 checkpoint |
| `physflow_guarded_adv_w095_lr1e6_0605d` | failed early | 新增 `--skip-initial-eval` 后启动到建环境阶段；8 rank 均已读取 4208 motions，但 IsaacGym `_create_simulation` 出现 CUDA illegal memory access / invalid resource handle，疑似每 rank 256 env 的 GPU pipeline 稳定性问题 |
| `physflow_guarded_adv_w095_lr1e6_env128_0605e` | failed at epoch 0 train | `NUM_ENVS=128` 成功过 IsaacGym env creation（8 rank 创建 128 env/rank，并完成 Epoch 0 collect），但 AMP discriminator 默认 `discriminator_batch_size=4096`，而 env128 rollout 只有 2048，TensorDict 报 `batch dimension mismatch` |
| `physflow_guarded_adv_w095_lr1e6_env128_nodr_pack_0605f` | success / evaluating | `NUM_ENVS=128`、`BATCH_SIZE=2048`、显式 `agent.amp_parameters.discriminator_batch_size=2048`；使用 no-DR smoke config，`PACK_MOTION_LIB=1`，成功完成 Epoch 0-14 并产出 `last.ckpt`。自然结束原因：warmstart 已有 `step_count`，final checkpoint `step_count=491520`，接近 `training_max_steps=500000` |
| `physflow_tracker_eval_0605f_lafan_amass` | running | 复用已有 LAFAN/AMASS motion shard cache，在 1 台 8 卡上顺序评估 official `g1-bones-deploy` 与 0605f final checkpoint |

修复文件：`scripts/embodied/build_weighted_motion_manifest.py`、`scripts/embodied/run_guarded_adversarial_tracker_train.sh`。修复后 manifest：`output/guarded_adversarial_tracker/_manifest_w095_adv004_jump001_seed0_containerpath/weighted_motion_manifest.yaml`，4208 条 motion，bad prefix 0，missing 0。

0605f 训练内曲线（TensorBoard event，15 epochs）：`info/episode_reward` 41.84→438.01，`rewards/task_rewards` 3.045→3.198，`env/terminate_mean` 0.00659→0.00186，`losses/discriminator_loss` 0.414→0.342。外部 benchmark 仍是最终准绳。

## 目标

在线对抗闭环：用 **G1 tracker（judge）** 给 **KIMODO-G1 T2M** 打物理/可跟踪奖励，生成器用 **RAFT** 优化。

```
prompt → KIMODO-G1 CSV → .motion → MuJoCo ONNX tracker → adversarial_score
```

**评测重点**：paired `base` vs `PhysFlow 优化后` 的 **物理真实性与机器人可跟踪性**（非 HumanML3D FID/R-Precision）。

**标准测试集**：`configs/experiments/physflow_kimodo_g1/physflow_bench_hml3d_test.jsonl`（40 prompts）。

---

## 当前主线（2026-06）

| 项 | 路径 / 说明 |
|----|-------------|
| 生成器训练基配置 | `configs/physflow/physflow_online_adv_v3.py` |
| v3 单机 8×V100 | `work_dirs/physflow_online_adv_v3`，**已完成** `checkpoint-iter_3000` |
| **多机正式训练** | `configs/physflow/physflow_online_adv_mn.py` → `work_dirs/physflow_online_adv_mn` |
| mn 启动 | `tools/physflow_mn_start.sh` + `tools/taiji_dist_train.sh`（4 节点 × 8 V100） |
| mn 续训 | `load_from` = v3@3000，`load_scope=model`，`max_iters=1500` |
| 语料 | `configs/experiments/physflow_kimodo_g1/physflow_text_train.jsonl`（~11241） |
| 文本特征（hml3d test） | `data/kimodo_text_feature/kimodo_g1_llm2vec_hml3dtest/` |

**交接时 mn 进度**：约 step 904/1500；已有 `checkpoint-iter_{200,400,600,800}`。训满后应对 **最终 ckpt** 再跑一轮配对评测。

---

## Judge / Tracker（当前部署）

- **MuJoCo judge ONNX（在用）**：  
  `ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx`
- **算法**：PPO + AMP + BeyondMimic（released rehearsal 配方），**不是**已废弃的 xy_offset position-aware 线。
- **废弃**：`active_tracker_v1`（iter1_v2 / A_e609 等 position-aware 微调）— 在标准动作与 KIMODO 上均劣于 released；勿再用于 viz / reward。

**历史结论（一句）**：xy_offset 微调 + 错误 warmstart 曾导致「像不会跟踪」；根因是 warmstart 通道错位与 OOD 遗忘，非算法本质不行。详见 backup §CRITICAL / §OVERFIT。

---

## 配对评测结果（HumanML3D test, n=40）

`adversarial_score` = **惩罚，越低越好**（`scripts/embodied/physflow_g1_scoring.py`）。

| 指标 | base | v3@1050 量级 | **mn@800** | 方向 |
|------|------|--------------|------------|------|
| jerk | 335 | −55 vs base | **236 (−99)** | ↓优 |
| foot_skate_speed | 0.263 | −0.076 | **0.198 (−0.065)** | ↓优 |
| foot_skate_ratio | 0.270 | +0.017 ❌ | **0.244 (−0.026)** | ↓优 |
| adversarial_score | 1.639 | +0.079 ❌ | **1.646 (+0.007)** | ↓优 |
| root_trajectory_error_m | 0.615 | +0.097 ❌ | **0.661 (+0.046)** | ↓优 |
| joint_std | 0.089 | −0.003 | **0.093 (+0.005)** | 未坍缩 |

产物：

- v3 对比：`work_dirs/physflow_online_adv_v3/viz/hml3dtest_compare.json`
- mn@800：`work_dirs/physflow_online_adv_mn/viz/hml3dtest_compare.json`
- 三列 viz manifest：`work_dirs/physflow_online_adv_mn/viz/hml3dtest_compare_manifest/manifest.json`

```
/physflow_triplet?manifest=work_dirs/physflow_online_adv_mn/viz/hml3dtest_compare_manifest/manifest.json
```

三列含义：**base 生成 | PhysFlow 优化 | 优化→MuJoCo tracker rollout**。

评测脚本：`scripts/embodied/physflow_coevolve_viz.py` + `build_compare_manifest.py`。

---

## 环境与命令

| 用途 | Python / 路径 |
|------|----------------|
| KIMODO 生成 + MuJoCo 打分 | `/usr/local/bin/python3`（Taiji vermo 容器同理） |
| IsaacGym tracker 训练 | `/root/physflow_isaacgym_py38_cu118/bin/python` |
| 调试机 8×V100 | Taiji `physflow_trainee_gpu_v2` |
| HF 离线 | `HF_HUB_OFFLINE=1`，`TEXT_ENCODERS_DIR=checkpoints/kimodo/text_encoders` |
| 分布式防 SIGHUP | `setsid` 启动 `accelerate` |

提交多机：`python3 tools/taiji_submit.py <name> configs/physflow/physflow_online_adv_mn.py --host_num 4`

Taiji viz 再生：`scripts/embodied/regen_viz_hml3dtest.sh`（含 pip mujoco/onnxruntime/dm_control/typer）。

---

## 关键 gotchas（仍有效）

- `kimodo.scripts.generate` **每条 motion 单独起进程**，扫库很慢。
- ProtoMotions `--training-max-steps` 为**全局步数**，resume 不读 CLI 覆盖；续训须 **新 experiment_name** 或提高 max-steps。
- Tracker 打分：已对 ONNX/MjModel 做模块缓存（`run_g1_rl_tracker_export.py`），避免每步重复加载。
- 训练 pool 文件会被轮转删除；离线分析用静态 `v3/tracker_motion_pool`。
- 场景/物体依赖 prompt 污染：见 `docs/temp/physflow_scene_dependence_issue.md`（~8.4% HIGH+MED，暂未过滤主训练）。
- ProtoMotions `inference_agent.py` 要求 checkpoint 同目录存在 `resolved_configs_inference.pt`；保存到 `output/.../checkpoints` 的快照必须同步 `resolved_configs*`。

---

## 2026-06-05 Tracker 迭代（AMASS-G1 / LAFAN1-G1）

### 评测/训练脚本修复

- `scripts/embodied/run_guarded_adversarial_tracker_train.sh`
  - 训练结束后同步 `last.ckpt` + `resolved_configs*.{pt,yaml}` + `config.yaml` + `experiment_config.py` 到 `${OUT_ROOT}/checkpoints/`。
  - 保留 `PACK_MOTION_LIB`、`REBUILD_GYMTORCH`、`PHYSFLOW_EXTRA_OVERRIDES`。
- `scripts/embodied/run_tracker_ckpt_lafan_amass_eval.sh`
  - 新增 checkpoint 对 LAFAN+AMASS 通用评测入口。
  - 新增 `RUN_AMASS=0` / `RUN_LAFAN=0`，用于快速门禁或 AMASS-only。
  - AMASS 默认改为 `8 shards × 128 envs`，总 env 不变但用满 8 卡；早先已提交的 0605f/0605g AMASS 仍按旧 `4 × 256` 跑。

### 0605f：guarded adversarial smoke

- 任务：`physflow_guarded_adv_w095_lr1e6_env128_nodr_pack_0605f`
- ckpt：`output/guarded_adversarial_tracker/physflow_guarded_adv_w095_lr1e6_env128_nodr_pack_0605f/checkpoints/final_epoch14_last.ckpt`
- 配方：native/adversarial/jump = `0.95/0.04/0.01`，`task_reward_w=0.5`，`discriminator_reward_w=2.0`，no-DR，env128，disc batch 2048。
- 内部曲线：`episode_reward 41.84→438.01`，`task_rewards 3.045→3.198`，`terminate_mean 0.00659→0.00186`。
- LAFAN1-G1 600-step（`output/lafan1_g1_proto_baseline_eval/physflow_0605f_cfgfix_600step/summary.md`）：

| baseline | success | gt mean mm | jerk | 结论 |
|---|---:|---:|---:|---|
| official `protomotions_g1_bones` | 0.800 | 796.454 | 2425.24 | baseline |
| `physflow0605f` | 0.775 | 700.019 | 2173.51 | 误差/jerk 好，但 success 降，不能算有效 |

- AMASS-G1 full 600-step：`physflow_tracker_eval_0605f_cfgfix` / `8b1d899c9e921b41019e94124dd90297`
  - summary：`output/amass_g1_proto_baseline_eval/physflow_0605f_cfgfix_600step/summary.md`

| baseline | success | gt mean mm | jerk | 结论 |
|---|---:|---:|---:|---|
| official `protomotions_g1_bones` | 0.879700 | 685.897 | 2456.47 | baseline |
| `physflow0605f` | 0.879869 | 594.081 | 2208.78 | AMASS 大样本小幅正向，但 LAFAN success 负向 |

### 0605g：track-first adversarial

- 任务：`physflow_guarded_adv_trackfirst_0605g` / `8b1d810c9e921bcc019e941b40de0280`，成功结束。
- ckpt：`output/guarded_adversarial_tracker/physflow_guarded_adv_trackfirst_0605g/checkpoints/last.ckpt`
- 配方：从 0605f warm-start，native/adversarial/jump = `0.98/0.015/0.005`，`task_reward_w=2.0`，`discriminator_reward_w=0.25`，`discriminator_reward_threshold=0.0`，`discriminator_max_cumulative_bad_transitions=1000000`。
- 内部曲线：`episode_reward 41.08→491.80`，`task_rewards 3.070→3.230`，`terminate_mean 0.00497→0.00131`，`discriminator_loss 0.394→0.344`。
- LAFAN1-G1 600-step（实际目录 `output/lafan1_g1_proto_baseline_eval/20260605_033818/summary.md`；提交时 `OUT_ROOT` 未 export，但 checkpoint spec 正确）：

| baseline | success | gt mean mm | jerk | 结论 |
|---|---:|---:|---:|---|
| official `protomotions_g1_bones` | 0.800 | 728.022 | 2307.73 | baseline |
| `physflow0605g` | 0.825 | 707.408 | 2270.37 | success、GT error、jerk 均正向 |

- AMASS-G1 full 600-step：
  - `physflow_tracker_eval_0605g_amass` / `8b1d81e79e920c67019e943131f5027a` 因坏节点失败：driver CUDA 11.0，IsaacGym 需要 11.4，GPU pipeline 被禁用后 illegal memory access。
  - retry：`physflow_tracker_eval_0605g_amass_retry1` / `8b1d81e79e920c67019e9435886c027e`，节点 driver CUDA 11.4，成功完成。
  - summary：`output/amass_g1_proto_baseline_eval/physflow_0605g_amass_600step_retry1/summary.md`

| baseline | success | gt mean mm | jerk | 结论 |
|---|---:|---:|---:|---|
| official `protomotions_g1_bones` | 0.879361 | 684.993 | 2462.68 | baseline |
| `physflow0605g` | 0.877724 | 574.931 | 2176.32 | 平均误差/jerk 大幅变好，但 success 下降；仍不能算有效 |

0605g 的 failure 分解显示 relative-body failure 从 `0.086316` 升到 `0.087953`，anchor-height failure 从 `0.103308` 降到 `0.103026`，success 降幅主要来自 body-pose 阈值越界。下一版不再从已带成功率损失的 0605f 继续，而是从 official-equivalent warmstart 重新起，进一步降低 adversarial 干扰。

### 0605h：official anchor / ultra-light adversarial

- 首次任务：`physflow_guarded_adv_trackfirst_0605h` / `8b1d81459e921b47019e946a0bc702bd`，抽到 driver CUDA 11.0 节点，已主动停止，避免重演 IsaacGym bad-node failure。
- retry：`physflow_guarded_adv_trackfirst_0605h_retry1` / `8b1d80739e921bcd019e946bd0df02f1`，节点 driver CUDA 11.4，成功结束。
- ckpt：`output/guarded_adversarial_tracker/physflow_guarded_adv_anchor_0605h_retry1/checkpoints/last.ckpt`
- 配方：official-equivalent `g1_released_warmstart_epoch0.ckpt`，native/adversarial/jump = `0.99/0.0075/0.0025`，`task_reward_w=3.0`，`discriminator_reward_w=0.1`，actor LR `5e-7`，critic/disc LR `1e-6`，no-DR，env128，disc batch 2048，700k steps。
- LAFAN1-G1 600-step：
  - 首次 eval 抽到 driver 450 bad node，已停。
  - retry2：`physflow_tracker_eval_0605h_lafan_retry2` / `8b1d89b89e921b41019e95d0d1160635`，driver 470，成功完成。
  - summary：`output/lafan1_g1_proto_baseline_eval/physflow_0605h_lafan_600step_retry2/summary.md`

| baseline | success | gt mean mm | jerk | 结论 |
|---|---:|---:|---:|---|
| official `protomotions_g1_bones` | 0.800 | 720.598 | 2300.65 | baseline |
| `physflow0605h` | 0.800 | 724.226 | 2169.32 | success 持平，jerk/max-joint/gr 改善，mean GT 小幅变差 |

- AMASS-G1 full 600-step：
  - retry1 发现 eval wrapper 将 8-shard request 错误 symlink 到 4-shard cache，已主动停止并修复：`run_tracker_ckpt_lafan_amass_eval.sh` 默认改回 `4 shards x 256 envs`，并加 cache shard-count guard；`run_amass_g1_proto_baseline_eval.sh` 加 packed shard count guard，避免混 shard。
  - retry2：`physflow_tracker_eval_0605h_amass_retry2` / `8b1d81c89e921b54019e95e46ea50678`，driver 470，4 V100，full AMASS shard logs 完整；Taiji false 是聚合阶段脚本错误，已本地手动聚合。
  - summary：`output/amass_g1_proto_baseline_eval/physflow_0605h_amass_600step_retry2/summary.md`

| baseline | success | gt mean mm | jerk | relative-body fail | anchor-height fail | 结论 |
|---|---:|---:|---:|---:|---:|---|
| official `protomotions_g1_bones` | 0.878797 | 686.815 | 2460.63 | 0.086880 | 0.103760 | baseline |
| `physflow0605h` | 0.880603 | 602.642 | 2228.48 | 0.084679 | 0.101671 | AMASS full 上 success、GT、jerk、两个 failure 均正向 |

- ONNX：`output/guarded_adversarial_tracker/physflow_guarded_adv_anchor_0605h_retry1/compiled_onnx/unified_pipeline.onnx`
- HML3D 40-case 四列可视化：
  - manifest：`work_dirs/physflow_online_adv_mn_hymotion_real/viz/hml3dtest_hymotion_real_latest2_20260604_iter1400_fourway_physflow0605h_manifest/manifest.json`
  - metrics：`work_dirs/physflow_online_adv_mn_hymotion_real/viz/hml3dtest_hymotion_real_latest2_20260604_iter1400_physflow0605h_metrics.json`
  - manifest 校验：40 rows x 4 columns = 160 ready slots，missing paths = 0。

HML3D 40-case 可视化指标（iter1400 KIMODO reference，Tracker before 为官方 released tracker，Tracker after 为 `physflow0605h`）：

| arm | completion | fall | adversarial score | joint err mean | joint err max | root traj mean |
|---|---:|---:|---:|---:|---:|---:|
| KIMODO before | 0.975 | 0.025 | 1.638565 | 0.605783 | 0.920420 | 0.615110 |
| KIMODO after + Tracker before | 1.000 | 0.000 | 1.466987 | 0.618059 | 1.638148 | 0.506459 |
| KIMODO after + Tracker after (`physflow0605h`) | 1.000 | 0.000 | 1.406594 | 0.590206 | 1.610836 | 0.471167 |

- 当前结论：0605h 是第一版相对 official pretrained tracker 在 AMASS full 上达成正向的 PhysFlow tracker；在 HML3D 40-case visualization 上也相对 Tracker before 改善 adversarial score、mean joint error、root trajectory error，并保持 completion/fall 不退化。LAFAN1-G1 success 仅持平、GT mean 略差，因此不能把它作为 SOTA 证明；之后仍必须对 OpenTrack/Any2Track、BeyondMimic、PBHC/KungfuBot、ASAP 等 SOTA tracker baseline 做同表复现。

---

## Tracker 跳跃能力（待做，数据阻塞）

**配方层**（仓库内已有，当前 judge **未开**）：`contact_match_rew`、`rh_rew` / `global_anchor_pos_rew`、`init_start_prob=0`（RSI 含腾空帧）、放宽 `anchor_height_error_term(0.25)`、actor `root_height_obs=True`。无内置 phase-aware 腾空终止，需自写或提高 threshold。

**数据层（硬阻塞）**：现有 G1 retarget 库无真·动态大跳参考。

| 数据 | 结论 |
|------|------|
| `g1_bones_seed_mini.pt`（58 clip） | 名含 jump 的 pelvis 升幅 ~0.13m；大 Z-rise 多为 crawl/roll |
| `physflow_g1_released_rehearsal_v1_pool` | 仅 `train_small_hop` |
| accept-gated pool | jump 标签中位 pelvis 升 ~5cm |

**用户意图**：先完成当前版本评测 → 再跳跃 judge v2；曾选激进腾空目标 + 立即起训，但需先补动态跳跃 mocap → `pipeline_motion_to_robot.py`（含 contact）或接受浅跳上限。

实验模板：`ref_repo/ProtoMotions/examples/experiments/mimic/mlp.py`（contact+rh）+ BM 元素；启动 `scripts/embodied/launch_position_aware_g1_tracker_train.sh`。

---

## 待办

1. 盯 mn 至 1500 step；**最终 ckpt** 复跑 hml3dtest 配对评测 + 更新 compare manifest。
2. 与用户确认跳跃路线：补 mocap / 仅配方验证浅跳 / 并行 / 论文限定范围。
3. collapse 监控：`sel_joint_std_mean < 0.04`。
4. （可选）语料场景过滤 → `physflow_text_train.feasible.jsonl`。

---

## 关键脚本索引

```
hftrainer/trainers/motion/physflow_trainer.py
scripts/embodied/physflow_g1_scoring.py
scripts/embodied/physflow_coevolve_viz.py
scripts/embodied/physflow_kinematic_metrics.py
scripts/embodied/run_g1_rl_tracker_export.py
scripts/embodied/build_compare_manifest.py
ref_repo/ProtoMotions/deployment/export_bm_tracker_onnx.py
motion_annot_web/embodied_viz/app.py
```
