# HyMotion M2M v2 训练实验清单

> 日期: 2026-04-14
> 共 6 个实验，每个 8 节点 × 8 GPU = 64 V100

---

## 实验总览

| # | 实验名称 | task_flag | 类型 | 弹性 | 续训起点 | T2M占比 | max_epochs |
|---|---------|-----------|------|------|---------|---------|-----------|
| 1 | Caption Local Phase1 | `m2m_v2_clp1` | Phase1 纯T2M | **非弹性** | caption_local epoch 183 | **100%** | 50 |
| 2 | Caption Global Phase1 | `m2m_v2_cgp1` | Phase1 纯T2M | **非弹性** | caption_global epoch 213 | **100%** | 50 |
| 3 | Caption Local 单阶段 | `m2m_v2_cl` | 混合训练 | **非弹性** | caption_local epoch 183 | 16% | 10000 |
| 4 | Caption Global 单阶段 | `m2m_v2_cg` | 混合训练 | **非弹性** | caption_global epoch 213 | 16% | 10000 |
| 5 | Uncond Local | `m2m_v2_ul` | 混合训练 | 弹性 | uncond_local epoch 177 | 12% | 10000 |
| 6 | Uncond Global | `m2m_v2_ug` | 混合训练 | 弹性 | uncond_global epoch 271 | 12% | 10000 |

**优先级**: 1=2 > 3=4 > 5=6

---

## 各实验详情

### 实验 1: Caption Local Phase1 (纯T2M)

- **目的**: 用纯T2M数据强化text-motion对应能力，后续接Phase2混合训练
- **Config**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase1.py`
- **续训**: 从 `work_dirs/hymotion_m2m_v2_caption_local_046b/checkpoint-epoch_183/model.safetensors` 加载权重（仅权重，epoch重新计数）
- **Work dir**: `work_dirs/hymotion_m2m_v2_caption_local_phase1/`
- **关键参数**: mask全1, cond_mask_prob=0.1, mask_aware_noise=False, max_epochs=50

```bash
python3 tools/taiji_submit.py m2m_v2_clp1 \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase1.py \
    --host_num 8
```

### 实验 2: Caption Global Phase1 (纯T2M)

- **目的**: 同实验1，global rotation版本
- **Config**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_phase1.py`
- **续训**: 从 `work_dirs/hymotion_m2m_v2_caption_global_046b/checkpoint-epoch_213/model.safetensors` 加载权重
- **Work dir**: `work_dirs/hymotion_m2m_v2_caption_global_phase1/`
- **关键参数**: mask全1, cond_mask_prob=0.1, mask_aware_noise=False, max_epochs=50

```bash
python3 tools/taiji_submit.py m2m_v2_cgp1 \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_phase1.py \
    --host_num 8
```

### 实验 3: Caption Local 单阶段混合

- **目的**: 不分阶段，直接在已有checkpoint上用提升后的T2M比例继续训练（对照组）
- **Config**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py`
- **续训**: 自动从 `work_dirs/hymotion_m2m_v2_caption_local_046b/checkpoint-epoch_183/` resume（--auto-resume，恢复optimizer+epoch）
- **Work dir**: `work_dirs/hymotion_m2m_v2_caption_local_046b/`（在原work_dir继续）
- **关键参数**: Tier2 pure_gen=40% (全局16%), cond_mask_prob=0.1, mask_aware_noise=True

```bash
python3 tools/taiji_submit.py m2m_v2_cl \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py \
    --host_num 8
```

### 实验 4: Caption Global 单阶段混合

- **目的**: 同实验3，global rotation版本
- **Config**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_046b.py`
- **续训**: 自动从 `work_dirs/hymotion_m2m_v2_caption_global_046b/checkpoint-epoch_213/` resume
- **Work dir**: `work_dirs/hymotion_m2m_v2_caption_global_046b/`
- **关键参数**: Tier2 pure_gen=40% (全局16%), cond_mask_prob=0.1, mask_aware_noise=True

```bash
python3 tools/taiji_submit.py m2m_v2_cg \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_046b.py \
    --host_num 8
```

### 实验 5: Uncond Local

- **目的**: 无text条件，单阶段混合训练，提升pure_gen比例
- **Config**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py`
- **续训**: 自动从 `work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_177/` resume
- **Work dir**: `work_dirs/hymotion_m2m_v2_uncond_local_046b/`
- **关键参数**: Tier2 pure_gen=30% (全局12%), cond_mask_prob=0.0, mask_aware_noise=True

```bash
python3 tools/taiji_submit.py m2m_v2_ul \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py \
    --host_num 8 --elastic
```

### 实验 6: Uncond Global

- **目的**: 同实验5，global rotation版本
- **Config**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_global_046b.py`
- **续训**: 自动从 `work_dirs/hymotion_m2m_v2_uncond_global_046b/checkpoint-epoch_271/` resume
- **Work dir**: `work_dirs/hymotion_m2m_v2_uncond_global_046b/`
- **关键参数**: Tier2 pure_gen=30% (全局12%), cond_mask_prob=0.0, mask_aware_noise=True

```bash
python3 tools/taiji_submit.py m2m_v2_ug \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_global_046b.py \
    --host_num 8 --elastic
```

---

## Taiji 设置说明

所有实验共享的固定设置（已写入 `tools/taiji_template.json`）:

| 设置 | 值 | 说明 |
|------|---|------|
| GPUName | V100 | |
| host_gpu_num | 8 | 每节点8卡 |
| host_num | 8 | 8节点，总共64卡 |
| is_enable_rdma | true | RDMA 加速通信 |
| rdma_in_same_module | true | 同模块调度 |
| is_resource_waiting | true | 排队等资源 |
| priority_level | HIGH | |
| task_queuing_priority | P2 | |
| location | cq | 重庆集群 |

**弹性 vs 非弹性**：
- 实验 1-4（caption）: `--elastic` 不加 → `is_elasticity=false`（非弹性，优先调度）
- 实验 5-6（uncond）: 加 `--elastic` → `is_elasticity=true`（弹性卡，可能被抢占）

**启动命令**:
```
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/
python3 -c 'from mmengine.config import Config; Config.fromfile("<CONFIG_PATH>")'
bash tools/taiji_dist_train.sh <CONFIG_PATH> --auto-resume
```
先单进程预热 mmengine Config 解析（避免 yapf pickle 多进程竞争），再启动分布式训练。`--auto-resume` 自动从 work_dir 中最新 checkpoint 恢复。

---

## 相对之前训练的变化

| 项目 | 之前 | 现在 |
|------|------|------|
| **cond_mask_prob** (caption) | 0.3 | **0.1** |
| **Tier2 pure_gen** (caption) | 20% (全局8%) | **40% (全局16%)** |
| **Tier2 pure_gen** (uncond) | 20% (全局8%) | **30% (全局12%)** |
| **max_epochs** | 1000 | **10000** (Phase1除外=50) |
| **FK consistency loss** | silent except→始终为0 | **正常计算** |
| **caption数据格式** | "short caption"格式报错 | **兼容空格/下划线两种格式** |
| **log精度** | `.4f` (小值显示0.0000) | **自适应** (小值用科学计数法) |

---

## Phase 1 完成后的操作

实验1、2（Phase1）训完50 epoch后，用对应的Phase2 config启动：

```bash
# Phase2: Caption Local (从Phase1的checkpoint续训)
python3 tools/taiji_submit.py m2m_v2_clp2 \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2.py \
    --host_num 8

# Phase2: Caption Global
python3 tools/taiji_submit.py m2m_v2_cgp2 \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_phase2.py \
    --host_num 8
```

Phase2 config 中 `load_from` 指向 Phase1 的 `checkpoint-epoch_50/model.safetensors`。如果Phase1实际训练epoch数不同，需要手动修改 Phase2 config 中的 checkpoint 路径。
