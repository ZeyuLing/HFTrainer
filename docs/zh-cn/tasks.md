# 任务矩阵

## 当前可运行

| 任务 | Bundle | Trainer | Pipeline | 示例 Config |
| --- | --- | --- | --- | --- |
| 动作生成（PRISM） | `PrismBundle` | `PrismTrainer` | `PrismPipeline` | `configs/prism/prism_1b_tp2m_motionhub.py` |
| 动作生成（PRISM MCM） | `PrismMCMBundle` | `PrismMCMTrainer` | `PrismMCMPipeline` | `configs/prism/prism_mcm_smoke.py` |
| 动作生成 / 理解（VerMo） | `VermoBundle` | `VermoTrainer` | `VermoPipeline` | `configs/vermo/vermo_pretrain_4k_llama1b_wavtokenizer.py` |
| 动作编辑 / 补全（HyMotion M2M） | `HyMotionM2MBundle` | `HyMotionM2MTrainer` | `HyMotionM2MPipeline` | `configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py` |
| 文本到动作（HyMotion T2M） | `HyMotionT2MBundle` | `HyMotionT2MTrainer` | `HyMotionT2MPipeline` | `configs/hymotion_t2m/hymotion_t2m_smoke.py` |

## Validation 输出约定

- 动作生成 / 编辑：按任务返回动作产物（`rot6d`、`transl`、`keypoints3d` 等）
- VerMo 多任务：按任务返回模态字典（`motion`、`caption`、`audio` 等）
