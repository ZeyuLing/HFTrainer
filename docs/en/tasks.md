# Task Matrix

## Runnable Today

| Task | Bundle | Trainer | Pipeline | Example Config |
| --- | --- | --- | --- | --- |
| Motion generation (PRISM) | `PrismBundle` | `PrismTrainer` | `PrismPipeline` | `configs/prism/prism_1b_tp2m_motionhub.py` |
| Motion generation (PRISM MCM) | `PrismMCMBundle` | `PrismMCMTrainer` | `PrismMCMPipeline` | `configs/prism/prism_mcm_smoke.py` |
| Motion generation / understanding (VerMo) | `VermoBundle` | `VermoTrainer` | `VermoPipeline` | `configs/vermo/vermo_pretrain_4k_llama1b_wavtokenizer.py` |
| Motion editing / completion (HyMotion M2M) | `HyMotionM2MBundle` | `HyMotionM2MTrainer` | `HyMotionM2MPipeline` | `configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py` |
| Text-to-motion (HyMotion T2M) | `HyMotionT2MBundle` | `HyMotionT2MTrainer` | `HyMotionT2MPipeline` | `configs/hymotion_t2m/hymotion_t2m_smoke.py` |

## Validation Output Convention

- Motion generation/editing: task-specific motion artifacts (`rot6d`, `transl`, `keypoints3d`, etc.)
- VerMo multi-task outputs: modality-specific dicts (`motion`, `caption`, `audio`) depending on task
