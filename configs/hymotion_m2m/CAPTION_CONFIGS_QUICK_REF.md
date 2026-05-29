# HyMotion M2M v2 Caption Configs - Quick Reference

## ALL 16 CAPTION/TEXT CONFIGS AT A GLANCE

### Main Caption Baselines (4)
1. **caption_local_046b** - Standard caption + local rotation
2. **caption_global_046b** - Standard caption + global rotation  
3. **smpl_caption_046b** - E2: SMPL + caption + keypoint supervision
4. **kimodo_caption_046b** - E4: KIMODO + caption + ADMM smoothing

### Curriculum Phases - Local (3)
5. **caption_local_phase1** - Phase 1: Pure T2M (100% mask=1)
6. **caption_local_phase2** - Phase 2: Mixed T2M(16%) + completion(84%)
7. **caption_local_phase2b** - Phase 2b: Component-mean loss

### Curriculum Phases - Global (2)
8. **caption_global_phase1** - Phase 1: Pure T2M (global)
9. **caption_global_phase2** - Phase 2: Mixed T2M + completion (global)

### PerMo Data Variants (2)
10. **smpl_caption_permo_046b** - E2 + PerMo dataset
11. **kimodo_caption_permo_046b** - E4 + PerMo dataset

### T2M Transfer Learning (3)
12. **t2m_pretrained** - Default: freeze encoders
13. **t2m_no_freeze** - Ablation: all trainable
14. **t2m_full_freeze** - Ablation: freeze all

### SOAR Post-Training (2)
15. **caption_local_046b_soar** - SOAR on caption_local
16. **caption_global_046b_soar** - SOAR on caption_global

---

## TRAINING CHAINS

### Primary Chain: Caption Local
```
Base → caption_local_046b@183 → phase1@50 → phase2@3370 → phase2b@3320 → SOAR@498
```

### Alternative Chain: Caption Global
```
Base → caption_global_046b@213 → phase1@50 → phase2
                                          ↓
                                      SOAR@548
```

### Experiment Variants (All start from Phase 2@3370)
- E2: smpl_caption_046b
- E4: kimodo_caption_046b  
- E2+PerMo: smpl_caption_permo_046b
- E4+PerMo: kimodo_caption_permo_046b

---

## KEY PARAMETERS BY CONFIG

| Parameter | Base | Local | Global | Phase1 | Phase2 | E2/E4 | E2+PerMo/E4+PerMo | SOAR |
|-----------|------|-------|--------|--------|--------|-------|-------------------|------|
| uncondition_mode | True | False | False | False | False | False | False | False |
| cond_mask_prob | 0.0 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| batch_size | 28 | 20 | 20 | 20 | 20 | 20 | 20 | 10 |
| mask_sampler | v2 | v2 | v2 | full | v3 | v3 | v3 | v3 |
| K=0 (T2M) | N/A | 16% | 16% | 100% | 16% | 16% | 16% | 16% |
| mask_aware_noise | True | True | True | False | True | True | True | True |
| keypoints3d_weight | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 | 10.0 | 10.0 |
| velocity_loss_reduction | default | default | default | default | default | component | component | component |
| trans_dim_weight | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |

### Phase 2b Exception
- velocity_loss_reduction = 'component_mean'
- trans_dim_weight = 1.0

### KIMODO Exception
- exclude_bundle_keys=['mean', 'std'] (critical!)

---

## CHECKPOINT PATHS

```
Base
├─ Load: checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt

caption_local_046b / caption_global_046b
├─ Load: T2M pretrained (auto from base)

caption_local_phase1
├─ Load: work_dirs/hymotion_m2m_v2_caption_local_046b/checkpoint-epoch_183
├─ Patch: T2M pretrained (null embeddings)

caption_local_phase2
├─ Load: work_dirs/hymotion_m2m_v2_caption_local_phase1/checkpoint-epoch_50
├─ Patch: T2M pretrained

caption_local_phase2b
├─ Load: work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3320
├─ Patch: T2M pretrained

smpl_caption_046b (E2)
├─ Load: work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370
├─ Patch: T2M pretrained

kimodo_caption_046b (E4)
├─ Load: work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370
├─ exclude_bundle_keys=['mean', 'std']  ← CRITICAL!
├─ Patch: T2M pretrained

caption_local_046b_soar
├─ Load: work_dirs/hymotion_m2m_v2_caption_local_046b/checkpoint-epoch_498

caption_global_046b_soar
├─ Load: work_dirs/hymotion_m2m_v2_caption_global_046b/checkpoint-epoch_548
```

---

## MOTION REPRESENTATIONS

### SMPL Root (198-dim)
- [0:3] Translation (3D)
- [3:135] Body rotations (22 joints × 6D rot6d)
- [135:198] Joint positions (21 joints × 3D, relative to pelvis)

### KIMODO Root (198-dim)
- [0:3] Translation (ADMM smoothed, 6cm margin on XZ)
- [3:9] Root rotation (6D)
- [9:135] Body rotations (21 joints × 6D)
- [135:198] Joint positions (21 joints × 3D)

**Key difference**: ADMM smoothing on pelvis trajectory for embodied tasks

---

## LOSS CONFIGURATION SUMMARY

### Base / Caption Standard
- fk_consistency_weight = 0.0 (KIMODO aux active)
- keypoints3d_weight = 0.0
- velocity_loss_reduction = 'element_mean' (default)
- trans_dim_weight = 5.0

### Phase 1
- fk_consistency_weight = 0.1 (enabled during pure T2M)
- Rest same as base

### Phase 2 / Phase 2b / Experiments
- fk_consistency_weight = 0.0
- Phase 2b + E2/E4:
  - keypoints3d_weight = 10.0
  - velocity_loss_reduction = 'component_mean'
  - trans_dim_weight = 1.0 (Phase 2b only)

---

## TEXT ENCODING

All caption configs use:
- **Text encoder**: QWEN3 (4096-dim) + CLIP-L (768-dim)
- **Loading**: `LoadPreExtractedTextEmbedding`
- **CFG**: cond_mask_prob = 0.1 (10% unconditional during training)
- **Data**: Pre-extracted embeddings (not on-the-fly)

---

## COMMON PITFALLS & SOLUTIONS

1. **KIMODO loading bug**: Must use `exclude_bundle_keys=['mean', 'std']`
   - Otherwise SMPL stats overwrite KIMODO stats

2. **Null embeddings issue**: Phase checkpoints have all-zero null embeddings
   - Solution: Patch with T2M pretrained via `null_embedding_source`

3. **Memory pressure**: Text tokens add ~6GB per batch
   - Solution: Reduce batch_size from 28 to 20 for caption configs

4. **Mask sampler mismatch**: Phase 2 switched from v2 (tier2_prob) to v3 (Rank-K)
   - Phase 1: Full mask (100% pure T2M)
   - Phase 2: v3 with k_weights=(0.16, 0.513, 0.233, 0.065, 0.029)

---

## LAUNCHING CONFIGS

### Single GPU
```bash
python tools/train.py configs/hymotion_m2m/hymotion_m2m_caption_local_046b.py
python tools/train.py configs/hymotion_m2m/hymotion_m2m_caption_local_phase1.py
```

### Multi-GPU
```bash
bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_caption_local_046b.py 8
bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_caption_local_phase2.py 8 \
  --load-from work_dirs/hymotion_m2m_v2_caption_local_phase1/checkpoint-epoch_200/model.safetensors
```

### Taiji Cluster (64 GPUs)
```bash
python tools/taiji_submit.py m2m_v2_caption_local_p1 \
  configs/hymotion_m2m/hymotion_m2m_caption_local_phase1.py --host_num 8

python tools/taiji_submit.py m2m_v2_caption_local_p2 \
  configs/hymotion_m2m/hymotion_m2m_caption_local_phase2.py --host_num 8
```

---

## FILE STATISTICS

| Type | Count | Files |
|------|-------|-------|
| Caption configs | 11 | caption_{local,global}_046b, caption_{local,global}_phase{1,2,2b}, smpl_caption_*, kimodo_caption_* |
| T2M variants | 3 | t2m_pretrained, t2m_no_freeze, t2m_full_freeze |
| SOAR post-training | 2 | caption_local_046b_soar, caption_global_046b_soar |
| **Total** | **16** | See list above |

Total size: ~55KB across all caption configs
