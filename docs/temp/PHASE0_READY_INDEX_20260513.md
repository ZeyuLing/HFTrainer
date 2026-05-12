# Phase 0 Ready Index — May 13, 2026

Quick reference guide to Phase 0 experiments (E1-E4) and related documentation.

---

## 📋 Phase 0 Experiments at a Glance

| Exp | Name | Root Repr. | Conditioning | Config File | Status | Next |
|-----|------|-----------|--------------|-------------|--------|------|
| **E1** | SMPL Uncond | SMPL | None | `smpl_uncond_046b.py` | ✅ Ready | Train |
| **E2** | SMPL Caption | SMPL | Text | `smpl_caption_046b.py` | ✅ Ready | Train |
| **E3** | KIMODO Uncond | KIMODO+ADMM | None | `kimodo_uncond_046b.py` | ✅ Ready* | Compute stats |
| **E4** | KIMODO Caption | KIMODO+ADMM | Text | `kimodo_caption_046b.py` | ✅ Ready* | Compute stats |

*E3-E4 await `data/hymotion_m2m_data/_stats_198dim_kimodo_root/` computation

---

## 📂 Config Files

All configs located in: `configs/hymotion_m2m_v2/`

### E1: SMPL Unconditioned (Baseline)
```
hymotion_m2m_v2_smpl_uncond_046b.py
├─ Base: _base_hymotion_m2m_v2_046b.py
├─ Root: SMPL (no transforms)
├─ Loss Overrides:
│  ├─ keypoints3d_weight: 10.0 (position supervision)
│  └─ timestep_squared_weighting: False (standard weighting)
├─ Work Dir: work_dirs/hymotion_m2m_v2_smpl_uncond_E1
└─ Batch Size: 28
```

### E2: SMPL + Caption
```
hymotion_m2m_v2_smpl_caption_046b.py
├─ Base: _base_hymotion_m2m_v2_046b.py (extends E1)
├─ Text Encoding: QWEN3 (4096d) + CLIP-L (768d)
├─ CFG Training: cond_mask_prob=0.1 (10% unconditional)
├─ Loss Overrides: Same as E1
├─ Work Dir: work_dirs/hymotion_m2m_v2_smpl_caption_E2
└─ Batch Size: 20 (reduced for text memory)
```

### E3: KIMODO Unconditioned
```
hymotion_m2m_v2_kimodo_uncond_046b.py
├─ Base: _base_hymotion_m2m_v2_046b.py
├─ Root: KIMODO (ADMM smoothing via SmplTransToKimodoRootOnline)
├─ Data Pipeline Change:
│  └─ Adds: SmplTransToKimodoRootOnline(admm_margin_m=0.06)
├─ Mean/Std: _stats_198dim_kimodo_root (pending computation)
├─ Loss Overrides: Same as E1
├─ Work Dir: work_dirs/hymotion_m2m_v2_kimodo_uncond_E3
└─ Batch Size: 28
```

### E4: KIMODO + Caption
```
hymotion_m2m_v2_kimodo_caption_046b.py
├─ Base: _base_hymotion_m2m_v2_046b.py
├─ Root: KIMODO (ADMM smoothing)
├─ Data Pipeline: Same as E3
├─ Text Encoding: Same as E2
├─ CFG Training: cond_mask_prob=0.1
├─ Mean/Std: _stats_198dim_kimodo_root (pending)
├─ Loss Overrides: Same as E2
├─ Work Dir: work_dirs/hymotion_m2m_v2_kimodo_caption_E4
└─ Batch Size: 20
```

---

## 🔧 New Transform: SmplTransToKimodoRootOnline

**File**: `hftrainer/datasets/motion/motionhub/transforms/smpl_trans_to_kimodo_root.py`

**Purpose**: Online (during dataset loading) conversion from SMPL Root to KIMODO Root

**Algorithm**:
1. Smooth translation [0:3]: Frame-to-frame XZ distance ≤ 6cm margin
   - Y-axis preserved (vertical motion unsmoothed)
   - Forward+backward pass for smoothness
2. Preserve rotation [3:135]: Unchanged
3. Adjust position [135:198]: Update reference frame for smooth pelvis

**Integration**:
- E1-E2: Not used (SMPL Root bypass)
- E3-E4: Integrated in data pipeline
  ```python
  dict(
      type='SmplTransToKimodoRootOnline',
      key='motion',
      admm_margin_m=0.06,  # 6cm threshold
  )
  ```

---

## 📊 Loss Configuration Summary

All E1-E4 use **identical loss settings** for fair comparison:

| Loss | Weight | Purpose | Change from Base |
|------|--------|---------|------------------|
| velocity | 1.0 | Main prediction target | Unchanged |
| keypoints3d | 10.0 | Position supervision | ↑ 0.0 → 10.0 |
| joint_pos | 50.0 | FK-derived joint pos | Unchanged |
| joint_vel | 500.0 | Joint velocity | Unchanged |
| fk_consistency | 1500.0 | Rotation↔Position match | Unchanged |
| timestep_squared_weighting | False | Loss scaling | Changed from True |

**Key Insight**: `keypoints3d_weight=10.0` enables position channels as direct supervision signal, improving structural consistency.

---

## 🚀 Launch Commands

### Local Training (8 GPUs)

E1:
```bash
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_uncond_046b.py 8 --auto-resume
```

E2:
```bash
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py 8 --auto-resume
```

E3 (after mean/std computed):
```bash
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_uncond_046b.py 8 --auto-resume
```

E4 (after mean/std computed):
```bash
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py 8 --auto-resume
```

### Taiji Submission (64 GPUs, 8 hosts)

Template:
```bash
python tools/taiji_submit.py <job_name> <config_path> --host_num 8
```

E1:
```bash
python tools/taiji_submit.py m2m_v2_smpl_uncond_E1 configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_uncond_046b.py --host_num 8
```

---

## 📍 Key File Locations

### Configs
```
configs/hymotion_m2m_v2/
├── _base_hymotion_m2m_v2_046b.py           (base config)
├── hymotion_m2m_v2_smpl_uncond_046b.py     (E1)
├── hymotion_m2m_v2_smpl_caption_046b.py    (E2)
├── hymotion_m2m_v2_kimodo_uncond_046b.py   (E3)
└── hymotion_m2m_v2_kimodo_caption_046b.py  (E4)
```

### Transform
```
hftrainer/datasets/motion/motionhub/transforms/
└── smpl_trans_to_kimodo_root.py            (ADMM smoothing)
```

### Mean/Std (will be computed in Phase 0-Step 2)
```
data/hymotion_m2m_data/
├── _stats_198dim/                          (SMPL Root, exists)
└── _stats_198dim_kimodo_root/              (KIMODO Root, pending)
```

### Loss Code (already correct, no changes needed)
```
hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py
└─ keypoints3d loss: computed relative-to-root (pelvis subtracted)
```

---

## ✅ Validation Status

| Item | Status | Notes |
|------|--------|-------|
| E1 config | ✅ Ready | Tested, no deps |
| E2 config | ✅ Ready | Tested, no deps |
| E3 config | ✅ Ready (pending stats) | Tested, awaits mean/std |
| E4 config | ✅ Ready (pending stats) | Tested, awaits mean/std |
| SmplTransToKimodoRootOnline | ✅ Tested | Smoke test passed |
| Position loss | ✅ Verified | Already relative-to-root |
| Git commits | ✅ Clean | 3 commits, all committed |

---

## 📈 Expected Phase 0 Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| **0-Step 1** | Config + Transform prep | 1 day | ✅ COMPLETE |
| **0-Step 2** | Mean/std computation | 3-4 days | ⏳ Pending |
| **0-Step 2** | Single-step validation | 1 day | ⏳ Pending |
| **0-Step 3** | Taiji submission | 1 day | ⏳ Pending |
| **0-Training** | Full experiments (all 4) | 7-10 days | ⏳ Pending |

**Total Phase 0**: 1-2 weeks for full results

---

## 📚 Related Documentation

### Design & Rationale
- `docs/temp/hymotion_m2m_next_gen_proposal_20260511.md` — Full proposal with experimental design
- `docs/temp/PHASE0_READINESS_STATUS_20260512.md` — Implementation checklist
- `docs/temp/PHASE0_STEP1_COMPLETION_20260513.md` — This step's completion report

### KIMODO Root Details
- `docs/temp/KIMODO_HEADING_QUICK_ANSWER.md` — Root representation overview
- `docs/temp/KIMODO_ROOT_ANALYSIS.md` — Technical deep-dive

### Reference
- `docs/temp/HYMOTION_M2M_V2_TRAINING_CONFIG_REPORT.md` — Training config details
- `docs/temp/HYMOTION_M2M_V2_SYSTEM_OVERVIEW.md` — System architecture

---

## 🎯 Next Actions

### Immediate (Now)
- [ ] Review configs and validate with team
- [ ] Decide whether to start E1-E2 training on available GPUs
- [ ] Plan E3-E4 mean/std computation schedule

### Phase 0-Step 2 (Next)
- [ ] Compute KIMODO Root mean/std statistics
- [ ] Single-step validation on both root representations
- [ ] Prepare Taiji submission for all 4 experiments

### Phase 0-Step 3
- [ ] Submit to Taiji with 64 GPU per experiment
- [ ] Monitor first epoch convergence
- [ ] Set up evaluation pipeline

---

## 💡 Quick Tips

**To validate configs load correctly**:
```bash
python -c "
from mmengine.config import Config
cfg = Config.fromfile('configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_uncond_046b.py')
print('Config loads OK:', cfg.work_dir)
"
```

**To test SmplTransToKimodoRootOnline**:
```python
from hftrainer.datasets.motion.motionhub.transforms import SmplTransToKimodoRootOnline
import torch

t = SmplTransToKimodoRootOnline(admm_margin_m=0.06)
dummy = {'motion': torch.randn(10, 198)}
output = t(dummy)
print(f'Shape OK: {output["motion"].shape}')
```

**To check position loss computation**:
```bash
grep -A 5 "local_keypoints3d =" hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py
```

---

## 📞 Support

For questions about:
- **Configs**: Check corresponding config file headers for detailed comments
- **Transform**: See `smpl_trans_to_kimodo_root.py` docstrings
- **Loss**: See `m2m_loss.py` for implementation details
- **Proposal**: See `hymotion_m2m_next_gen_proposal_20260511.md` §6-9 for rationale

---

**Index Version**: 1.0  
**Date**: May 13, 2026  
**Status**: ✅ Phase 0-Step 1 Complete — Ready for Training
