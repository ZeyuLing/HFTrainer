# HyMotion M2M v2 — Complete Documentation Index

## 📚 Reference Documents (Just Created)

### 1. **HYMOTION_M2M_V2_SYSTEM_OVERVIEW.md** (START HERE)
- High-level system architecture
- The three loss tracks (M2MLoss, KIMODO, smoothness)
- Phase 2b config decisions and why they matter
- Quick lookup table: "Where to look for what"
- Critical notes and gotchas
- **Best for:** Getting oriented, understanding system design

### 2. **HYMOTION_M2M_V2_CRITICAL_FILES.md**
- Deep code dive for each component
- Bundle class structure (lines, methods, attributes)
- Config files explained in detail
- Base config inheritance chain
- SMPL-22 kinematic tree
- Loss function internals
- **Best for:** Understanding implementation details, debugging

### 3. **HYMOTION_M2M_V2_LINE_REFERENCE.md**
- Exact line numbers for every critical element
- Quick lookup tables organized by file
- Component ranges for motion dimensions
- Instantiation sequences
- **Best for:** Finding code, quick reference while editing

---

## 🎯 Quick Navigation by Task

### I need to understand the system
1. Read SYSTEM_OVERVIEW.md (sections: Key Facts, Three Loss Tracks, Config Decisions)
2. Look at motion representation diagrams
3. Review SMPL-22 kinematic tree

### I need to modify a loss weight
1. Check SYSTEM_OVERVIEW.md → "Where to Look for What"
2. Find the config file and line number in LINE_REFERENCE.md
3. Edit the config
4. Check if it interacts with other losses (e.g., component_mean + trans_dim_weight)

### I found a bug, need to trace code
1. Check SYSTEM_OVERVIEW.md → "Code Instantiation Flow"
2. Look up exact line in LINE_REFERENCE.md
3. Read implementation in CRITICAL_FILES.md

### I need to understand foot skating suppression
1. Read CRITICAL_FILES.md → Section 5 (Loss Functions)
2. Look at KimodoStyleAuxLoss (lines 70–121 in implementation)
3. Review the three loss terms: joint_pos, joint_vel, fk_consistency

### I need motion 198-dim details
1. SYSTEM_OVERVIEW.md → "Motion Dimensionality: Detailed Breakdown"
2. Or CRITICAL_FILES.md → Section 7 (Motion Representation Details)
3. Component ranges in LINE_REFERENCE.md

### I need to resume training
1. SYSTEM_OVERVIEW.md → "Quick Action: Reproducing Phase 2b Training"
2. Check checkpoint paths in LINE_REFERENCE.md
3. Review load_from config in CRITICAL_FILES.md

---

## 📍 Critical File Locations

All files are in the project directory. Key paths:

```
hftrainer/models/motion/hymotion_m2m/
├─ bundle.py                          ← HyMotionM2MBundle class
├─ network/
│  ├─ m2m_loss.py                    ← M2MLoss (main velocity loss)
│  ├─ kimodo_aux_loss.py             ← KIMODO auxiliary losses
│  ├─ smpl_lite.py                   ← FK body model
│  └─ geometry.py                    ← rot6d ↔ matrix conversions

hftrainer/datasets/motion/motionhub/transforms/
└─ fk_utils.py                        ← SMPL22_PARENTS, FK/IK utils

configs/hymotion_m2m_v2/
├─ _base_hymotion_m2m_v2_046b.py     ← Base config (v2 0.46B)
├─ hymotion_m2m_v2_caption_local_phase2b.py   ← Caption Phase 2b (CURRENT)
└─ hymotion_m2m_v2_uncond_local_cmean.py      ← Unconditioned cmean (CURRENT)

data/hymotion_m2m_data/
├─ _stats_198dim/
│  ├─ Mean.npy                       ← Per-dim mean (198,)
│  └─ Std.npy                        ← Per-dim std (198,)
└─ bone_offsets_22.pt                ← Fallback bone offsets
```

---

## 🔑 Key Concepts at a Glance

### The 198-Dim Motion
```
[0:3]       Translation (3D world position)
[3:135]     22 joints × 6D rot6d (132D, row-major format)
[135:198]   21 joints × 3D position (63D, XZ rel-pelvis, Y absolute)
```

### The Three Loss Tracks (Phase 2b)

**1. M2MLoss (main velocity loss)**
- `velocity_loss_reduction='component_mean'` → equal 25% to each of 4 groups
- `trans_dim_weight=1.0` → no per-dim upweighting (already handled by component_mean)
- `motion_smoothness_weight=0.5` → temporal consistency

**2. KIMODO Auxiliary Loss**
- `joint_pos_weight=50.0` → suppress pelvis cheating
- `joint_vel_weight=500.0` → main skating killer
- `fk_consistency_weight=1500.0` → enforce pos↔rot consistency
- All three run in denormalised metres (world space)

**3. Smoothness Regularization**
- Built into M2MLoss as `motion_smoothness_weight`

### Component Breakdown (component_mean mode)
```
trans       | 3D   | 25% of loss
root_rot    | 6D   | 25% of loss
body_rot    | 126D | 25% of loss (was drowning out translation before!)
joint_pos   | 63D  | 25% of loss
```

### Rotation Space
- **Local** SMPL (not global)
- Stored in `bundle.rotation_space = 'local'`
- Config: `rotation_space='local'` in all v2 configs

### Normalization
- Input: normalized by per-dim mean/std
- Mean/Std: `data/hymotion_m2m_data/_stats_198dim/Mean.npy`, `Std.npy`
- Denorm: `latent_denorm = latent * std + mean` (clamped to avoid div-by-zero)

### VACE Mode
- 'no_inactive' (v2 slim)
- Model input = x_t + reactive + mask = 3×198 = 594-dim
- Saves GPU memory vs v1 'split_reactive' (3×198 full VACE)

---

## 💡 Critical Insights

### Why Phase 2b Config Changed
1. **component_mean:** Translation was ~1.5% of loss (body_rot dominated). Now 25%.
2. **trans_dim_weight=1.0:** Avoid overcorrection. At 5.0 with component_mean → 55% (too much).
3. **KIMODO enabled:** Direct world-space supervision for skating prevention.

### Why KIMODO Has 3 Terms
1. **joint_pos (γ₃):** Prevents pelvis translating without moving legs
2. **joint_vel (γ₄):** Catches any residual skating (velocity mismatch)
3. **fk_consistency (γ₇):** Teaches pos channels to match FK(rot)

### Why Foot Skating Is Hard
- Relative-pelvis representation in 198-dim allows "cheating" (pelvis moves, legs static)
- Element-mean loss hides this under body_rot dominance
- KIMODO uses world-space supervision that cannot be cheated

### Why Row-Major rot6d Matters
- Training data: row-major `[R00, R01, R10, R11, R20, R21]`
- FK utils conversion: row-major ↔ col-major ↔ matrix ↔ FK/IK
- Bundle decode: uses row-major natively via `geometry.py`

---

## 📊 Loss Weight Magnitudes Explained

### M2MLoss under component_mean (Phase 2b)
```
Per-component loss = smooth_l1(pred_comp, gt_comp, reduction='none')
                   × mask_comp × trans_dim_weight

trans_dim_weight=1.0 because component_mean already weights each component equally (25%)
```

### KIMODO in denormalised metres
```
joint_pos_loss = smooth_l1(fk_world_pos, gt_world_pos) * 50
joint_vel_loss = smooth_l1(fk_world_vel, gt_world_vel) * 500
fk_consistency_loss = smooth_l1(pos_channels, fk_rel_pelvis_pos) * 1500

With timestep_squared_weighting=True, multiply by (t/1000)²
```

### Proportion in total loss
- Velocity loss: ~97% (dominant)
- KIMODO aux: ~3% (but concentrated on skating suppression)
- Total: velocity_weight + sum(KIMODO weights × typical base value)

---

## ⚠️ Common Pitfalls

| Pitfall | Why | Fix |
|---------|-----|-----|
| Translation drowning in loss | element_mean with 126-dim body_rot | Use component_mean |
| Overcorrection in translation | trans_dim_weight=5.0 with component_mean | Lower to 1.0 |
| Feet still skate | joint_pos alone insufficient | Use all 3 KIMODO terms |
| FK consistency disabled | fk_consistency_weight=0.0 in M2MLoss | Use KIMODO version instead |
| Position channels wrong | Compute198DimPosition runs AFTER LocalToGlobalRotation | Reorder pipeline |
| Null embeddings wrong | Old intermediate checkpoint | Patch from T2M pretrained |
| Cannot find bone_offsets | Fallback file missing | Run precompute script or use body model |

---

## 🚀 Starting Points

### "I'm new to this system"
→ Read SYSTEM_OVERVIEW.md in order

### "I need to train or resume"
→ SYSTEM_OVERVIEW.md → "Quick Action: Reproducing Phase 2b Training"

### "I need to fix a bug"
→ SYSTEM_OVERVIEW.md → "Where to Look for What" → CRITICAL_FILES.md → code

### "I need to change a loss weight"
→ LINE_REFERENCE.md → find config and line → edit → check interactions

### "I need to understand foot skating"
→ CRITICAL_FILES.md → Section 5: KimodoStyleAuxLoss

### "I need motion representation details"
→ SYSTEM_OVERVIEW.md → "Motion Dimensionality" or CRITICAL_FILES.md → Section 7

---

## 📞 Questions & Answers

**Q: What's the difference between phase 2b and base config?**
A: See CRITICAL_FILES.md Section 6, or SYSTEM_OVERVIEW.md "Critical Config Decisions"

**Q: Why are there 3 loss terms in KIMODO?**
A: See CRITICAL_FILES.md Section 5, or SYSTEM_OVERVIEW.md "Loss Weight Magnitudes"

**Q: How do I change the foot skating suppression strength?**
A: Modify `joint_vel_weight` in `kimodo_aux_loss_cfg`. See LINE_REFERENCE.md for line numbers.

**Q: What's the SMPL-22 kinematic tree?**
A: See SYSTEM_OVERVIEW.md "SMPL-22 Kinematic Tree" with ASCII diagram, or fk_utils.py lines 29–52

**Q: How does component_mean work?**
A: See CRITICAL_FILES.md Section 5 (m2m_loss.py), or LINE_REFERENCE.md under m2m_loss.py

**Q: Which losses are mutually exclusive?**
A: M2MLoss.fk_consistency and KIMODO.fk_consistency (use only one)

**Q: How do I resume training from phase 2?**
A: See SYSTEM_OVERVIEW.md "Quick Action" section

---

## 📈 Document Statistics

| Document | Size | Sections | Focus |
|----------|------|----------|-------|
| SYSTEM_OVERVIEW.md | 6.5 KB | 12 | Architecture & decisions |
| CRITICAL_FILES.md | 14 KB | 9 | Implementation details |
| LINE_REFERENCE.md | 7.1 KB | 8 | Code locations |
| **Total** | **27.6 KB** | **29** | Complete reference |

Last updated: 2026-05-12
Created for: HyMotion M2M v2 motion editing system
Scope: Bundle, losses, configs, FK/kinematics

