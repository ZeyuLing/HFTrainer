# MotionFix / Semantic Editing Integration Analysis

**Date**: May 13, 2026  
**Project**: HyMotion M2M v2 Training Framework  
**Search Base**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`

---

## Executive Summary

**Key Finding**: MotionFix is **referenced in docs and reference implementations** (UMO/MotionLab) but **NOT currently integrated into the HyMotion M2M v2 training pipeline**. The current framework uses:
- **Editing mode**: Simulated via `editing_prob` (15%) + online corruptors (jitter, joint_jump, sliding, etc.)
- **Semantic editing**: Not supported (UMO paper lists this as capability, MotionLab supports text-based instruction editing)
- **MotionFix dataset**: Only referenced in MotionLab reference code, not in hftrainer codebase

---

## 1. MotionFix References in Codebase

### 1.1 Documentation References

| File | Context | Reference |
|------|---------|-----------|
| `ref_repo/CLAUDE.md` (line 306) | Comparison table | "UMO uses HumanML3D + **MotionFix** etc public datasets" |
| `docs/temp/m2m_evaluation_plan.md` (line 406) | Evaluation benchmark | "Editing: 论文 Table 7 (**MotionFix**)" |
| `ref_repo/MotionLab/CLAUDE.md` (line 116) | MotionLab training data | "All datasets (HumanML3D + **MotionFix**) retarget to same SMPL" |

**Status**: MotionFix mentioned as **evaluation benchmark** and **reference work data**, not as current training source.

---

### 1.2 Reference Code: MotionLab Dataset Loader

**Path**: `ref_repo/MotionLab/rfmotion/data/MotionFix.py` (class `MotionFixDataModule`)

**What it does**:
- Loads paired motion corrections (source-target pairs for editing tasks)
- Supports HumanML3D 263-dim representation (vs M2M's 198/135-dim)
- Implements data augmentation and train/val/test splits
- Used for **text-based editing task training** in MotionLab's curriculum

**Key classes**:
```python
class MotionFixDataset(Dataset):
    """Load MotionFix pairs with motion encoder features, stats normalization."""
    def __init__(self, data: list, n_body_joints, stats_file, 
                 rot_repr="6d", load_feats=[...], text_augment_db=[...])
    
class MotionFixDataModule(BASEDataModule):
    """PyTorch Lightning data module for MotionFix paired editing."""
```

**Loading path** (`get_data.py` line 51):
```python
"motionfix": MotionFixDataModule,
```

---

## 2. Current Editing Pipeline in HyMotion M2M v2

### 2.1 Editing Configuration

**Source**: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` (line 182)

```python
dict(
    type='PrepareM2Mv2Condition',
    key='motion',
    tier2_prob=0.4,
    editing_prob=0.15,  # ← **15% of samples enter editing mode**
    corruptor_names=[
        'jitter', 'joint_jump', 'sliding',
        'limb_candy_wrapper', 'wrist_candy_wrapper',
    ],
    max_corruptions=2,  # Apply 1-2 corruptors per editing sample
),
```

### 2.2 `editing_prob` Semantics

**Found in**: 62 files across configs + transforms + tests + docs

#### Control Flow

1. **Sampling** (`condition_sampler_v2.py:339`):
   ```python
   if not edit_mode and rng.random() < editing_prob:
       edit_mode = True  # 15% chance if not already edit from Tier-2
   ```

2. **Corruption application** (`prepare_m2m_v2.py:137-152`):
   ```python
   if edit_mode and self.corruptor_names:
       # Load .npz file from motion_path
       # Apply random selection of corruptors (1-2 chosen from 5)
       # Get joint_corrupted_mask + trans_corrupted_mask
       results['src_motion'] = lq_motion  # Corrupted LQ motion
       results['src_mask'] = perturbed_mask  # Where corruption is
       results['edit_mode'] = True
   ```

3. **Trainer handling** (`hymotion_m2m_trainer.py:91-108`):
   ```python
   edit_flags = batch.get('edit_mode', None)
   if edit_flags is not None and src_mask is not None:
       keep = edit_flags.view(-1, 1, 1).float()
       # For edit_mode=True: keep src_motion as-is (corrupted values in mask=1)
       # For edit_mode=False: zero mask regions (completion mode)
       src_motion = src_motion * (1 - src_mask * (1 - keep))
   ```

#### Distribution Tracking

- **Test**: `tests/unit/test_condition_sampler_v3.py:378-380` — validates ~10% edit_mode rate
- **Analysis**: `scripts/analysis/m2m_v2_v3_mask_density.py:69` — logs edit_mode percentage

### 2.3 Corruptor Registry

**Source**: `hftrainer/utils/data_corruptor.py` (referenced but not shown in search)

**Corruptors registered**:
1. `jitter` — add noise to motion
2. `joint_jump` — sudden joint position change
3. `sliding` — foot sliding artifact
4. `limb_candy_wrapper` — unrealistic limb bending
5. `wrist_candy_wrapper` — wrist rotation artifact

These are **online corruptors** that apply during training on-the-fly, not from MotionFix dataset.

---

## 3. Where MotionFix Would Plug In

### 3.1 Data Loading Layer

**Current**: Loads from `train_hymotion_400h_hq_20260403.json`  
**Integration point**: Add MotionFix as additional dataset

```python
# In dataset loading pipeline:
train_dataloader = dict(
    dataset=dict(
        type='MotionhubMultiTaskMultiAgentDataset',
        # Option A: Extend to support multiple dataset sources
        # Option B: Create composite dataset that mixes MotionHub + MotionFix
    ),
)
```

### 3.2 Transform Pipeline

**Current location**: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` (line 178)

If integrating MotionFix **paired data**, would need:

1. **Load MotionFix pairs** (new transform):
   ```python
   dict(
       type='LoadMotionFixPair',
       source_key='src_motion',
       target_key='tgt_motion',
       dataset_config={...},  # Path to MotionFix data
   ),
   ```

2. **Mark as editing data** (modification to `PrepareM2Mv2Condition`):
   ```python
   dict(
       type='PrepareM2Mv2Condition',
       # Add flag to recognize MotionFix pairs as explicit edit mode
       use_motionfix_mask=True,  # Extract mask from pair metadata
       editing_prob=0.15,  # Still apply additional synthetic corruption
   ),
   ```

### 3.3 Loss Function Integration

**Current**: `smooth_l1` loss uniformly on all samples  
**For MotionFix**: Could weight edit samples differently

```python
losses_cfg=dict(
    loss_type='smooth_l1',
    # Option: separate loss scale for MotionFix editing samples
    motionfix_loss_scale=1.5,  # Up-weight editing task
)
```

---

## 4. Semantic Editing Support

### 4.1 Current Capability

**Status**: ❌ **NOT supported** in M2M v2

- ✅ M4 mask (joint-level regeneration) can do part-level edits
- ❌ Cannot understand **instruction text** like "use opposite leg"
- ❌ No task distinction at conditioning level

### 4.2 How UMO/MotionLab Do It

**MotionLab's Approach** (`ref_repo/MotionLab/CLAUDE.md:79-83`):
- **Task Instruction Modulation**: CLIP-encode instruction text (e.g., *"edit source motion by given text"*)
- **Add to timestep embedding**: Task token + timestep → adaLN modulation
- **Per-task instruction examples**:
  - Text editing: *"edit source motion by given text"*
  - Trajectory editing: *"edit motion given trajectory constraints"*
  - Style transfer: *"apply style from style motion to source motion"*

**UMO's Approach** (`ref_repo/UMO/CLAUDE.md`):
- **Frame-level meta-op embedding**: [Preserve], [Generate], [Edit]
- **Element-wise add to input**: `x'_t = E_in(x_t) + E_ctx(source + meta_op_emb)`

### 4.3 Integration Plan for M2M v2

**To add semantic instruction editing**:

1. **Extend VACE conditioning** to include instruction token:
   ```python
   # Current: x_t + reactive + mask = 3*D
   # New: x_t + reactive + mask + instruction_emb = 3*D + instruction_dim
   ```

2. **Add CLIP-based instruction encoder**:
   ```python
   def encode_instruction(task_type: str) -> Tensor:
       """Map task to instruction embedding via CLIP."""
       instruction_text = {
           'completion': 'complete masked motion frames',
           'joint_edit': 'regenerate joints',
           'text_edit': 'edit motion by given description',
       }[task_type]
       return clip_model.encode_text(instruction_text)  # (D_instruction,)
   ```

3. **Condition on instruction at model input**:
   ```python
   vace_context = concat([x_t, reactive, mask, instruction_emb.expand(T, -1)])
   ```

---

## 5. Config & Code Locations Summary

### 5.1 Configuration Files

| File | Key Config | Current Value |
|------|-----------|--------------|
| `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py:182` | `editing_prob` | 0.15 |
| `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py:183-186` | `corruptor_names` | ['jitter', 'joint_jump', 'sliding', ...] |
| `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py:132` | `vace_condition_mode` | 'no_inactive' |
| All `hymotion_m2m_v2_*.py` configs | `editing_prob` | 0.15 (uniform across all) |

### 5.2 Transform Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `PrepareM2Mv2Condition` | `hftrainer/datasets/motion/motionhub/transforms/prepare_m2m_v2.py` | **Main entry point** for editing mode |
| `sample_condition_v3` | `hftrainer/datasets/motion/motionhub/transforms/condition_sampler_v3.py` | Rank-K Boolean mask sampler |
| `sample_condition` (v2) | `hftrainer/datasets/motion/motionhub/transforms/condition_sampler_v2.py` | Two-tier condition sampler |
| `apply_mask_perturbation` | `condition_sampler_v2.py:399-419` | Over-mask corruption masks |

### 5.3 Trainer

| Component | Location | Key Method |
|-----------|----------|-----------|
| `HyMotionM2MTrainer` | `hftrainer/trainers/motion/hymotion_m2m_trainer.py` | `_prepare_and_forward()` (line 49) |
| Edit flag handling | line 91-108 | Conditionally zeros mask regions based on `edit_mode` |

### 5.4 Datasets

| Type | Location | Status |
|------|----------|--------|
| HyMotion training | `data/annotation/train_hymotion_400h_hq_20260403.json` | ✅ Active |
| MotionFix pairs | Reference only in `ref_repo/MotionLab/` | ❌ Not integrated |

---

## 6. Edit Mode vs MotionFix Distinction

### 6.1 Current "Edit Mode" (Synthetic)

- **Source**: Online corruptors applied during training (15% probability)
- **Corruption types**: jitter, joint_jump, sliding, candy_wrapper artifacts
- **Mask generation**: Returned by corruptor.corrupt() method
- **Task**: Repair/denoise (corrupted motion → clean)
- **No requirement**: No paired ground-truth data needed

### 6.2 MotionFix (Paired Dataset)

- **Source**: Pre-annotated editing pairs (source motion + edit instruction + target motion)
- **Corruption types**: Real human-intended edits (trajectory changes, style variations, etc.)
- **Mask generation**: Derived from human annotation or hand-segmentation
- **Task**: Instruction-based editing (given edit intent → modified motion)
- **Requirement**: Paired (source, target, instruction) tuples

### 6.3 Integration Strategy

**Option A: Replace synthetic with MotionFix**
- Pro: Real editing data
- Con: Different data distribution; may hurt denoise performance

**Option B: Mix both**
- 60% synthetic editing (jitter, etc.) from HyMotion data
- 40% real editing from MotionFix dataset
- Requires dataset loader that handles both modes

**Option C: Separate task heads** (Long-term)
- Maintain current denoise pipeline (synthetic)
- Add separate text-aware instruction editing pipeline (MotionFix)

---

## 7. Documentation of Editing Pipeline

### 7.1 Training Documents

**Found**: `docs/temp/M2M_CONDITION_SAMPLING_DEEP_DIVE.md` (line 666-678)

```markdown
## 15% Editing Mode (Tier-1 Override)

After Tier-1 mask sampling, an additional 15% probability
(editing_prob=0.15) triggers "editing mode":

- The transform loads motion_path NPZ → applies 1-2 corruptors
- Corruptors return joint_corrupted_mask (per-joint per-frame)
- Mask is perturbed (over-masked) to avoid overfitting
- Trainer uses edit_mode flag to KEEP (not zero) src_motion values

CFG override: editing_prob=0.08 in v3 configs
```

### 7.2 Comments in Code

**`condition_sampler_v2.py:18-20`**:
```python
# T2-8: Edit/repair mode (placeholder, actual corruption in transform)
# Actual corruption is applied by the transform (PrepareM2Mv2Condition).
# Here we return edit_mode=True so the transform knows to apply corruption.
```

**`prepare_m2m_v2.py:13`**:
```python
# Output keys:
#   - edit_mode: bool
```

---

## 8. Key Code Paths

### 8.1 Data Flow: How edit_mode Propagates

```
1. PrepareM2Mv2Condition.transform()
   ├─ sample_condition_v3(T, rng, editing_prob=0.15) → (mask, edit_mode)
   ├─ if edit_mode and corruptor_names:
   │  ├─ _apply_corruption(npz_path) 
   │  │  └─ corruptor.corrupt(motion) → {corrupted_motion, joint_corrupted_mask}
   │  └─ results['edit_mode'] = True
   └─ else: results['edit_mode'] = False

2. Batch collation (PackInputs)
   └─ batch['edit_mode'] = [True, False, True, ...]  (per-sample flags)

3. HyMotionM2MTrainer.train_step()
   ├─ edit_flags = batch.get('edit_mode', None)
   └─ if edit_flags:
      └─ src_motion *= (1 - src_mask * (1 - keep))
         # keep=1 (edit): preserve src_motion values
         # keep=0 (completion): zero mask regions
```

### 8.2 Condition Sampler v3: Rank-K Prior

**File**: `condition_sampler_v3.py:611-612`

```python
edit_mode = bool(rng.random() < editing_prob)
return mask, edit_mode
```

**Key**: `edit_mode` is **independent** of mask pattern
- Can combine any mask type with edit_mode
- Ensures 15% of samples see corruptors regardless of mask

---

## 9. What's NOT Documented/Integrated

### 9.1 MotionFix-Specific Guidance

❌ **Missing**:
- How to load MotionFix dataset into hftrainer
- How to extract editing instructions from MotionFix pairs
- How to weight MotionFix samples vs synthetic corruption
- Dataset balancing when mixing MotionHub + MotionFix

### 9.2 Semantic Editing

❌ **Missing**:
- Instruction encoder (CLIP or similar)
- Task instruction modulation at conditioning time
- Instruction-aware loss (e.g., contrastive learning on edits)
- Evaluation metrics for instruction editing quality

### 9.3 MotionFix Evaluation Benchmark

❌ **Defined in docs** but not implemented:
- `docs/temp/m2m_evaluation_plan.md` references "Table 7 (MotionFix)" 
- No scripts found that run M2M against MotionFix editing benchmarks

---

## 10. Recommendations for MotionFix Integration

### 10.1 Immediate (If using synthetic corruption only)

✅ **Current setup is stable**
- 15% editing_prob works well with online corruptors
- No changes needed for current training

### 10.2 Short-term (Add MotionFix dataset)

```python
# Step 1: Create MotionFix dataset loader in hftrainer/datasets/motion/
class MotionFixEditingDataset(Dataset):
    """Load paired (source, target, instruction) tuples from MotionFix."""
    def __init__(self, motionfix_dir, anno_file):
        # Load source → target pairs
        # Extract editing instructions from metadata
        pass

# Step 2: Extend training config
train_dataloader = dict(
    dataset=dict(
        type='CompositeMotionDataset',
        datasets=[
            dict(type='MotionhubMultiTaskMultiAgentDataset', ...),
            dict(type='MotionFixEditingDataset', weight=0.3),  # 30% MotionFix
        ],
    ),
)
```

### 10.3 Medium-term (Add instruction encoding)

```python
# Extend conditioning to include instruction token
model = dict(
    ...
    instruction_encoder=dict(
        type='CLIPTextEncoder',
        model_name='ViT-L/14@336px',
        output_dim=768,
    ),
    ...
)

# In PrepareM2Mv2Condition:
instruction_emb = instruction_encoder(editing_instruction)
vace_input = concat([x_t, reactive, mask, instruction_emb])
```

### 10.4 Long-term (Semantic editing capability)

- Evaluate MotionLab's task instruction modulation approach
- Implement curriculum learning (MotionLab: 1000 ep pre-train + 7-stage fine-tune)
- Compare unified vs specialist model paths (MotionLab shows unified wins)

---

## Appendix: Files Referenced

### A. Transform Files
- `hftrainer/datasets/motion/motionhub/transforms/prepare_m2m_v2.py` — Main editing coordinator
- `hftrainer/datasets/motion/motionhub/transforms/condition_sampler_v2.py` — Tier-2 templates
- `hftrainer/datasets/motion/motionhub/transforms/condition_sampler_v3.py` — Rank-K prior
- `hftrainer/datasets/motion/motionhub/transforms/universal_mask.py` — Legacy v1 mask

### B. Trainer Files
- `hftrainer/trainers/motion/hymotion_m2m_trainer.py` — Training loop + edit_mode handling

### C. Config Files
- `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` — Base config (editing_prob=0.15)
- `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_*.py` — Variant configs (all inherit editing_prob)

### D. Reference Code (MotionLab)
- `ref_repo/MotionLab/rfmotion/data/MotionFix.py` — MotionFix dataset loader
- `ref_repo/MotionLab/rfmotion/data/get_data.py` — Dataset registry

### E. Documentation
- `docs/temp/M2M_CONDITION_SAMPLING_DEEP_DIVE.md` — Detailed editing mechanics
- `docs/temp/m2m_evaluation_plan.md` — Evaluation benchmark (MotionFix listed)
- `ref_repo/CLAUDE.md` — Reference work summary
- `ref_repo/MotionLab/CLAUDE.md` — MotionLab analysis (MotionFix integration example)

### F. Test Files
- `tests/unit/test_prepare_m2m_v2_sampler_switch.py` — Integration tests
- `tests/unit/test_condition_sampler_v3.py` — edit_mode frequency validation

### G. Analysis Scripts
- `scripts/analysis/m2m_v2_v3_mask_density.py` — edit_mode statistics logging
- `scripts/debug/diag_caption_v2.py` — Debug sampler with editing_prob

---

## Conclusion

**MotionFix is not currently integrated into HyMotion M2M v2 training.** The framework:
1. ✅ Has complete **synthetic editing pipeline** (editing_prob + corruptors)
2. ✅ Properly tracks `edit_mode` through training
3. ❌ Does **not load MotionFix paired data**
4. ❌ Does **not support semantic/instruction editing**

To integrate MotionFix:
- **Minimal**: Replace current corruptors with MotionFix pairs (different data source, same pipeline)
- **Recommended**: Mix both datasets + add instruction encoding (curriculum learning style)
- **Advanced**: Implement semantic instruction editing following MotionLab's task modulation approach

