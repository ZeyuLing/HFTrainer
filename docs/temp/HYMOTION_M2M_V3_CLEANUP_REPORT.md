# HyMotion M2M v3/CRFM Cleanup Report
**Date**: 2026-05-12  
**Repository**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/`

---

## Executive Summary

This report identifies **ALL HyMotion M2M v3-related files** (v3 variant = "CRFM", Condition-Routed Flow Matching with Text-Awareness Loss / TAL) that need to be cleaned up. The v3 implementation is **complete but experimental** and should be removed to avoid confusion with the production v2 models.

**Total Files to Delete**: 15  
**Total Files to Edit**: 9  
**Total Directories to Delete**: 1

---

## 1. CONFIGURATION FILES (7 files to DELETE)

### Directory: `configs/hymotion_m2m_v3/`

All files in this directory are v3-specific experimental configurations and should be **DELETED ENTIRELY**.

| File | Size | Lines | Status | Reason |
|------|------|-------|--------|--------|
| `_base_hymotion_m2m_v3_046b.py` | 6.4 KB | ~150 | DELETE | Base config for v3 model (DSCF architecture) |
| `hymotion_m2m_v3_caption_046b.py` | 3.0 KB | ~70 | DELETE | v3 caption variant config |
| `hymotion_m2m_v3_caption_local_046b.py` | 3.9 KB | ~90 | DELETE | v3 caption local variant config |
| `hymotion_m2m_v3_debug.py` | 3.3 KB | ~80 | DELETE | v3 debug config |
| `hymotion_m2m_v3_smoke.py` | 2.9 KB | ~70 | DELETE | v3 smoke test config |
| `hymotion_m2m_v3_uncond_046b.py` | 2.7 KB | ~65 | DELETE | v3 unconditional config |
| `hymotion_m2m_v3_uncond_local_046b.py` | 3.0 KB | ~75 | DELETE | v3 unconditional local config |

**Action**: **DELETE ENTIRE DIRECTORY** `configs/hymotion_m2m_v3/` (7 files, total ~25 KB)

**References in directory**:
```
configs/hymotion_m2m_v3/
├── _base_hymotion_m2m_v3_046b.py          # model: HyMotionM2Mv3Bundle, trainer: HyMotionM2Mv3Trainer
├── hymotion_m2m_v3_caption_046b.py        # inherits _base_hymotion_m2m_v3_046b.py
├── hymotion_m2m_v3_caption_local_046b.py  # inherits _base_hymotion_m2m_v3_046b.py
├── hymotion_m2m_v3_debug.py               # inherits _base_hymotion_m2m_v3_046b.py
├── hymotion_m2m_v3_smoke.py               # inherits _base_hymotion_m2m_v3_046b.py
├── hymotion_m2m_v3_uncond_046b.py         # inherits _base_hymotion_m2m_v3_046b.py
└── hymotion_m2m_v3_uncond_local_046b.py   # inherits _base_hymotion_m2m_v3_046b.py
```

---

## 2. MODEL BUNDLE FILES (2 files)

### File: `hftrainer/models/motion/hymotion_m2m/bundle_v3.py`

- **Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/models/motion/hymotion_m2m/bundle_v3.py`
- **Size**: ~15 KB
- **Class**: `HyMotionM2Mv3Bundle` (extends `ModelBundle`)
- **Status**: **DELETE**
- **Purpose**: Implements v3 model bundle for Dual-Stream Condition Fusion (DSCF) architecture. Removed from production in favor of v2.

**Key Methods**:
```python
class HyMotionM2Mv3Bundle(ModelBundle):
    def __init__(...)  # Lines 60-150
    def prepare_padding(...)
    def prepare_vace_input(...)
    def predict_flow(...)
    def decode_motion_from_latent(...)
    def mask_text_cond(...)
    def encode_text(...)
```

**Pycache files to delete**:
- `hftrainer/models/motion/hymotion_m2m/__pycache__/bundle_v3.cpython-39.pyc`
- `hftrainer/models/motion/hymotion_m2m/__pycache__/bundle_v3.cpython-311.pyc`

---

### File: `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit_v3.py`

- **Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit_v3.py`
- **Size**: ~30+ KB
- **Class**: `HunyuanMotionMMDiTv3` (extends `nn.Module`)
- **Status**: **DELETE**
- **Purpose**: v3 transformer architecture with Condition Density Embedding and Dual-Stream fusion.

**Key Components**:
- `DualCondMMDiTBlock`: Motion + text cross-attention with adaptive fusion gate
- `MotionCondEncoder`: 128 queries, 4 layers for motion condition encoding
- `RoleEmbedding`: Per-frame KEEP/GENERATE/EDIT role embeddings
- `TimestepAdaptiveFusionGate`: Learns text vs motion balance per timestep

**Pycache file to delete**:
- `hftrainer/models/motion/hymotion_m2m/network/__pycache__/hymotion_mmdit_v3.cpython-311.pyc`

---

## 3. NETWORK/CONDITION ROUTING (1 file)

### File: `hftrainer/models/motion/hymotion_m2m/network/condition_routing.py`

- **Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/models/motion/hymotion_m2m/network/condition_routing.py`
- **Size**: ~15 KB
- **Status**: **DELETE**
- **Purpose**: CRFM-specific modules:
  - `ConditionDensityEmbedding` (CDE): Encodes mask density via sinusoidal positional encoding + MLP
  - `TextAttentionPreservation` (TAP): Gradient scaling for text-related parameters
  - `text_awareness_loss()` (TAL): Regularization ensuring text conditioning affects generated regions

**Key Functions**:
```python
class ConditionDensityEmbedding(nn.Module):  # Lines 26-68
class TextAttentionPreservation(nn.Module):  # Lines 71-120
def text_awareness_loss(...):                # Lines 123-200
```

**Pycache file to delete**:
- `hftrainer/models/motion/hymotion_m2m/network/__pycache__/condition_routing.cpython-39.pyc`

---

## 4. TRAINER FILES (2 files)

### File: `hftrainer/trainers/motion/hymotion_m2m_crfm_trainer.py`

- **Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/trainers/motion/hymotion_m2m_crfm_trainer.py`
- **Size**: ~20 KB
- **Class**: `HyMotionM2MCRFMTrainer` (extends `HyMotionM2MTrainer`)
- **Status**: **DELETE**
- **Purpose**: CRFM trainer adding Text-Awareness Loss (TAL) regularization.

**Key Methods**:
```python
class HyMotionM2MCRFMTrainer(HyMotionM2MTrainer):
    def __init__(...)           # Lines 46-70: tal_weight, tal_interval, tal_density_threshold
    def train_step(batch)       # Lines 80-130: Adds TAL computation
    def _compute_tal_loss(...)  # Lines 132-180: TAL regularization
```

**Pycache files to delete**:
- `hftrainer/trainers/motion/__pycache__/hymotion_m2m_crfm_trainer.cpython-311.pyc`
- `hftrainer/trainers/motion/__pycache__/hymotion_m2m_crfm_trainer.cpython-39.pyc`

---

### File: `hftrainer/trainers/motion/hymotion_m2m_v3_trainer.py`

- **Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/trainers/motion/hymotion_m2m_v3_trainer.py`
- **Size**: ~25 KB
- **Class**: `HyMotionM2Mv3Trainer` (extends `BaseTrainer`)
- **Status**: **DELETE**
- **Purpose**: v3 trainer for DSCF architecture training. No VACE context, condition_mask + known_motion passed directly to transformer.

**Key Methods**:
```python
class HyMotionM2Mv3Trainer(BaseTrainer):
    def __init__(...)
    def train_step(batch)          # Main training loop
    def _prepare_and_forward(...)  # No VACE preparation
    def _compute_loss(...)
    def _sync_orphan_param_grads(...)
```

**Pycache files to delete**:
- `hftrainer/trainers/motion/__pycache__/hymotion_m2m_v3_trainer.cpython-311.pyc`
- `hftrainer/trainers/motion/__pycache__/hymotion_m2m_v3_trainer.cpython-39.pyc`

---

## 5. DOCUMENTATION FILES (3 files to DELETE)

### Directory: `docs/temp/`

| File | Size | Status | Reason |
|------|------|--------|--------|
| `m2m_v3_crfm_implementation_plan.md` | 33 KB | DELETE | CRFM v3 implementation design document |
| `hymotion_m2m_v3_dual_stream_condition_fusion_plan.md` | 35 KB | DELETE | v3 DSCF architecture design document |
| `survey_motion_gen_embodied_v3_20260512.md` | 33 KB | DELETE | v3 survey/research document |

**These are design/proposal documents for the experimental v3 variant.**

---

## 6. REGISTRY/IMPORT CHANGES (9 files to EDIT)

### File 1: `hftrainer/models/motion/hymotion_m2m/__init__.py`

**Current content**:
```python
from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
from hftrainer.models.motion.hymotion_m2m.bundle_v3 import HyMotionM2Mv3Bundle

__all__ = ['HyMotionM2MBundle', 'HyMotionM2Mv3Bundle']
```

**Action: EDIT**
```python
from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle

__all__ = ['HyMotionM2MBundle']
```

**Lines to remove**:
- Line 4: `from hftrainer.models.motion.hymotion_m2m.bundle_v3 import HyMotionM2Mv3Bundle`
- Line 6: Update `__all__` to remove `'HyMotionM2Mv3Bundle'`

---

### File 2: `hftrainer/models/motion/hymotion_m2m/network/__init__.py`

**Current content**:
```python
from hftrainer.models.motion.hymotion_m2m.network.hymotion_mmdit import HunyuanMotionMMDiT
from hftrainer.models.motion.hymotion_m2m.network.hymotion_dit import HunyuanMotionDiT
from hftrainer.models.motion.hymotion_m2m.network.hymotion_mmdit_v3 import HunyuanMotionMMDiTv3
...
if not HF_MODELS.get('HunyuanMotionMMDiTv3'):
    HF_MODELS.register_module(name='HunyuanMotionMMDiTv3', module=HunyuanMotionMMDiTv3, force=True)
__all__ = ['HunyuanMotionMMDiT', 'HunyuanMotionDiT', 'HunyuanMotionMMDiTv3']
```

**Action: EDIT** — Remove v3 imports and registrations

**Lines to remove**:
- Line 14-16: `from hftrainer.models.motion.hymotion_m2m.network.hymotion_mmdit_v3 import HunyuanMotionMMDiTv3`
- Line 29-32: Registration of `HunyuanMotionMMDiTv3` (4 lines)
- Line 34: Update `__all__` to remove `'HunyuanMotionMMDiTv3'`

---

### File 3: `hftrainer/trainers/motion/__init__.py`

**Current content**:
```python
from hftrainer.trainers.motion.hymotion_m2m_trainer import HyMotionM2MTrainer
from hftrainer.trainers.motion.hymotion_m2m_soar_trainer import HyMotionM2MSoarTrainer
from hftrainer.trainers.motion.hymotion_m2m_crfm_trainer import HyMotionM2MCRFMTrainer
from hftrainer.trainers.motion.hymotion_m2m_v3_trainer import HyMotionM2Mv3Trainer
...
__all__ = [
    'PrismTrainer', 'VermoTrainer',
    'HyMotionM2MTrainer', 'HyMotionM2MSoarTrainer', 'HyMotionM2MCRFMTrainer',
    'HyMotionM2Mv3Trainer',
    'HyMotionT2MTrainer', 'HyMotionUMOTrainer',
    'MotionCLIPTrainer',
]
```

**Action: EDIT**

**Lines to remove**:
- Line 7: `from hftrainer.trainers.motion.hymotion_m2m_crfm_trainer import HyMotionM2MCRFMTrainer`
- Line 8: `from hftrainer.trainers.motion.hymotion_m2m_v3_trainer import HyMotionM2Mv3Trainer`
- Line 15: Remove `'HyMotionM2MCRFMTrainer'` from `__all__`
- Line 16: Remove `'HyMotionM2Mv3Trainer'` from `__all__`

**After edit**:
```python
from hftrainer.trainers.motion.hymotion_m2m_trainer import HyMotionM2MTrainer
from hftrainer.trainers.motion.hymotion_m2m_soar_trainer import HyMotionM2MSoarTrainer
from hftrainer.trainers.motion.hymotion_t2m_trainer import HyMotionT2MTrainer
...
__all__ = [
    'PrismTrainer', 'VermoTrainer',
    'HyMotionM2MTrainer', 'HyMotionM2MSoarTrainer',
    'HyMotionT2MTrainer', 'HyMotionUMOTrainer',
    'MotionCLIPTrainer',
]
```

---

### File 4: `hftrainer/models/motion/hymotion_m2m/bundle.py`

**Current content** (lines ~180-200):
```python
def predict_flow(self, x_t, ...):
    ...
    from hftrainer.models.motion.hymotion_m2m.network.condition_routing import (
        ConditionDensityEmbedding,
    )
```

**Action: EDIT** — Remove the conditional import of `ConditionDensityEmbedding`

This is a lazy import that checks if model uses CRFM. Since CRFM models are being deleted, this import can be removed.

---

### File 5: `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py`

**Current content** (check for conditional v3 imports):
```python
if 'cond_encoder' in config_keys:
    from .condition_routing import ConditionDensityEmbedding
```

**Action: EDIT** — Remove conditional v3 code block

Find and remove any:
- Conditional imports checking for v3 features
- `ConditionDensityEmbedding` usage
- CDE (Condition Density Embedding) initialization

---

### File 6: `hftrainer/trainers/motion/hymotion_m2m_trainer.py`

**Check for conditional CRFM imports**:
```python
def _compute_tal_loss(...):  # Only exists in CRFM trainer
    ...

def train_step(self, batch):
    ...
    if hasattr(self, 'tal_weight'):  # CRFM-specific logic
        tal_loss = self._compute_tal_loss(...)
```

**Action: EDIT** — Check if there are any conditional CRFM branches and remove them

Search for:
- `'tal_'` parameters
- `_compute_tal_loss` method
- Conditional checks for `HyMotionM2MCRFMTrainer`

If none found, **NO ACTION NEEDED** (likely base trainer is clean).

---

### File 7: `hftrainer/models/motion/hymotion_m2m/network/__init__.py`

(Already covered in File 2 above, listed here for completeness)

---

### File 8: `CLAUDE.md` (Root level)

**Check for v3/CRFM references**:
```bash
grep -n "v3\|crfm\|CRFM\|bundle_v3\|HyMotionM2Mv3\|condition_routing" /path/to/CLAUDE.md
```

**Action**: If references found, **EDIT** to remove v3/CRFM mentions from the main documentation

---

### File 9: Any test files referencing v3/CRFM

**Check for**:
```bash
find tests/ -name "*.py" -exec grep -l "v3\|crfm\|CRFM\|bundle_v3" {} \;
```

**Action: EDIT or DELETE** any test files that only test v3/CRFM functionality

---

## 7. SEARCH RESULTS SUMMARY

### All files containing v3/CRFM references:

```
hftrainer/models/motion/hymotion_m2m/network/condition_routing.py       ← DELETE
hftrainer/trainers/motion/hymotion_m2m_crfm_trainer.py                 ← DELETE
hftrainer/models/motion/hymotion_m2m/bundle.py                         ← EDIT (remove CDE import)
hftrainer/models/motion/hymotion_m2m/bundle_v3.py                      ← DELETE
hftrainer/trainers/motion/__init__.py                                  ← EDIT (remove imports)
hftrainer/models/motion/hymotion_m2m/__init__.py                       ← EDIT (remove imports)
hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py         ← EDIT (check for v3 code)
hftrainer/datasets/motion/motionhub/transforms/prepare_m2m_v2.py       ← CHECK (v3 refers to mask sampler, not model)
hftrainer/trainers/motion/hymotion_m2m_v3_trainer.py                   ← DELETE
```

---

## 8. PYCACHE FILES TO DELETE

All `__pycache__` directories containing compiled v3/CRFM code:

```
hftrainer/models/motion/hymotion_m2m/__pycache__/
  - bundle_v3.cpython-39.pyc
  - bundle_v3.cpython-311.pyc

hftrainer/trainers/motion/__pycache__/
  - hymotion_m2m_crfm_trainer.cpython-39.pyc
  - hymotion_m2m_crfm_trainer.cpython-311.pyc
  - hymotion_m2m_v3_trainer.cpython-39.pyc
  - hymotion_m2m_v3_trainer.cpython-311.pyc

hftrainer/models/motion/hymotion_m2m/network/__pycache__/
  - hymotion_mmdit_v3.cpython-39.pyc
  - hymotion_mmdit_v3.cpython-311.pyc
  - condition_routing.cpython-39.pyc (if present)
```

---

## 9. CLEANUP CHECKLIST

### Phase 1: Delete Files
- [ ] Delete `configs/hymotion_m2m_v3/` directory (7 config files)
- [ ] Delete `hftrainer/models/motion/hymotion_m2m/bundle_v3.py`
- [ ] Delete `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit_v3.py`
- [ ] Delete `hftrainer/models/motion/hymotion_m2m/network/condition_routing.py`
- [ ] Delete `hftrainer/trainers/motion/hymotion_m2m_crfm_trainer.py`
- [ ] Delete `hftrainer/trainers/motion/hymotion_m2m_v3_trainer.py`
- [ ] Delete `docs/temp/m2m_v3_crfm_implementation_plan.md`
- [ ] Delete `docs/temp/hymotion_m2m_v3_dual_stream_condition_fusion_plan.md`
- [ ] Delete `docs/temp/survey_motion_gen_embodied_v3_20260512.md`

### Phase 2: Edit Import Files
- [ ] Edit `hftrainer/models/motion/hymotion_m2m/__init__.py`
- [ ] Edit `hftrainer/models/motion/hymotion_m2m/network/__init__.py`
- [ ] Edit `hftrainer/trainers/motion/__init__.py`
- [ ] Edit `hftrainer/models/motion/hymotion_m2m/bundle.py` (remove CDE import)
- [ ] Edit `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py` (remove v3 code)
- [ ] Check `hftrainer/trainers/motion/hymotion_m2m_trainer.py` for CRFM logic
- [ ] Check `hftrainer/models/motion/CLAUDE.md` for v3 references
- [ ] Check `CLAUDE.md` (root) for v3 references

### Phase 3: Clean Pycache
- [ ] Run `find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null` (or delete manually)

### Phase 4: Verify
- [ ] Run `grep -r "bundle_v3\|HyMotionM2Mv3\|CRFM\|crfm_trainer\|hymotion_m2m_v3\|condition_routing" hftrainer/ --include="*.py"` — should return NO results (except in comments/docs)
- [ ] Run `python -c "from hftrainer.models.motion.hymotion_m2m import *; from hftrainer.trainers.motion import *"` — should import cleanly
- [ ] Verify no broken imports in `__all__` exports

---

## 10. IMPACT ANALYSIS

### What Will Break (if not cleaned up)
- **Registry collision**: `HyMotionM2Mv3Bundle` and `HyMotionM2Mv3Trainer` remain registered but unused
- **Import confusion**: Users might accidentally load v3 config/checkpoint expecting v2 behavior
- **Maintenance burden**: Dead code requires updates when SMPL/transformer APIs change
- **Training bloat**: All imports are evaluated on startup, even though v3 is never used

### What Won't Break
- ✅ v2 models (`HyMotionM2MBundle`, `HyMotionM2MTrainer`, `HyMotionM2MSoarTrainer`) — unaffected
- ✅ Dataset transforms (`condition_sampler_v3` for mask sampling) — NOT deleted, only model v3 is removed
- ✅ Existing v2 checkpoints and configs — fully compatible
- ✅ Pipelines (inference) — no v3 pipeline exists, so nothing breaks

---

## 11. SPECIAL NOTE: `condition_sampler_v3.py` (DO NOT DELETE)

**File**: `hftrainer/datasets/motion/motionhub/transforms/condition_sampler_v3.py`

⚠️ **This should NOT be deleted.** The "v3" here refers to the v3 **mask sampling strategy** (Rank-K Boolean Tensor Prior), which is used by v2 models:
- Referenced in `prepare_m2m_v2.py`: `from .condition_sampler_v3 import sample_condition_v3`
- Used by v2 configs with `sampler_version='v3'`
- Independent of the model v3 CRFM implementation

**Action**: **KEEP** `condition_sampler_v3.py` — this is a dataset utility, not a model variant.

---

## 12. FILES TO DELETE — COMPLETE LIST

### Code Files (6 files)
1. `hftrainer/models/motion/hymotion_m2m/bundle_v3.py`
2. `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit_v3.py`
3. `hftrainer/models/motion/hymotion_m2m/network/condition_routing.py`
4. `hftrainer/trainers/motion/hymotion_m2m_crfm_trainer.py`
5. `hftrainer/trainers/motion/hymotion_m2m_v3_trainer.py`
6. (Entire directory) `configs/hymotion_m2m_v3/` — 7 config files

### Documentation Files (3 files)
7. `docs/temp/m2m_v3_crfm_implementation_plan.md`
8. `docs/temp/hymotion_m2m_v3_dual_stream_condition_fusion_plan.md`
9. `docs/temp/survey_motion_gen_embodied_v3_20260512.md`

### Pycache Files (~9 files)
10. `hftrainer/models/motion/hymotion_m2m/__pycache__/bundle_v3.cpython-*.pyc` (2 files)
11. `hftrainer/models/motion/hymotion_m2m/network/__pycache__/hymotion_mmdit_v3.cpython-*.pyc` (2 files)
12. `hftrainer/models/motion/hymotion_m2m/network/__pycache__/condition_routing.cpython-*.pyc` (1 file)
13. `hftrainer/trainers/motion/__pycache__/hymotion_m2m_crfm_trainer.cpython-*.pyc` (2 files)
14. `hftrainer/trainers/motion/__pycache__/hymotion_m2m_v3_trainer.cpython-*.pyc` (2 files)

---

## 13. FILES TO EDIT — COMPLETE LIST

| File | Lines to Remove/Change | Type |
|------|---|---|
| `hftrainer/models/motion/hymotion_m2m/__init__.py` | Line 4, update line 6 `__all__` | Remove v3 import |
| `hftrainer/models/motion/hymotion_m2m/network/__init__.py` | Lines 14-16, 29-32, update line 34 | Remove v3 registration |
| `hftrainer/trainers/motion/__init__.py` | Lines 7-8, update lines 15-16 | Remove CRFM/v3 imports |
| `hftrainer/models/motion/hymotion_m2m/bundle.py` | Search for conditional CDE import | Remove lazy import |
| `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py` | Search for v3-specific code blocks | Remove v3-only branches |
| `hftrainer/trainers/motion/hymotion_m2m_trainer.py` | Search for CRFM logic | Verify no CRFM code |
| `hftrainer/models/motion/CLAUDE.md` | Search for "v3", "crfm", "CRFM" | Remove v3 references |
| `CLAUDE.md` (root) | Search for "v3", "crfm", "CRFM" | Remove v3 references |
| Test files (if any) | Remove v3/CRFM-specific tests | Verify tests |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| **Files to DELETE** | 15 (6 code + 3 docs + 9 pycache) |
| **Files to EDIT** | 8-9 |
| **Lines of code to DELETE** | ~100-150 KB |
| **Directories to DELETE** | 1 (`configs/hymotion_m2m_v3/`) |
| **Time to cleanup** | ~30 minutes (manual deletion + verification) |

---

**End of Report**
