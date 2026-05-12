# Investigation Findings Index — May 12, 2026

Quick reference guide to the three major investigations completed in the previous session.

---

## 1. KIMODO t² (Timestep Squared) Weighting

**Question**: Does KIMODO use t² weighting in its training losses?

**Answer**: ✅ **NO** — KIMODO uses fixed γ weights, not timestep-dependent weighting.

**Key Finding**: Our `timestep_squared_weighting` flag in `kimodo_aux_loss.py` is a **custom addition** we made.

**Documents**:
- Quick answer: `KIMODO_T2_WEIGHTING_ANALYSIS.md` (5.1KB)
- Explains: Why we added t² weighting (down-weight high-noise timesteps)

**Evidence**:
- KIMODO uses fixed γ values (gamma): γ₁=γ₃=γ₅=10, γ₂=2, γ₄=3, γ₆=4, γ₇=5
- No timestep-dependent weighting found in diffusion.py or training code
- Parameter `timestep_squared_weighting` never appears in KIMODO source
- Our docstring explicitly references "existing motion198_fk_loss t-weighting" (our own loss)

**Implication**: No changes needed. Our addition is well-justified for auxiliary FK losses.

---

## 2. KIMODO Heading Representation & 3D Rotation Preservation

**Question**: Does KIMODO's 2D heading [cos(ψ), sin(ψ)] representation lose pitch/roll information?

**Answer**: ✅ **NO** — Full 3D root rotation is preserved and perfectly recoverable.

**Key Finding**: The 2D heading is a **summary feature for canonicalization**, not the actual rotation storage.

**The Real Root Rotation**: Lives in `global_rot_data[0]` as a **6D continuous representation**.

**Documents**:
- Comprehensive analysis: `KIMODO_HEADING_ANALYSIS.md` (13.8KB)
- Quick answer: `KIMODO_HEADING_QUICK_ANSWER.md` (4.2KB)
- Root representation: `KIMODO_ROOT_REPRESENTATION_SUMMARY.md` (11.8KB)

**Evidence**:
- 333-dim feature vector structure (lines 26-41 of kimodo_motionrep.py):
  - dims [3:5]: `global_root_heading` [2] = yaw only
  - dims [86:248]: `global_rot_data` [27×6] = full 3D rotations (includes root at index 0)
- Inverse reconstruction (lines 162-215):
  - Converts `global_rot_data[0]` [6D] → rotation matrix [3×3] → local rotations
  - Never uses `global_root_heading` for reconstruction
- SMPL conversion (exports/smplx.py):
  - Root matrix [3×3] → axis-angle [3D] = full pitch, roll, yaw

**Implication**: KIMODO's design is sound. Can fully convert to SMPL with no information loss.

---

## 3. Height Estimation Fix for Motion Retargeting

**Question**: How to measure human height accurately for robot IK scaling?

**Answer**: ✅ **IMPLEMENTED & TESTED** — FK-based height from joint positions.

**Key Finding**: Height estimated from SMPL-X FK joint positions (head to feet distance).

**Documents**:
- Implementation summary: `EXECUTION_SUMMARY.md` (7.4KB)
- Index and guide: `HEIGHT_FIX_INDEX.md` (8.0KB)
- Implementation guide: `HEIGHT_IMPLEMENTATION_GUIDE.md`

**What Was Fixed**:
- **Before**: Height hardcoded as 1.66m (because `betas[0]` always zero in motion_135)
- **After**: Height estimated from motion (1.4–2.2m range)
- **IK Scaling Impact**: Varies by human size (e.g., 0.91 for 1.55m, 1.09 for 1.85m)

**Evidence**:
- Function added: `estimate_human_height_from_joints()` (lines 11-45, smpl.py)
- Used in: `load_smplx_file()` (lines 88-105) and `load_gvhmr_pred_file()` (lines 157-174)
- Algorithm: Median over middle 50% of frames + clamping to [1.4m, 2.2m]
- Tests: All 7 scenarios passing, ±1mm accuracy on clean data, ±44mm with noise

**Status**: ✅ Code already in production at `ref_repo/GMR/general_motion_retargeting/utils/smpl.py`

**Implication**: Correct robot limb proportions for any human height in retargeting pipeline.

---

## Related Documentation

For deeper understanding, see these complementary documents:

### KIMODO Root Representation (Complete)
- `EXECUTIVE_SUMMARY.md` (6.9KB) — Overview of root design choices
- `KIMODO_ROOT_ANALYSIS.md` (21.4KB) — Detailed technical breakdown
- `KIMODO_vs_HyMotion_Translation.md` (20.4KB) — Comparison and mapping

### Configuration & Training
- `HYMOTION_M2M_V2_TRAINING_CONFIG_REPORT.md` — New Phase 2b config details
- `HYMOTION_M2M_V2_SYSTEM_OVERVIEW.md` — System architecture

### Reference Materials
- `KIMODO_QUICK_REFERENCE.txt` — Visual reference guide
- `KIMODO_DOCUMENTATION_INDEX.md` — Complete file inventory
- `SMPL_SKELETON_REFERENCE.txt` — Joint definitions and mapping

---

## Background Investigations (In Progress)

### Task: null_ctxt_embedding in Uncond Configs
**Status**: Still running (agent abe62ea44ef31d64f)
**Questions**:
- Do uncond configs (cond_mask_prob=0.0) use null embeddings during training?
- Should we add null_embedding_source to uncond configs?

**Target Code**: 
- `configs/hymotion_m2m_v2/` (configs)
- `hftrainer/models/motion/hymotion_m2m/bundle.py` (null logic)
- `hftrainer/trainers/motion/hymotion_m2m_trainer.py` (training)
- `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` (inference)

### Task: KIMODO Heading Conversion Verification
**Status**: Still running (agents a8401204e8de759e1, aa9caf3dd86ae6eb4)
**Purpose**: Verify KIMODO heading can be fully converted back to SMPL

---

## Session Summary

See `SESSION_SUMMARY_20260512.md` for complete session overview including:
- Documentation organization work
- Git status and new config files
- Recommendations for next steps

---

## Quick Navigation

| Topic | Primary Document | Secondary |
|-------|-----------------|-----------|
| **t² Weighting** | `KIMODO_T2_WEIGHTING_ANALYSIS.md` | — |
| **Heading & 3D Rotation** | `KIMODO_HEADING_QUICK_ANSWER.md` | `KIMODO_HEADING_ANALYSIS.md` |
| **Height Estimation** | `EXECUTION_SUMMARY.md` | `HEIGHT_FIX_INDEX.md` |
| **Root Representation** | `EXECUTIVE_SUMMARY.md` | `KIMODO_ROOT_ANALYSIS.md` |
| **Overall Session** | `SESSION_SUMMARY_20260512.md` | — |

---

**Index Created**: May 12, 2026  
**All Analysis Documents**: Located in `docs/temp/`  
**Status**: Ready for reference and next investigation phase
