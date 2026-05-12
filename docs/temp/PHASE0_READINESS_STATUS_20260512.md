# Phase 0 Implementation Readiness Status
**Date:** 2026-05-12  
**Status:** Design finalized, implementation preparation phase

---

## Document Status: ✅ PROPOSAL V1.7 FINALIZED

### What's Complete
1. **Proposal Document (v1.7)**
   - ✅ All dimensional references fixed (198-dim throughout)
   - ✅ KIMODO Root simplified to ADMM translation replacement only
   - ✅ Online conversion specified (dataset __getitem__, not offline)
   - ✅ Phase 0 experiments (E1-E4) fully specified
   - ✅ Loss configuration defined uniformly
   - ✅ All inconsistencies resolved
   - **Location:** `docs/temp/hymotion_m2m_next_gen_proposal_20260511.md`
   - **Recent commit:** `edeedcc` - "proposal v1.7: Finalize KIMODO Root simplification and Phase 0 planning"

---

## Implementation Checklist: Phase 0 Tasks

### Task Group A: Configuration Preparation

#### A1: Config E1 — SMPL Root + Uncond (Baseline)
- **Base:** `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py`
- **Modifications needed:**
  ```python
  # losses_cfg overrides:
  keypoints3d_weight = 10.0  # from 0.0 → enable FK keypoint loss
  
  # kimodo_aux_loss_cfg overrides:
  timestep_squared_weighting = False  # from True → remove t² weighting
  ```
- **Note:** No data pipeline changes needed
- **Status:** Ready for implementation

#### A2: Config E2 — SMPL Root + Caption  
- **Base:** `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py`
- **Modifications needed:** Same as E1 + verify `null_embedding_source` is present
- **Reference:** `hymotion_m2m_v2_caption_local_phase2.py` already has `null_embedding_source='pretrained'`
- **Status:** Ready for implementation

#### A3: Config E3 — KIMODO Root + Uncond
- **Base:** Copy from E1, rename to `hymotion_m2m_v2_kimodo_uncond_046b.py`
- **Modifications needed:**
  ```python
  # Dataset pipeline: add ADMM smoothing after LoadSmplx55
  dict(
      type='SmplTransToKimodoRootOnline',  # New transform
      admm_margin=0.06,
      key='motion',
  ),
  
  # mean_std_dir: point to KIMODO Root statistics
  mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root',
  
  # Same loss config as E1
  keypoints3d_weight = 10.0
  timestep_squared_weighting = False
  ```
- **Requires:** New transform `SmplTransToKimodoRootOnline` implementation
- **Requires:** New mean/std computation for KIMODO Root (198-dim)
- **Status:** Awaiting transform implementation

#### A4: Config E4 — KIMODO Root + Caption
- **Base:** Combine E2 + E3 modifications
- **Modifications:** Same as E3 + `null_embedding_source` from E2
- **Status:** Awaiting E3 prerequisite

### Task Group B: Data Preprocessing

#### B1: Compute KIMODO Root mean/std statistics
- **Input:** Full training dataset (456K high-quality samples)
- **Process:**
  1. Load each SMPL motion (198-dim)
  2. Apply online ADMM smoothing: `smpl_trans_to_smooth_trans(motion_198, margin=0.06)`
  3. Accumulate mean/std for resulting 198-dim
- **Output:** `data/hymotion_m2m_data/_stats_198dim_kimodo_root/mean.npy`, `std.npy`
- **Estimated time:** < 30 minutes (single-pass through training data)
- **Status:** Specified in proposal §6.3

#### B2: Implement SmplTransToKimodoRootOnline transform
- **Location:** `hftrainer/datasets/motion/transforms/` (new file)
- **API:**
  ```python
  class SmplTransToKimodoRootOnline:
      def __init__(self, admm_margin: float = 0.06, key: str = 'motion'):
          pass
      def __call__(self, results: dict) -> dict:
          # Applies smpl_trans_to_smooth_trans() in __getitem__
          pass
  ```
- **Required helper function:**
  ```python
  def smpl_trans_to_smooth_trans(motion_198: Tensor, admm_margin: float = 0.06) -> Tensor:
      # Implementation in proposal §6.3
      pass
  ```
- **Dependencies:** ADMM smoothing algorithm (XZ-only, margin ≤ 6cm)
- **Status:** Algorithm specified in proposal

### Task Group C: Loss Configuration Alignment

#### C1: Position loss in relative-to-root space
- **File:** `hftrainer/losses/motion/m2m_loss.py` (line ~222)
- **Current code:** Computes position loss in absolute space
- **Required change:**
  ```python
  # Before:
  pos_loss = smooth_l1(pred_x1[..., 135:198], target_x1[..., 135:198])
  
  # After (both versions A & B):
  pred_pos_rel = pred_x1[..., 135:198] - expand_to_joints(pred_x1[..., 0:3])
  target_pos_rel = target_x1[..., 135:198] - expand_to_joints(target_x1[..., 0:3])
  pos_loss = smooth_l1(pred_pos_rel, target_pos_rel)
  ```
- **Applies to:** All 4 experiments (E1-E4)
- **Status:** Single unified change for all configs

#### C2: Remove t² timestep weighting
- **File:** Already overridable in config via `timestep_squared_weighting`
- **Config change only:** Set to `False` in E1-E4 configs
- **No code changes needed**
- **Status:** Ready

### Task Group D: Validation & Testing

#### D1: Unit tests for online conversion
- **Test:** `smpl_trans_to_smooth_trans()` → `smooth_trans_to_smpl_trans()` roundtrip
- **Accuracy target:** 
  - Translation: max_error < 1e-6 (float32 precision)
  - Rotation: exact (zero error, passthrough)
- **Spec:** Proposal §6.3 (test_roundtrip_conversion pseudocode)
- **Status:** Specified

#### D2: Debug single-step training
- **E1:** Run 1 step with E1 config, verify loss decreases
- **E3:** Run 1 step with E3 config, verify ADMM conversion works
- **Location:** Recommend `lzy_debug_machine_1` or `lzy_debug_machine_2`
- **Status:** Specified in proposal §15

---

## Implementation Order (Recommended)

1. **Phase 0-Step 1:** Config & Loss Implementation (2-3 days)
   - [ ] A1: E1 config (copy + modify base)
   - [ ] A2: E2 config (copy E1 + add caption)
   - [ ] C1: Position loss relative-to-root (code change)
   - [ ] D1: Single-step debug E1 & E2

2. **Phase 0-Step 2:** KIMODO Root Implementation (3-4 days)
   - [ ] B2: Implement `SmplTransToKimodoRootOnline` transform
   - [ ] B1: Compute KIMODO Root mean/std
   - [ ] A3: E3 config (copy E1 + add ADMM)
   - [ ] A4: E4 config (copy E3 + add caption)
   - [ ] D2: Single-step debug E3 & E4

3. **Phase 0-Step 3:** Submit to Taiji (1 day)
   - [ ] Verify all 4 configs
   - [ ] Submit E1-E4 (48×V100 each)
   - [ ] Monitor loss curves

---

## Key References

### In Proposal v1.7
- **Root representation:** §6 (198-dim, both versions)
- **Online conversion algorithm:** §6.3 (lines 469-512)
- **Loss alignment:** §6.4 (position loss relative-to-root, t² weighting)
- **Config specifications:** §10.2 (E1-E4 detailed tables)
- **Phase 0 tasks:** §15 (complete task list)

### Existing Production Code
- **Base config:** `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`
- **Loss code:** `hftrainer/losses/motion/m2m_loss.py` (position loss calculation)
- **KIMODO aux losses:** `hftrainer/losses/motion/kimodo_aux_loss.py` (has `timestep_squared_weighting` flag)

---

## Estimated Effort

| Task Group | Estimated Time | Resource |
|-----------|----------------|----------|
| A: Configs | 4-6 hours | 1 engineer |
| B: Data & transforms | 1-2 days | 1 engineer |
| C: Loss code | 2-4 hours | 1 engineer |
| D: Testing | 1-2 hours | 1 engineer |
| **Total Phase 0 prep** | **2-3 days** | **1 engineer** |
| **Phase 0 training** | **5-7 days per exp** | **48 V100 × 4 exp** |

---

## Sign-Off Checklist

- [x] Proposal v1.7 finalized and committed
- [x] All references consistent (198-dim throughout)
- [x] Phase 0 experiments fully specified
- [x] Loss configuration defined
- [x] Implementation tasks enumerated
- [ ] Configs E1-E4 created and tested
- [ ] Transforms implemented and tested
- [ ] Mean/std computed for KIMODO Root
- [ ] Single-step validation passed (E1-E4)
- [ ] Submitted to Taiji

---

**Next Step:** Begin Phase 0-Step 1 (Config & Loss Implementation)

