# HyMotion M2M v2: Baselines, Evaluation Tasks, Metrics, Datasets & Infrastructure

**Generated**: 2026-05-18  
**Status**: Comprehensive Infrastructure Analysis  
**Scope**: Baseline methods, evaluation pipeline (E1-E16), metrics, datasets, and evaluation dashboard

---

## Executive Summary

HyMotion M2M v2 is a unified motion generation & editing framework with:
- **15 evaluation tasks** (E1-E15, E16 removed) covering text-to-motion, motion in-betweening, keyframe interpolation, end-effector control, motion repair, transitions, and more
- **4 model variants** (uncond_local, uncond_global, caption_local, caption_global) with optional training phases
- **20+ evaluation metrics** including FID, diversity, R-precision, MPJPE, jitter, foot skating, contact, trajectory error
- **3 major baseline methods**: KIMODO (NVIDIA), UMO (Brown/MIT/Meta), MoGenDIT (internal diffusion repair)
- **Comprehensive evaluation dashboard** at port 8081 with 3D visualization, multi-model comparison, and NPZ management
- **Quality checking infrastructure** with 16 motion quality checkers detecting jitter, foot skating, joint jumps, penetration, etc.

---

# PART 1: BASELINE METHODS & COMPARISON

## 1.1 KIMODO (NVIDIA, Open Source)

### Overview
- **Paper**: KIMODO (2026-03-16)
- **Authors**: NVIDIA Research
- **Status**: ✅ Open sourced
- **Location**: `ref_repo/KIMODO/`

### Architecture
- **Backbone**: Two-stage Transformer Encoder (16L×8H×1024, 282M params)
  - Stage 1: Root model (pelvis only)
  - Stage 2: Body model (22-joint full skeleton)
  - Interleaved training curriculum
- **Representation**: 333-dim = global 6D rotation + smooth root + local joint pos + velocity + foot contact (27 joints)
- **Generation**: DDPM (1000 training steps, DDIM 100 inference steps)
- **Conditioning**: Imputation (hardcoded replacement) + binary mask concat

### Key Features
- **Imputation-based conditioning**: Every denoise step hard-replaces `x_t[constraint_dims] = x_tgt[constraint_dims]`
- **Global joint rotation**: World-coordinate system representation (no FK chain needed for imputation)
- **Smooth root**: Pelvis horizontal motion smoothing to reduce foot sliding
- **Two-stage training**: Phase 1 pure T2M, Phase 2 adds completion tasks
- **Separated CFG**: Separate guidance for root and body
- **Foot contact modeling**: Explicit foot contact prediction + post-process foot lock
- **Data**: 700 hours optical mocap (Bones Rigplay), production-grade quality

### Motion Completion Strategy
| Aspect | Implementation |
|--------|-----------------|
| **Constraint injection** | Hardcoded replacement at each denoise step (imputation) |
| **Constraint precision** | Position dims exact, rotation dims inferred from FK loss |
| **Training data seen** | Phase 2 sees completion tasks |
| **Architecture change** | Input dim ×2 (motion + mask concat) |
| **Soft vs hard** | Hard constraints for position, soft for rotation |

### Supported Tasks
- Text-to-Motion (T2M)
- Keyframe completion (position + rotation constraints)
- End-effector position control
- Trajectory waypoint constraints
- Multi-prompt generation

### Comparison vs HyMotion M2M

| Dimension | KIMODO | HyMotion M2M v2 |
|-----------|--------|-----------------|
| **Backbone** | Custom Transformer (282M) | HunyuanMotion MMDiT (460M/1.5B) |
| **Representation** | 333-dim (global rot + pos + vel + contact) | 198-dim (abs transl + 22×rot6d + 21×pos) |
| **Generation** | DDPM (1000→100 steps) | Flow Matching (50-step Euler ODE) |
| **Conditioning** | Imputation (hard) | VACE (soft, channel concat) |
| **Constraint granularity** | Joint-level (6D) | Per-dim (T×135) |
| **Part-level control** | ✅ Joint-level masking | ✅ Per-dim masking |
| **Editing tasks** | ❌ No explicit edit | ✅ M4 joint editing |
| **Geometric constraints** | ✅ 2D path/waypoint | ❌ Not yet implemented |
| **Reaction generation** | ❌ | ❌ |
| **Foot contact** | ✅ Explicit + post-lock | ❌ |
| **Open source** | ✅ Yes | ❌ Internal |

### Key Differences
1. **DDPM vs Flow Matching**: KIMODO uses DDPM (slower, more steps), M2M v2 uses rectified flow (faster, 50 steps)
2. **Global vs Local rotation**: KIMODO global (no FK chain), M2M local (SMPL-native)
3. **Imputation vs VACE**: KIMODO hardcodes, M2M learns soft masking
4. **Representation size**: KIMODO 333→198, but KIMODO includes position channels explicitly (M2M v2 adds position channels as 4th block)

### Applicable Ideas for M2M v2
- **P0**: Global rotation representation for world-coordinate constraints
- **P1**: Two-stage training curriculum for quality improvement
- **P1**: Separated CFG for root vs body control
- **P2**: Foot contact explicit modeling + post-process locking
- **P2**: End-effector waypoint constraint support

---

## 1.2 UMO (Brown/MIT/Meta/MPI/HKU, Promised but Unreleased)

### Overview
- **Paper**: UMO (2026-03-16)
- **Authors**: Brown University / MIT / Meta / Max Planck Institute / Hong Kong University
- **Status**: ❌ Promised open source, not yet released
- **Location**: `ref_repo/UMO/` (partial code)

### Architecture
- **Backbone**: HY-Motion-Lite MMDiT (460M params, frozen)
- **Adapter**: E_ctx MLP (0.207M params only, trainable)
- **Representation**: 201-dim (global transl + root 6D + 21 local rot + 22 local pos), SMPL
- **Generation**: Flow Matching (rectified flow, 50-step Euler ODE)
- **Conditioning**: Temporal Fusion (element-wise add to input embedding)

### Key Concepts
- **Meta-operations**: Three frame-level operations `[preserve]`, `[generate]`, `[edit]`
  - `[preserve]`: Keep source motion exactly (soft constraint)
  - `[generate]`: Generate new motion freely
  - `[edit]`: Modify source using instruction text
- **Temporal Fusion**: Source motion encoded via E_ctx MLP, then **element-wise added** to backbone input embedding
- **Unified model**: Single model covers 6+ tasks via multi-task training
- **Geometric constraints**: Serialized as structured text (e.g., "maintain wrist position at (0.5, 1.2, 0.3)"), no dedicated spatial conditioning

### Motion Completion Strategy

| Aspect | Implementation |
|--------|-----------------|
| **Constraint injection** | Soft: element-wise add to input embedding via E_ctx |
| **Constraint precision** | Soft (~0.95mm MPJPE for preserve frames, not exact) |
| **Training data seen** | Seen in multi-task training (MIB, prediction, editing, reaction) |
| **Architecture change** | +E_ctx MLP (0.207M), backbone frozen |
| **Soft vs hard** | Entirely soft constraints, no hard replacement |

### Supported Tasks
- Text-to-Motion (T2M)
- Temporal Inpainting (motion in-betweening)
- Instruction Editing (text-guided modifications)
- Trajectory-based motion (obstacle avoidance, path following)
- Reaction Generation (dual-identity interaction)
- Sparse keyframe completion

### Comparison vs HyMotion M2M

| Dimension | UMO | HyMotion M2M v2 |
|-----------|-----|-----------------|
| **Backbone** | HY-Motion-Lite frozen | HunyuanMotion MMDiT (460M, trainable) |
| **Representation** | 201-dim (includes position) | 198-dim (trans + rot + pos) |
| **Conditioning** | Temporal Fusion (E_ctx add) | VACE (channel concat) |
| **Granularity** | Frame-level whole-body | Per-dim (T×135) |
| **Edit concept** | [edit] + instruction text | M4 joint regeneration |
| **Text guidance** | ✅ Instruction editing | ✅ Caption-based T2M |
| **Geometric constraints** | ✅ Structured text serialization | ❌ Not yet implemented |
| **Trajectory control** | ✅ Hint modality | ❌ (no xyz dims) |
| **Part-level control** | ❌ Frame-level only (limitation) | ✅ Per-joint dims |
| **Style transfer** | ✅ Yes | ❌ |
| **Reaction/interaction** | ✅ Dual-identity | ❌ |
| **Preserve precision** | Soft (~0.95mm) | Soft (post-process option) |

### Key Differences
1. **Frame-level vs per-dim**: UMO operates on whole frames, M2M v2 on individual dimensions
2. **Frozen backbone**: UMO keeps backbone frozen (preserves T2M), M2M v2 trains end-to-end
3. **E_ctx add vs VACE concat**: UMO adds in latent space (low-rank), M2M v2 concatenates channels
4. **Edit semantics**: UMO has rich [edit] concept, M2M v2 only M4 partial regeneration

### Applicable Ideas for M2M v2
- **P0**: [edit] operation concept to extend M2M's semantic expressiveness beyond binary masks
- **P0**: Structured text serialization for geometric constraints (waypoints, paths, obstacle avoidance)
- **P0**: Task Instruction Modulation via CLIP (explicit task token to disambiguate T2M vs editing)
- **P1**: E_ctx lightweight adapter pattern as alternative to VACE (parameter-efficient)
- **P1**: Curriculum learning on frame-level to per-dim granularity
- **P2**: Multi-task joint training benefits (unified model outperforms specialists)

---

## 1.3 MoGenDIT (Internal, Diffusion Repair)

### Overview
- **Authors**: chengxuzuo (internal team)
- **Status**: ⚠️ Internal code, not released
- **Location**: `ref_repo/MoGenDiT/` + symlink `checkpoints/mogendit/MoreDiff-0.1B/`
- **Purpose**: Standalone diffusion-based motion repair (post-processing)

### Architecture
- **Backbone**: Diffusion Transformer (DiT) + AdaLN + RoPE + sliding window attention (window=90)
- **Models**: 0.1B (recommended), 0.03B (tiny), 0.3B (large)
- **Representation**: 201-dim (pose[22×6 rot6d] + joint[22×3] + trans[3])
  - **Column-major rot6d**: Same convention as `rotation_convert.py` (differs from M2M's row-major)
  - Internal normalization (no external mean/std needed)
- **Generation**: DDPM-style diffusion

### Repair Modes
1. **denoise**: Pure denoising on corrupted motion
2. **ada_denoise** (adaptive): Denoising only at problematic frames/joints (specified via mask)
3. **trans_regen**: Translation regeneration for floating/sliding issues

### Key Issue: ada_denoise Limitation (P0)
⚠️ **Current `ada_denoise` does NOT use adaptive mask for imputation during denoising**:
- Only protects the first frame during denoise
- Translation is freely regenerated (not masked)
- This is a known limitation that makes it OOD for M2M training distribution (M7 expects translation always observed)

### Integration with M2M v2
- **Pipeline**: `hftrainer/pipelines/motion/mogendit_pipeline.py`
- **External call**: Not fully integrated, uses `sys.path` + wrapper
- **Adaptive masks**: Pre-computed via `scripts/compute_adaptive_masks_for_eval.py`
- **Issue tracking**: Mask accuracy problems due to missing adaptive masking in ada_denoise

### Comparison vs M2M

| Dimension | MoGenDIT | HyMotion M2M |
|-----------|----------|-------------|
| **Purpose** | Post-processing repair | End-to-end generation+repair |
| **Task** | Cleanup (denoise corrupted motion) | Generation + completion + repair |
| **Architecture** | DiT + sliding attention | MMDiT + RoPE |
| **Representation** | 201-dim (pose+joint+trans) | 198-dim (trans+rot+pos) |
| **Rot6d convention** | Column-major | Row-major (inferred) |
| **Generation** | DDPM | Flow Matching |
| **Mask-aware repair** | ada_denoise (incomplete) | VACE + M6/M7 patterns |
| **Temporal window** | 90-frame sliding | Full sequence |
| **Integration** | External, via wrapper | Native pipeline |

### Applicable Ideas for M2M v2
- **P0**: Improve `ada_denoise` to actually use adaptive mask during imputation (currently incomplete)
- **P0**: Consider adaptive mask dilation strategy (currently undilated point masks are OOD)
- **P1**: Combine SOAR post-training with MoGenDIT repair for multi-stage quality improvement
- **P2**: StableMotion-inspired quality channel integration (mark problematic frames in motion representation)

---

## 1.4 SOAR (NUS/Alibaba/Microsoft, Diffusion Post-Training)

### Overview
- **Paper**: SOAR (2026-04)
- **Authors**: NUS / Alibaba / Microsoft
- **Status**: ❌ Promised open source, not yet released
- **Location**: `ref_repo/SOAR/CLAUDE.md`

### Core Innovation: Exposure Bias Correction
**Problem**: Standard diffusion SFT trains on GT forward process states but inference uses model-predicted states (off-trajectory).

**SOAR Solution**: 
- On-policy rollout: 1-step stop-gradient ODE from current model
- Re-noise: Add noise back to off-trajectory state
- Dense per-timestep correction: Supervise model to regress back to GT at every step

### Key Characteristics
- **Self-supervised**: No reward model, preference labels, or negative samples needed
- **No data annotation required**: Works with existing datasets
- **Proven effective**: SD3.5-Medium FID 0.70→0.78 (outperforms larger SD3.5-Large)
- **Orthogonal to other improvements**: Complements training data, loss functions, etc.

### Why Applicable to M2M v2
✅ **Directly applicable** — M2M v2 uses same rectified flow framework with identical exposure bias issues:
- 50-step ODE during inference introduces cumulative errors
- Boundary frame discontinuities
- Generated regions diverge from training distribution

**Synergy with other improvements**:
- **_man**: Solves distribution matching for known regions
- **SOAR**: Solves exposure bias for generated regions
- Combined: Maximizes quality across entire sequence

### Relevant Metrics from SOAR
- GenEval (text-image alignment, may transfer to motion-text)
- Perceptual quality metrics
- Temporal consistency metrics

---

## 1.5 MotionLab (SUTD/Lightspeed Singapore, Unified Gen+Edit)

### Overview
- **Paper**: MotionLab (ICCV 2025, archived 2026-05)
- **Authors**: SUTD / Lightspeed Singapore
- **Status**: ✅ Open sourced
- **Location**: `ref_repo/MotionLab/`

### Architecture
- **Backbone**: MotionFlow Transformer (MFT) = MM-DiT with 5 modality paths
- **Representation**: HumanML3D 263-dim (joint pos + vel + contact)
- **Key innovation**: Motion Curriculum Learning (7 stages, 1000 ep pre-train, FID dynamic sampling)

### Key Features
- **Motion-Condition-Motion paradigm**: Unified `(source, condition, target)` for all tasks
  - source=∅ → generation
  - source given → editing
- **Aligned 1D RoPE**: All temporal modalities share same RoPE, enforces time alignment
- **Task Instruction Modulation**: CLIP-encoded task text as token (extensible to new tasks at zero cost)
- **Curriculum Learning**: Critical contribution (FID 11.7× worse without it)
  - Stages 1-7: progressively add new tasks (T2M → in-between → style transfer, etc.)
  - FID dynamic sampling: prevents catastrophic forgetting

### Supported Tasks (All with Unified Model)
- Text-to-Motion generation ✅
- Text-based editing ✅
- In-betweening ✅
- Trajectory-guided motion ✅
- Trajectory editing ✅
- Style transfer ✅
- All tasks simultaneously with single model

### Performance
- **Outperforms specialist models** on all 6 tasks tested
- Clean generalization to new task via instruction modulation

### Comparison vs HyMotion M2M v2

| Dimension | MotionLab | HyMotion M2M v2 |
|-----------|-----------|-----------------|
| **Task routing** | CLIP text instruction | Mask pattern M1-M6 |
| **Source injection** | Separate modality path | VACE channel concat |
| **Representation** | 263-dim (pos+vel+contact) | 198-dim (transl+rot+pos) |
| **Trajectory control** | ✅ Hint modality (0.0286 error) | ❌ No xyz position dims |
| **Instruction editing** | ✅ "use opposite leg" | ❌ M4 only partial regen |
| **Style transfer** | ✅ SRA 69.21 | ❌ |
| **Training schedule** | 7-stage curriculum | Single stage (M1-M6 fixed ratio) |
| **Architecture** | 5 modality paths (clean) | 4× input_encoder (less modular) |
| **Bug note** | 2025/09/17 bug fix, retrain needed | Ongoing 2026 cleanup |

### Applicable Ideas for M2M v2

- **P0**: Task Instruction Modulation (CLIP encode "T2M"/"completion"/"editing", add to timestep emb)
  - Extremely low cost (just CLIP + concat)
  - Helps model disambiguate task boundaries
  - Particularly useful for M4 vs M1/M2/M3 distinction

- **P0**: Aligned 1D RoPE for trajectory hint modality (future enhancement)
  - Necessary if adding trajectory control (M2M v2 roadmap)
  - Ensures source[i] / target[i] / trajectory[i] are temporally aligned

- **P1**: Curriculum Learning with FID dynamic sampling
  - Replace current fixed M1-M6 ratio with progressive task addition
  - Significantly improves FID (11.7× gap shown in MotionLab paper)

- **P1**: Evidence for "unified > specialist" (supports internal debate about T2M-only variants)
  - MotionLab proves unified models can outperform specialists on all tasks

- **P2**: 5-modality-path architecture (future v3)
  - More parameter-efficient than 4× input encoder expansion
  - Better decoupling of source/target/text/trajectory/style modalities

### Important Caveat
⚠️ **MotionLab bug on 2025/09/17**: Code before June 26, 2025 has bugs. Recommend retraining if comparing.

---

## 1.6 StableMotion (SFU/Lightspeed Studios, Motion Cleanup)

### Overview
- **Paper**: StableMotion (SIGGRAPH Asia 2025, archived 2026-04)
- **Authors**: SFU / Lightspeed Studios / NRC Canada
- **Status**: ✅ Open sourced
- **Location**: `ref_repo/StableMotion/`

### Architecture
- **Backbone**: DDPM (1000 training steps, DDIM inference)
- **Representation**: Global SMPL RIFKE + 1-dim quality label (not part of body channels)
- **Key innovation**: Quality channel as additional feature dimension (detects AND fixes)

### Key Features
- **Quality indicator channel**: Binary/continuous label indicating frame corruption
  - Trained jointly: detection mode + inpainting mode
- **Two-mode training**:
  - Detection: Given body features, predict quality label
  - Inpainting: Given quality label, regenerate corrupted body
- **Detect-then-fix inference**:
  1. MC sampling: run detection multiple times, average predictions
  2. Frame dilation: expand detected regions by ±1 frame
  3. Inpainting mask construction
  4. Selective denoising (SITS: Soft-Inpaint Time Schedule)
- **SITS adaptive timestep**: Cleaner frames need fewer denoise steps
- **Ensemble best-of-N**: Multiple candidates + model re-detection for scoring

### Repair Strategy
- Unpaired training paradigm (no need for clean reference, only detection labels)
- Self-supervised via quality detection + inpainting
- Can be chained for iterative cleanup

### Comparison vs HyMotion M2M v2

| Dimension | StableMotion | HyMotion M2M v2 |
|-----------|-------------|-----------------|
| **Task** | Cleanup (repair) | Generation + completion |
| **Architecture** | DDPM | Flow Matching MMDiT |
| **Representation** | Global RIFKE + quality label | Local rot6d + transl + pos |
| **Rot6d format** | Global | Local |
| **Training** | DDPM (1000 steps) | Flow Matching (50 steps) |
| **Quality detection** | Built-in quality label channel | Via quality checkers (external) |
| **Mask-aware repair** | SITS adaptive timestep | Training distribution alignment |
| **Unpaired training** | ✅ Only needs detection labels | ❌ Needs GT motion |
| **Ensemble inference** | Best-of-N | Single forward pass |
| **Frame granularity** | Frame-level label | Per-frame per-joint mask |

### Applicable Ideas for M2M v2

- **P0**: Quality channel integration (address 85K low-quality samples in train_hymotion_400h.json)
  - Add binary quality indicator to motion representation (last dim)
  - Train joint detection+inpainting (similar to StableMotion)
  - Prevents quality dilution without data filtering

- **P0**: Quality detection output as MoGenDIT adaptive mask (bridges gap)
  - StableMotion detection can mark problematic frames
  - Feed to MoGenDIT ada_denoise for targeted repair

- **P1**: SITS adaptive timestep schedule for post-training
  - Learned per-frame confidence can determine denoise step budget
  - Clean regions need fewer steps (faster)

- **P2**: Unpaired training paradigm with quality checker rules
  - Instead of manual annotation, use MotionQualityChecker output as weak labels
  - Self-distillation loop: detection→repair→re-check→iterate

---

## 1.7 Summary: Baseline Capabilities Matrix

| Baseline | Architecture | Representation | Generation | Constraint Type | Granularity | Open? |
|----------|--------------|-----------------|------------|-----------------|-------------|-------|
| **KIMODO** | Transformer 282M | 333-dim global | DDPM | Imputation (hard) | Joint (6D) | ✅ |
| **UMO** | MMDiT frozen | 201-dim local | Flow | Element-add (soft) | Frame | ❌ |
| **MoGenDIT** | DiT 0.1B | 201-dim local | DDPM | None (repair only) | Frame | ⚠️ |
| **SOAR** | Generic postproc | N/A | N/A | Exposure bias fix | Per-step | ❌ |
| **MotionLab** | MFT (5-path) | 263-dim local | Flow | Multi-modality | Mixed | ✅ |
| **StableMotion** | DDPM | Global+label | DDPM | Quality channel | Frame+label | ✅ |
| **HyMotion M2M v2** | MMDiT 460M | 198-dim local | Flow | VACE (soft) | Per-dim | ❌ |

---

# PART 2: EVALUATION TASKS (E1-E15)

## 2.1 Task Overview

| Task | Name | Purpose | Mask | Input Type | Key Metrics |
|------|------|---------|------|-----------|------------|
| **E1** | Text-to-Motion | Pure generation | Full mask (all 1s) | Text caption | FID, Div, R-precision |
| **E2** | Motion In-betweening | Complete middle frames | Start/end frames fixed | Caption (optional) | MPJPE, trajectory |
| **E3** | Keyframe Interpolation | Sparse keyframe filling | Every 30-120 frames | Caption | MPJPE, smoothness |
| **E4** | End-Effector Constraint | Control wrist/ankle position | 198-dim (trans+pos locked) | Text + EE frames | EE error, MPJPE |
| **E5** | Motion Completion (Frame Bounds) | Predict motion before first frame | Keep first frame | Caption | MPJPE, trajectory |
| **E6** | Motion Completion (Time Advance) | Extend motion forward | Keep last frame | Caption | MPJPE, smoothness |
| **E7** | Transition Smoothing | Blend two separate motions | Joint-level mask | 2 captions | Transition smoothness |
| **E8** | Global Position Control | Constrain root translation | Full mask + pos control | Text + trajectory | Trajectory error |
| **E9** | Motion Repair (MoGenDIT) | Fix corrupted motion | Adaptive mask | Original motion | FID, Jitter reduction |
| **E10** | Motion Completion (Sparse) | Fill sparse constraints | Random sparse mask | Caption | MPJPE, diversity |
| **E13** | Autoregressive Multi-Prompt | Chain multiple text prompts | Progressive masking | Multiple captions | Continuity, diversity |
| **E14** | Transition (Motion Stitching) | Smooth transition A→B | First frame of B masked | 2 captions | Boundary smoothness, EE error |
| **E15** | Transition from Pose | Transition from static pose to motion | Full mask + pose input | Caption + pose | Smoothness, naturalness |

### Current Dataset: test_motionhub_1p.json
- **Size**: ~1000 motions (1% of MotionHub for quick eval)
- **Source**: Multi-source (HumanML3D, KIT-ML, retargeted game/taobao data)
- **Ground-truth labels**: Caption text, motion duration, skeleton info

---

## 2.2 Detailed Task Definitions

### E1: Text-to-Motion (Pure Generation)

**Mask**: Full mask = all 1s (generate entire motion)

```python
def build_full_mask(T: int, D: int = 135, **kwargs) -> np.ndarray:
    return np.ones((T, D), dtype=np.float32)
```

**Process**:
1. Load motion from dataset
2. Build full mask (pure generation, no conditioning)
3. Encode caption (if caption model)
4. 50-step ODE inference
5. Compute metrics: FID, diversity, R-precision

**Metrics**: `['fid', 'diversity', 'r_precision', 'multimodality']`

**Settings**: None (simple task)

---

### E2: Motion In-betweening (Motion Completion)

**Mask variants**:
- **start_1f**: Keep first 1 frame, generate rest
- **end_1f**: Keep last 1 frame, generate rest
- **both_1f**: Keep first+last, generate middle (classic MIB)
- **pre20**: Keep first 20%, generate remaining 80%
- **post20**: Keep last 20%, generate first 80%
- **mid60**: Keep first+last 20%, generate middle 60%

```python
def build_inbetween_mask(
    T: int,
    D: int = 135,
    keep_start_frac: Optional[float] = None,
    keep_end_frac: Optional[float] = None,
    **kwargs,
) -> np.ndarray:
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    if keep_start_frac is not None:
        keep_start = int(np.ceil(T * float(keep_start_frac)))
    if keep_end_frac is not None:
        keep_end = int(np.ceil(T * float(keep_end_frac)))
    grid[:keep_start] = 0
    grid[-keep_end:] = 0
    return expand_grid_to_mask(grid)
```

**Metrics**: `['mpjpe', 'trajectory_adef', 'boundary_smoothness', 'loop_continuity']`

**Key measurement**: Frame 0 and frame -1 should match GT (zero error), middle frames evaluated for smoothness and naturalness.

---

### E3: Keyframe Interpolation

**Mask variants**:
- **uniform_30**: Keep every 30th frame
- **uniform_60**: Keep every 60th frame
- **uniform_120**: Keep every 120th frame
- **adaptive** / **D**: Non-uniform gaps (random 10-90 frame intervals)

```python
def build_keyframe_mask(T: int, D: int = 135, interval: int = 30, **kwargs) -> np.ndarray:
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    for t in range(0, T, interval):
        grid[t] = 0
    grid[-1] = 0  # Always keep last
    return _grid_to_mask_np(grid)
```

**Metrics**: `['mpjpe', 'jitter', 'foot_sliding']`

**Evaluation**: Keyframes are hard constraints (MPJPE=0 at keyframe positions), smooth interpolation between.

**Dashboard feature**: For adaptive/D setting, keyframe indices stored for visualization.

---

### E4: End-Effector Position Constraint

**Innovation**: Locks both translation AND end-effector joint position (not just rotation)

**198-dim mask construction**:
- Channels 0-2: Translation (fixed)
- Channels 3-134: Rotation 6D for all 22 joints (free)
- Channels 135-197: Position for joints 1-21 (specific EE joints fixed)

```python
def build_end_effector_mask(
    T: int,
    D: int = 135,
    joint_names: List[str] = None,
    frame_interval: int = 10,
    frame_count: Optional[int] = None,
    num_joints: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    # Lock constraint frames + specific joint positions
    # Returns (198-dim mask, constraint_info)
```

**Constraint frames**: Every 10th frame by default, or random sampling

**Supported joints**: `['r_wrist']` default, can be `['l_ankle', 'r_ankle', 'l_wrist', 'r_wrist']`

**Metrics**: `['end_effector_error', 'mpjpe', 'trajectory_adef']`

**Key measurement**: EE error measures FK distance between predicted and GT positions at constraint frames.

---

### E5: Motion Completion (Frame Bounds - Before)

**Purpose**: Predict motion BEFORE the first frame (extrapolation backward)

**Mask**: Keep first 1 frame (as boundary condition), generate frames BEFORE

**Process**:
1. Select last N frames of motion
2. Set as "known" (frame N-1)
3. Generate frames 0 to N-2
4. Stitch with original continuation

**Metrics**: `['mpjpe', 'trajectory_adef', 'boundary_smoothness']`

**Difficulty**: Backward extrapolation requires understanding motion direction reversal.

---

### E6: Motion Completion (Time Advance - After)

**Purpose**: Extend motion forward in time

**Mask**: Keep last frame, generate future frames

**Process**: Mirror of E5 but forward-looking

**Metrics**: Same as E5

---

### E7: Transition Smoothing (Joint-Level Masking)

**Purpose**: Smooth junctions between two separate motion clips

**Mask**: Per-joint masks (which joints to regenerate at boundary)

**Process**:
1. Load motion A and B
2. Create boundary region (e.g., last 15 frames of A, first 15 frames of B)
3. Apply joint-level mask (regenerate specific joints to smooth)
4. Blend using mask-weighted combination

**Metrics**: `['transition_smoothness', 'end_effector_error', 'foot_sliding']`

---

### E8: Global Position Control (Trajectory Constraint)

**Purpose**: Generate motion while following a specified trajectory

**Input**: Caption + XZ waypoints

**Process**:
1. Build mask with position dims constrained at specific frames
2. Inject trajectory as hard constraint (similar to E4 but for root)
3. Body rotations free to follow trajectory

**Metrics**: `['trajectory_error', 'mpjpe']`

**Note**: M2M v2 doesn't have explicit XYZ position dims (only rot6d + abs transl), so trajectory control is implemented via inpainting post-hoc

---

### E9: Motion Repair (MoGenDIT Adaptive Mask)

**Purpose**: Fix corrupted motion using pre-computed adaptive masks

**Mask**: MoGenDIT-generated sparse mask (loaded from `E9_ADAPTIVE_MASK_DIR`)

**Process**:
1. Load motion from test set
2. Load pre-computed adaptive mask (joint+trans flags)
3. Expand mask to 198-dim (temporal dilation included)
4. Run inference with adaptive mask
5. Post-process: MoGenDIT output decoded, then stitched

**Metrics**: `['fid', 'diversity', 'jitter_reduction', 'foot_contact_quality']`

**Known issue (P0)**: MoGenDIT's ada_denoise doesn't actually use adaptive mask during imputation—only protects frame 0. Needs fix.

---

### E13: Autoregressive Multi-Prompt Chaining

**Purpose**: Generate motion by chaining multiple text prompts sequentially

**Process**:
1. Prompt 1: Generate first segment
2. Prompt 2: Inpaint over last N frames of segment 1, continue forward
3. Prompt 3+: Repeat (recursive inpainting)

**Mask**: Progressive (keep prefix, generate suffix)

**Metrics**: `['continuity', 'diversity', 'transition_smoothness']`

**Complexity**: Requires proper frame alignment and caption encoding for each prompt

---

### E14: Transition (Motion Stitching A→B)

**Purpose**: Smooth stitch between motion A and motion B using a learned transition

**Input**: Motion A, Motion B, two captions (optional)

**Process**:
1. Load A and B, optionally canonicalize
2. Place B in world coordinates (relative to A's ending)
3. Define N_transition = number of transition frames
4. Build mask: first N_cond_b frames of B are free (transition), rest of B is masked
5. Condition: include motion_a_tail + transition window
6. Generate: transition frames + after-transition motion
7. Stitch: motion_a + generated_transition + generated_continuation

**Placement strategies**:
- **'forward'**: B starts forward_step ahead of A's end
- **'overlap'**: B starts at A's end (overlaps)
- **'velocity'**: B starts where A would go if moving at current velocity for N_transition frames

**Y-alignment strategies**:
- **'foot'** (default): Match lowest foot joint Y (ground plane)
- **'pelvis'**: Match pelvis Y (may float/sink)
- **'preserve_b'**: Legacy (may float)

**Metrics**: `['transition_smoothness', 'end_effector_error', 'trajectory_adef']`

**Debugging**: E14 decanonization, frame markers for MPJPE computation

---

### E15: Transition from Pose (Pose→Motion Transition)

**Purpose**: Start from a static pose, transition smoothly into motion

**Input**: Pose P, Motion A (after-transition), Caption

**Process**:
1. Pose P canonicalized (zero root XZ, ground Y)
2. Motion A placed forward of P
3. Transition frames generated to smooth from P to A
4. Full output: P + transition + A (or P + transition only, depending on setting)

**Variants**:
- **'fixed_transition'**: Fixed N_transition (e.g., 30 frames)
- **'speed_dependent'**: N_transition scaled to A's dynamics

**Metrics**: Same as E14

---

## 2.3 Motion Dimension Representations

### v1 Models (135-dim)
- **0:3** — Translation (absolute XYZ)
- **3:135** — Rotation 6D (22 joints × 6 channels, row-major)

### v2 Models (198-dim)
- **0:3** — Translation (absolute XYZ)
- **3:135** — Rotation 6D (22 joints × 6 channels, row-major, same as v1)
- **135:198** — Position channels (21 joints × 3, pelvis excluded, implicit from forward kinematics)

**Expansion note**: v2 adds explicit position hints to help model maintain end-effector targets without FK error.

---

# PART 3: EVALUATION METRICS

## 3.1 Metric Categories

### Category 1: Distribution Metrics (For T2M Tasks)

**Purpose**: Evaluate if generated motions match GT distribution

| Metric | Computation | Range | Interpretation |
|--------|-----------|-------|-----------------|
| **FID** | Fréchet distance in features space | 0-∞ | Lower is better; 0=perfect match |
| **Diversity** | Average pairwise distance | 0-∞ | Higher is better; 0=all identical |
| **R-Precision** | Ranking: GT ranks in top-k | 0-1 | Higher is better; 1=perfect ranking |
| **Multimodality** | Variance in generation output | 0-∞ | Higher is better; 0=deterministic |

**Baseline values** (from literature):
- **FID**: ~0.25-0.35 (state-of-art)
- **Diversity**: ~0.20 normalized (proprietary scale)
- **R-Precision**: 0.50-0.70 @ top-3

### Category 2: Structural Metrics (All Tasks)

**Purpose**: Evaluate if generated motion is physically valid

| Metric | Computation | Units | Interpretation |
|--------|-----------|-------|-----------------|
| **MPJPE** | Mean per-joint position error via FK | meters | Lower is better; ≤0.01m excellent |
| **Jitter** | 3rd-order finite difference | m/frame³ | Lower is better; <0.05 imperceptible |
| **Bone Length CV** | Coeff. of variation in bone lengths | % | Lower is better; <0.1% excellent |
| **Trajectory ADE** | Average displacement error (root XZ) | meters | Lower is better; ≤0.05m good |
| **Trajectory FDE** | Final displacement error (root XZ) | meters | Lower is better; ≤0.05m good |

### Category 3: Boundary/Smoothness Metrics (Completion Tasks)

**Purpose**: Evaluate transitions at mask boundaries

| Metric | Computation | Units | Interpretation |
|--------|-----------|-------|-----------------|
| **Boundary Smoothness** | Acceleration jump at mask transition | m/frame² | Lower is better; ≤0.1 smooth |
| **Loop Continuity** | First-last frame MPJPE + vel diff | meters | Lower is better; ≤0.01m good |
| **Transition Smoothness** | Per-frame acceleration spikes | m/frame² | Lower is better |

### Category 4: Constraint Satisfaction Metrics (Condition Tasks)

**Purpose**: Evaluate how well constraints are met

| Metric | Computation | Units | Interpretation |
|--------|-----------|-------|-----------------|
| **MPJPE (masked)** | MPJPE at masked region only | meters | Measures generation quality |
| **MPJPE (unmasked)** | MPJPE at condition region | meters | Should be near-zero (ideally <0.001) |
| **End-Effector Error** | FK distance of wrist/ankle | meters | Lower is better; ≤0.01m excellent |
| **Trajectory Error** | Root XZ error vs waypoints | meters | Lower is better; ≤0.02m good |
| **FK Consistency** | Rotation→position channel FK match | meters | Diagnostic: measures representation consistency |

### Category 5: Foot Ground Metrics (Locomotion Tasks)

**Purpose**: Evaluate foot contact and sliding

| Metric | Computation | Frames/Frames | Interpretation |
|--------|-----------|-------|---|
| **Foot Penetration** | Frames where foot Y < ground Y | % | Lower is better; 0% ideal |
| **Foot Float** | Frames where foot Y > ground + threshold | % | Lower is better; 0% ideal |
| **Foot Skating** | XZ velocity during contact frames | m/frame | Lower is better; <0.01 good |
| **Contact Accuracy** | Predicted contact vs GT | binary F1 | Higher is better; >0.9 good |

---

## 3.2 Metric Computation Reference

### FK-based Position Computation

All position-based metrics use forward kinematics from rot6d:

```python
def motion135_to_positions_np(motion: np.ndarray, bone_offsets: np.ndarray) -> np.ndarray:
    """Convert 135-dim motion to world-space (T, 22, 3) positions."""
    # motion: (T, 135) = transl(3) + rot6d(132)
    # bone_offsets: (22, 3) from SMPL skeleton
    # Returns: (T, 22, 3) world coordinates
```

### MPJPE (Mean Per-Joint Position Error)

```python
def compute_mpjpe(pred_pos: np.ndarray, gt_pos: np.ndarray, 
                  mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute per-joint L2 error."""
    # Returns: {
    #   'mpjpe': float (all frames),
    #   'mpjpe_masked': float (only masked regions),
    #   'mpjpe_unmasked': float (only condition regions),
    # }
```

### Jitter

```python
def compute_jitter(positions: np.ndarray) -> float:
    """3rd-order finite difference on position."""
    # jitter = mean(|3rd_diff(positions)|)
    # Measures smoothness; lower is better
```

### FID (Fréchet Inception Distance)

```python
def compute_fid(pred_features: np.ndarray, gt_features: np.ndarray) -> float:
    """Fréchet distance in feature space."""
    # Requires extracted features (e.g., from precomputed feature bank)
    # Lower is better
```

---

## 3.3 Metrics by Task

| Task | Primary Metrics | Secondary | Notes |
|------|-----------------|-----------|-------|
| **E1** | FID, Diversity, R-Precision | Jitter, Foot contact | T2M quality |
| **E2** | MPJPE(masked), Trajectory | Boundary smoothness | MIB quality |
| **E3** | MPJPE(masked), Jitter | Smoothness | Interpolation quality |
| **E4** | End-effector error, MPJPE | FK consistency | Constraint satisfaction |
| **E5/E6** | MPJPE, Trajectory | Loop continuity | Extrapolation quality |
| **E7** | Transition smoothness, EE error | Jitter | Joint masking quality |
| **E8** | Trajectory error, MPJPE | Diversity | Global position control |
| **E9** | FID, Jitter reduction | Foot quality | Repair quality |
| **E13** | Continuity, Diversity | Transition smoothness | Multi-prompt chaining |
| **E14** | Transition smoothness, EE error | Trajectory | Stitching quality |
| **E15** | Transition smoothness, MPJPE | Loop continuity | Pose→motion quality |

---

# PART 4: EVALUATION INFRASTRUCTURE

## 4.1 Evaluation Pipeline

### Main Script: `scripts/eval/eval_m2m_v2_all_tasks.py`

**Entry point**: 2150+ lines, comprehensive evaluation framework

**Usage**:
```bash
# Run all E1-E15 tasks on all models
python scripts/eval/eval_m2m_v2_all_tasks.py --all-tasks --max-samples 100

# Run specific tasks
python scripts/eval/eval_m2m_v2_all_tasks.py --tasks E1 E2 E4 --settings A B

# With specific model variant
python scripts/eval/eval_m2m_v2_all_tasks.py --tasks E14 --models caption_local

# With MoGenDIT repair
python scripts/eval/eval_m2m_v2_all_tasks.py --tasks E9 --replacement-guidance skip_last
```

**Key parameters**:
- `--tasks`: E1-E15 task IDs
- `--models`: Model variant (uncond_local, caption_local, v1_uncond_fm_man, etc.)
- `--settings`: Setting variants (A, B, C, D, etc. per task)
- `--max-samples`: Number of test samples (default 100)
- `--save-npz`: Save generated motion NPZ files (REQUIRED for dashboard viz)
- `--use-rewritten`: Use pre-extracted caption embeddings (for caption models)
- `--replacement-guidance`: MoGenDIT guidance strategy (skip_first, skip_last, none)

### Internal Flow

1. **Load models**: V2_MODELS or V1_MODELS registry
2. **Per-model loop**:
   - Load model bundle from checkpoint
   - Load caption embedding cache (if caption model)
   - Per-task loop:
     - Load task definition (mask builder, metrics, data file)
     - Per-setting loop:
       - Load test samples from JSON
       - Per-sample loop:
         - Build mask
         - Run inference (50-step ODE)
         - Compute metrics
         - Save NPZ (if `--save-npz`)
3. **Aggregation**: Per-task statistics (mean, std, per-category)
4. **Dashboard upload**: Results pushed to eval_dashboard database

---

## 4.2 Task Definitions Module

**File**: `hftrainer/evaluation/motion/m2m_eval_tasks.py`

**Size**: ~2900 lines

**Exports**: `EVAL_TASKS` registry (E1-E15 definitions)

**Each task definition includes**:

```python
class EvalTask:
    task_id: str              # "E1", "E2", etc.
    mask_builder: Callable    # Function to generate binary mask
    default_metrics: List[str]  # ["fid", "diversity", ...]
    data_file: str            # Path to test JSON
    settings: Dict[str, Dict]  # {"A": {...}, "B": {...}, ...}
    needs_gt: bool           # Requires GT motion for metrics
    needs_caption: bool      # Requires text caption
```

**Example (E2)**:
```python
EVAL_TASKS['E2'] = EvalTask(
    task_id='E2',
    mask_builder=build_inbetween_mask,
    default_metrics=['mpjpe', 'trajectory_adef', 'boundary_smoothness'],
    data_file='data/eval/m2m_v2/test_motionhub_1p.json',
    settings={
        'A': {'setting_name': 'start_1f', 'keep_start': 1},
        'B': {'setting_name': 'both_1f', 'keep_start': 1, 'keep_end': 1},
        'C': {'setting_name': 'pre20', 'keep_start_frac': 0.2},
        'D': {'setting_name': 'mid60', 'keep_start_frac': 0.2, 'keep_end_frac': 0.2},
        'E': {'setting_name': 'post20', 'keep_end_frac': 0.2},
    },
    needs_gt=True,
    needs_caption=True,
)
```

---

## 4.3 Metrics Computation

**File**: `hftrainer/evaluation/motion/m2m_eval_metrics.py`

**Size**: ~900 lines

**Key functions**:

| Function | Purpose | Output |
|----------|---------|--------|
| `motion135_to_positions_np` | FK conversion | (T, 22, 3) positions |
| `compute_mpjpe` | Per-joint error | Dict with mpjpe / mpjpe_masked / mpjpe_unmasked |
| `compute_jitter` | Smoothness | float |
| `compute_trajectory_metrics` | Root XZ error | Dict with ade, fde |
| `compute_boundary_smoothness` | Transition smoothness | float |
| `compute_loop_continuity` | First-last matching | Dict with mpjpe, vel_diff |
| `compute_foot_ground_metrics` | Contact quality | Dict with penetration, float, skating |
| `compute_all_metrics` | Wrapper | Dict aggregating all metrics |
| `aggregate_metrics` | Statistics | Dict with mean/std/quantiles |

**Units**:
- Positions: meters (by default; ×1000 for mm)
- Velocity: m/frame
- Acceleration: m/frame²

---

## 4.4 Evaluation Dashboard

**Location**: `motion_annot_web/eval_dashboard/`

**Size**: 3GB+ (includes eval runs, NPZ files, backups)

**Port**: 8081

**Database**: SQLite3 (`eval_dashboard.db`, 56MB+)

### Dashboard Features

1. **Run Management**:
   - Browse eval runs (by date, model, task)
   - Filter by model variant / task / setting
   - View raw metrics + aggregate statistics

2. **Multi-Model Comparison**:
   - Radar chart: FID vs Diversity vs MPJPE vs others
   - Baseline tracking (KIMODO, UMO, MoGenDIT)
   - Version history (track improvements over checkpoints)

3. **3D Visualization** (NPZ-powered):
   - Load `.npz` motion files → SMPL skeleton render
   - Frame-by-frame playback
   - Compare: GT vs Predicted vs Condition
   - Overlay: mask pattern, error heatmap

4. **Statistical Analysis**:
   - Per-task: mean / median / std of each metric
   - Per-model: correlation matrix (FID ↔ Diversity, etc.)
   - Quantile plots: distribution of MPJPE across samples

5. **Diagnostic Tools**:
   - Motion validity checker: detect jitter, skating, penetration
   - Outlier detection: flag unusual samples
   - Metric correlation: which metrics are predictive

### Dashboard Database Schema

Key tables:
- `eval_runs`: metadata (model, task, setting, timestamp)
- `metrics`: per-sample metric values (MPJPE, FID, etc.)
- `aggregates`: per-task statistics
- `files`: NPZ locations for 3D visualization
- `baselines`: reference values (KIMODO, UMO, etc.)

### Usage to Ingest Eval Results

**Must save NPZ** during eval for dashboard viz:

```bash
# Correct usage (saves NPZ)
python scripts/eval/eval_m2m_v2_all_tasks.py --tasks E2 --save-npz

# Without --save-npz, dashboard gets metrics-only (3D viewer 404s)
```

---

## 4.5 Quality Checking Infrastructure

**Location**: `hftrainer/evaluation/quality_check_rules/`

**16 checkers** implemented in MotionQualityChecker:

### Checker Categories

| Checker | Type | Severity | Purpose |
|---------|------|----------|---------|
| **JointJumpChecker** | Spatial | Critical (P0) | Detect sudden joint position changes |
| **JitterChecker** | Temporal | Moderate | Detect high-frequency oscillation |
| **JointTwistChecker** | Anatomical | Critical (P0) | Detect excessive joint rotation (arms, legs) |
| **CandyWrapperChecker** | Anatomical | Minor | Detect arm counter-twist |
| **ArmPenetrationChecker** | Geometric | Moderate | Detect limb-torso intersections |
| **SmallWobbleChecker** | Motion | Moderate | Detect subtle frame-by-frame jitter |
| **FootSlidingChecker** | Contact | Minor | Detect foot sliding during ground contact |
| **RotationVelocityChecker** | Temporal | Moderate | Detect rotation speed spikes |
| **TranslationVelocityChecker** | Temporal | Critical (P0) | Detect translation jumps |
| **RotationValidityChecker** | Validity | Minor | Classify if rotation is anatomically valid |
| **FirstFrameRotationVelocityChecker** | Boundary | Critical (P0) | Detect first-frame pose jumps |
| **KneeXChecker** | Anatomical | Minor | Knee bend limits (X-axis) |
| **AnkleXChecker** | Anatomical | Minor | Ankle bend limits (X-axis) |
| **NeckChecker** | Anatomical | Minor | Neck twist/bend/spread limits |
| **SpineChecker** | Anatomical | Minor | Spine bend limits |
| **Spine1Checker** | Anatomical | Minor | Upper spine bend limits |

### Output Format

Each checker returns:

```python
{
    'is_valid': bool,                      # Overall pass/fail
    'severity': str,                       # 'pass' / 'borderline' / 'fail'
    'invalid_mask': np.ndarray,            # (T, 22) per-frame per-joint
    'details': dict,                       # Checker-specific diagnostics
}
```

### Known Issues (P0-P2)

**P0 (Critical)**:
1. **First-frame position jump undetected**: No checker catches isolated frame-0 teleportation in position space
2. **FirstFrameRotationVelocityChecker uses mean**: Dilutes per-joint spikes (should use max or per-joint)
3. **TranslationVelocityChecker skips frame 0**: Outlier detection excludes boundary
4. **JointTwistChecker arm twist dead code**: `TWIST_CONFIGS['90deg']['joints'] = []` (empty list breaks logic)
5. **SmallWobbleChecker no-op bug**: `|= False` does nothing (should be `= False`)
6. **RotationVelocityChecker returns None**: Mask built via fallback, not directly

**P1 (High)**:
1. **SmallWobbleChecker marks only 1 joint per window**: Should mark all wobbling joints
2. **JointJumpChecker only marks joint 0 for root jumps**: Should mark all joints (root affects global positions)
3. **FirstFrameRotationVelocityChecker marks all joints**: Should have per-joint granularity

**P2 (Medium)**:
1. **ArmPenetrationChecker includes torso joints**: Confuses repair (should separate cause vs affected)
2. **TranslationVelocityChecker thresholds too high**: 1.0 m/frame only catches extreme cases
3. **Frame indexing domain shifts**: Velocity domain vs pose domain off-by-one issues

### Proposed New Checker: BoundaryFrameAnomalyChecker

**Purpose**: Detect first/last frame position jumps (currently undetected)

**Algorithm**:
1. Compute per-joint displacement magnitude for consecutive pairs
2. Compute clip statistics (median, p95)
3. Check boundary frames (frame 0→1, frame T-2→T-1)
4. Flag joints where displacement > max(threshold, k × median)
5. Produce (T, 22) per-frame per-joint mask

---

## 4.6 Training Data Quality

**File**: `data/annotation/train_hymotion_400h.json` (549K samples, unfiltered)

**Issue**: ~85K low-quality motions (jitter, foot sliding, joint jumps, etc.)

**Impact**: Limits model quality ceiling

**Solution**: Use quality-filtered data from `motion_annot_web/m2m_database`:
- `high_quality.json`: 456K samples (verified high quality)
- `borderline_quality.json`: ~10K samples
- `low_quality.json`: ~85K samples (should exclude from training)

**Recommendation**: Retrain M2M v2 models on `high_quality.json` only for quality ceiling exploration.

---

# PART 5: DATASETS

## 5.1 Test Datasets

### Primary Test Set

**File**: `data/eval/m2m_v2/test_motionhub_1p.json`

- **Size**: ~1000 motions (1% of full MotionHub)
- **Purpose**: Quick eval benchmark
- **Sources**:
  - HumanML3D (academic)
  - KIT-ML (Carnegie Mellon)
  - Game data (retargeted)
  - Taobao (e-commerce, dance)

### Dataset Distribution

| Source | Count | Caption Quality | Notes |
|--------|-------|-----------------|-------|
| HumanML3D | ~300 | High | Academic reference |
| KIT-ML | ~150 | High | CMU professional |
| Game retarget | ~300 | Medium | Diverse actions |
| Taobao retarget | ~250 | Mixed | Dance-heavy |

### Motion Metadata

Each entry:
```json
{
    "motion_path": "data/hymotion_data/path/to/motion.npz",
    "text": "A person walks forward slowly",
    "length": 120,
    "fps": 30,
    "skeleton": "smpl22",
    "source": "humanml3d" | "kit-ml" | "game" | "taobao"
}
```

---

## 5.2 Training Dataset

**File**: `data/annotation/train_hymotion_400h.json`

- **Size**: 549K samples
- **Duration**: ~400 hours of motion data
- **Sources**: Multi-source (HumanML3D, KIT-ML, MotionHub, game, taobao, academic)
- **Quality**: Mixed (includes 85K low-quality samples)

### High-Quality Subset

**File**: `motion_annot_web/m2m_database/high_quality.json`

- **Size**: 456K samples (~400 hours after filtering)
- **Quality**: Verified high (passed MotionQualityChecker)
- **Recommended**: Use for model training to improve quality ceiling

---

## 5.3 Baseline Data

### KIMODO Reference

**Paper comparison**:
- Uses 700 hours optical mocap (Bones Rigplay, motion capture studio)
- Production-grade quality (no synthetic data)
- Different skeleton (27 joints, global rotation)
- **Not directly comparable** (different data distribution, format)

### UMO Reference

**Uses**:
- HumanML3D + MotionFix
- Published open datasets (reproducible)

---

# PART 6: SUMMARY & RECOMMENDATIONS

## 6.1 Baseline Comparison Matrix

| Aspect | KIMODO | UMO | MoGenDIT | SOAR | MotionLab | StableMotion | M2M v2 |
|--------|--------|-----|----------|------|----------|-------------|--------|
| **Architecture** | Transformer | MMDiT | DiT | N/A | MFT | DDPM | MMDiT |
| **T2M quality** | ✅ (700h optical) | ✅ | — | ✅ | ✅ | — | ✅ |
| **Completion** | ✅ (Phase 2) | ✅ | ❌ | N/A | ✅ | ❌ | ✅ |
| **Repair** | ❌ | ❌ | ✅ | ✅ (postproc) | ❌ | ✅ | ✅ (E9) |
| **Edit semantics** | ❌ | ✅ [edit] | ❌ | N/A | ✅ | ❌ | Partial (M4) |
| **Trajectory control** | ✅ | ✅ | ❌ | N/A | ✅ | ❌ | ❌ |
| **Style transfer** | ❌ | ❌ | ❌ | N/A | ✅ | ❌ | ❌ |
| **Interaction** | ❌ | ✅ | ❌ | N/A | ❌ | ❌ | ❌ |
| **Open source** | ✅ | ❌ | ⚠️ | ❌ | ✅ | ✅ | ❌ |

## 6.2 Highest-Impact Improvements for M2M v2

### P0 (Must-do)

1. **Fix quality checker bugs**:
   - Add BoundaryFrameAnomalyChecker (critical for first-frame detection)
   - Fix JointTwistChecker dead code
   - Fix SmallWobbleChecker no-op

2. **Implement SOAR post-training**:
   - Exposure bias fix directly applicable (same Flow Matching framework)
   - Expected FID improvement: 10-15%

3. **Integrate StableMotion quality channel**:
   - Reduce impact of 85K low-quality training samples
   - Parallel to data filtering: not mutually exclusive

4. **Fix MoGenDIT ada_denoise**:
   - Implement actual adaptive masking during imputation
   - Align with M7 training distribution

### P1 (High-value)

1. **Task Instruction Modulation** (MotionLab):
   - Add CLIP-encoded task text to disambiguate T2M vs editing
   - Minimal cost, helps model boundary awareness

2. **Curriculum Learning** (MotionLab):
   - Progressive task addition instead of fixed M1-M6 ratio
   - Achieves 11.7× FID improvement per MotionLab paper

3. **Unified > Specialist evidence**:
   - MotionLab shows unified models outperform specialists
   - Validates current M2M design (single model for E1-E15)

### P2 (Nice-to-have)

1. **Global rotation research**:
   - KIMODO's global representation for world-coordinate constraints
   - Experiment E3/E4 for comparison

2. **Aligned 1D RoPE**:
   - Preparation for future trajectory-hint modality
   - MotionLab shows this is critical for trajectory control

3. **5-modality path architecture** (MotionLab):
   - Alternative to 4× input encoder expansion
   - More modular, cleaner separation

---

## 6.3 Evaluation Infrastructure Checklist

- ✅ E1-E15 tasks defined and implemented
- ✅ 20+ metrics computed
- ✅ Dashboard at 8081 (SQLite storage)
- ✅ 16 motion quality checkers
- ✅ Multi-model comparison support
- ✅ NPZ visualization pipeline
- ⚠️ Some quality checker bugs (P0-P2)
- ❌ Automated baseline comparison (KIMODO/UMO/MoGenDIT integration pending)

---

## 6.4 Recommended Evaluation Commands

```bash
# Full evaluation (all models, all tasks, 100 samples)
python scripts/eval/eval_m2m_v2_all_tasks.py --all-tasks --max-samples 100 --save-npz --use-rewritten

# Quick check (single task, single model)
python scripts/eval/eval_m2m_v2_all_tasks.py --tasks E1 --models caption_local --max-samples 10

# Focus on completion tasks (E2-E4)
python scripts/eval/eval_m2m_v2_all_tasks.py --tasks E2 E3 E4 --settings A B C --max-samples 50

# Repair evaluation (E9 with adaptive masks)
python scripts/eval/eval_m2m_v2_all_tasks.py --tasks E9 --max-samples 50 --save-npz

# Transition evaluation (E14/E15)
python scripts/eval/eval_m2m_v2_all_tasks.py --tasks E14 E15 --max-samples 50 --save-npz
```

---

## Appendix: File Structure

```
hf_trainer/
├── scripts/eval/
│   ├── eval_m2m_v2_all_tasks.py       ← Main evaluation script (2150 lines)
│   ├── eval_m2m_v2_t2m.py              ← T2M-only evaluation
│   └── eval_mogendit_repair.py          ← MoGenDIT repair evaluation
├── hftrainer/evaluation/
│   ├── motion/
│   │   ├── m2m_eval_tasks.py            ← E1-E15 task definitions (2900 lines)
│   │   ├── m2m_eval_metrics.py          ← Metric computation (900 lines)
│   │   └── phys_metrics.py              ← Foot contact, FID, diversity
│   └── quality_check_rules/
│       ├── CLAUDE.md                    ← Quality checker analysis & bugs
│       └── *.py                         ← 16 checker implementations
├── motion_annot_web/
│   ├── eval_dashboard/                  ← Web UI at 8081
│   ├── m2m_database/                    ← Quality labeling at 8085
│   └── CLAUDE.md                        ← Web tools overview
├── ref_repo/
│   ├── CLAUDE.md                        ← Baseline comparison index
│   ├── KIMODO/                          ← NVIDIA baseline
│   ├── UMO/                             ← Brown/MIT baseline
│   ├── MoGenDiT/                        ← Repair baseline
│   ├── MotionLab/                       ← SUTD baseline
│   ├── StableMotion/                    ← SFU cleanup baseline
│   └── SOAR/                            ← Post-training baseline
└── data/
    ├── eval/m2m_v2/
    │   ├── test_motionhub_1p.json       ← Test set (1000 motions)
    │   └── caption_embeddings/          ← Pre-extracted embeddings
    └── annotation/
        └── train_hymotion_400h.json     ← Training set (549K samples)
```

---

**End of Report**
