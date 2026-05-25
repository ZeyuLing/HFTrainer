# PRISM TMM2026 Revision: Novelty Strengthening Proposal

## Status: PROPOSAL — Pending User Decision

## 1. Problem Diagnosis

ECCV2026 rejection (scores 2/3/2) primary reason: **"incremental novelty."** The paper currently presents two contributions:

1. **Latent–generator alignment principle** (joint-factorized VAE with independent KL regularization)
2. **Per-token timestep conditioning** (Diffusion Forcing instantiation for unified streaming)

Reviewer criticism: both are "principled integration of existing building blocks" — the paper explicitly acknowledges this. The current paper needs **at least one novel module/technique** that goes beyond integration.

## 2. Key Discovery: KAFS (Kinematic-Adaptive Flow Scheduling)

**KAFS is fully implemented in `prism_backend.py` (lines 134-221, 383-387) but completely absent from the paper.**

### What it is
- An **inference-time** technique that assigns per-joint timestep scaling factors α_j based on kinematic tree depth
- Formula: `t_j = t × α_j` where α ∈ [0.85, 1.15]
- Root/pelvis: α=0.85 (denoised faster → trajectory locked early)
- Distal joints (wrist): α=1.15 (denoised later → more refinement steps)

### Why it's novel
1. **Only possible on joint-factorized latents** — can't exist in monolithic encoding. This creates a unique synergy that elevates both contributions.
2. **Kinematic-aware inference schedule** — no prior flow-matching work conditions the denoising schedule on the kinematic structure of the human body.
3. **No retraining required** — purely inference-time, works with existing checkpoint.
4. **Physical motivation**: proximal joints need fewer denoising steps because their motion is simpler (low DoF, low frequency); distal joints require more steps due to cascading FK effects. This mirrors the FK supervision insight.
5. **5 ablation modes already implemented**: none, depth_driven, uniform, random, custom.

### Novelty assessment
This is a **novel module** (satisfying reviewer criterion §1.5 in paper-review.md: "at least one clear novelty type: task/pipeline/module/design finding/insight"). It provides:
- **Novel design choice**: kinematic-depth-aware denoising schedule
- **Novel insight**: the optimal denoising schedule is non-uniform across the kinematic tree; proximal joints converge faster and benefit from fewer steps while distal joints need more refinement
- **Natural extension of the alignment principle**: just as latent structure mirrors kinematics, inference schedule should mirror kinematic complexity

## 3. Proposal: Add KAFS as Third Contribution

### 3.1 Paper Structure Change

**Method §3.2 — New subsection "§3.2.X Kinematic-Adaptive Flow Scheduling (KAFS)"**

Position: After "Per-token timestep conditioning" (§3.2.1) and before "Autoregressive streaming" (§3.2.3). 

Structure following method.md three-element pattern:

1. **Motivation**: Standard flow-matching uses a single global timestep for all tokens. With joint-factorized latents, each token has physical meaning. Proximal joints (pelvis, spine) have simple, low-frequency dynamics; distal joints (wrists, feet) exhibit complex, high-frequency motion cascading through the kinematic chain. Using the same denoising schedule for all joints is suboptimal.

2. **Design**: KAFS assigns a per-joint timestep scaling factor α_j to each latent token:
   ```
   t_j = t × α_j, where α_j ∈ [α_min, α_max]
   ```
   α_j is derived from the joint's depth in the SMPL kinematic tree:
   - Depth 0 (root, pelvis): α = 0.85
   - Depth 1-2 (hips, spine): α = 0.90-1.00
   - Depth 3-4 (ankles, feet, shoulders): α = 1.05-1.10
   - Depth 5-6 (elbows, wrists): α = 1.12-1.15
   
   During denoising, each joint's effective timestep is modulated by its α, meaning proximal joints are denoised faster (reaching clean state earlier) while distal joints retain noise longer (getting more refinement steps).

3. **Technical advantages**:
   - Matches denoising schedule to joint dynamics complexity
   - Early root denoising stabilizes global trajectory for subsequent limb refinement
   - Purely inference-time — zero training cost
   - Enabled uniquely by joint-factored latents (not possible with monolithic encoding)

### 3.2 Required Experiments (All No-Retrain)

| Experiment | KAFS Mode | Dataset | Metric | Purpose |
|------------|-----------|---------|--------|---------|
| Baseline | none | HumanML3D, MotionHub | FID, R-P, MM-D | Control |
| KAFS-depth | depth_driven | HumanML3D, MotionHub | FID, R-P, MM-D | Main result |
| KAFS-uniform | uniform (α=1.0) | HumanML3D, MotionHub | FID, R-P, MM-D | Ablation control |
| KAFS-random | random | HumanML3D, MotionHub | FID, R-P, MM-D | Random does not help |
| KAFS + TP2M | depth_driven | HumanML3D | FID, R-P, MM-D | Works across tasks |
| KAFS + BABEL | depth_driven | BABEL | Subseq + Trans | Works for streaming |

**Expected narrative**: depth_driven > none > uniform ≈ random. This shows the improvement comes from kinematic structure, not arbitrary per-joint scaling.

### 3.3 New Ablation Table

```
Table X: KAFS ablation on T2M generation.
Mode         | FID↓  | R-P T3↑ | MM-D↓
-------------|-------|---------|------
none         |  ...  |   ...   |  ...
uniform      |  ...  |   ...   |  ...  
random       |  ...  |   ...   |  ...
depth_driven |  ...  |   ...   |  ...
```

### 3.4 Contribution List Revision

Current contributions (2):
1. Latent–generator alignment principle
2. Per-token timestep conditioning for unified streaming

Proposed contributions (3):
1. **Latent–generator alignment principle** (unchanged)
2. **Per-token timestep conditioning for unified streaming** (unchanged)  
3. **Kinematic-Adaptive Flow Scheduling (KAFS)**: A training-free inference technique that modulates the denoising schedule per joint based on kinematic tree depth. KAFS is uniquely enabled by the joint-factorized latent: because each token is a kinematic unit, the denoising schedule can be adapted to the complexity of each joint's dynamics. Root/pelvis tokens are denoised faster (stabilizing global trajectory early), while distal limb tokens retain noise longer (receiving more refinement). This provides FID improvement without any retraining.

### 3.5 How KAFS Strengthens the Overall Paper Story

**Current weakness**: The paper's two contributions are both "principled integration" — reviewers called this incremental.

**With KAFS**: The paper now has a three-act story:
1. **Structural contribution** (latent design): align latent with kinematics → each token = one joint
2. **Scheduling contribution** (per-token timestep): leverage per-token structure for unified conditioning
3. **Inference contribution** (KAFS): leverage per-joint structure for kinematic-aware denoising

Each contribution builds on the previous one. KAFS cannot exist without per-token timestep conditioning, which cannot be physically meaningful without joint-factored latents. This creates a **coherent contribution chain** rather than isolated add-ons, and demonstrates that the joint-factored latent opens a new design space for motion generation that has not been explored.

## 4. Secondary Strengthening (Writing-Level, No New Experiments)

### 4.1 Trajectory Rollout Supervision (L_tr)
Currently described in one sentence. The cumsum formulation (supervising cumulative trajectory rather than per-frame displacement) is non-trivial:
- Standard approach: L1 on Δp_i (per-frame displacement)
- Our approach: L1 on cumsum(Δp_i) (trajectory)
- Why it matters: forces gradient to account for accumulated drift, preventing jitter

**Action**: Expand description in §3.1.4 with explicit motivation and equation.

### 4.2 Translation/Rotation Loss Balancing
The loss computation separates translation and rotation channels to prevent translation loss dilution (translation is 1/23 of total tokens ≈ 4.3%). This is implemented in the trainer but not mentioned in the paper.

**Action**: Add a paragraph in §3.2 or Implementation noting this design choice.

### 4.3 2D RoPE on Kinematic Grid  
2D RoPE along time × joint axes is mentioned but not emphasized. The joint-axis positional encoding encodes kinematic proximity, which is unique.

**Action**: Expand 1-2 sentences in §3.2 to note this physical interpretation.

## 5. Experimental Execution Plan

### Phase 1: KAFS Eval Script (1 hour)
- Modify `tools/infer.py` to accept `--kafs-mode` argument  
- OR: Write standalone eval script `scripts/eval/eval_prism_kafs.py`
- Use existing checkpoint: `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000`

### Phase 2: Run T2M Evaluation (2-4 hours per mode, can parallel)
- 4 modes × 2 datasets = 8 runs
- Uses existing TMR evaluator infrastructure
- Can run on taiji GPU machines

### Phase 3: Run TP2M + BABEL Evaluations (2-4 hours)
- KAFS depth_driven on TP2M and BABEL benchmarks
- Compare with existing baseline numbers in paper

### Phase 4: Compile Results and Write Sections (1 day)
- Create `tab_abl_kafs.tex`
- Write KAFS method subsection 
- Revise introduction contributions
- Update abstract and conclusion

## 6. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| KAFS shows no improvement | Medium | Paper still has 2 contributions + deeper analysis. If uniform ≈ depth_driven, it disproves kinematic scheduling but the negative result itself is informative. |
| KAFS hurts quality | Low | α range [0.85, 1.15] is conservative. worst case: remove from paper. |
| Eval infrastructure issues | Low | All eval tools already exist, just need KAFS flag. |
| Reviewer still says "incremental" | Medium | KAFS + strengthened writing + more ablations significantly raise the bar. Three chained contributions > two integrated contributions. |

## 7. Recommendation

**Proceed with KAFS as primary novelty addition.** Rationale:
1. It's already fully implemented with 5 ablation modes
2. Zero training cost — pure inference experiment
3. Creates a novel contribution that is unique to joint-factored latents
4. Builds naturally on the existing contribution chain
5. Has strong physical motivation (kinematic depth → denoising complexity)
6. Ablation infrastructure already exists

**Secondary actions** (writing-level): expand L_tr, mention translation/rotation loss balancing, expand 2D RoPE physical interpretation. These require no experiments.
