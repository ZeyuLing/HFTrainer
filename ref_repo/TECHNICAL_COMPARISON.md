# Technical Comparison: UMO vs MotionLab vs HyMotion M2M
## Detailed Analysis of Control Mechanisms, Architecture, and Training

---

## 1. UMO: Frame-Level Meta-Operations Architecture

### 1.1 Control Mechanism: P/G/E Meta-Operations

**Core Concept**: Every frame in any motion task falls into **exactly one** of three mutually exclusive categories:
- **[P] Preserve**: Keep this frame unchanged from source
- **[G] Generate**: Generate this frame from scratch (no source reference)
- **[E] Edit**: Modify this frame based on source motion

**Key Detail: These are FRAME-LEVEL only** — no per-joint granularity.

```
Example: Keyframe In-filling Task
source motion:  [frame_1, 0,        0,        ..., 0,        frame_T]  (T, 201)
meta-op τ:      [P,       G,        G,        ..., G,        P       ]  (T,)
```

The mapping:
- Preserve frames: `s_i = actual_source_motion[i]` + `Emb(τ=P)`
- Generate frames: `s_i = 0` + `Emb(τ=G)`
- Edit frames: `s_i = actual_source_motion[i]` + `Emb(τ=E)`

### 1.2 Temporal Fusion: How Control Enters the Model

**Architecture Choice**: Tested 4 alternatives on Keyframe Infilling:

| Method | Extra Params | Extra FLOPs | Extra Latency | [P]-MPJPE | FID |
|--------|-------------|------------|---------------|-----------|-----|
| **Temporal Fusion** (WINNER) | **0.207M** | **0.140G** | **0.01s** | **0.95** | **0.476** |
| AdaLN | 4.4M | 1.66G | 0.02s | 11.1 | 8.86 |
| Sequential Concat | 0.207M | 198.6G | 0.89s | 2.04 | 11.77 |
| ControlNet | 234M | 85.12G | 0.49s | 5.19 | 6.52 |

**Temporal Fusion Formula**:
```python
# Encode source + meta-op into context
s̃_i = s_i + Emb(τ_i)          # τ_i ∈ {P, G, E}

# Inject at input embedding level (element-wise add)
x'_t = E_in(x_t) + E_ctx(s̃)

# where:
# E_in = pretrained input encoder from HY-Motion
# E_ctx = MLP encoder (0.207M), initialized as COPY of E_in
```

**Why it wins**:
1. **Minimal overhead**: Only 0.207M parameters (compare to AdaLN's 4.4M, ControlNet's 234M)
2. **No backbone changes**: All MMDiT blocks remain frozen, T2M quality fully preserved
3. **E_ctx initialization trick**: Copy pretrained E_in weights → inherits representation learning
4. **Fast**: No latency overhead (just matrix addition at input level)

### 1.3 Frame-Level vs Per-Joint: The Core Limitation

**What UMO can do**:
```
Whole-body editing: "Speed up your motion"
[P]  [E]  [E]  [E]  ... [E]  [P]   ← Applies to ALL 22 joints simultaneously
```

**What UMO CANNOT do (Paper Limitation §5)**:
```
Part-level editing: "Speed up RIGHT ARM only, keep rest unchanged"
← Impossible: τ is per-frame, not per-joint
```

This is explicitly listed as a limitation in the paper. **M2M has clear advantage here** with per-dimension masking.

### 1.4 Adapter Design: E_ctx MLP

```python
class TemporalFusion(nn.Module):
    def __init__(self, motion_dim=201):
        super().__init__()
        # E_ctx: Two-layer MLP
        self.linear1 = nn.Linear(motion_dim, 2048)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(2048, motion_dim)
    
    def forward(self, s_tilde):
        # s_tilde = (T, 201)
        return self.linear2(self.gelu(self.linear1(s_tilde)))
```

**Initialization**:
```python
# Copy pretrained input encoder weights
E_ctx.linear1.weight = E_in.linear1.weight.clone()
E_ctx.linear1.bias = E_in.linear1.bias.clone()
# ... same for linear2
```

**Training**:
- Backbone: Frozen (HY-Motion-Lite 460M)
- Trainable: E_ctx (0.207M) + 3 meta-op embeddings (201×3)
- Multi-task joint training (MIB, prediction, editing, reaction simultaneously)
- **No curriculum learning** — all tasks trained from the start with equal probability

### 1.5 Motion Representation

**UMO's 201-dim representation** (vs M2M's 138-dim):
```
Global root translation:     3D    (3 dims)
Root orientation:           6D    (6 dims)  
21 local joint rotations:    21×6D (126 dims)
22 local joint positions:    22×3D (66 dims)  ← M2M lacks this!
─────────────────────────────────────────
Total:                       201 dims
```

**Implication**: UMO can directly control xyz trajectory (through joint positions), M2M cannot.

---

## 2. MotionLab: Multi-Modality Path Architecture

### 2.1 Motion-Condition-Motion Paradigm

**Unified abstraction**: Every task is defined as `(source_motion, condition, target_motion)`:

| Task | Source | Condition | Target | Note |
|------|--------|-----------|--------|------|
| Unconditional gen | ∅ | ∅ | ✓ | Pure generation |
| In-betweening | ∅ | keyframes | ✓ | M6 in M2M |
| Text-based gen | ∅ | text | ✓ | T2M |
| **Text-based editing** | ✓ | text | ✓ | **M2M lacks** |
| Trajectory-based gen | ∅ | traj hint + text | ✓ | **M2M lacks** |
| **Style transfer** | ✓ | style_motion | ✓ | **M2M lacks** |

### 2.2 MotionFlow Transformer (MFT): 5 Modality Paths

**Architecture**: MM-DiT variant with 5 **independent** modality paths:

```python
class JointAttention(nn.Module):
    # 5 modalities with independent processing before shared attention
    
    def forward(self, x_source, x_target, x_text, x_hint, x_style):
        # Each modality has independent:
        # - Linear projection
        # - adaLN modulation (timestep + task_instruction dependent)
        # - FFN (independent to each modality)
        # - LayerNorm
        
        # But SHARED attention (QKV concatenation):
        Q = concat([Q_source, Q_target, Q_text, Q_hint, Q_style])  # (5*L, D)
        K = concat([K_source, K_target, K_text, K_hint, K_style])
        V = concat([V_source, V_target, V_text, V_hint, V_style])
        
        attn_out = attention(Q, K, V)  # Cross-modality interaction
        
        # Split back to 5 modalities
        out_source = attn_out[:L]
        out_target = attn_out[L:2L]
        out_text = attn_out[2L:3L]
        out_hint = attn_out[3L:4L]
        out_style = attn_out[4L:]
        
        return out_source, out_target, out_text, out_hint, out_style
```

**Key Design**:
- 5 modalities: `source_motion (M_S)` / `noisy_target (M_T)` / `text (CLIP)` / `trajectory_hint (66D)` / `style (512+256D)`
- Each has **isolated adaLN, FFN, LayerNorm** 
- **Shared attention** → forced cross-modal interaction
- Token dim = 512 (lite version vs standard MM-DiT's 1536)

### 2.3 Aligned 1D RoPE: Time Synchronization

**Critical Innovation for Multi-Modality**: 

```python
from rotary_embedding_torch import RotaryEmbedding

# ALL time-dependent modalities (source, target, trajectory) share SAME 1D RoPE
rope = RotaryEmbedding(dim=512, freq_base=10000)

# Key insight:
# source[i] and target[i] have IDENTICAL position embeddings
# → strong implicit constraint that they refer to same frame i
```

**Why this matters**:
- MotionFix dataset has only 6.7k paired examples (very sparse)
- Without aligned time axis, attention would need to learn frame correspondence implicitly
- Aligned 1D RoPE makes time correspondence **explicit architectural bias**

**Ablation impact**: Without Aligned ROPE, trajectory error increases **2.6×** (0.0334 → 0.0886)

### 2.4 Task Instruction Modulation (TIM)

**Mechanism**:
```python
# Task instruction as natural language
instruction = "edit source motion by given text"  # or many other tasks

# Encode with CLIP-L/14
task_token = CLIP.encode_text(instruction)  # (1, 768)

# Add to timestep embedding → adaLN sees it globally
y = timestep_emb + task_token  # (batch, 768)

# Passed to all 5 modality paths' adaLN
out = adaLN(x, y)  # All paths affected by task_instruction
```

**Advantages over alternatives**:
- vs learnable `[TASK]` token: Can handle arbitrary new tasks without retraining
- vs one-hot encoding: Leverages CLIP's semantic understanding
- vs implicit task routing: Task boundary is **explicit**, helps model focus

### 2.5 Motion Curriculum Learning: 7-Stage Schedule

**Stage 1: Self-Supervised Pre-training (1000 epochs)**

```
- Masked source reconstruction: Mask 0-100% of source frames uniformly
- Masked trajectory reconstruction: Random mask on joint trajectories
- Implicit in-between learning (from 0-100% masking)

Effect: Model learns motion priors without task labels
```

**Stage 2: Supervised Fine-tuning (7 stages, each 200 epochs)**

```
Stage ① (ep 0-200):      text-based generation
        ↓ add 45% old tasks to training mix
Stage ② (ep 200-400):    + style-based generation  
        ↓ add 45% old tasks (with FID-based importance weighting)
Stage ③ (ep 400-600):    + trajectory-based editing (no text)
        ↓
Stage ④ (ep 600-800):    + text-based editing
        ↓
Stage ⑤ (ep 800-1000):   + style transfer
        ↓
Stage ⑥ (ep 1000-1200):  + motion in-between + trajectory generation
        ↓
Stage ⑦ (ep 1200-1400):  + trajectory editing (final)

Total training time: 4× RTX 4090D × 4 days
```

**Anti-Forgetting Mechanism**:
```python
# Dynamic task sampling at each stage
# Sample composition:
#   45% new task
#   45% old tasks (with adaptive weights)
#    5% unconditional generation
#    5% masked reconstruction

# Old task sampling weight updates based on last eval:
# If FID of task X went up by 20% last eval → sample it more
# If FID of task X stable → sample it less
```

**Ablation Study Result**:
```
WITH curriculum learning:     FID = 0.167
WITHOUT curriculum learning:  FID = 1.956
Degradation: 11.7×  ← MOST CRITICAL CONTRIBUTION
```

### 2.6 Motion Representation

**263-dim HumanML3D format** (vs M2M's 138-dim):
```
Root velocity:           3D    (3 dims)
22 joint positions:      22×3D (66 dims) ← M2M LACKS
22 joint velocities:     22×3D (66 dims) ← M2M LACKS
22 joint 6D rotations:   22×6D (132 dims)
Foot contact:            4D    (4 dims)  ← M2M LACKS
────────────────────────────────────────
Total: 263 dims
```

**Trajectory Hint (separate modality)**:
```
22 joints × 3 xyz = 66-dim trajectory hint
(Independent of source/target motion representation)
```

---

## 3. HyMotion M2M: VACE Binary Mask Architecture

### 3.1 Control Mechanism: Per-Dimension Binary Mask

**Core Concept**: Define controllable regions at **T × 138 granularity** (frame × dimension):

```
Mask shape: (T, 138)
  - mask[t, d] = 0 → dimension d at frame t is KNOWN (take from source)
  - mask[t, d] = 1 → dimension d at frame t is UNKNOWN (generate)

Source motion shape: (T, 138)
  - If mask[t, d] = 0: Use src_motion[t, d] as condition
  - If mask[t, d] = 1: Model must generate it
```

**Example: M4 Joint Editing (Left Arm Only)**
```
Suppose left arm joints are dims [30-42] in 138-dim vector

M4 mask (joint editing):
mask[:, 0:30] = 0    # Keep body/root unchanged
mask[:, 30:42] = 1   # Left arm dims to regenerate
mask[:, 42:138] = 0  # Keep everything else

Result: Only left arm regenerated, rest preserved
```

### 3.2 VACE Conditioning: Channel Concatenation

**Formula**:
```python
x_input = concat([
    x_t,                          # (T, 138) noisy motion
    inactive,                     # (T, 138) known values
    reactive,                     # (T, 138) masked values (for model awareness)
    src_mask                      # (T, 138) binary mask
], dim=-1)
# Shape: (T, 552)  [4× original motion_dim]

# Then through input_encoder
x_emb = input_encoder(x_input)  # (T, hidden_dim)
```

**inactive/reactive Split Logic**:
```python
inactive = src_motion * (1 - src_mask)   # Known regions
reactive = src_motion * src_mask         # Masked regions
```

**Interpretation**:
- `inactive`: "Here's what I know for sure" (guide model away from source in these dims)
- `reactive`: "Here are the masked dims (I know source values but will ignore them)"
- `src_mask`: "These are the regions you need to generate"

### 3.3 Six Mask Strategies (M1-M6)

**Training Mix** (fixed ratio):
```
M1 Random Cell:       25%  → Random (t, d) cells masked
M2 Random Block:      15%  → Random rectangular blocks in (T, d) plane
M3 Temporal:          25%  → Temporal segments (all dims masked for contiguous frames)
M4 Joint Editing:     15%  → Specific joint dims masked across all T
M5 Full Mask:         5%   → All dims masked (pure T2M)
M6 Keyframe:          15%  → Sparse keyframes at dims, middle frames masked
```

**Key Difference from UMO**:
- UMO: Frame-level meta-op (entire frame is P/G/E)
- M2M: Dimension-level mask (each of 138 dims independent)

### 3.4 Per-Joint vs Whole-Frame Control Comparison

| Aspect | UMO Frame-Level | M2M Per-Dim |
|--------|-----------------|------------|
| Can mask right arm only? | ❌ No (whole frame) | ✅ Yes (dims 30-42) |
| Mask granularity | (T,) — single value per frame | (T, 138) — unique per dim |
| Task: "Keep left leg" | ❌ Impossible | ✅ Set mask=0 for those dims |
| Complexity for per-joint editing | N/A — Not supported | Supported natively |

---

## 4. Mask/Conditioning Design Comparison

### 4.1 Mask Semantics

| Framework | Mask Encoding | Semantics | Constraints |
|-----------|---------------|-----------|-------------|
| **UMO** | τ_i ∈ {P, G, E} (3 classes) | P=preserve, G=generate, E=edit | Per-frame only |
| **MotionLab** | source_motion + task_instruction | Implicit (task type from instruction) | Per-modality level |
| **M2M** | Binary mask (T, 138) | 0=condition, 1=generate | Per-dimension |

### 4.2 How Conditioning Information Flows

**UMO - Element-wise Add to Input Embedding**:
```
Inference step:
1. x_t_noisy = current noisy motion (201)
2. s̃ = source_motion + Emb(meta_op)  (201)
3. input_emb = E_in(x_t_noisy) + E_ctx(s̃)  ← ADD at embedding level
4. Backbone (frozen) sees modified input embedding
5. No hard enforcement: model CAN violate [P] frames if it wants
```

**MotionLab - Independent Modality Path + Joint Attention**:
```
Inference step:
1. x_t_noisy = noisy target motion (263)
2. s = source motion (263)
3. E_source = independent source embedding + positional encoding
   E_target = independent target embedding + positional encoding
   E_text = CLIP text embedding
   E_hint = trajectory coordinates (66D projected)
4. All 4 embeddings go through JointAttention with ALIGNED 1D RoPE
5. Cross-attention naturally aligns source[i] with target[i]
```

**M2M - Channel Concatenation to Input Encoder**:
```
Inference step:
1. x_t_noisy = noisy motion (138)
2. inactive = source * (1 - mask) (138)
3. reactive = source * mask (138)
4. x_input = [x_t_noisy, inactive, reactive, mask]  (552)
5. input_encoder projects from 552 → hidden_dim
6. Backbone sees explicit conditioning in input space (not latent)
7. No hard enforcement: soft constraint through training
```

### 4.3 Constraint Precision Trade-offs

| Framework | Constraint Type | Precision | Soft/Hard |
|-----------|-----------------|-----------|-----------|
| **KIMODO** (for comparison) | Imputation (hard replace) | Position dims exact, rotation imprecise | Hard |
| **UMO** | Element-wise add to embedding | [P]-MPJPE ≈ 0.95mm (soft) | Soft (learned) |
| **MotionLab** | Modality path + Aligned ROPE | Trajectory error 0.0286m (very precise) | Soft + architectural bias |
| **M2M** | VACE channel concat | No explicit evaluation | Soft (learned) |

---

## 5. Training Mechanics Comparison

### 5.1 Training Data Distribution

| Framework | Pre-training | Main Training | Curriculum |
|-----------|--------------|---------------|-----------|
| **UMO** | None | 100k steps, multi-task joint (no schedule) | None |
| **MotionLab** | 1000 epochs masked reconstruction | 1400 epochs, 7-stage curriculum | Yes (critical) |
| **M2M** | From scratch | M1-M6 fixed ratio from epoch 1 | No (parallel not sequential) |

### 5.2 Architectural Freezing Strategies

**UMO**: 
```
- Backbone: FROZEN (HY-Motion-Lite, 460M)
- Trainable: E_ctx MLP (0.207M) + 3 meta-op embeddings
- Reasoning: Preserve T2M pretrain quality
```

**MotionLab**:
```
- Backbone: TRAINED (MFT, 512 token_dim)
- From scratch (no pretrained checkpoint)
- Reasoning: Small model (lite MM-DiT) allows full training
```

**M2M**:
```
- Backbone: TRAINED (HunyuanMotion, 0.46B-1.5B)
- input_encoder: Re-initialized to accommodate 4× input dims
- Reasoning: Large foundation model → different training regime
```

### 5.3 Multi-Task Learning Approaches

**UMO - No Curriculum**:
```python
for step in range(100k):
    task = randomly_choose_from([T2M, keyframe, inpainting, editing, ...])
    loss = train_on_task(task)
```

**MotionLab - Strict Curriculum**:
```python
# Stage 1 (1000 epochs)
for epoch in range(1000):
    tasks = [masked_reconstruction]  # Single task

# Stage 2 (1400 epochs)
for stage_idx, new_task in enumerate([T2M, style_gen, traj_edit, ...]):
    for epoch in range(200):
        # Mix new task + all previous tasks (weighted by FID feedback)
        samples = sample_mixture(old_tasks, new_task, fid_weights)
```

**M2M - Fixed Ratio Mixture**:
```python
for epoch in training_epochs:
    for batch in dataloader:
        # Pre-computed mask distribution:
        # M1: 25%, M2: 15%, M3: 25%, M4: 15%, M5: 5%, M6: 15%
        # Ratio stays constant throughout training
```

---

## 6. Technical Differences Summary Table

| Dimension | UMO | MotionLab | M2M |
|-----------|-----|-----------|-----|
| **Motion Representation** | 201D (with positions) | 263D (with velocities + contacts) | 138D (rotation only) |
| **Conditioning Method** | Element-wise add (input emb) | Modality paths + JointAttention | Channel concat (input encoder) |
| **Condition Granularity** | Frame-level (T,) | Modality-level + trajectory hint | Dimension-level (T, 138) |
| **Mask/Control Semantics** | 3-class (P/G/E) | Implicit (task-dependent) | Binary (0/1) |
| **Task Instruction** | Implicit (from data) | Explicit CLIP text | Implicit (mask pattern) |
| **Part-Level Control** | ❌ (Paper limitation) | ⚠️ Via trajectory + text | ✅ (Native) |
| **Adapter Params** | 0.207M (E_ctx only) | Full MFT trained | ~4× input_encoder |
| **Backbone Freezing** | Frozen (T2M preserved) | Trained from scratch | Trained |
| **Curriculum Learning** | None (parallel) | 7-stage with FID feedback | Fixed ratio |
| **Supported: Text Gen** | ✅ | ✅ | ✅ |
| **Supported: Trajectory** | ❌ (no position dims) | ✅ (Aligned ROPE + hint) | ❌ (no position dims) |
| **Supported: Instruction Editing** | ✅ via [E] + text | ✅ (explicit task text) | ❌ (M4 only does part regen) |
| **Supported: Style Transfer** | ❌ | ✅ (style modality) | ❌ |
| **Training Steps** | 100k | 1400 epochs (curriculum) | From scratch (unknown steps) |
| **FID on HumanML3D** | 9.46 (T2M SOTA) | 0.167 (gen SOTA) | ~0.15-0.2 (MotionHub) |
| **Trajectory Error** | 18.78cm (90× faster than optimization) | **0.0286m** (OmniControl killer) | N/A (no xyz dims) |
| **Instruction Edit R@3** | 100% (vs PartMotionEdit 90%) | 56.34 R@1 | N/A |

---

## 7. What Makes Each Approach Unique

### UMO's Distinctive Features
1. **Extreme parameter efficiency** (0.207M adapter on frozen 460M backbone)
2. **[Edit] semantic** — explicit "modify based on source" operation
3. **Language-only geometry** — trajectories as structured text
4. **All-tasks-at-once** training (no curriculum)

### MotionLab's Distinctive Features
1. **5-modality path architecture** — each modality independently processed before Joint Attention
2. **Aligned 1D RoPE** — force-aligns source[i] and target[i] at embedding level
3. **Curriculum learning with FID-based weight** — most impactful contribution (11.7× FID degradation without it)
4. **Task instruction as CLIP text** — true zero-shot task routing
5. **263D representation with velocities + contacts** — richer motion features

### M2M's Distinctive Features
1. **Per-dimension mask** (T, 138) — most fine-grained control of all three
2. **VACE three-channel split** (inactive/reactive/mask) — explicit source handling at input
3. **Six mask strategies** — comprehensive coverage of completion patterns
4. **Large foundation model** (0.46B-1.5B) on large dataset (549k samples)

---

## 8. Critical Gaps in M2M vs Competitors

| Gap | Impact | How UMO/MotionLab does it | Effort to fix M2M |
|-----|--------|---------------------------|-------------------|
| No xyz position control | Can't do trajectory hints | UMO: 66D local pos / MotionLab: explicit hint modality | Add position dims to representation + FK loss |
| No instruction editing | Can't understand "speed up" / "opposite leg" | UMO/MotionLab: text instruction | Add CLIP task instruction modulation |
| No style transfer | Limited editing scope | MotionLab: style modality + MCM-LDM encoder | Add style encoder (can reuse existing) |
| No explicit curriculum | May be undertrained on T2M | MotionLab: Proven 11.7× FID improvement | Replace fixed ratio with staged schedule |
| No task awareness token | Model doesn't "know" which task | MotionLab: CLIP instruction in adaLN | Easy: add instruction to timestep emb |

---

## 9. Recommendations for M2M Evolution

### Priority 0 (Quick wins, <1 week each)
1. **Add Task Instruction Modulation**: Encode task text with CLIP, add to timestep embedding
   - Helps model focus on specific task during multi-task training
   - Extends to instruction editing later

2. **Implement Motion Curriculum Learning**: Replace fixed M1-M6 ratio with 3-4 stage schedule
   - Stage 1: M5 (T2M) + M3 (temporal) only
   - Stage 2: Add M6 (keyframe)
   - Stage 3: Add M1/M2/M4 (fine-grained)
   - Include FID-based dynamic weighting of older tasks

### Priority 1 (Significant improvements, 1-2 weeks)
3. **Add position dimensions to representation** (if trajectory control needed)
   - Extend from 138D to 141D (add root xyz)
   - Retrain input_encoder
   - Add FK loss supervision

4. **Modality Path Upgrade** (if editing accuracy is critical)
   - Split source motion into independent modality path (like MotionLab)
   - Add Aligned 1D RoPE for time synchronization
   - Reduces "over-copying" source in editing tasks

### Priority 2 (Long-term, architectural)
5. **Style Transfer Module**: Add MCM-LDM style encoder as 6th modality path

---

## 10. Genuine Technical Differentiation of M2M

Despite similarities, M2M has unique strengths:

1. **Finest granularity control**: (T, 138) vs UMO's (T,) or MotionLab's modality-level
   - Enables true per-joint manipulation impossible for others
   - Example: "Keep torso, regenerate only right arm" — native support

2. **No reliance on pretrained foundation model**:
   - UMO needs frozen HY-Motion backbone (dependency on external model)
   - M2M trains from scratch → full control + potential for better task fit

3. **Representation simplicity vs. explicitness**:
   - MotionLab: 263D with implicit semantics (velocity, contact channels)
   - M2M: 138D explicitly just (position + rotation) → more interpretable masking

4. **VACE split (inactive/reactive) novelty**:
   - Unlike UMO's element-wise add, explicitly separates "what I know" vs "what's masked"
   - Unlike MotionLab's Joint Attention, doesn't require architectural explosion

---

## Conclusion

**UMO** is the parameter-efficiency champion (0.207M on frozen backbone), best for quick adaptation to new tasks.

**MotionLab** is the curriculum/architecture champion, with curriculum learning being its secret weapon (11.7× FID difference).

**M2M** is the control granularity champion — per-dimension masking is genuinely novel and enables capabilities neither competitor offers. The key to unlocking M2M's potential is:
1. Add Task Instruction Modulation (easy, high impact)
2. Implement curriculum learning (medium effort, proven huge gains)
3. Consider modality path architecture for editing tasks (larger effort, cleaner design)

