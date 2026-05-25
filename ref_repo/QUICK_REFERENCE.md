# Quick Reference: UMO vs MotionLab vs M2M

## 1. How Control Works (The Essence)

### UMO: Frame-Level Meta-Operations
```
INPUT SIDE (Inference):
    source_motion (T, 201)        ← Full motion for frames marked [P] or [E]
    meta_ops τ (T,)               ← [P], [G], or [E] for each frame
                ↓
    s̃ = source + Emb(τ)          ← Add to motion
    x'_emb = E_in(x_t) + E_ctx(s̃) ← Add to input embedding
                ↓
    FROZEN Backbone (HY-Motion-Lite)
                ↓
    OUTPUT: Denoised motion

RESULT: Whole-frame operations only. Cannot selectively edit one arm while keeping rest.
```

### MotionLab: Multi-Modality Joint Attention
```
INPUT SIDE (Inference):
    source_motion (T, 263)        ← Independent modality
    target_motion (T, 263)        ← Noisy target (what we're denoising)
    text (CLIP 768D)              ← Independent modality
    trajectory_hint (T, 66)       ← 22 joints × 3 xyz, independent modality
    style_feat (512+256D)         ← Optional style modality
                ↓
    Each gets: independent embedding + adaLN + FFN + LayerNorm
                ↓
    JointAttention (QKV from all 5 modalities concatenated)
    ALL USE ALIGNED 1D RoPE      ← Forces source[i] ↔ target[i] time sync
                ↓
    TRAINED Backbone (MFT)
                ↓
    OUTPUT: Each modality path has output

RESULT: Per-modality operations. Trajectory hint is explicit separate channel (0.0286m precision).
```

### M2M: Binary Mask + VACE Channels
```
INPUT SIDE (Inference):
    x_t (T, 138)                  ← Noisy motion
    src_motion (T, 138)           ← Source motion (may have many zeros if mask=1)
    inactive (T, 138)             ← src * (1-mask), known values
    reactive (T, 138)             ← src * mask, masked values
    src_mask (T, 138)             ← Binary 0/1 indicator
                ↓
    x_input = [x_t; inactive; reactive; mask]  ← Concatenate = (T, 552)
                ↓
    input_encoder (552 → hidden_dim)
                ↓
    TRAINED Backbone (HunyuanMotion)
                ↓
    OUTPUT: Denoised motion (T, 138)

RESULT: Per-dimension operations. Can mask dims 30-42 (left arm) while keeping 0-30 (body).
```

---

## 2. Conditioning Injection Methods (Side-by-Side)

```python
# UMO: Element-wise Add at Embedding Level
def umo_condition(x_t, source_motion, meta_op):
    s_tilde = source_motion + Emb(meta_op)  # (T, 201)
    x_emb = E_in(x_t) + E_ctx(s_tilde)      # Element-wise add at embedding
    # Backbone sees modified embedding, doesn't know about condition directly
    return backbone(x_emb)

# MotionLab: Independent Paths → Joint Attention
def motionlab_condition(x_t_noisy, source, text, traj_hint):
    # Each modality independent
    source_emb = embed_source(source)                    # (L, D)
    target_emb = embed_target(x_t_noisy)                # (L, D)
    text_emb = encode_text(text)                        # (L, D)
    hint_emb = embed_hint(traj_hint)                    # (L, D)
    
    # Concatenate for attention (force interaction)
    Q = concat([Q_source, Q_target, Q_text, Q_hint])    # (4L, D)
    K = concat([K_source, K_target, K_text, K_hint])    # (4L, D)
    V = concat([V_source, V_target, V_text, V_hint])    # (4L, D)
    
    attn_out = attention(Q, K, V)  # All modalities interact
    # Split back
    return backbone(source_out, target_out, text_out, hint_out)

# M2M: Channel Concatenation to Input Encoder
def m2m_condition(x_t, src_motion, src_mask):
    inactive = src_motion * (1 - src_mask)              # Known values
    reactive = src_motion * src_mask                    # Masked values (zero if mask=1)
    
    x_input = concat([x_t, inactive, reactive, src_mask], dim=-1)  # (T, 552)
    x_emb = input_encoder(x_input)  # Projects 552 → hidden_dim
    # Backbone sees conditioning already in feature space
    return backbone(x_emb)
```

---

## 3. What Each Can Do (Task Coverage)

| Task | UMO | MotionLab | M2M |
|------|-----|-----------|-----|
| **T2M (text to motion)** | ✅ | ✅ | ✅ |
| **Keyframe In-fill** | ✅ | ✅ | ✅ (M6) |
| **Temporal completion** | ✅ | ✅ | ✅ (M3) |
| **Random mask recovery** | ✅ | ✅ | ✅ (M1/M2) |
| **Part-level control** (e.g., "left arm only") | ❌ | ⚠️ Not native | ✅ Native |
| **Trajectory following** | ❌ | ✅ (0.0286m) | ❌ |
| **Instruction editing** ("speed up") | ✅ | ✅ | ❌ |
| **Style transfer** | ❌ | ✅ | ❌ |
| **Multi-person / reaction** | ✅ (in paper) | ❌ | ❌ |

---

## 4. The Adapter/Architecture Pattern

### UMO's Minimalism
```
Pretrained HY-Motion (460M) ─→ [FROZEN]
                                  ↑
                                  │ (x'_emb = E_in(x_t) + E_ctx(source))
                                  │
                           E_ctx (0.207M)
                           ├─ Linear1: 201 → 2048
                           ├─ GELU
                           └─ Linear2: 2048 → 201
                           (Initialized as copy of E_in)
```
**Cost**: 0.207M parameters
**Benefit**: Preserves all T2M quality, extremely efficient
**Tradeoff**: Can't do per-joint control

### MotionLab's Modularity
```
MFT (Full MM-DiT) ──→ [TRAINED]
    ├─ Source Path: embed → adaLN → FFN → LayerNorm
    ├─ Target Path: embed → adaLN → FFN → LayerNorm
    ├─ Text Path:   embed → adaLN → FFN → LayerNorm
    ├─ Hint Path:   embed → adaLN → FFN → LayerNorm
    └─ Style Path:  embed → adaLN → FFN → LayerNorm
    
    All 5 paths share: JointAttention (QKV concat)
                     + Aligned 1D RoPE (time sync)
```
**Cost**: Full architecture retraining
**Benefit**: Clean modality separation, extensible
**Tradeoff**: Larger model

### M2M's Concatenation
```
input_encoder (540 → hidden_dim) ──→ [RETRAINED for 4× input]
                ↑
        [x_t | inactive | reactive | mask]
        (138)  (138)      (138)      (138)
          ↓
        (552-dim input)
```
**Cost**: input_encoder weights reshape/reinit
**Benefit**: Per-dimension granularity
**Tradeoff**: More input parameters

---

## 5. Training Strategies

### UMO
```
Epoch 0:   [T2M] [Inpainting] [Editing] [Reaction] ...  ← All tasks mixed
Epoch 1:   [T2M] [Inpainting] [Editing] [Reaction] ...  ← Same mix
...
Epoch 99:  [T2M] [Inpainting] [Editing] [Reaction] ...  ← Constant ratio

100k total steps, backbone frozen, only E_ctx trained
No curriculum, no task awareness token
```

### MotionLab
```
Pre-train (1000 epochs):
    Epoch 0-1000: [Masked Reconstruction]  ← Self-supervised

Fine-tune (1400 epochs, 7 stages):
    Stage 1 (ep 0-200):     [T2M]                   (45% new + 45% old + 10% other)
                ↓ add to old task pool
    Stage 2 (ep 200-400):   [T2M] [StyleGen]        (45% new + 45% old + 10% other)
                ↓ old tasks weighted by FID change
    Stage 3 (ep 400-600):   [T2M] [StyleGen] [TrajEdit]
                ↓ NEW: if FID of T2M went up, sample it more
    ...
    Stage 7 (ep 1200-1400): All 7 tasks             (anti-forgetting via FID weight)

Result: Curriculum proven essential (11.7× degradation without it)
        Task instruction modulation: CLIP text → adaLN
```

### M2M
```
Epoch 0:   M1(25%) M2(15%) M3(25%) M4(15%) M5(5%) M6(15%)
Epoch 1:   M1(25%) M2(15%) M3(25%) M4(15%) M5(5%) M6(15%)
...
Epoch N:   M1(25%) M2(15%) M3(25%) M4(15%) M5(5%) M6(15%)

Fixed ratio from day 1, no curriculum, no task awareness
Question: Is M5 (5% pure T2M) enough to maintain quality?
Question: Does multi-task training hurt single-task specialization?
```

---

## 6. Representation Comparison

### What's in Each Motion Vector?

| Component | UMO (201D) | MotionLab (263D) | M2M (138D) |
|-----------|-----------|-----------------|-----------|
| Root translation | 3D ✅ | 3D ✅ | 3D ✅ |
| Root velocity | — | 3D ✅ | — |
| Root orientation | 6D ✅ | 6D ✅ | 6D ✅ |
| Local joint rotation (21/22 joints) | 21×6D ✅ | 22×6D ✅ | 22×6D ✅ |
| **Local joint position** | 22×3D ✅ | 22×3D ✅ | ❌ MISSING |
| **Joint velocity** | — | 22×3D ✅ | — |
| **Foot contact** | — | 4D ✅ | — |
| **Total dims** | **201** | **263** | **138** |

**Implication of missing dims in M2M**:
- Can't directly control xyz trajectory (e.g., "walk to point (x,y,z)")
- MotionLab: explicit 66D trajectory hint modality → 0.0286m error
- M2M: would need to add position dims + FK loss + retrain

---

## 7. The Three Approaches Visualized

```
┌─ UMO: "Adapter on Frozen Foundation"
│  ├─ Backbone frozen (HY-Motion pretrain)
│  ├─ Add 0.207M adapter (E_ctx)
│  ├─ Temporal fusion: element-wise add to embedding
│  └─ Result: Param-efficient, T2M quality preserved, but no per-joint control
│
├─ MotionLab: "Multi-Modality Fusion"
│  ├─ 5 independent modality paths (source, target, text, trajectory, style)
│  ├─ JointAttention with Aligned 1D RoPE
│  ├─ Curriculum learning (11.7× FID gain from scheduling)
│  ├─ Task instruction modulation (CLIP text)
│  └─ Result: Clean architecture, trajectory precision, instruction editing, style transfer
│
└─ M2M: "Per-Dimension Masking at Input"
   ├─ (T, 138) binary mask for each dimension
   ├─ VACE 3-channel split (inactive/reactive/mask)
   ├─ Channel concat to input_encoder
   ├─ M1-M6 fixed ratio training
   └─ Result: Ultra-fine control (per-joint), but no trajectory, no curricula
```

---

## 8. Decision Tree: Which Approach for What?

### **If you want parameter efficiency + frozen backbone:**
→ **UMO's temporal fusion** (0.207M adapter)

### **If you want fine-grained control (per-joint):**
→ **M2M's VACE masking** (T, 138 binary mask)

### **If you want trajectory control + instruction editing:**
→ **MotionLab's modality paths** + curriculum learning

### **If you want to boost M2M:**
1. Add curriculum learning → likely 2-10× improvement on T2M
2. Add task instruction modulation → explicit task routing
3. Add position dimensions if trajectory needed

---

## 9. Key Technical Insights

### Insight 1: Curriculum Learning is Critical
**Evidence**: MotionLab ablation shows 11.7× FID degradation without curriculum
- Implication: Fixed ratio (M2M) may underutilize potential
- Recommendation: Try staged curriculum on M2M

### Insight 2: Frame-Level vs Dimension-Level is Hard Tradeoff
- **Frame-level (UMO)**: Simple, efficient, but can't do partial editing
- **Dimension-level (M2M)**: Complex, but enables per-joint control
- **Modality-level (MotionLab)**: Heavy, but very clean separation

### Insight 3: Aligned Time Encoding Matters
- **MotionLab**: Forces source[i] ↔ target[i] sync via 1D RoPE → 2.6× trajectory error improvement
- **M2M**: Channel concat → implicit time alignment (may be sufficient)
- **UMO**: No separate source modality → N/A

### Insight 4: Task Awareness Token Helps
- **Explicit (MotionLab)**: CLIP instruction text added to adaLN
- **Implicit (UMO, M2M)**: Model learns from data/mask pattern
- **Benefit**: Helps model focus on specific task in multi-task setting

### Insight 5: Representation Richness vs Simplicity
- **263D (MotionLab)**: Rich (velocity, contacts) but implicit
- **138D (M2M)**: Simple (just pos+rot) but explicit
- **201D (UMO)**: Moderate (has positions, no velocity)

---

## 10. One-Sentence Summary

| Framework | Summary |
|-----------|---------|
| **UMO** | Extremely efficient (0.207M) adapter for frozen backbone, but whole-frame operations only |
| **MotionLab** | Multi-modality paths + curriculum learning = highest FID gains and trajectory precision |
| **M2M** | Per-dimension masking = ultra-fine control, but needs curriculum + task awareness boost |

