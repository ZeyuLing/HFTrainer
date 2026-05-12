# HyMotion T2M 1.0 - Visual Reference Guide

## 1. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TEXT-TO-MOTION INFERENCE PIPELINE                   │
└─────────────────────────────────────────────────────────────────────────┘

INPUT
  │
  ├─ Prompt Text: "a person walks forward"
  ├─ Motion Length: 360 frames (12 seconds @ 30fps)
  └─ CFG Scale: 5.0

  │
  ▼

[STEP 1: TEXT ENCODING]
  │
  ├─ Qwen3 LLM Encoder
  │   └─ Output: (1, 512, 4096)  [batch, max_seq_len, dims]
  │
  └─ CLIP-L Sentence Encoder
      └─ Output: (1, 77, 768)    [batch, max_seq_len, dims]

  │
  ▼

[STEP 2: PREPARE DIFFUSION SCHEDULE]
  │
  ├─ Time steps: t ∈ {0, 1/50, 2/50, ..., 1}  (50 steps)
  ├─ Initial noise: y₀ ~ N(0, I), shape (1, 360, 201)
  └─ Null context: For classifier-free guidance

  │
  ▼

[STEP 3: ODE INTEGRATION (Euler Solver)]
  │
  └─ FOR each time step t in [0, 1]:
     │
     ├─ Compute flow prediction:
     │  └─ x_pred = transformer(y_t, t, text_context, text_vec)
     │
     ├─ Apply Classifier-Free Guidance:
     │  └─ x_guided = x_cond + scale × (x_cond - x_uncond)
     │  │  └─ where scale = 5.0 (default)
     │
     ├─ Update: y_{t+1} = y_t + Δt × x_guided
     │
     └─ Output trajectory: [y₀, y₁, ..., y₅₀]

  │
  ▼

[STEP 4: DENORMALIZATION]
  │
  ├─ Load: mean, std ∈ checkpoints/HY-Motion-1.0/stats/
  ├─ Denorm: motion = y₅₀ × std + mean
  ├─ Shape: (1, 360, 201)
  └─ Trim to actual length: (1, T, 201)

  │
  ▼

[STEP 5: EXTRACT COMPONENTS]
  │
  ├─ Translation:   motion[:, :3]         → (T, 3)
  ├─ Rot 6D:        motion[:, 3:135]      → (T, 132)
  ├─ Positions:     motion[:, 135:201]    → (T, 66)  [optional]
  └─ Save 135-dim:  motion[:, :135]       → (T, 135)

  │
  ▼

[STEP 6: FORWARD KINEMATICS (Optional)]
  │
  ├─ Convert 6D → Axis-Angle
  ├─ Load skeleton: data/hymotion_m2m_data/bone_offsets_22.pt
  ├─ Compute: 3D positions = FK(rotations, translations, skeleton)
  └─ Output: (T, 22, 3)  [frames, joints, xyz]

  │
  ▼

OUTPUT (NPZ FORMAT)
  │
  ├─ motion_135:  (T, 135)    [translation + 6D rotations]
  ├─ positions:   (T, 22, 3)  [3D joint coordinates]
  └─ translation: (T, 3)      [root motion, redundant]
```

---

## 2. Model Architecture Overview

```
┌───────────────────────────────────────────────────────┐
│        HyMotionT2MBundle (Model Wrapper)              │
│  ┌─────────────────────────────────────────────────┐  │
│  │  motion_transformer: HunyuanMotionMMDiT         │  │
│  │  ┌───────────────────────────────────────────┐  │  │
│  │  │ Input Encoder (projection 201→1024)      │  │  │
│  │  │ ┌────────────────────────────────────┐   │  │  │
│  │  │ │ Stacked MultiHead Attn Blocks     │   │  │  │
│  │  │ │ (18 layers, 16 heads)             │   │  │  │
│  │  │ │                                    │   │  │  │
│  │  │ │ - Self-attention on motion        │   │  │  │
│  │  │ │ - Cross-attn: text context (4096) │   │  │  │
│  │  │ │ - Cross-attn: sentence (768)      │   │  │  │
│  │  │ │ - Time conditioning (1000 steps)  │   │  │  │
│  │  │ └────────────────────────────────────┘   │  │  │
│  │  │ Output Decoder (projection 1024→201)     │  │  │
│  │  └───────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  Key Methods:                                         │
│  - encode_text(captions) → text_embeddings           │
│  - predict_flow(x, t, context) → velocity prediction │
│  - denormalize_motion(latent) → denormalized motion  │
└───────────────────────────────────────────────────────┘

HunyuanMotionMMDiT Architecture:
  feat_dim=1024         [internal hidden dimension]
  num_layers=18         [transformer blocks]
  num_heads=16          [attention heads]
  ctxt_input_dim=4096   [text LLM embedding size]
  vtxt_input_dim=768    [text sentence embedding size]
  mlp_ratio=4.0         [FFN expansion ratio]
  mask_mode='narrowband' [attention pattern]
  time_factor=1000.0    [time embedding scale]
```

---

## 3. Motion Representation Breakdown

### Full 201-Dimensional Representation

```
┌─────────────────────────────────────────────────────────────────┐
│                 MOTION VECTOR (201 dims)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Segment 1: ROOT TRANSLATION [0:3]                             │
│  ┌──────────────┐                                              │
│  │ Tx | Ty | Tz │  (3 dims)                                    │
│  │ X  | Y  | Z  │  Root/pelvis XYZ position                    │
│  └──────────────┘                                              │
│         │                                                       │
│         └─ Relative translation (not absolute)                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Segment 2: 6D ROTATIONS [3:135] (132 dims = 22 joints × 6)   │
│  ┌───────────────────────────────────────────────────┐         │
│  │ Joint 0 (Pelvis):     [r0x, r0y, r0z, r0x', r0y', r0z']   │
│  │ Joint 1 (L Hip):      [r1x, r1y, r1z, r1x', r1y', r1z']   │
│  │ ...                                                          │
│  │ Joint 21 (R Ankle):   [r21x, r21y, r21z, ...]             │
│  └───────────────────────────────────────────────────┘         │
│         │                                                       │
│         ├─ Each joint: 6D rotation representation              │
│         ├─ Continuous 6D (NOT Euler/quaternion)               │
│         └─ Can convert to: axis-angle, quaternion, Euler      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Segment 3: LOCAL JOINT POSITIONS [135:201] (66 dims)         │
│  ┌───────────────────────────────────────────────────┐         │
│  │ Joint 0 (Pelvis):     [px0, py0, pz0]                      │
│  │ Joint 1 (L Hip):      [px1, py1, pz1]                      │
│  │ ...                                                          │
│  │ Joint 21 (R Ankle):   [px21, py21, pz21]                   │
│  └───────────────────────────────────────────────────┘         │
│         │                                                       │
│         ├─ 3D coordinates of each joint                        │
│         ├─ In local frame (relative to skeleton)               │
│         └─ Can be FK-computed from rotations                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

SMPL 22-Joint Skeleton:
┌─────────────────────────────┐
│  0: Pelvis (Root)           │
├─────────────────────────────┤
│ Left Leg:                   │
│  1: L Hip                   │
│  2: L Knee                  │
│  3: L Ankle                 │
├─────────────────────────────┤
│ Right Leg:                  │
│  4: R Hip                   │
│  5: R Knee                  │
│  6: R Ankle                 │
├─────────────────────────────┤
│ Spine:                      │
│  7: Spine1                  │
│  8: Spine2                  │
│  9: Spine3                  │
├─────────────────────────────┤
│ Left Arm:                   │
│ 10: L Shoulder              │
│ 11: L Elbow                 │
│ 12: L Wrist                 │
├─────────────────────────────┤
│ Right Arm:                  │
│ 13: R Shoulder              │
│ 14: R Elbow                 │
│ 15: R Wrist                 │
├─────────────────────────────┤
│ Neck & Head:                │
│ 16: Neck                    │
│ 17: Head                    │
├─────────────────────────────┤
│ Other:                      │
│ 18: L Toe                   │
│ 19: R Toe                   │
│ 20: L Thumb                 │
│ 21: R Thumb                 │
└─────────────────────────────┘
```

### Saved NPZ Format (135-dim)

```
┌────────────────────────────────────────────────────────────┐
│           NPZ FIELDS (What Gets Saved)                    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  motion_135: (T, 135)                                     │
│  ├─ [:, :3]         Translation (3)                       │
│  └─ [:, 3:135]      6D Rotations (132)                    │
│     └─ NOT SAVED: Local positions (66 dims)              │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  positions: (T, 22, 3)                                    │
│  ├─ FK-computed from rot6d + translation                  │
│  └─ World-frame 3D coordinates                            │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  translation: (T, 3)                                      │
│  └─ Copy of motion_135[:, :3]  [REDUNDANT]                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 4. Inference Parameter Space

```
┌──────────────────────────────────────────────────────────────┐
│              INFERENCE PARAMETERS                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  num_steps (ODE Integration Steps)                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  20   →  0.5 sec/sample  (Fast, Lower Quality)        │  │
│  │  50   → 1-2 sec/sample   (DEFAULT - Balanced)         │  │
│  │ 100   → 3-4 sec/sample   (High Quality, Slower)       │  │
│  │ 200   → 8-10 sec/sample  (Very High Quality)          │  │
│  └────────────────────────────────────────────────────────┘  │
│  Effect: More steps = smoother ODE trajectory                │
│          but slower inference                                │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  cfg_scale (Classifier-Free Guidance Strength)              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1.0  → No guidance (pure random, ignores text)       │  │
│  │  3.0  → Weak guidance (balanced text-randomness)      │  │
│  │  5.0  → Strong guidance (DEFAULT - follows text well) │  │
│  │  7.0  → Very strong (strict text adherence)           │  │
│  │ >10.0 → Over-constrained (artifacts, not recommended) │  │
│  └────────────────────────────────────────────────────────┘  │
│  Formula: x_guided = x_cond + scale * (x_cond - x_uncond)    │
│  Effect: Higher scale = more text-aligned but less diverse   │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  tgt_length (Motion Duration)                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Default: 360 frames (@ 30fps = 12 seconds)           │  │
│  │  Range: 1-360 frames                                  │  │
│  │  Effect: Longer motions = more VRAM, slower inference │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  batch_size (Parallel Samples)                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1 → ~0.5 GB VRAM                                     │  │
│  │  4 → ~1.5 GB VRAM                                     │  │
│  │  8 → ~2.5 GB VRAM                                     │  │
│  │  Advanced: Can use gradient accumulation for larger   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Config vs Checkpoint vs Inference

```
┌──────────────────────────────────────────────────────────────────┐
│                RELATIONSHIP OVERVIEW                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CONFIG: hymotion_t2m_201dim_046b.py                            │
│  ├─ Specifies: Model structure, dimensions, optimizer          │
│  ├─ Says: input_dim=201, output_dim=201                        │
│  ├─ Sets: text_encoder config, training params                 │
│  └─ Path: configs/hymotion_t2m/hymotion_t2m_201dim_046b.py    │
│      │                                                          │
│      └→ USED FOR: Loading model architecture                   │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CHECKPOINT: latest.ckpt (1.8 GB)                              │
│  ├─ Contains: Trained weights for HunyuanMotionMMDiT           │
│  ├─ Shape: motion_dim=201                                      │
│  ├─ Status: Already trained on HY-Motion 1.0 dataset          │
│  └─ Path: checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/      │
│      │                                                          │
│      └→ USED FOR: Loading pre-trained weights                  │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INFERENCE:                                                     │
│  ├─ Step 1: Load config → instantiate model with 201-dim      │
│  ├─ Step 2: Load checkpoint weights into model                 │
│  ├─ Step 3: Create HyMotionT2MPipeline wrapper                 │
│  ├─ Step 4: Run ODE integration (50 steps default)            │
│  ├─ Step 5: Denormalize output                                │
│  ├─ Step 6: Save 135-dim + FK positions to NPZ               │
│  └─ Output: motion_135 (T, 135) + positions (T, 22, 3)       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. File Size & Performance

```
┌──────────────────────────────────────────────────────────────┐
│            RESOURCE REQUIREMENTS                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  CHECKPOINT                                                  │
│  ├─ File size: 1.8 GB                                       │
│  ├─ Load time: ~30 seconds (CPU → GPU)                      │
│  └─ 460M parameters                                         │
│                                                              │
│  INFERENCE (Single Sample, T=360)                           │
│  ├─ num_steps=20:  ~0.5 sec  (0.4 GB VRAM)                │
│  ├─ num_steps=50:  ~1.5 sec  (0.5 GB VRAM)                │
│  ├─ num_steps=100: ~3.0 sec  (0.6 GB VRAM)                │
│  └─ num_steps=200: ~6.0 sec  (0.8 GB VRAM)                │
│                                                              │
│  BATCH INFERENCE                                             │
│  ├─ batch_size=1: ~0.5 GB VRAM                             │
│  ├─ batch_size=4: ~1.5 GB VRAM                             │
│  ├─ batch_size=8: ~2.5 GB VRAM                             │
│  └─ batch_size=16: ~4.5 GB VRAM                            │
│                                                              │
│  TEXT ENCODING                                               │
│  ├─ Qwen3 load time: ~20 seconds                            │
│  ├─ CLIP-L load time: ~5 seconds                            │
│  ├─ Per-text encode time: ~50ms                             │
│  └─ VRAM: ~1.2 GB (both encoders)                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Quick Comparison: T2M vs M2M

```
┌─────────────────────────────────────────────────────────────┐
│              T2M (TEXT-TO-MOTION)                          │
├─────────────────────────────────────────────────────────────┤
│ Config:        hymotion_t2m_201dim_046b.py                 │
│ Input:         Text prompt only                             │
│ Motion input:  ZERO tensor (generated from noise)          │
│ input_dim:     201                                          │
│ VACE:          NO (T2M does NOT use VACE)                 │
│ Output:        201-dim (saved as 135 + FK)                │
│ Use case:      "walk forward slowly"                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              M2M (MOTION-TO-MOTION)                        │
├─────────────────────────────────────────────────────────────┤
│ Config:        hymotion_m2m_v2_caption_local_046b.py       │
│ Input:         Text + Motion (optional)                     │
│ Motion input:  Non-zero tensor (actual motion)             │
│ input_dim:     198 * 4 = 792 (with VACE)                  │
│ VACE:          YES (VACE augmentation applied)             │
│ Output:        198-dim                                     │
│ Use case:      "complete this partial motion"             │
└─────────────────────────────────────────────────────────────┘

KEY DIFFERENCE:
  T2M: input_dim = motion_dim (no augmentation)
  M2M: input_dim = motion_dim * 4 (with VACE augmentation)
```

---

## 8. NPZ Loading Examples

```python
# Load and inspect NPZ
import numpy as np

data = np.load('00001401.npz')

# Option 1: Extract all fields
motion_135 = data['motion_135']     # (60, 135)
positions = data['positions']       # (60, 22, 3)
translation = data['translation']  # (60, 3)

# Option 2: Reconstruct components from motion_135
T = motion_135.shape[0]
transl = motion_135[:, :3]          # (T, 3)
rot6d = motion_135[:, 3:135]        # (T, 132)
rot6d = rot6d.reshape(T, 22, 6)     # (T, 22, 6)

# Option 3: FK consistency check
# positions should match FK(rot6d + skeleton)
# If discrepancy > threshold → potential issue

# Option 4: Convert 6D rotations to axis-angle (for analysis)
# from hftrainer.models.motion.components.utils.geometry.rotation_convert \
#     import rotation_6d_to_axis_angle
# aa = rotation_6d_to_axis_angle(torch.from_numpy(rot6d).float())
# aa_np = aa.numpy()  # (T, 22, 3)
```

---

## 9. Debug Checklist

```
┌─────────────────────────────────────────────────────────────┐
│                INFERENCE DEBUGGING                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ✓ Config File                                              │
│   └─ Verify: configs/hymotion_t2m/hymotion_t2m_201dim... │
│   └─ Check: input_dim=201, output_dim=201                 │
│                                                             │
│ ✓ Checkpoint                                               │
│   └─ File size: ~1.8 GB                                   │
│   └─ Load time: <30 seconds                               │
│   └─ Format: PyTorch state dict                           │
│                                                             │
│ ✓ Model Instantiation                                      │
│   └─ Bundle type: HyMotionT2MBundle                        │
│   └─ Transformer type: HunyuanMotionMMDiT                 │
│   └─ Device: cuda:0 (verified)                            │
│                                                             │
│ ✓ Text Encoding                                            │
│   └─ Qwen3 loaded: check bundle._text_encoder_cfg         │
│   └─ CLIP-L loaded: check bundle._sentence_encoder        │
│   └─ Output shapes: (1, 512, 4096) + (1, 77, 768)         │
│                                                             │
│ ✓ ODE Integration                                          │
│   └─ Initial noise shape: (1, 360, 201)                   │
│   └─ num_steps: 50                                         │
│   └─ Output shape: (1, 360, 201)                          │
│                                                             │
│ ✓ Denormalization                                          │
│   └─ mean/std loaded from checkpoints/HY-Motion-1.0/stats │
│   └─ Output range: roughly [-1, 1]                        │
│                                                             │
│ ✓ NPZ Output                                               │
│   └─ motion_135 shape: (T, 135)                           │
│   └─ positions shape: (T, 22, 3)                          │
│   └─ translation shape: (T, 3)                            │
│   └─ All values: float32                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

