# HyMotion T2M 1.0 - Direct Answers to Your Questions

## Question 1: Read `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` fully ✅

### Key Findings:

**Config Location:**
```
configs/hymotion_t2m/hymotion_t2m_201dim_046b.py
```

**Model Configuration:**
- **Type**: HyMotionT2MBundle + HunyuanMotionMMDiT
- **Motion Dimension**: _motion_dim = 201
- **Architecture**:
  - feat_dim=1024
  - num_layers=18
  - num_heads=16
  - ctxt_input_dim=4096 (Qwen3 LLM)
  - vtxt_input_dim=768 (CLIP-L)
  - mask_mode='narrowband'
  - time_factor=1000.0

**Important Config Notes:**
1. **NO VACE**: Unlike M2M models, input_dim = motion_dim (NOT multiplied by 4)
2. **Text Encoders**: Auto-injected at runtime with default config:
   ```python
   {
       'llm_type': 'qwen3_embedding',
       'max_length_llm': 512,
       'sentence_emb_type': 'clipl',
       'max_length_sentence_emb': 77,
   }
   ```
3. **Data Pipeline Limitation**: Current LoadSmplx55 outputs 135 dims, not full 201
4. **Checkpoint Loading**: Direct loading from `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`

**Inference Settings:**
- Default num_steps: 50 (ODE solver integration steps)
- Default cfg_scale: 5.0 (classifier-free guidance)
- Motion length: 360 frames (12 seconds @ 30fps)
- CFG dropout: 0.1 (10% of samples drop text for unconditional branch)

---

## Question 2: Search for existing T2M eval scripts ✅

### Found Scripts:

**Primary Eval Script:**
```
scripts/eval/eval_m2m_v2_t2m.py (751 lines)
```

**Key Features:**
- **Two modes**:
  - Mode A: Per-model parallelism (legacy, one GPU per model)
  - Mode B: CFG-sweep parallelism (recommended, chunk-based)
- **Multi-GPU support**: Can use 1-8+ GPUs
- **CFG ablation**: Run multiple cfg scales sequentially under one model load
- **Batch inference**: 240 prompts from `data/eval/t2m/251125_yiran_subset.json`

**Usage Examples:**

```bash
# Legacy: 4 models in parallel at cfg=5
python scripts/eval/eval_m2m_v2_t2m.py

# CFG ablation: 5 cfgs, 8 GPUs
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local \
    --cfg-sweep 1.0 1.5 2.5 4.0 7.5 \
    --prompt-chunks 8 \
    --gpus 0 1 2 3 4 5 6 7
```

---

## Question 3: Check for T2M 1.0 eval outputs in `work_dirs/` ✅

### Found Directories:

```
work_dirs/
├── m2m_v2_t2m_eval/                           (Main results)
│   ├── caption_global/
│   │   ├── npz/              (240 motion files)
│   │   └── result.json
│   ├── caption_local/
│   │   ├── npz/              (240 motion files)
│   │   └── result.json
│   ├── uncond_global/
│   └── uncond_local/
│
├── m2m_v2_t2m_eval_cfg_ablation_2860_unpatched/  (CFG ablation)
│   ├── cfg1/npz/
│   ├── cfg1.5/npz/
│   ├── cfg2.5/npz/
│   ├── cfg4/npz/
│   └── cfg7.5/npz/
│
├── m2m_v2_t2m_eval_cfg_ablation_v2/
├── m2m_v2_t2m_eval_compare/
└── kimodo_t2m_eval/
```

**Sample Data:** `work_dirs/m2m_v2_t2m_eval/caption_local/npz/00001401.npz`

---

## Question 4: Read eval script to understand NPZ format ✅

### Inference Flow (eval_m2m_v2_t2m.py, Lines 296-330):

```python
# Prepare input
T = min(prompt['frames'], 360)
D = 198  # ← Motion dimension used in input
src_motion = torch.zeros(1, 360, D, device=device)
src_mask = torch.zeros(1, 360, D, device=device)
src_mask[:, :T, :] = 1.0

# Encode text
text_out = bundle.encode_text([text])
batch = {
    'src_motion': src_motion,
    'src_mask': src_mask,
    'src_length': [T],
    'tgt_length': [T],
    'text_vec_raw': text_out['text_vec_raw'],
    'text_ctxt_raw': text_out['text_ctxt_raw'],
}

# Run pipeline (ODE integration)
output = pipeline(batch)
```

### Output Processing (Lines 327-330):

```python
sampled = output['latent']  # Raw tensor from ODE solver
output_denorm = bundle.denormalize_motion(sampled)[0].cpu()
output_denorm = output_denorm[:T]

# Extract 135-dim motion
output_135 = output_denorm[:, :135].numpy()  # (T, 135)
```

### NPZ Saving (Lines 371-377):

```python
np.savez_compressed(
    npz_path,
    motion_135=output_135,          # (T, 135)
    positions=pos_np,               # (T, 22, 3) - FK computed
    translation=transl,             # (T, 3)
)
```

---

## Question 5: Check 201-dim representation & NPZ fields ✅ **[CRITICAL DISCOVERY]**

### **NPZ Output Format - VERIFIED**

**Inspected Sample File:**
```
work_dirs/m2m_v2_t2m_eval/caption_local/npz/00001401.npz

Keys found: ['motion_135', 'positions', 'translation']
```

**Field Details:**

```
1. motion_135: (T, 135) float32
   ├─ [:, :3]      = translation (3 dims)
   ├─ [:, 3:135]   = 6D rotations (132 dims = 22 joints × 6)
   └─ Range: [-0.8892, 1.1765]

2. positions: (T, 22, 3) float32
   ├─ Computed via forward kinematics from rot6d
   ├─ 3D joint coordinates in world frame
   └─ Range: [-0.4378, 1.3356]

3. translation: (T, 3) float32
   ├─ Redundant copy of motion_135[:, :3]
   └─ Range: [-0.0013, 1.1765]
```

### **Critical Finding: NO motion_201 field**

The NPZ **DOES NOT** contain a single `motion_201` field.

**What actually happens:**

```
Full 201-dim representation (theoretical):
├─ [0:3]       Translation (3)
├─ [3:135]     6D Rotations (132)
└─ [135:201]   Local Joint Positions (66)  ← NOT SAVED

Saved to NPZ:
├─ motion_135  → [0:135] (Translation + 6D rot)
├─ positions   → FK-computed 3D coordinates
└─ translation → Redundant copy

Not explicitly saved:
└─ Dims [135:201] local positions (can be reconstructed from FK)
```

### **Why Only 135 Dims Saved?**

The eval script (lines 417-434) checks for `pos_channel` (dims 135-198) but only uses it for consistency metrics:

```python
if output_denorm.shape[-1] >= 198:
    pos_channel = output_denorm[:, 135:198].numpy()  # (T, 63)
    # Used for metrics only, NOT saved to NPZ
```

This suggests:
- Model can output up to 198 dims (T + 6D×22)
- But eval script is conservative and only saves 135 + FK-positions
- The full 201-dim representation likely exists internally but is not required for evaluation

### **Data Pipeline Limitation**

From config notes (lines 12-17):
```python
# TODO: The current data pipeline with LoadSmplx55 (smpl_type='smpl_22')
# outputs 135 dims. For 201 dims, LoadSmplx55 needs to be extended to also
# output local joint positions (22 joints × 3 dims = 66 dims), giving
# 135 + 66 = 201 dims total. For now, this config uses 135 dims.
```

---

## Summary Answer Table

| Question | Answer |
|----------|--------|
| **Config file location?** | `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` |
| **Checkpoint location?** | `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` |
| **T2M eval scripts?** | `scripts/eval/eval_m2m_v2_t2m.py` (751 lines) |
| **Existing eval results?** | Yes, in `work_dirs/m2m_v2_t2m_eval/` and variants |
| **NPZ fields present?** | `motion_135`, `positions`, `translation` |
| **motion_135 field?** | YES - (T, 135) = [3 trans + 132 rot6d] |
| **motion_201 field?** | NO - only 135-dim saved + FK-computed positions |
| **positions field?** | YES - (T, 22, 3) FK-computed 3D coordinates |
| **motion_135 to smplx mapping?** | Lines 3-135: translation (3) + 6D rotations (132) |
| **Why not 201 dims?** | Data pipeline limitation; full 201 would require local positions |
| **ODE solver steps?** | 50 (default, configurable) |
| **Text guidance scale?** | 5.0 (default, range 1.0-10.0) |
| **Input motion dim in eval?** | 198 (used in batch preparation, line 299) |
| **Config input_dim?** | 201 (theoretical, model input) |
| **Model parameters?** | 460M (0.46B HunyuanMotionMMDiT) |

---

## Implementation Checklist

To run T2M 1.0 inference:

- [x] Config: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` ✓
- [x] Checkpoint: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` (1.8 GB) ✓
- [x] Pipeline: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py` ✓
- [x] Bundle: `hftrainer/models/motion/hymotion_t2m/bundle.py` ✓
- [x] Eval script: `scripts/eval/eval_m2m_v2_t2m.py` ✓
- [x] Inference script: `scripts/misc/robot_sim/text_to_g1.py` ✓

**Recommended setup for batch inference:**
```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local \
    --gpus 0 1 2 3 \
    --cfg-sweep 1.0 3.0 5.0 7.0 \
    --prompt-chunks 4 \
    --num-steps 50 \
    --output-dir work_dirs/t2m_eval_new/
```

---

## References

- **Comprehensive Guide**: `HYMOTION_T2M_COMPREHENSIVE_GUIDE.md`
- **Visual Reference**: `HYMOTION_T2M_VISUAL_REFERENCE.md`
- **Quick Start**: `HYMOTION_T2M_QUICK_START.md`
- **Config Guide**: `HYMOTION_T2M_CONFIG_GUIDE.md`
- **Summary**: `HYMOTION_T2M_SUMMARY.txt`

