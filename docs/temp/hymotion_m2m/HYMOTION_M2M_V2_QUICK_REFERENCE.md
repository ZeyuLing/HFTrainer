# HyMotion M2M v2 Architecture - Quick Reference

## Key Code Locations

### 1. Block Count: 18 (6 + 12)
- **File:** `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py`
- **Lines 724-728:** Block configuration
  - `self.mm_double_blocks_layers = int(num_layers // 3)` → 6 blocks
  - `self.mm_single_blocks_layers = int(num_layers - num_layers // 3)` → 12 blocks
- **Lines 733-747:** Double-stream block creation (6 iterations)
- **Lines 753-767:** Single-stream block creation (12 iterations)

### 2. Input/Output Dimensions
- **File:** `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py`
- **Line 704:** `self.input_encoder = nn.Linear(in_features=input_dim, out_features=feat_dim)` 
  - input_dim=594 (VACE: x_t[198] + reactive[198] + mask[198])
  - feat_dim=1024
- **Lines 770-774 + encoders.py:76:** `self.final_layer = FinalLayer(feat_dim=1024, out_dim=198)`

### 3. Text Encoders
- **File:** `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py`

**Context (ctxt):** Qwen3-8B output → 1024D hidden
- **Line 706:** `self.ctxt_encoder = nn.Linear(in_features=4096, out_features=1024)`

**Vector text (vtxt):** CLIP-L → 1024D hidden  
- **Line 708:** `self.vtxt_encoder = MLPEncoder(in_dim=768, feat_dim=1024, num_layers=2, act_type="silu")`
  - See implementation: `encoders.py:54-67`
  - Layer 1: Linear(768→1024) + SiLU
  - Layer 2: Linear(1024→1024)

### 4. Transfer Learning
- **File:** `hftrainer/models/motion/hymotion_m2m/checkpoint_loading.py`

**Reusable modules (transferred from T2M):** Lines 49-56
- ctxt_encoder
- vtxt_encoder  
- timestep_encoder
- text_refiner
- double_blocks (all 6)
- single_blocks (all 12)
- **Total: ~443M parameters**

**Non-reusable modules (reinitialized):** Lines 59-62
- input_encoder (shape mismatch: 135→594)
- final_layer (shape mismatch: 135→198)
- **Reinitialization method:** `xavier_uniform_` for 2D+ params, `zeros_` for bias

### 5. Attention Configuration
- **File:** `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py:47`
- **Mode:** `mask_mode='narrowband'` → Local attention window
- **File:** `hymotion_mmdit.py:689-890`
  - Line 690: `self.narrowband_length = narrowband_length * 30.0` (convert seconds to frames @ 30fps)
  - Lines 874-882: Narrowband mask creation (forms local attention band)

### 6. RoPE (Rotary Position Embeddings)
- **File:** `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py:48`
- **Config:** `apply_rope_to_single_branch=False` → Apply RoPE to both motion AND text
- **Implementation:** `hymotion_mmdit.py:253-256, 519-531`
  - Double-stream: RoPE only on motion if `apply_rope_to_single_branch=True`
  - Single-stream: Can split and apply differently

---

## Architecture Summary

```
Input: 594D (VACE)
  ↓
Input Encoder: 594→1024
  ↓
Timestep + vtxt (768) encoding → Adapter
  ↓
Ctxt (4096) encoding via Qwen3 → 1024D context
  ↓
Double-Stream Blocks (6×):
  - Motion stream: 1024D
  - Text stream: 1024D
  - Joint attention with T→M blocking
  ↓
Single-Stream Blocks (12×):
  - Concatenated: 1024D
  - T→M blocking maintained
  ↓
Final Layer: 1024→198
  ↓
Output: 198D (motion)
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` | Base config (0.46B, 18 blocks, 594→1024→198) |
| `configs/hymotion_m2m_v2/hymotion_m2m_v2_046b_t2m_pretrained.py` | Load T2M pretrained + freeze strategy |

---

## Parameter Summary

| Component | Params | Details |
|-----------|--------|---------|
| input_encoder | 594K | 594×1024 linear |
| ctxt_encoder | 4M | 4096×1024 linear |
| vtxt_encoder | 2M | 768→1024→1024 (2-layer MLP) |
| timestep_encoder | 2M | Sinusodial embedding + MLP |
| double_blocks (6×) | 150M | 25M per block |
| single_blocks (12×) | 280M | ~23M per block |
| final_layer | 1M | 1024×198 linear + modulation |
| **TOTAL** | **~0.46B** | - |

---

## Key Verification Checkpoints

1. ✅ Block count formula in code: `num_layers // 3 = 18 // 3 = 6` (double), `18 - 6 = 12` (single)
2. ✅ Input dimension in config: `_motion_dim * 3 = 198 * 3 = 594`
3. ✅ Hidden dimension: `feat_dim=1024` (consistent across all encoders)
4. ✅ Output dimension: `output_dim=_motion_dim = 198`
5. ✅ Text encoders: `ctxt_encoder` is simple Linear, `vtxt_encoder` is 2-layer MLP
6. ✅ Transfer strategy: Only encoders + blocks transferred; input/final layers reinitialized
7. ✅ Attention: narrowband mode with 60-frame window @ 30fps
8. ✅ RoPE: Applied to entire (motion+text) sequence due to `apply_rope_to_single_branch=False`

