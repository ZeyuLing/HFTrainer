# HyMotion M2M CFG: Quick Fix Guide

## The Problem in 30 Seconds

**Caption guidance isn't working because:**
- Model receives `ctxt_input` (40K-80K floats of semantic info) in BOTH CFG branches
- Only `vtxt_input` (768 floats) differs between branches
- Guidance signal ≈ (40K-80K identical + 768 different) ≈ nearly zero

## The Fix in 1 Line

**In your training config, add:**
```python
model = dict(
    type='HyMotionM2MBundle',
    enable_ctxt_null_feat=True,  # <-- THIS LINE
    cond_mask_prob=0.1,          # Also add this for CFG training
    # ... rest of config
)
```

## Verification After Training

```python
import torch
bundle = torch.load('checkpoint/model.pt', map_location='cpu')['model']

# Check null embeddings are learned (not near-zero)
print(f"null_vtxt_feat norm: {bundle.null_vtxt_feat.norm().item():.4f}")
print(f"null_ctxt_input norm: {bundle.null_ctxt_input.norm().item():.4f}")

# If either is < 0.01, use null_embedding_source in load_from
```

## Inference-time Changes

```python
from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

# Increase guidance scale (default is 1.0, effectively off)
pipeline = HyMotionM2MPipeline(
    bundle,
    num_steps=50,
    text_guidance_scale=7.5  # Increase from 1.0
)
```

## Code Locations

| What | File | Line |
|-----|------|------|
| Enable flag default | `bundle.py` | 166 |
| CFG masking logic | `hymotion_m2m_pipeline.py` | 231 |
| Null param init | `bundle.py` | 212 |

## Testing Caption Responsiveness

```python
# Simple test: does model respond to different captions?
captions = ["standing", "walking", "running"]
for cap in captions:
    motion = pipeline({"caption": cap, "tgt_length": [120]})
    # Should produce visibly different motions
```

## If Still Not Working

1. ✓ Check `cond_mask_prob > 0` during training (e.g., 0.1)
2. ✓ Check `enable_ctxt_null_feat=True` is actually in config
3. ✓ Verify null embeddings norm > 0.1 after training
4. ✓ If null embeddings are zero, use `null_embedding_source`:
   ```yaml
   load_from:
     path: m2m_checkpoint.pt
     null_embedding_source: checkpoints/HY-Motion-1.0-Lite/latest.ckpt
   ```
5. ✓ Increase `text_guidance_scale` to 7.5-10.0 at inference
