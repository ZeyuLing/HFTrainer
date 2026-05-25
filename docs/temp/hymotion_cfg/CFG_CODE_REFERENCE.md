# CFG Code Reference - Exact Implementation Details

## Overview

This document provides line-by-line references to where CFG is configured, passed, and applied.

---

## 1. Eval Script Configuration

### File: `scripts/eval/eval_m2m_v2_all_tasks.py`

#### 1.1 CLI Argument Definition (lines 3797-3798)
```python
parser.add_argument('--text-guidance-scale', type=float, default=5.0,
                    help='CFG scale for text-conditioned models (5.0 standard for flow matching)')
```
- **Default**: 5.0
- **Type**: float
- **Usage**: `args.text_guidance_scale`

#### 1.2 Main Evaluation Loop - Conditional Assignment (lines 4046-4051)
```python
metrics, output_135 = evaluate_sample(
    bundle, pipeline, sample, task, setting_name,
    model_info, bone_offsets, args.device,
    replacement_guidance=args.replacement_guidance,
    text_guidance_scale=(
        args.text_guidance_scale
        if model_info.get('has_caption') else 1.0),  # ← CONDITIONAL LOGIC
    num_steps=args.num_steps,
    sample_seed=seed,
)
```

**Logic**:
```
IF model_info['has_caption'] == True:
    text_guidance_scale = args.text_guidance_scale  (5.0)
ELSE:
    text_guidance_scale = 1.0  (CFG disabled)
```

#### 1.3 evaluate_sample() Function Signature (lines 1751-1759)
```python
def evaluate_sample(
    bundle,
    pipeline,
    sample,
    task,
    setting_name: str,
    model_info: Dict,
    bone_offsets: np.ndarray,
    device: str,
    replacement_guidance: str = 'skip_last',
    text_guidance_scale: float = 1.0,  # ← Receives from main loop
    num_steps: int = 50,
    sample_seed: Optional[int] = None,
) -> Dict:
```

#### 1.4 Pipeline Assignment (line 2905)
```python
# Set pipeline parameters
pipeline.replacement_guidance = replacement_guidance
pipeline.text_guidance_scale = text_guidance_scale  # ← SET HERE
pipeline.num_steps = num_steps
pipeline.sdedit_tau = float(setting_kwargs.get('_sdedit_tau', 0.0))
```

**Direct assignment**: `pipeline.text_guidance_scale = text_guidance_scale`

#### 1.5 E13 Multi-Prompt Path - Function Signature (lines 1438-1447)
```python
def _evaluate_e13_multiprompt_chain(
    bundle,
    pipeline,
    sample,
    task,
    setting_name: str,
    model_info: Dict,
    bone_offsets: np.ndarray,
    device: str,
    replacement_guidance: str = 'skip_last',
    text_guidance_scale: float = 1.0,  # ← Receives parameter
    num_steps: int = 50,
    num_prompts: int = 3,
    overlap_frames: int = 5,
) -> Tuple[Dict, Optional[np.ndarray]]:
```

#### 1.6 E13 Pipeline Assignment (line 1632)
```python
pipeline.replacement_guidance = replacement_guidance
pipeline.text_guidance_scale = text_guidance_scale  # ← SET HERE
pipeline.num_steps = num_steps
pipeline.sdedit_tau = 0.0
```

#### 1.7 E13 Call from evaluate_sample() (lines 1810-1818)
```python
return _evaluate_e13_multiprompt_chain(
    bundle, pipeline, sample, task, setting_name, model_info,
    bone_offsets, device,
    replacement_guidance=replacement_guidance,
    text_guidance_scale=text_guidance_scale,  # ← PASSED
    num_steps=num_steps,
    num_prompts=num_prompts,
    overlap_frames=overlap_frames,
)
```

---

## 2. Pipeline Implementation

### File: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py`

#### 2.1 Class Initialization (lines 81-90)
```python
def __init__(
    self,
    bundle,
    num_steps: int = 50,
    text_guidance_scale: float = 1.0,  # ← DEFAULT 1.0 (overridden by eval script)
    replacement_guidance: str = 'none',
    position_constraint_interval: int = 5,
    max_text_len: int = 128,
    sdedit_tau: float = 0.0,
):
```

#### 2.2 Instance Variable Assignment (line 98)
```python
self.text_guidance_scale = text_guidance_scale
```

#### 2.3 CFG Activation Check (line 220)
```python
do_cfg = self.text_guidance_scale > 1.0 and not self.bundle.uncondition_mode
```

**Condition**:
- `self.text_guidance_scale > 1.0`: If guidance scale > 1, enable CFG
- `not self.bundle.uncondition_mode`: Don't apply CFG if model is in uncondition mode

#### 2.4 Null Context Construction for CFG (lines 228-236)
```python
if do_cfg:
    null_vtxt = self.bundle.null_vtxt_feat.to(dtype=model_dtype).expand_as(vtxt_input)
    # Expand null_ctxt to match ctxt_input's sequence length (same as
    # training-time mask_text_cond which does null_ctxt_input.expand_as(ctxt)).
    # This ensures torch.cat along batch dim works correctly.
    null_ctxt = self.bundle.null_ctxt_input.to(dtype=model_dtype).expand(
        ctxt_input.shape[0], ctxt_input.shape[1], -1
    ).contiguous()
    null_ctxt_mask = ctxt_mask_temporal  # SAME attention coverage
```

**What's being nulled**:
- `null_vtxt`: 768-dim sentence embedding set to null vector (learned during training)
- `null_ctxt`: Token-level context embeddings set to null
- `null_ctxt_mask`: Attention mask matches text sequence length

#### 2.5 ODE Function - Batch Construction (lines 243-255)
```python
def fn(t: Tensor, x: Tensor) -> Tensor:
    x_input = torch.cat([x, vace_context], dim=-1)
    if do_cfg:
        x_input = torch.cat([x_input, x_input], dim=0)  # Duplicate for CFG
    x_pred = self.bundle.predict_flow(
        x_input=x_input,
        ctxt_input=(
            ctxt_input if not do_cfg
            else torch.cat([null_ctxt, ctxt_input], dim=0)  # [NULL, TEXT]
        ),
        vtxt_input=(
            vtxt_input if not do_cfg
            else torch.cat([null_vtxt, vtxt_input], dim=0)  # [NULL, TEXT]
        ),
        ...
    )
```

**CFG Batch Structure**:
- First half: Null context (unconditional)
- Second half: Full context (conditioned)

#### 2.6 ODE Prediction Handling (lines 267-277)
```python
if self.bundle.pred_type == 'x1':
    t_eps = 0.05
    if do_cfg:
        x_pred = (x_pred - torch.cat([x, x], dim=0)) / (1.0 - t).clamp_min(t_eps)
    else:
        x_pred = (x_pred - x) / (1.0 - t).clamp_min(t_eps)

if do_cfg:
    pred_basic, pred_text = x_pred.chunk(2, dim=0)
    x_pred = pred_basic + self.text_guidance_scale * (pred_text - pred_basic)  # ← CFG FORMULA
return x_pred
```

**CFG Application**:
- Split predictions: `[pred_basic, pred_text]`
- Apply formula: `pred = pred_basic + scale * (pred_text - pred_basic)`
- Scale: `self.text_guidance_scale` (default 5.0 from eval)

#### 2.7 Text Conditioning Input (lines 190-211)
```python
if 'text_vec_raw' in batch:
    # CONDITIONED PATH
    vtxt_input = batch['text_vec_raw'].to(device=device, dtype=model_dtype)
    ctxt_raw = batch['text_ctxt_raw'].to(device=device, dtype=model_dtype)
    ctxt_input = _pad_ctxt(ctxt_raw, True)
    ctxt_length = batch['text_ctxt_raw_length'].to(device).clamp(max=pad_len)
    ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)
else:
    # UNCONDITIONED PATH
    vtxt_input = self.bundle.null_vtxt_feat.to(dtype=model_dtype).expand(B, 1, -1)
    ctxt_input = self.bundle.null_ctxt_input.to(dtype=model_dtype).expand(B, 1, -1).contiguous()
    ctxt_length = torch.ones(B, dtype=torch.long, device=device)
    ctxt_mask_temporal = _length_to_mask(ctxt_length, 1)
```

**For Captioned Inference**:
- Uses `text_vec_raw`: Sentence embedding from cache
- Uses `text_ctxt_raw`: Token embeddings from cache
- Both padded to 128 tokens to match training

**For Unconditioned Inference**:
- Uses null embeddings (learned during training)
- Single token instead of 128

---

## 3. Caption Embedding Cache

### File: `scripts/eval/eval_m2m_v2_all_tasks.py` (lines 50-107)

#### 3.1 Cache Path Definition (lines 50-51)
```python
CAPTION_EMBED_CACHE_PATH = Path(__file__).resolve().parents[2] / \
    'data' / 'eval' / 'm2m_v2' / 'caption_embeddings' / 'cache.pt'
```

#### 3.2 Cache Loading Function (lines 56-91)
```python
def _load_caption_embed_cache() -> Dict[str, Dict[str, torch.Tensor]]:
    global _CAPTION_EMBED_CACHE
    if _CAPTION_EMBED_CACHE is not None:
        return _CAPTION_EMBED_CACHE
    if not CAPTION_EMBED_CACHE_PATH.is_file():
        print(f'  WARNING: caption embedding cache not found at '
              f'{CAPTION_EMBED_CACHE_PATH}. Caption models will run '
              f'unconditioned. Run scripts/caption/extract_eval_caption_embeddings.py '
              f'first.')
        _CAPTION_EMBED_CACHE = {}
        return _CAPTION_EMBED_CACHE
    data = torch.load(str(CAPTION_EMBED_CACHE_PATH), map_location='cpu',
                      weights_only=False)
    ...
    cache = data.get('cache', {}) if isinstance(data, dict) else {}
    ...
    print(f'  Loaded {len(cache)} caption embeddings from '
          f'{CAPTION_EMBED_CACHE_PATH}')
    _CAPTION_EMBED_CACHE = cache
    return _CAPTION_EMBED_CACHE
```

#### 3.3 Caption Lookup (lines 94-106)
```python
def _lookup_caption_embedding(caption: str
                              ) -> Optional[Dict[str, torch.Tensor]]:
    """Return {'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length'} or None."""
    cache = _load_caption_embed_cache()
    key = caption.strip()
    if not key or key not in cache:
        return None
    entry = cache[key]
    return {
        'text_vec_raw': entry['text_vec_raw'],
        'text_ctxt_raw': entry['text_ctxt_raw'],
        'text_ctxt_raw_length': entry['text_ctxt_raw_length'],
    }
```

**Returns**:
- `text_vec_raw`: 768-dim sentence embedding
- `text_ctxt_raw`: (seq_len, embed_dim) token embeddings
- `text_ctxt_raw_length`: Number of valid tokens

---

## 4. Reference Implementation: tools/infer.py

### File: `tools/infer.py`

#### 4.1 T2M Pipeline (lines 277-287)
```python
def infer_hymotion_t2m(bundle, args):
    """Run HyMotion-T2M text-to-motion inference."""
    import torch
    import numpy as np
    from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline

    pipeline = HyMotionT2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps or 50,
        text_guidance_scale=getattr(args, 'guidance_scale', 5.0) or 5.0,  # ← DEFAULT 5.0
    )
```

#### 4.2 M2M Pipeline (lines 224-233)
```python
def infer_hymotion_m2m(bundle, args):
    """Run HyMotion-M2M motion-to-motion inference."""
    import torch
    import numpy as np
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    pipeline = HyMotionM2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps or 50,
    )
```

**Note**: M2M pipeline creation doesn't set `text_guidance_scale` directly here - it's done in eval script later.

---

## 5. Summary of Values Through Pipeline

```
Step 1: CLI Default
├─ args.text_guidance_scale = 5.0

Step 2: Main Loop Conditional
├─ Caption model: text_guidance_scale = 5.0
└─ Uncond model: text_guidance_scale = 1.0

Step 3: evaluate_sample() Function
├─ Receives: text_guidance_scale (5.0 or 1.0)

Step 4: Pipeline Assignment
├─ pipeline.text_guidance_scale = text_guidance_scale

Step 5: ODE Solver Activation
├─ do_cfg = text_guidance_scale > 1.0
├─ Caption (5.0): do_cfg = TRUE → CFG enabled
└─ Uncond (1.0): do_cfg = FALSE → CFG disabled

Step 6: CFG Application
├─ x_pred = pred_basic + 5.0 * (pred_text - pred_basic)
```

---

## Key Takeaways

1. **Default**: 5.0 (NOT 1.0)
2. **Assignment**: Direct via `pipeline.text_guidance_scale = ...`
3. **Activation**: Conditional check `> 1.0`
4. **Formula**: `pred_basic + scale * (pred_text - pred_basic)`
5. **Cache**: 328MB pre-extracted embeddings
6. **Null Context**: Both vtxt (768-dim) and ctxt (token-level) are nulled

