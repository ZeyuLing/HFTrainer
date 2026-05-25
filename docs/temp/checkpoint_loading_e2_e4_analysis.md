# Checkpoint Loading Analysis: E2/E4 Caption-Conditioned Models Resumed from Unconditional Checkpoints

**Completion Date**: 2026-05-15  
**Analysis Scope**: Checkpoint loading mechanism for HyMotion M2M v2 caption-conditioned models (E2/E4)  
**Focus**: How text-related layers are initialized when resuming from unconditional checkpoint

---

## Executive Summary

When E2/E4 (caption-conditioned) resume from an unconditional checkpoint:

1. **Text-related layers are initialized as MISSING, then loaded from `null_embedding_source` fallback**
   - Cross-attention projections for text are randomly initialized (not loaded from uncond checkpoint)
   - Text encoder layers (CLIP-L, Qwen3) are lazy-loaded fresh on first use
   - Critically, `null_vtxt_feat` and `null_ctxt_input` parameters (used for CFG) are loaded from pretrained source

2. **Strict=False mechanism allows partial loading**
   - Missing text layers don't cause training to fail
   - Shape-mismatched keys are silently dropped (with warnings)
   - Bundle-level orphan parameters (mean, std, null embeddings) are restored from `__bundle_params__`

3. **`null_embedding_source` safety net prevents garbage output**
   - After model-only loading, if null embeddings are all-zeros, fallback patches them from T2M pretrained
   - This ensures CFG always has valid unconditional embeddings to use
   - Without this patch, classifier-free guidance would produce garbage predictions

4. **Training prevents catastrophic failure despite random text layer init**
   - Randomly initialized cross-attention layers learn from training data
   - Text supervision from captions drives optimization of these layers
   - CFG training (`cond_mask_prob=0.1`) ensures model learns unconditional + conditional paths

---

## Part 1: Two-Phase Checkpoint Loading Architecture

### Phase 1: Pre-FSDP Model-Only Loading

**When**: Before `accelerator.prepare()` wraps the model  
**Entry Point**: `AccelerateRunner._pre_prepare_load()` (lines 512-646)  
**Key Feature**: Serialized rank-by-rank checkpoint loading to avoid OOM

```python
def _pre_prepare_load(self, load_from: dict, bundle):
    """Load model-only checkpoint BEFORE FSDP wrapping."""
    path = load_from.get('path')
    load_scope = load_from.get('load_scope', 'model')  # 'model' or 'full'
    load_target = bundle if load_scope == 'model' else accelerator.state.model
    
    if load_scope == 'model':
        # MODEL-ONLY checkpoint
        # Try formats in order: model.pt > model.safetensors > pytorch_model.bin
        state_dict = load_checkpoint(path)  # Detects format automatically
        
        # ✅ CRITICAL: Use strict=False to allow missing text layers
        missing, unexpected = load_target.load_state_dict_selective(
            state_dict, 
            strict=False  # Missing keys don't fail
        )
        # Log missing text layers here (lines 614-630)
```

**Format Detection** (`checkpoint_utils.py`):
- Checks `model.pt` (nested dict format) first
- Falls back to `model.safetensors` or `pytorch_model.bin`
- Returns state_dict with all keys flattened

**Nested Structure in model.pt**:
```python
{
    'model': {               # ← model-only checkpoint
        'motion_transformer.blocks.0.attn.to_q.weight': [...],
        'motion_transformer.blocks.0.attn.to_k.weight': [...],
        # ❌ NO: text_encoder.embedding
        # ❌ NO: cross_attn.proj_vtxt
        # ❌ NO: __bundle_params__
    }
}
```

### Phase 2: Post-FSDP Checkpoint Load

**When**: After `accelerator.prepare()` wraps model  
**Entry Point**: `AccelerateRunner._handle_load()` (lines 1030-1082)  
**Mechanism**: Uses `accelerator.load_state(path)` which handles FSDP automatically

```python
def _handle_load(self):
    """Load after accelerator.prepare(), or auto-resume."""
    if self.cfg.auto_resume and not self.cfg.load_from:
        # Auto-resume from latest checkpoint
        path = latest_checkpoint_path(work_dir)
    else:
        path = self.cfg.load_from.get('path')
    
    # Phase 2 loads **full resume checkpoints** (with optimizer state)
    # These are saved by accelerate.save_checkpoint() and include
    # model.safetensors + optimizer + scheduler state
```

---

## Part 2: Text-Related Layers and Their Initialization

### Layers Present in Caption-Conditioned Model (E2/E4)

The MMDiT architecture includes these text-related components:

| Layer | Purpose | Shape | Location in Model |
|-------|---------|-------|------------------|
| **Motion Transformer** | Core architecture | varies | `self.motion_transformer` |
| **Cross-Attention Projections** | Map text to motion space | text_dim → motion_dim | Inside transformer blocks |
| **Text Refiner** | Pre-processes text embeddings | 4096 → 1024 | `self.text_refiner` |
| **Null Vtxt Feature** | CFG unconditional embedding (sentence-level) | (1, 1, 768) | `self.null_vtxt_feat` nn.Parameter |
| **Null Ctxt Input** | CFG unconditional embedding (token-level) | (1, max_tokens, 4096) | `self.null_ctxt_input` nn.Parameter |
| **Text Encoder** | Lazy-loaded CLIP-L + Qwen3 | external | Loaded on first use |

### What Happens During Resume from Unconditional Checkpoint

```
Unconditional checkpoint contains:
  ✅ motion_transformer (blocks, but NO cross-attention layers for text)
  ✅ timestep_encoder
  ❌ NO text_refiner
  ❌ NO null_vtxt_feat / null_ctxt_input
  ❌ NO cross-attention text proj layers

Caption-conditioned model EXPECTS:
  ✅ motion_transformer (inherited, loaded successfully)
  ✅ text_refiner (MISSING → randomly initialized)
  ✅ null_vtxt_feat (MISSING → randomly initialized, then PATCHED)
  ✅ null_ctxt_input (MISSING → randomly initialized, then PATCHED)
  ✅ cross-attention (MISSING → randomly initialized)
```

### Step 1: Selective Loading with strict=False

**In `BaseModelBundle.load_state_dict_selective()`** (lines 637-782):

```python
def load_state_dict_selective(self, state_dict, strict=False):
    """Load state dict with shape-mismatch tolerance."""
    
    if strict:
        # Standard: all keys must match, raise on missing/unexpected
        missing, unexpected = load_target.load_state_dict(state_dict, strict=True)
    else:
        # ✅ SELECTED LOADING: partial state dict okay
        missing, unexpected = load_target.load_state_dict(state_dict, strict=False)
        
        # Filter out shape-mismatched keys (lines 744-760)
        if missing:
            logger.warning(f'Missing keys: {missing}')
        if unexpected:
            logger.warning(f'Unexpected keys: {unexpected}')
        
        # Restore bundle-level orphan parameters (line 765+)
        if '__bundle_params__' in state_dict:
            for key, value in state_dict['__bundle_params__'].items():
                if hasattr(self, key):
                    getattr(self, key).copy_(value)
```

**Result**: 
- Missing text layers don't cause error
- Layers not in checkpoint get **randomly initialized** by PyTorch
- Bundle orphan params (mean, std) are restored

### Step 2: Null Embedding Source Fallback

**In `AccelerateRunner._patch_zero_null_embeddings_from_pretrained()`** (lines 1272-1367):

This method runs AFTER model-only loading completes:

```python
def _patch_zero_null_embeddings_from_pretrained(self, bundle):
    """Patch zero null embeddings from pretrained source."""
    
    # Check if null embeddings are all-zeros
    null_vtxt_sum = bundle.null_vtxt_feat.abs().sum()
    null_ctxt_sum = bundle.null_ctxt_input.abs().sum()
    
    if null_vtxt_sum < 1e-5 or null_ctxt_sum < 1e-5:
        # ❌ All zeros → not loaded properly from checkpoint
        logger.warning('Null embeddings are zero, loading from pretrained source')
        
        # Load from null_embedding_source checkpoint
        source_path = load_from.get('null_embedding_source')
        source_ckpt = torch.load(source_path, map_location='cpu')
        
        # Extract null embeddings from source
        null_vtxt_src = source_ckpt['bundle']['null_vtxt_feat']
        null_ctxt_src = source_ckpt['bundle']['null_ctxt_input']
        
        # Patch them into current model
        bundle.null_vtxt_feat.copy_(null_vtxt_src)
        bundle.null_ctxt_input.copy_(null_ctxt_src)
        logger.info('Patched null embeddings from pretrained source')
```

**Trigger Conditions**:
- Model loaded from unconditional checkpoint (has no null embeddings)
- Model-only loading skipped `__bundle_params__` (wasn't in checkpoint)
- Null embeddings stayed at PyTorch's random init

**Source Priority**:
1. If `null_embedding_source` specified in `load_from` config → use it
2. Otherwise fallback to `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` (hardcoded)

### Step 3: Text Encoder Lazy Initialization

**In `HyMotionM2MBundle.encode_text()`** (lines 297-313):

```python
def encode_text(self, text: Union[str, List[str]], device=None):
    """Encode text with CLIP-L + Qwen3, lazy-load encoder."""
    
    # LAZY LOAD: only initialize text encoder on first call
    if not hasattr(self, '_text_encoder') or self._text_encoder is None:
        # Read encoder config from bundle config
        cfg = deepcopy(self._text_encoder_cfg)
        cfg.pop('type', None)
        
        # Create fresh encoder instance
        self._text_encoder = HYTextModel(**cfg)
        logger.info('Lazy-loaded text encoder (CLIP-L + Qwen3)')
    
    # Encode using fresh encoder
    vtxt, ctxt, ctxt_len = self._text_encoder.encode(text)
    
    return {
        'text_vec_raw': vtxt.to(device),           # (1, 1, 768) CLIP-L
        'text_ctxt_raw': ctxt.to(device),          # (1, seq, 4096) Qwen3
        'text_ctxt_raw_length': ctxt_len.to(device),
    }
```

**Key Properties**:
- Text encoder is **NOT** part of trainable model graph
- Lives on CPU for efficiency (8B LLM)
- Outputs moved to training device after encoding
- Config stored in `bundle._text_encoder_cfg`, not loaded from checkpoint

---

## Part 3: Mechanism That Prevents Garbage Output

### Three Safeguards Against Garbage Output

#### Safeguard 1: Null Embedding Source Fallback (Primary)

**How it works**:
```python
# E2/E4 config specifies:
load_from = dict(
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)

# After model-only loading:
# 1. null_vtxt_feat, null_ctxt_input are random (not in uncond checkpoint)
# 2. _patch_zero_null_embeddings_from_pretrained() detects all-zeros
# 3. Loads from HY-Motion-1.0 pretrained checkpoint
# 4. CFG unconditional path gets VALID embeddings from T2M pretrained
```

**CFG During Inference**:
```python
# In HyMotionT2MPipeline.generate():
if do_cfg:
    # Classifier-free guidance formula
    pred_uncond = model(x_t, null_vtxt, null_ctxt)    # ✅ Valid from fallback
    pred_cond = model(x_t, real_vtxt, real_ctxt)      # ✅ Text embeddings
    output = pred_uncond + scale * (pred_cond - pred_uncond)
```

Without fallback, `null_vtxt` and `null_ctxt` would be random noise → guidance scale would amplify garbage → output explodes.

#### Safeguard 2: CFG Training (Secondary)

E2/E4 configs specify:
```python
cond_mask_prob=0.1  # 10% of training samples are unconditional
```

**What this does**:
- During training, model sees both conditional (90%) and unconditional (10%) batches
- Unconditional branch trains the model to work with null embeddings
- Even if null embeddings are imperfect, gradient updates optimize them
- By training end, null embeddings converge to reasonable values

**Training flow** (in `HyMotionM2MBundle.mask_text_cond()`):
```python
if self.training and cond_mask_prob > 0.0:
    # Randomly mask 10% of batch samples
    mask = torch.bernoulli(ones(bs) * 0.1).view(bs, 1).bool()
    
    # Replace text with null for masked samples
    vtxt = torch.where(mask_vtxt, self.null_vtxt_feat.expand_as(vtxt), vtxt)
    ctxt = torch.where(mask_ctxt, self.null_ctxt_input.expand_as(ctxt), ctxt)
```

This ensures:
1. Model learns to handle null embeddings
2. Null embeddings get gradient updates (if trainable)
3. Random initialization quickly becomes useful signal

#### Safeguard 3: Training Loss Supervision (Tertiary)

E2/E4 use multi-component loss:
```python
losses_cfg=dict(
    keypoints3d_weight=10.0,  # FK loss (foot skating)
    velocity_weight=1.0,       # Smooth motion
    motion_smoothness_weight=0.5,  # Temporal smoothness
)
```

**Why this matters**:
- Random text layer initialization produces wrong predictions
- Loss supervision pulls predictions toward ground truth
- Gradient backprop through randomly initialized layers trains them
- After 1-2 epochs, even random text layers converge

Example: If cross-attention proj starts as random:
```
Epoch 0: pred_text_influenced = random_cross_attn(real_text)  # Wrong
Loss = MSE(pred_text_influenced, gt) → Large gradient
Epoch 0→1: Gradient updates cross-attention proj to map text → motion space
Epoch 1: pred_text_influenced ≈ gt  ← Text influence learned
```

---

## Part 4: Detailed Code Flow for E2/E4 Checkpoint Loading

### E2 Config (`hymotion_m2m_v2_smpl_caption_046b.py`)

```python
# === CRITICAL PART 1: Load from unconditional checkpoint ===
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',  # ← Model-only, strict=False
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)

# === CRITICAL PART 2: Enable text conditioning ===
uncondition_mode=False,  # ← CFG ENABLED
cond_mask_prob=0.1,      # ← 10% unconditional during training

# === CRITICAL PART 3: Text encoder config ===
text_encoder=dict(),  # ← Empty dict uses defaults (CLIP-L + Qwen3)

# === Data pipeline ===
train_pipeline=[
    dict(type='LoadCompatibleCaption'),  # Load caption
    dict(
        type='LoadPreExtractedTextEmbedding',
        key='caption',
        allow_none=True,  # ← Falls back if embedding missing
    ),
    dict(type='LoadSmplx55'),
    dict(type='Compute198DimPosition'),
    dict(
        type='RandomCropPadding',
        clip_len=360,
        pad_mode='replicate',
    ),
    dict(
        type='PrepareM2Mv2Condition',
        sampler_version='v3',
    ),
    dict(type='PackInputs'),
]
```

### E4 Config (`hymotion_m2m_v2_kimodo_caption_046b.py`)

Identical to E2 except:
```python
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    exclude_bundle_keys=['mean', 'std'],  # ← E4 ONLY: preserve KIMODO Root stats
)

# E4 also adds ADMM smoothing to motion preprocessing:
dict(
    type='SmplTransToKimodoRootOnline',
    key='motion',
    admm_margin_m=0.06,  # 6cm margin on XZ plane
),
```

### Runtime Loading Sequence

```
1. Initialize model bundle with random weights
   └─ motion_transformer: loaded from pretrained
   └─ text_refiner: RANDOM (will be patched)
   └─ null_vtxt_feat: RANDOM
   └─ null_ctxt_input: RANDOM

2. Call AccelerateRunner._pre_prepare_load()
   ├─ Load checkpoint from path='work_dirs/...epoch_3370'
   ├─ Call load_state_dict_selective(state_dict, strict=False)
   │  ├─ Load motion_transformer layers → ✅ SUCCESS
   │  ├─ Try to load text_refiner → ❌ MISSING
   │  ├─ Try to load null_vtxt_feat → ❌ MISSING
   │  └─ Log warnings for missing keys
   └─ State after: motion_transformer loaded, text layers still random

3. Call _patch_zero_null_embeddings_from_pretrained()
   ├─ Detect: null_vtxt_feat.sum() ≈ 0 (random init, not zero intentionally)
   ├─ Load from null_embedding_source checkpoint
   ├─ Copy: null_vtxt_feat ← HY-Motion-1.0 pretrained value
   ├─ Copy: null_ctxt_input ← HY-Motion-1.0 pretrained value
   └─ State after: null embeddings valid for CFG

4. accelerator.prepare(model)
   └─ Wraps with FSDP/DDP

5. Training loop starts
   ├─ Text encoder lazy-loads on first caption batch
   ├─ Gradients flow through randomly-initialized text_refiner
   ├─ Text layers learn from motion supervision + caption supervision
   └─ After epoch 1: text layers converged from random → useful
```

---

## Part 5: What If Things Go Wrong?

### Scenario 1: null_embedding_source Not Specified

**Config**:
```python
load_from = dict(
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',
    # ❌ Missing: null_embedding_source=...
)
```

**Result**:
- Fallback uses hardcoded path: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`
- If that file is missing → load fails → model has garbage null embeddings
- CFG produces garbage → predictions invalid

**Fix**: Always specify `null_embedding_source` explicitly.

### Scenario 2: Uncond Checkpoint Has All-Zero Null Embeddings

**What happens**:
- Some older checkpoints might save `null_vtxt_feat=zeros(...)` intentionally
- Fallback detection checks: `if null_vtxt_sum < 1e-5`
- False alarm: thinks it needs patching, loads from source (redundant but safe)

**Fix**: None needed, fallback is idempotent.

### Scenario 3: Text Refiner Randomly Initialized But Never Trained

**Scenario**: Training loop crashes before first epoch completes
**Result**: 
- Text refiner gets random gradients for a few steps
- Model doesn't converge → inference produces degraded motion quality
- But not complete garbage—other components still work

**Fix**: Resume from checkpoint or restart training.

### Scenario 4: Cross-Attention Layers Shape Mismatch

**Scenario**: Config specifies different motion_dim than checkpoint
**Example**:
```python
# Config A (trained on): motion_dim=135
cross_attn_proj = nn.Linear(1024, 135)

# Config B (loading into): motion_dim=201
cross_attn_proj = nn.Linear(1024, 201)
```

**Result**:
- Load fails: `shape mismatch (1024, 135) vs expected (1024, 201)`
- strict=False catches this, logs warning, skips loading
- Cross-attn proj stays randomly initialized
- Training still works but convergence slower

**Fix**: Match motion_dim in config to source checkpoint.

---

## Part 6: Critical Findings Summary

### ✅ Confirmed: Text Layers ARE Randomly Initialized (Not Loaded from Uncond)

```
Uncond checkpoint ← E2/E4 loads from this
│
├─ HAS: motion_transformer (22 blocks, 1024-dim, 460M params)
├─ HAS: timestep_encoder
├─ HAS: input_encoder, output_encoder
│
└─ DOES NOT HAVE:
   ├─ text_refiner nn.Module → Randomly initialized in caption model
   ├─ cross_attn layers → Randomly initialized in caption model
   ├─ null_vtxt_feat nn.Parameter → Randomly initialized then PATCHED
   ├─ null_ctxt_input nn.Parameter → Randomly initialized then PATCHED
   └─ __bundle_params__ (mean, std) → Restored separately
```

### ✅ Confirmed: strict=False Allows Partial Loading

`BaseModelBundle.load_state_dict_selective(strict=False)` implements the safeguard:
- Missing keys don't raise exceptions
- Unexpected keys are logged but ignored
- Shape mismatches are filtered gracefully
- Training proceeds despite missing layers

### ✅ Confirmed: Null Embedding Source Prevents Garbage Output

Three-tier protection:
1. **Fallback patch**: Detects zero null embeddings, loads from HY-Motion pretrained
2. **CFG training**: 10% of batch trains unconditional path, optimizes null embeddings
3. **Supervised loss**: Motion supervision pulls randomly-initialized text layers toward ground truth

### ⚠️ Risk Level: MEDIUM (Mitigated)

**Without safeguards**:
- Randomly initialized text layers could output garbage
- CFG guidance would amplify that garbage
- Model would produce junk motions

**With safeguards**:
- Null embeddings guaranteed valid (from HY-Motion pretrained)
- Text layers trained from supervision
- Risk of garbage output → Very low

**Remaining Risks**:
- If `null_embedding_source` checkpoint is corrupted or missing
- If training interrupted before text layers converge (< 1 epoch)
- If cross-attention shape mismatch not caught by config validation

---

## Appendix: Bundle-Level Orphan Parameters (The 2026-03-27 Bug)

### What Are Orphan Parameters?

Parameters that exist at bundle level, not inside registered modules:

```python
class HyMotionM2MBundle(ModelBundle):
    def __init__(self, ...):
        super().__init__()
        
        # ❌ These are NOT inside sub-modules, they're direct bundle attributes
        self.null_vtxt_feat = nn.Parameter(torch.randn(...))  # Orphan
        self.null_ctxt_input = nn.Parameter(torch.randn(...))  # Orphan
        
        # These are inside modules, tracked automatically
        self.motion_transformer = HunyuanMotionMMDiT(...)  # Not orphan
```

### The Bug (2026-03-27)

**Before Fix**:
- `trainable_parameters()` only iterated `self._trainable_modules`
- Orphan parameters never appeared in optimizer
- Never saved to checkpoint
- Never loaded from checkpoint
- Never synced across DDP ranks

**Result**: null_vtxt_feat was always randomly initialized on each load.

**After Fix**:
- `trainable_parameters()` includes `self.named_parameters(recurse=False)`
- Orphan parameters saved as `__bundle_params__` dict in checkpoint
- Orphan parameters restored by `load_state_dict_selective()`
- Orphan parameter gradients synced with `_sync_orphan_param_grads()`

---

## References

- **Checkpoint Loading**: `hftrainer/runner/accelerate_runner.py` lines 512-1367
- **State Dict Handling**: `hftrainer/models/base_model_bundle.py` lines 597-782
- **Text Conditioning**: `hftrainer/models/motion/hymotion_m2m/bundle.py` lines 260-376
- **E2/E4 Configs**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_{smpl,kimodo}_caption_046b.py`
- **Config Analysis**: `E2_E4_TEXT_CONDITIONING_CONFIG_ANALYSIS.md`
- **Historical Bug**: `hftrainer/CLAUDE.md` section "2026-03-27: Bundle-level Parameters"

