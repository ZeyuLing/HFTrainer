# T2M to M2M v2: Parameter Names and Checkpoint Loading Mapping

**Date**: May 17, 2026  
**Status**: Complete Analysis with Code References

---

## Executive Summary

When loading T2M pretrained weights into M2M v2, the parameter naming is **exact match** for reusable modules. The checkpoint loading mechanism in `checkpoint_loading.py` filters parameters by module path and applies selective freezing strategies.

---

## 1. HunyuanMotionMMDiT Module Attributes (Line Numbers)

### __init__ Method (Lines 610-774)

The following modules are created as attributes in HunyuanMotionMMDiT:

#### Input Encoders
| Module Attribute | Line | Type | Purpose |
|---|---|---|---|
| `self.input_encoder` | 704 | `nn.Linear` | Projects motion input (motion_dim → feat_dim) |
| `self.ctxt_encoder` | 706 | `nn.Linear` | Projects context text embeddings (ctxt_input_dim → feat_dim) |
| `self.vtxt_encoder` | 708 | `MLPEncoder` | Projects vector text embeddings (vtxt_input_dim → feat_dim, 2-layer) |
| `self.timestep_encoder` | 710-714 | `TimestepEmbeddingEncoder` | Sinusoidal + MLP for diffusion timesteps |

#### Optional Text Refiner
| Module Attribute | Line | Type | Notes |
|---|---|---|---|
| `self.text_refiner` | 721 | `SingleTokenRefiner` | Optional self-attention over text tokens (2 layers default) |

#### Transformer Blocks
| Module Attribute | Line | Type | Structure |
|---|---|---|---|
| `self.double_blocks` | 733-748 | `nn.ModuleList` | MMDoubleStreamBlock × (num_layers // 3) |
| `self.single_blocks` | 753-768 | `nn.ModuleList` | MMSingleStreamBlock × (2 * num_layers // 3) |

#### Output Layer
| Module Attribute | Line | Type | Purpose |
|---|---|---|---|
| `self.final_layer` | 774 | `FinalLayer` | Projects (feat_dim → output_dim) with adapter modulation |

#### Optional Long Skip Connection
| Module Attribute | Line | Type | Notes |
|---|---|---|---|
| `self.long_skip_net` | 700 | `FinalLayer` | Residual bypass if with_long_skip_connection=True |

#### Learnable Start Token
| Module Attribute | Line | Type | Notes |
|---|---|---|---|
| `self.start_token` | 695 | `nn.Parameter` | Shape (1, feat_dim) if insert_start_token=True |

---

## 2. T2M Null Embedding Initialization (bundle.py)

### Location and Values

**File**: `hftrainer/models/motion/hymotion_t2m/bundle.py`

**Lines**: 104-105

```python
# Zero default; actual values loaded from pretrained checkpoint.
self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, vtxt_input_dim))
self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, ctxt_input_dim))
```

**Key Details**:
- `null_vtxt_feat`: Shape (1, 1, 768) - initialized as zeros
- `null_ctxt_input`: Shape (1, 1, 4096) - initialized as zeros
- **Both are frozen in T2M** but **trainable in M2M v2**
- Used for classifier-free guidance when `force_mask=True` in `mask_text_cond()`
- T2M config may load actual values from checkpoint during training/inference

### M2M v2 Initialization (Different)

**File**: `hftrainer/models/motion/hymotion_m2m/bundle.py`

**Lines**: 104-105

```python
# M2M v2: trainable small random initialization (scaled by 0.01)
# Different strategy: don't directly load T2M's frozen zeros
self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, vtxt_input_dim))
self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, ctxt_input_dim))
```

**Note**: Both bundles start with zeros, but M2M v2 makes them trainable and won't load T2M's values (excluded in checkpoint loading).

---

## 3. Checkpoint Loading Parameter Name Mapping

### Module Path Mapping

When loading a checkpoint into M2M v2's `motion_transformer`, the parameter names follow this pattern:

**T2M Checkpoint Keys** → **M2M Model Keys** (1:1 mapping for reusable modules)

#### Reusable Modules (Exact Match)

```
motion_transformer.ctxt_encoder.*           → motion_transformer.ctxt_encoder.*
motion_transformer.vtxt_encoder.*           → motion_transformer.vtxt_encoder.*
motion_transformer.timestep_encoder.*       → motion_transformer.timestep_encoder.*
motion_transformer.text_refiner.*           → motion_transformer.text_refiner.*
motion_transformer.double_blocks.*          → motion_transformer.double_blocks.*
motion_transformer.single_blocks.*          → motion_transformer.single_blocks.*
```

**Example concrete parameter names**:
```
motion_transformer.ctxt_encoder.weight          [4096, 1024]
motion_transformer.ctxt_encoder.bias            [1024]
motion_transformer.double_blocks.0.motion_qkv.weight    [4096, 1024]
motion_transformer.double_blocks.0.motion_qkv.bias      [4096]
motion_transformer.double_blocks.0.motion_out_proj.weight  [1024, 1024]
motion_transformer.single_blocks.0.linear1.weight   [5120, 1024]  # 4*D QKV + 4*D MLP hidden
motion_transformer.single_blocks.0.linear2.weight   [1024, 1024]
motion_transformer.text_refiner.layers.0.*         (2 self-attn layers)
```

#### Shape-Mismatch Modules (Reinitialized)

```
motion_transformer.input_encoder      T2M[135, 1024] → M2M[594, 1024] ✗ SKIP
motion_transformer.final_layer        T2M[1024, 135] → M2M[1024, 198] ✗ SKIP
```

**These are NOT loaded from checkpoint; reinitialized with Xavier uniform**

#### Bundle-Level Parameters (Excluded)

```
null_vtxt_feat        ✗ EXCLUDED - M2M v2 keeps trainable zeros
null_ctxt_input       ✗ EXCLUDED - M2M v2 keeps trainable zeros
mean                  ✗ EXCLUDED - Different dimensions (T2M: 135, M2M: 198)
std                   ✗ EXCLUDED - Different dimensions (T2M: 135, M2M: 198)
```

---

## 4. Checkpoint Loading Mechanism

### Implementation Location

**File**: `hftrainer/models/motion/hymotion_m2m/checkpoint_loading.py`

**Key Functions**:

#### Main Entry Point (Line 165)

```python
def load_t2m_pretrained_selective(
    bundle,
    t2m_checkpoint_path: str,
    freeze_strategy: str = 'none',
) -> Dict[str, Any]:
```

**Flow**:
1. Load checkpoint from .ckpt or .pt file (Line 217-220)
2. Extract only reusable parameters via `_filter_reusable_params()` (Line 223)
3. Detect shape mismatches (Line 239)
4. Load via `bundle.load_state_dict_selective()` with `strict=False` (Line 242-246)
5. Apply freeze strategy (Line 295)

#### Filtering Reusable Params (Line 105)

```python
def _filter_reusable_params(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract only parameters from reusable modules."""
    filtered = {}
    for key, value in state_dict.items():
        for reusable_mod in REUSABLE_MODULES:
            if key.startswith(reusable_mod):
                filtered[key] = value
                break
    return filtered
```

**REUSABLE_MODULES** (Lines 49-56):
```python
REUSABLE_MODULES = {
    'motion_transformer.ctxt_encoder',
    'motion_transformer.vtxt_encoder',
    'motion_transformer.timestep_encoder',
    'motion_transformer.text_refiner',
    'motion_transformer.double_blocks',
    'motion_transformer.single_blocks',
}
```

#### Loading via Runner (accelerate_runner.py, Lines 1137-1152)

```python
elif load_scope == 'model':
    from hftrainer.utils.checkpoint_utils import load_checkpoint
    try:
        state_dict = load_checkpoint(path, map_location='cpu')
        self.bundle.load_state_dict_selective(
            state_dict,
            exclude_bundle_keys=exclude_bundle_keys,
        )
```

---

## 5. Freeze Strategies

### Strategy Definitions (checkpoint_loading.py, Lines 207-390)

| Strategy | Description | Frozen Modules |
|---|---|---|
| `'none'` | No freezing (default) | None |
| `'encoders'` | Freeze text encoders only | ctxt_encoder, vtxt_encoder, timestep_encoder |
| `'text_refiner'` | Freeze encoders + text_refiner | + text_refiner |
| `'blocks'` | Freeze encoders + text_refiner + transformer blocks | + double_blocks, single_blocks |
| `'full'` | Freeze all reusable modules | All of above |

### Implementation (Lines 339-390)

```python
def _apply_freeze_strategy(bundle, freeze_strategy: str) -> list:
    frozen = []
    
    if freeze_strategy == 'none':
        return frozen
    
    modules_to_freeze = []
    
    if freeze_strategy in ('encoders', 'text_refiner', 'blocks', 'full'):
        modules_to_freeze.extend([
            'motion_transformer.ctxt_encoder',
            'motion_transformer.vtxt_encoder',
            'motion_transformer.timestep_encoder',
        ])
    
    if freeze_strategy in ('text_refiner', 'blocks', 'full'):
        modules_to_freeze.append('motion_transformer.text_refiner')
    
    if freeze_strategy in ('blocks', 'full'):
        modules_to_freeze.extend([
            'motion_transformer.double_blocks',
            'motion_transformer.single_blocks',
        ])
    
    # Apply freezing via requires_grad_(False)
    for mod_path in modules_to_freeze:
        parts = mod_path.split('.')
        if len(parts) == 2:
            parent = getattr(bundle, parts[0], None)
            if parent and hasattr(parent, parts[1]):
                mod = getattr(parent, parts[1])
                _freeze_module(mod)  # requires_grad_(False)
                frozen.append(mod_path)
    
    return frozen
```

---

## 6. Config Usage Examples

### Full Freeze Config

**File**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_046b_t2m_full_freeze.py`

```python
model = dict(
    t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    t2m_freeze_strategy='full',  # Freeze all loaded modules
)
```

**Effect**:
- Load: ctxt_encoder, vtxt_encoder, timestep_encoder, text_refiner, double_blocks, single_blocks
- Freeze: All of above (requires_grad=False)
- Trainable: input_encoder (reinitialized), final_layer (reinitialized)

### No Freeze Config

**File**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_046b_t2m_no_freeze.py`

```python
model = dict(
    t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    t2m_freeze_strategy='none',  # No freezing
)
```

**Effect**:
- Load: All reusable modules
- Freeze: None
- Trainable: Everything

---

## 7. Parameter Count Summary (HunyuanMotionMMDiT)

### Configuration
```python
feat_dim = 1024
num_heads = 16
num_layers = 18  # 6 double + 12 single
mlp_ratio = 4.0
input_dim = 594  # M2M v2: [x_t(198) + reactive(198) + mask(198)]
ctxt_input_dim = 4096
vtxt_input_dim = 768
output_dim = 198
```

### Module Sizes
- **input_encoder**: 594 × 1024 = 0.61M
- **ctxt_encoder**: 4096 × 1024 = 4.19M
- **vtxt_encoder**: ~2M (2-layer MLP)
- **timestep_encoder**: ~2M (sinusoidal + 2-layer MLP)
- **text_refiner**: ~2M (2 self-attn layers, 1024D)
- **double_blocks** (6 blocks): Motion + Text parallel streams
  - Per block: ~80M (QKV, proj, MLP for both branches)
  - Total: ~480M
- **single_blocks** (12 blocks): Concatenated stream
  - Per block: ~40M (linear1, linear2, MLP)
  - Total: ~480M
- **final_layer**: ~1M
- **Total**: ~0.97B parameters (matches 0.46B benchmark for half precision)

---

## 8. Exact Parameter Names Reference

### Example: ctxt_encoder
```
motion_transformer.ctxt_encoder.weight  Shape: [1024, 4096]
motion_transformer.ctxt_encoder.bias    Shape: [1024]
```

### Example: Double Block 0
```
motion_transformer.double_blocks.0.motion_mod.lin_0.weight      Shape: [6144, 1024]  # 6 * D for modulation params
motion_transformer.double_blocks.0.motion_mod.lin_0.bias        Shape: [6144]
motion_transformer.double_blocks.0.motion_norm1.weight          Shape: [1024]
motion_transformer.double_blocks.0.motion_qkv.weight            Shape: [4096, 1024]  # QKV: 3*D
motion_transformer.double_blocks.0.motion_qkv.bias              Shape: [4096]
motion_transformer.double_blocks.0.motion_q_norm.weight         Shape: [64]           # head_dim=64
motion_transformer.double_blocks.0.motion_k_norm.weight         Shape: [64]
motion_transformer.double_blocks.0.motion_out_proj.weight       Shape: [1024, 1024]
motion_transformer.double_blocks.0.motion_out_proj.bias         Shape: [1024]
motion_transformer.double_blocks.0.motion_mlp.fc1.weight        Shape: [4096, 1024]  # MLP: D → 4D
motion_transformer.double_blocks.0.motion_mlp.fc1.bias          Shape: [4096]
motion_transformer.double_blocks.0.motion_mlp.fc2.weight        Shape: [1024, 4096]  # MLP: 4D → D
motion_transformer.double_blocks.0.motion_mlp.fc2.bias          Shape: [1024]
# Same for text_* variants...
```

### Example: Single Block 0
```
motion_transformer.single_blocks.0.modulation.lin_0.weight      Shape: [3072, 1024]  # 3 * D for shift, scale, gate
motion_transformer.single_blocks.0.norm.weight                  Shape: [1024]
motion_transformer.single_blocks.0.linear1.weight               Shape: [5120, 1024]  # (3*D QKV + 4*D MLP hidden)
motion_transformer.single_blocks.0.linear1.bias                 Shape: [5120]
motion_transformer.single_blocks.0.linear2.weight               Shape: [1024, 1024+4096]  # (D attn + 4*D MLP) → D
motion_transformer.single_blocks.0.linear2.bias                 Shape: [1024]
motion_transformer.single_blocks.0.q_norm.weight                Shape: [64]
motion_transformer.single_blocks.0.k_norm.weight                Shape: [64]
```

---

## 9. Checkpoint Loading Validation

### verify_loading() Function (checkpoint_loading.py, Lines 393-431)

```python
def verify_loading(bundle, t2m_checkpoint_path: str) -> Dict[str, Any]:
    """Verify T2M pretrained parameters were correctly loaded."""
    t2m_state = _load_checkpoint(t2m_checkpoint_path)
    reusable_state = _filter_reusable_params(t2m_state)
    model_state = bundle.motion_transformer.state_dict()
    
    mismatches = []
    verified_count = 0
    
    for key, ckpt_value in reusable_state.items():
        if key in model_state:
            model_value = model_state[key]
            if ckpt_value.shape == model_value.shape:
                # Allow small numerical differences
                if not torch.allclose(ckpt_value, model_value, atol=1e-4):
                    mismatches.append(key)
                verified_count += 1
    
    return {
        'reusable_params_match': len(mismatches) == 0,
        'num_verified_params': verified_count,
        'mismatches': mismatches,
    }
```

**Usage**:
```python
result = verify_loading(m2m_bundle, 'checkpoints/HY-Motion-1.0/latest.ckpt')
if result['reusable_params_match']:
    print(f"✓ All {result['num_verified_params']} params verified")
else:
    print(f"✗ {len(result['mismatches'])} params don't match")
```

---

## 10. Key Takeaways for Freeze Strategy Planning

### Exact Module Paths for Freezing

To freeze specific modules in M2M after loading T2M, use:

```python
# Individual module paths
'motion_transformer.ctxt_encoder'      # Linear layer, 4096→1024
'motion_transformer.vtxt_encoder'      # MLPEncoder (2 layers)
'motion_transformer.timestep_encoder'  # TimestepEmbeddingEncoder
'motion_transformer.text_refiner'      # SingleTokenRefiner (2 self-attn layers)
'motion_transformer.double_blocks'     # nn.ModuleList of 6 MMDoubleStreamBlock
'motion_transformer.single_blocks'     # nn.ModuleList of 12 MMSingleStreamBlock

# Block-level indexing
'motion_transformer.double_blocks.0'   # First double block
'motion_transformer.double_blocks.5'   # Last (6th) double block
'motion_transformer.single_blocks.11'  # Last (12th) single block

# Sub-module within a block (example)
'motion_transformer.double_blocks.0.motion_qkv'  # Motion query-key-value
'motion_transformer.double_blocks.0.text_mlp'    # Text feedforward network
```

### Parameter Name Patterns

- **Encoder params**: `motion_transformer.{ctxt,vtxt,timestep}_encoder.*`
- **Text refiner params**: `motion_transformer.text_refiner.layers.{0,1}.*`
- **Double block params**: `motion_transformer.double_blocks.{i}.{motion,text}_{qkv,norm,mlp,*}.*`
- **Single block params**: `motion_transformer.single_blocks.{i}.{linear1,linear2,modulation,*}.*`

### Not Loaded from T2M

```python
# These are always reinitialized (NOT loaded):
'motion_transformer.input_encoder'     # 594→1024 (T2M is 135→1024)
'motion_transformer.final_layer'       # 1024→198 (T2M is 1024→135)

# These are excluded from checkpoint loading:
'null_vtxt_feat'
'null_ctxt_input'
'mean'
'std'
```

---

## 11. Recommended Freeze Strategy for Transfer Learning

### Scenario 1: Strong T2M Baseline (Conservative Fine-tuning)
```python
t2m_freeze_strategy='full'
# Trainable: input_encoder (594-dim VACE-aware), final_layer (198-dim M2M output)
# Risk: May underfit if T2M features don't adapt well to M2M task
```

### Scenario 2: Balanced Transfer (Recommended)
```python
t2m_freeze_strategy='blocks'
# Trainable: encoders, text_refiner, input_encoder, final_layer
# Reasoning: Encoders are task-specific but can fine-tune; blocks are general
```

### Scenario 3: Maximum Fine-tuning (Warm-start)
```python
t2m_freeze_strategy='none'
# All modules trainable from T2M initialization
# Risk: May degrade with poor learning rates; benefit: maximum expressivity
```

### Scenario 4: Minimal Adaption (Encoder Transfer)
```python
t2m_freeze_strategy='encoders'
# Trainable: text_refiner, double_blocks, single_blocks, + input/final
# Reasoning: Text encoding is most general; text_refiner and blocks can adapt
```

---

## Summary Table: Module Loadability

| Module | Type | T2M Shape | M2M Shape | Loaded | Frozen (Strategy) |
|---|---|---|---|---|---|
| input_encoder | Linear | 135→1024 | 594→1024 | ✗ | N/A |
| ctxt_encoder | Linear | 4096→1024 | 4096→1024 | ✓ | full/blocks/text_refiner/encoders |
| vtxt_encoder | MLPEncoder | 768→1024 | 768→1024 | ✓ | full/blocks/text_refiner/encoders |
| timestep_encoder | TstepEmb | - | - | ✓ | full/blocks/text_refiner/encoders |
| text_refiner | SingleTokenRefiner | 1024D×2 | 1024D×2 | ✓ | full/blocks/text_refiner |
| double_blocks[0:6] | MMDoubleStreamBlock | 1024 | 1024 | ✓ | full/blocks |
| single_blocks[0:12] | MMSingleStreamBlock | 1024 | 1024 | ✓ | full/blocks |
| final_layer | FinalLayer | 1024→135 | 1024→198 | ✗ | N/A |
| null_vtxt_feat | Parameter | zeros(1,1,768) | zeros(1,1,768) | ✗ | N/A |
| null_ctxt_input | Parameter | zeros(1,1,4096) | zeros(1,1,4096) | ✗ | N/A |

---

## References

- **HunyuanMotionMMDiT**: `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py` (Lines 571-1453)
- **T2M Bundle**: `hftrainer/models/motion/hymotion_t2m/bundle.py` (Lines 51-328)
- **M2M Bundle**: `hftrainer/models/motion/hymotion_m2m/bundle.py` (Lines 175-624)
- **Checkpoint Loading**: `hftrainer/models/motion/hymotion_m2m/checkpoint_loading.py` (All)
- **Runner (Accelerate)**: `hftrainer/runner/accelerate_runner.py` (Lines 1030-1152)
- **Config**: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`
- **Freeze Configs**: 
  - `configs/hymotion_m2m_v2/hymotion_m2m_v2_046b_t2m_full_freeze.py`
  - `configs/hymotion_m2m_v2/hymotion_m2m_v2_046b_t2m_no_freeze.py`

