# PrismBundle Text Encoder Investigation — COMPLETE

**Status**: ✅ COMPLETE  
**Date**: May 25, 2026  
**Session**: Continuation from prior investigation  

## Investigation Scope

Comprehensive analysis of how PrismBundle handles `text_encoder` and `tokenizer` attributes, including:
- Whether they can be set to None in config
- Whether the bundle's __init__ or from_config methods crash if these attributes are missing
- Whether mmengine-style `_delete_=True` patterns or similar skip mechanisms exist

## Key Findings

### Question 1: Can I set `text_encoder=None` in config to skip loading it?

**Answer**: ❌ **NO**

**Evidence**:
- `PrismBundle.__init__()` requires positional argument `text_encoder: dict`
- If omitted or None, raises TypeError immediately: `__init__() missing required argument: 'text_encoder'`
- Even if None is passed, `encode_prompt()` (lines 164-165, 183) crashes with AttributeError when calling `next(self.text_encoder.parameters())`
- `encode_prompt_with_mask()` has identical crash points (lines 217-218, 236)

### Question 2: Does the bundle crash if text_encoder is missing?

**Answer**: ❌ **CRASHES IMMEDIATELY**

**Failure Modes**:
1. **Init-time**: TypeError if omitted from config dict passed to `from_config()`
2. **Deferred**: AttributeError when `encode_prompt()` is first called if text_encoder was set to None
3. **Save-time**: AttributeError in `save_pretrained()` line 117 if text_encoder doesn't exist

### Question 3: Is there a `_delete_=True` pattern or similar?

**Answer**: ❌ **NO**

**Evidence**:
- ModelBundle uses custom `_build_modules()` method (base_model_bundle.py lines 394-493)
- No skip/delete mechanism exists in this method
- MMEngine patterns (`_delete_=True`, conditional module loading) are NOT supported
- All modules listed in the config dict passed to `_build_modules()` MUST be present and valid

## Architecture Details

### PrismBundle Initialization (bundle.py lines 34-53)

```python
def __init__(
    self,
    transformer: dict,
    vae: dict,
    tokenizer: dict,
    text_encoder: dict,        # ← REQUIRED
    scheduler: dict,
    smpl_pose_processor: dict,
):
    super().__init__()
    self._build_modules({
        'transformer': transformer,
        'vae': vae,
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,  # ← Passed directly
        'scheduler': scheduler,
        'smpl_pose_processor': smpl_pose_processor,
    })
```

### ModelBundle._build_modules() (base_model_bundle.py lines 394-493)

```python
def _build_modules(self, modules_cfg: dict):
    for name, sub_cfg in modules_cfg.items():  # Iterates over all provided
        # ... processing ...
        module = HF_MODELS.build(sub_cfg)      # Line 451: Will crash if sub_cfg invalid
        
        if isinstance(module, nn.Module):
            setattr(self, name, module)        # Registers attribute
        else:
            self._extra_attributes[name] = module
```

**Key Point**: No skip mechanism. Every module in modules_cfg MUST be buildable via HF_MODELS registry.

### Text Encoder Usage Patterns

| Location | Usage | Impact if None |
|----------|-------|-----------------|
| `encode_prompt()` line 164 | `next(self.text_encoder.parameters()).device` | AttributeError |
| `encode_prompt()` line 165 | `next(self.text_encoder.parameters()).dtype` | AttributeError |
| `encode_prompt()` line 183 | `self.text_encoder(input_ids=..., attention_mask=...)` | AttributeError |
| `encode_prompt_with_mask()` lines 217-218, 236 | Same as above | AttributeError |
| `save_pretrained()` line 117 | `self.text_encoder.save_pretrained(...)` | AttributeError |
| `_bundle_config_from_pretrained()` line 104 | Hardcoded loading | Silently loads T2M pretrained value |

## Workarounds

### Option 1: Subclass and Override _build_modules() ✅ RECOMMENDED

```python
class PrismBundleOptionalText(PrismBundle):
    """PrismBundle variant with optional text_encoder."""
    
    def __init__(self, transformer, vae, tokenizer, scheduler, 
                 smpl_pose_processor, text_encoder=None, **kwargs):
        # Skip parent init, set text_encoder conditionally
        nn.Module.__init__(self)
        modules_to_build = {
            'transformer': transformer,
            'vae': vae,
            'tokenizer': tokenizer,
            'scheduler': scheduler,
            'smpl_pose_processor': smpl_pose_processor,
        }
        if text_encoder is not None:
            modules_to_build['text_encoder'] = text_encoder
        
        self._build_modules(modules_to_build)
        # ... rest of init logic
```

**Pros**:
- Clean, explicit opt-out
- Minimal code changes
- Backward compatible

**Cons**:
- Subclass maintains text_encoder references anyway
- Must override all text_encoder-using methods

### Option 2: Create Dummy Text Encoder Wrapper

```python
class DummyTextEncoder(nn.Module):
    def __init__(self, hidden_dim=4096):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, input_ids, attention_mask=None):
        return type('obj', (object,), {'last_hidden_state': 
                    torch.zeros(input_ids.shape[0], input_ids.shape[1], 
                               self.hidden_dim)})()
    
    @property
    def parameters(self):
        return self._parameters.values()
```

**Pros**:
- No code changes to PrismBundle
- Compatible with existing config system

**Cons**:
- Misleading—appears to have text encoding but produces zeros
- Still loads DummyTextEncoder, uses memory

### Option 3: Use Lightweight T5 Model

Pre-load small T5 variant (e.g., T5-small) instead of T5-XXL to reduce memory:

```python
text_encoder=dict(
    type='T5EncoderModel',
    from_pretrained=dict(
        pretrained_model_name_or_path='google/t5-small',  # 60M params vs XXL's 11B
        load_in_8bit=True,  # Further reduce VRAM
    ),
    trainable=False,
    save_ckpt=False,
)
```

**Pros**:
- Actually functional text encoding
- No code changes needed

**Cons**:
- Different embedding space than XXL
- Requires retraining downstream models

## Historical Context

**PRISM Configuration** (configs/prism/prism_1b_tp2m_1frame.py):
- Uses T5-XXL (11B parameters) as text_encoder
- Marked with `trainable=False` and `save_ckpt=False` to minimize overhead
- Still loads and keeps text_encoder in memory during inference
- This is intentional: PRISM uses text conditioning even during pose-conditioned motion generation

## Recommendations

1. **Do NOT** try to skip text_encoder loading via config tricks
2. **DO** use Option 1 (subclass) if text_encoder truly not needed for your use case
3. **Verify** that your inference code actually needs text conditioning—PRISM may use it more than expected
4. **Consider** using smaller text encoder (Option 3) if memory is the blocker, rather than removing it entirely

## Related Code Files

- **Main investigation**: `hftrainer/models/motion/prism/bundle.py` (lines 1-324)
- **Base class**: `hftrainer/models/base_model_bundle.py` (lines 313-493)
- **Configuration example**: `configs/prism/prism_1b_tp2m_1frame.py` (lines 58-68)
- **Usage patterns**: `hftrainer/models/motion/prism/bundle.py` (lines 157-254)

## Conclusion

PrismBundle's design **requires** a text_encoder to be present and valid. The framework does not support optional module loading or deletion patterns. Any attempt to skip text_encoder must be done through subclassing or configuration of a dummy/lightweight alternative, not through configuration flags.

This is by design: PRISM's architecture assumes text conditioning is available, and attempting to bypass it could lead to silent failures or unexpected behavior downstream.
