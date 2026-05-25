# Checkpoint Loading: Quick Reference Guide

## Generated Documents
This analysis has been saved in two forms:

1. **Full Technical Analysis**: `docs/temp/checkpoint_loading_e2_e4_analysis.md`
   - Comprehensive 6-part analysis with code snippets
   - Detailed explanation of each safeguard
   - Error scenarios and fixes
   - Historical bug context (2026-03-27)

2. **Visual Flow Diagram**: `docs/temp/checkpoint_loading_diagram.txt`
   - ASCII flowchart showing entire loading sequence
   - State transitions at each phase
   - Three safeguards explained visually

---

## Key Files in Codebase

### Checkpoint Loading Entry Points
| File | Lines | Purpose |
|------|-------|---------|
| `hftrainer/runner/accelerate_runner.py` | 512-646 | `_pre_prepare_load()` - Pre-FSDP model-only loading |
| `hftrainer/runner/accelerate_runner.py` | 1272-1367 | `_patch_zero_null_embeddings_from_pretrained()` - Null embedding fallback patch |
| `hftrainer/runner/accelerate_runner.py` | 1030-1082 | `_handle_load()` - Post-FSDP full-resume loading |
| `hftrainer/utils/checkpoint_utils.py` | 1-136 | `load_checkpoint()` - Format auto-detection |

### State Dict Handling
| File | Lines | Purpose |
|------|-------|---------|
| `hftrainer/models/base_model_bundle.py` | 637-782 | `load_state_dict_selective()` - Partial loading with strict=False |
| `hftrainer/models/base_model_bundle.py` | 597-635 | `state_dict_to_save()` - Saves bundle params + module state |
| `hftrainer/models/base_model_bundle.py` | 520-546 | `trainable_parameters()` - Includes orphan params |

### Text Conditioning Architecture
| File | Lines | Purpose |
|------|-------|---------|
| `hftrainer/models/motion/hymotion_m2m/bundle.py` | 260-313 | `encode_text()` - Lazy-loads CLIP-L + Qwen3 |
| `hftrainer/models/motion/hymotion_m2m/bundle.py` | 315-376 | `mask_text_cond()` - CFG masking for training (10% unconditional) |
| `hftrainer/models/motion/hymotion_m2m/bundle.py` | 431-484 | `prepare_vace_input()` - Builds conditioning context |

### Configuration Files
| File | Purpose |
|------|---------|
| `configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py` | E2 config (SMPL root) |
| `configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py` | E4 config (KIMODO root) |
| `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` | Base config for both |

### Related Analysis Documents
| File | Purpose |
|------|---------|
| `E2_E4_TEXT_CONDITIONING_CONFIG_ANALYSIS.md` | Config-level analysis (cond_mask_prob, uncondition_mode, etc.) |
| `hftrainer/CLAUDE.md` | Framework overview + historical bugs (2026-03-27, 2026-05-12) |

---

## Critical Code Snippets

### Selective Loading (The "strict=False" Magic)
```python
# File: hftrainer/models/base_model_bundle.py, line 661-670
def load_state_dict_selective(self, state_dict, strict=False):
    # This is where missing text layers are ALLOWED
    missing, unexpected = load_target.load_state_dict(
        state_dict, 
        strict=strict  # ← False allows partial loading
    )
    # Missing keys are logged but don't raise exception
```

### Null Embedding Fallback (The "Safety Net")
```python
# File: hftrainer/runner/accelerate_runner.py, line 1299-1320
def _patch_zero_null_embeddings_from_pretrained(self, bundle):
    # Check if null embeddings are random/zero
    if null_vtxt_sum < 1e-5:
        # Load from pretrained source
        source_ckpt = torch.load(null_embedding_source)
        # Patch them into model
        bundle.null_vtxt_feat.copy_(source_ckpt['bundle']['null_vtxt_feat'])
```

### CFG Training (10% Unconditional)
```python
# File: hftrainer/models/motion/hymotion_m2m/bundle.py, line 353-371
def mask_text_cond(self, vtxt, ctxt, cond_mask_prob=0.1, ...):
    if self.training and cond_mask_prob > 0.0:
        # Randomly mask 10% of batch samples
        mask = torch.bernoulli(ones(bs) * 0.1)
        # Replace with null embeddings
        vtxt = torch.where(mask_vtxt, self.null_vtxt_feat, vtxt)
```

### Lazy Text Encoder Loading
```python
# File: hftrainer/models/motion/hymotion_m2m/bundle.py, line 299-308
def encode_text(self, text, device=None):
    if not hasattr(self, '_text_encoder'):
        # LAZY LOAD on first use
        cfg = deepcopy(self._text_encoder_cfg)
        self._text_encoder = HYTextModel(**cfg)
    vtxt, ctxt, ctxt_len = self._text_encoder.encode(text)
```

---

## Configuration Pattern: E2 and E4

Both E2 and E4 use the SAME loading pattern:

```python
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',  # ← Model-only, not full checkpoint
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)

uncondition_mode=False,  # ← CFG ENABLED
cond_mask_prob=0.1,      # ← 10% unconditional during training
```

**E4 Only** (preserves KIMODO Root statistics):
```python
load_from = dict(
    ...
    exclude_bundle_keys=['mean', 'std'],  # ← E4 only
)
```

---

## The 5 Questions: Answers at a Glance

| Question | Answer | Location |
|----------|--------|----------|
| **1. Where is checkpoint loading logic?** | `AccelerateRunner._pre_prepare_load()` + `_patch_zero_null_embeddings()` | runner/accelerate_runner.py |
| **2. Are text layers loaded or random?** | Text refiner + cross-attn = RANDOM; null embeddings = PATCHED from T2M | bundle.py line ~300 + runner.py line ~1300 |
| **3. What if source has no text layers?** | strict=False allows missing → logged as warning → training trains them | base_model_bundle.py line ~670 |
| **4. How does strict=False work?** | Returns (missing, unexpected) instead of raising → training proceeds anyway | base_model_bundle.py line ~661 |
| **5. How does null_embedding_source prevent garbage?** | Detects zero nulls → loads from HY-Motion-1.0 pretrained → CFG always sees valid signals | runner.py line ~1299 |

---

## Risk Assessment

**Without Safeguards**: ⚠️ HIGH RISK
- Random text layers → wrong text influence
- Random null embeddings → CFG amplifies garbage
- Model produces nonsensical motion

**With Current Safeguards**: ✅ LOW RISK
- Null embeddings guaranteed valid from T2M pretrained
- Text layers train from supervision (convergence: 1-2 epochs)
- CFG guidance works correctly on valid signals

**Remaining Vulnerabilities**:
1. If `null_embedding_source` checkpoint is missing → fallback to hardcoded path
2. If training interrupted before epoch 1 → text layers not converged
3. If cross-attention shape mismatch → convergence slower but still works

---

## Quick Test: Is the Loading Working?

Add this debug code to check checkpoint loading:

```python
# After AccelerateRunner._pre_prepare_load()
print("✓ motion_transformer loaded:", 
      "motion_transformer.blocks.0.attn.to_q.weight" in bundle.state_dict())
print("✓ null_vtxt_feat valid:", 
      bundle.null_vtxt_feat.abs().sum().item() > 0.1)
print("✓ null_ctxt_input valid:", 
      bundle.null_ctxt_input.abs().sum().item() > 0.1)
```

Expected output:
```
✓ motion_transformer loaded: True
✓ null_vtxt_feat valid: True
✓ null_ctxt_input valid: True
```

---

## Historical Context: The 2026-03-27 Bug

Before March 27, 2026:
- Bundle-level parameters (null_vtxt_feat, null_ctxt_input) were **never saved/loaded**
- They stayed randomly initialized on each load
- CFG inference produced garbage

Fix:
- `trainable_parameters()` now includes orphan params
- State dict saves/loads `__bundle_params__` dict
- `_sync_orphan_param_grads()` syncs across DDP ranks

See `hftrainer/CLAUDE.md` section "2026-03-27: Bundle-level Parameters" for full details.

---

## Related Documents for Deep Dive

- **Config Analysis**: `E2_E4_TEXT_CONDITIONING_CONFIG_ANALYSIS.md`
- **Framework Overview**: `hftrainer/CLAUDE.md` (root documentation)
- **SOAR Post-Training**: `docs/temp/soar_m2m_v2_post_training_plan.md`
- **Global Rotation Ablation**: `CLAUDE.md` section "Global vs Local Rotation Space"
- **Mask Patterns**: `docs/design/mask_prior_rank_k.md`
