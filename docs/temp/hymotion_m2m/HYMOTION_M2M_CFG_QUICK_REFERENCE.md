# HyMotion M2M: CFG Dropout — Quick Reference

## ⚡ One-Minute Summary

**What:** Classifier-Free Guidance dropout masks text embeddings during training to enable CFG at inference.

**How:** During training, ~10% of batch samples see learned "null" embeddings instead of real text.

**Why:** Model learns to distinguish text-conditioned vs. unconditional predictions, enabling CFG guidance.

---

## 🔧 Quick Code Reference

### Init (bundle.py line 212-213)
```python
self.null_vtxt_feat = nn.Parameter(torch.randn(1, 1, 768) * 0.01, requires_grad=True)
self.null_ctxt_input = nn.Parameter(torch.randn(1, 1, 4096) * 0.01, requires_grad=True)
```
**Key:** Trainable, initialized with small random values (N(0, 0.01²))

### Masking (bundle.py line 315-376)
```python
def mask_text_cond(vtxt, ctxt, cond_mask_prob=0.0, force_mask=False):
    if force_mask:
        return (null_vtxt_feat.expand_as(vtxt), null_ctxt_input.expand_as(ctxt))
    
    if self.training and cond_mask_prob > 0:
        mask = torch.bernoulli(ones(B) * cond_mask_prob)  # ~10% True
        # Expand mask to match dimensions
        vtxt = torch.where(mask, null_vtxt_feat.expand_as(vtxt), vtxt)
        ctxt = torch.where(mask, null_ctxt_input.expand_as(ctxt), ctxt)
    
    return (vtxt, ctxt)
```

### Training Usage
```python
# In trainer loop
text_vec, text_ctxt, text_available = bundle.mask_text_cond(
    text_vec_raw, text_ctxt_raw,
    cond_mask_prob=0.1,           # 10% dropout
    return_text_available=True,
)
# ~3 samples in batch of 32 get null embeddings

pred = model(motion, text_vec, text_ctxt, ...)
loss = mse_loss(pred, target)
loss.backward()  # Gradients flow into null_vtxt_feat, null_ctxt_input
```

### Inference Usage (CFG)
```python
# Two forward passes
pred_text = model(..., text_vec=text_vec, text_ctxt=text_ctxt, ...)
pred_null = model(..., text_vec=null_vtxt_feat, text_ctxt=null_ctxt_input, ...)

# Guided prediction
pred_guided = pred_null + 7.5 * (pred_text - pred_null)
```

---

## ❓ FAQ

### Q1: Does it zero out embeddings?
**A:** No. It **replaces** with `null_vtxt_feat` and `null_ctxt_input` via `torch.where()`.

### Q2: How is the ~10% chosen per batch?
**A:** Per-sample Bernoulli: `torch.bernoulli(ones(B) * 0.1)` → each sample independent.

### Q3: Could it accidentally mask 100%?
**A:** No. P(all 32 masked) = 0.1^32 ≈ 10^-31. Mathematically impossible.

### Q4: Why are null embeddings trainable?
**A:** Model needs to learn a unique "no text" representation. Frozen zeros cause CFG to fail.

### Q5: When is masking applied?
**A:** Only during training when `self.training=True` AND `cond_mask_prob > 0`.

### Q6: What's the difference between training and inference?
**A:** 
- **Training:** Probabilistic masking (~10% per batch)
- **Inference:** Deterministic (force_mask=True for CFG)

---

## 📊 Dimensions

| Embedding | Shape | Dim | Role |
|---|---|---|---|
| `text_vec` (vtxt) | (B, 1, 768) | 768 | Sentence-level semantic |
| `text_ctxt` (ctxt) | (B, L_c, 4096) | 4096 | Token-level fine-grained |
| `null_vtxt_feat` | (1, 1, 768) | 768 | Learned "no sentence" |
| `null_ctxt_input` | (1, 1, 4096) | 4096 | Learned "no tokens" |

---

## ✅ Verification Checklist

### For Trainers
- [ ] `cond_mask_prob > 0` in config
- [ ] `mask_text_cond()` called with embeddings
- [ ] `text_available` flag passed to loss computation
- [ ] `null_vtxt_feat.requires_grad == True`
- [ ] Gradients flowing into null embeddings during backward

### For Inference
- [ ] Model in eval mode
- [ ] `force_mask=True` when computing null prediction
- [ ] Two forward passes: one with real text, one with null
- [ ] CFG formula: `pred_null + scale * (pred_text - pred_null)`
- [ ] Loaded checkpoint contains trained null embeddings

### For Debugging
```python
# Check null embeddings were trained
print(bundle.null_vtxt_feat)
# Should see changed values from initial N(0, 0.01²)

# Check gradient flow
print(bundle.null_vtxt_feat.grad)
# Should see non-None gradient after backward()

# Check masking rate empirically
text_available_log = []  # Log from training
masking_rate = 1 - text_available_log.mean()
# Should be close to cond_mask_prob (e.g., ~0.1)
```

---

## 🎯 Key Equations

### Bernoulli Sampling
```
mask ~ Bernoulli(p)
E[mask] = p
P(all masked | B samples) = p^B  ≈ 0 for B>5, p<0.2
```

### torch.where
```
output[i] = true_val[i]  if condition[i] is True
            false_val[i] if condition[i] is False
```

### CFG Guidance
```
pred_guided = pred_null + scale * (pred_text - pred_null)

scale=1.0  → no guidance (uses pred_text)
scale=0.0  → anti-guidance (uses pred_null)
scale>1.0  → amplified text influence
```

---

## 🔗 Related Files

```
hftrainer/models/motion/hymotion_m2m/
├── bundle.py
│   └── mask_text_cond() — line 315
│   └── null_vtxt_feat init — line 212
│   └── null_ctxt_input init — line 213
│
hftrainer/trainers/motion/
├── hymotion_m2m_trainer.py
│   └── _prepare_and_forward() — uses mask_text_cond()
│
hftrainer/pipelines/motion/
└── hymotion_m2m_pipeline.py
    └── forward_pass() — uses force_mask=True
```

---

## 📚 Full Documentation

See detailed analysis:
- **HYMOTION_M2M_CFG_ANALYSIS.md** — Complete technical breakdown
- **HYMOTION_M2M_TEXT_FLOW.md** — Training & inference flow traces

---

**Last updated:** 2026-05-15
