# HyMotion M2M: CFG & Text Conditioning — Documentation Index

**Created:** 2026-05-15  
**Scope:** Comprehensive analysis of Classifier-Free Guidance dropout, null embedding initialization, and text conditioning flow in HyMotion M2M motion transformer.

---

## 📋 Document Map

### 1. **HYMOTION_M2M_CFG_QUICK_REFERENCE.md** ⭐ START HERE
**Read time:** 5 minutes  
**Purpose:** Quick answers to key questions  
**Contains:**
- One-minute summary of CFG dropout
- Code snippets (init, masking, training, inference)
- FAQ with answers
- Verification checklist
- Debugging guide

**When to read:** First time learning about CFG, need a quick answer, debugging checklist

---

### 2. **HYMOTION_M2M_CFG_ANALYSIS.md** 📖 DEEP DIVE
**Read time:** 20 minutes  
**Purpose:** Complete technical breakdown  
**Contains:**
- Executive summary (3 core questions answered)
- Full `mask_text_cond` implementation with line-by-line explanation
- Null embeddings initialization semantics
- Training vs. inference flow comparison
- Broadcasting in mask expansion
- Text embedding dimensions (sentence vs. token)
- Integration with training loss
- Common pitfalls checklist
- Key equations and math
- Visual diagrams

**When to read:** Understanding the implementation details, checking for bugs, learning the theory

---

### 3. **HYMOTION_M2M_TEXT_FLOW.md** 🔄 TRACING FLOWS
**Read time:** 25 minutes  
**Purpose:** Complete trace of data flow from input to model to loss  
**Contains:**
- Training flow: data loading → pre-extraction → CFG dropout → forward → loss
- Inference flow: setup → denoising loop → CFG computation → output
- Why trainable null embeddings matter (scenario comparison)
- Mathematical justification for learning null
- Text mask integration (padding mask vs. CFG mask)
- Debug scenarios with solutions
- Reference implementation (pseudo-code)
- Full iteration examples

**When to read:** Debugging unexplained behavior, understanding interaction between components, full system view

---

## 🎯 Navigation by Use Case

### "I have 5 minutes"
👉 Read **HYMOTION_M2M_CFG_QUICK_REFERENCE.md** — FAQ section

### "Is there a bug in mask_text_cond?"
👉 Read **HYMOTION_M2M_CFG_ANALYSIS.md** — Section 1 ("The Implementation") + Section 7 ("Common Pitfalls")

### "How does CFG actually work?"
👉 Read **HYMOTION_M2M_TEXT_FLOW.md** — Section 2.2–2.4 ("Denoising loop with CFG")

### "Why does my text guidance not work?"
👉 Read **HYMOTION_M2M_TEXT_FLOW.md** — Section 5 ("Common Debug Scenarios")

### "I want to understand trainable null embeddings"
👉 Read **HYMOTION_M2M_CFG_ANALYSIS.md** — Section 2 ("Null Embeddings Initialization") + **HYMOTION_M2M_TEXT_FLOW.md** — Section 3 ("Why Trainable Null Embeddings Matter")

### "How does the model learn CFG?"
👉 Read **HYMOTION_M2M_CFG_ANALYSIS.md** — Section 6 ("Integration with Training Loss") + **HYMOTION_M2M_TEXT_FLOW.md** — Section 1.3 ("Gradient Flow During Training")

### "Show me the complete training loop"
👉 Read **HYMOTION_M2M_TEXT_FLOW.md** — Section 1.2 + Section 6 ("Reference Implementation Flow")

### "Show me the complete inference loop"
👉 Read **HYMOTION_M2M_TEXT_FLOW.md** — Section 2 + Section 6 ("Reference Implementation Flow")

---

## ✅ Three Core Questions Answered

### Q1: What does `mask_text_cond` do?

**Quick answer:** Replaces embeddings with learned null embeddings via `torch.where()`.

**Details:**
- During training: ~10% of batch samples see null embeddings instead of real text
- Null embeddings are learned parameters (trainable, `requires_grad=True`)
- Initialized with small random values (N(0, 0.01²))
- Not zeroing, not padding — actual replacement with null tensors

**Reference:** 
- HYMOTION_M2M_CFG_QUICK_REFERENCE.md — Q1 FAQ
- HYMOTION_M2M_CFG_ANALYSIS.md — Section 1 (full code + operation flow)

---

### Q2: How does `cond_mask_prob` control masking?

**Quick answer:** Per-sample Bernoulli sampling. For batch B, ~`cond_mask_prob*100%` of samples masked.

**Details:**
```python
mask = torch.bernoulli(ones(B) * cond_mask_prob)  # Independent per sample
E[# masked] = B * cond_mask_prob
```
- Each sample draws independently: P(sample i masked) = cond_mask_prob
- Not aggregated, not sequential — true per-sample randomness
- Different forward passes produce different masks (stochastic)

**Reference:**
- HYMOTION_M2M_CFG_QUICK_REFERENCE.md — Q2 FAQ
- HYMOTION_M2M_CFG_ANALYSIS.md — Section 1, "Phase 2b: Probabilistic Masking Branch"
- HYMOTION_M2M_CFG_ANALYSIS.md — Section 8.1 ("Bernoulli Sampling" equations)

---

### Q3: Is there a bug causing 100% masking?

**Quick answer:** NO. Mathematically impossible.

**Why:**
- P(all B samples masked) = cond_mask_prob ^ B
- Example: cond_mask_prob=0.1, B=32 → P = 0.1^32 ≈ 10^-31
- Independent draws + no aggregation → no cascade to 100%
- No hardcoding or indexing error

**Common mistakes:**
- Trainer logic setting `cond_mask_prob=1.0` (trainer bug, not mask_text_cond bug)
- Force-masking for inference CFG (intentional, not a bug)

**Reference:**
- HYMOTION_M2M_CFG_QUICK_REFERENCE.md — Q3 FAQ
- HYMOTION_M2M_CFG_ANALYSIS.md — Section 1, "Why This Design Avoids the '100% Masking' Bug"

---

## 🔗 Code References

### Key Files
```
hftrainer/models/motion/hymotion_m2m/
├── bundle.py
│   ├── HyMotionM2MBundle class (line 142)
│   ├── __init__ method (line 151)
│   ├── mask_text_cond method (line 315)
│   ├── null_vtxt_feat init (line 212)
│   ├── null_ctxt_input init (line 213)
│   └── Documentation (line 205-211)
```

### Method Signatures
```python
# Initialization
self.null_vtxt_feat = nn.Parameter(torch.randn(1, 1, 768) * 0.01, requires_grad=True)
self.null_ctxt_input = nn.Parameter(torch.randn(1, 1, 4096) * 0.01, requires_grad=True)

# Masking
def mask_text_cond(
    self,
    vtxt: Tensor,
    ctxt: Tensor,
    force_mask: bool = False,
    cond_mask_prob: float = 0.0,
    return_text_available: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
```

---

## 📊 Summary Table

| Aspect | During Training | During Inference |
|---|---|---|
| **mask_text_cond call** | With `cond_mask_prob=0.1` (probabilistic) | With `force_mask=True` (deterministic) |
| **Masking rate** | ~10% of batch | 100% (both passes use null) |
| **null_vtxt_feat usage** | For masked samples | For unconditional pass |
| **Trainable** | Yes (gradients flow) | No (loaded from checkpoint) |
| **Purpose** | CFG training signal | CFG guidance computation |

---

## 🎓 Key Concepts

### Classifier-Free Guidance (CFG)
Training model to handle both conditioned (with text) and unconditional (without text) cases, enabling controllable generation at inference via:
```
pred_guided = pred_unconditional + scale * (pred_conditional - pred_unconditional)
```

### Null Embeddings
Learned parameters representing "no text condition." Must be trainable (not frozen zeros) so model learns to distinguish them from real text, making the CFG signal meaningful.

### CFG Dropout
During training, randomly masking text embeddings with null embeddings (~10% of batch). This forces model to learn the unconditional branch, enabling CFG to work.

### Bernoulli Sampling
Random binary distribution: each sample independently chosen with probability p. Used to determine which batch samples get masked.

### torch.where
Conditional tensor selection: `where(condition, true_val, false_val)` — element-wise choice based on condition.

---

## 🔍 Verification Checklist

### ✓ For Developers
- [ ] null_vtxt_feat has `requires_grad=True`
- [ ] null_ctxt_input has `requires_grad=True`
- [ ] mask_text_cond called with `cond_mask_prob > 0` during training
- [ ] Gradients flowing into null embeddings (check during backward)
- [ ] Inference uses `force_mask=True` for unconditional pass

### ✓ For Debugging
- [ ] Masking rate ~cond_mask_prob (log text_available stats)
- [ ] Model receives null embeddings correctly (check intermediate tensors)
- [ ] CFG signal non-zero (pred_text ≠ pred_null)
- [ ] No 100% masking (would require P = 0.1^32 ≈ 10^-31)

---

## 📚 Learning Path

**Beginner:** Quick reference (5 min) → FAQ (5 min) → Checklist (5 min)

**Intermediate:** Quick reference → Section 1 of CFG Analysis → Section 1 of Text Flow

**Advanced:** All three documents in full → Reference implementation (pseudo-code) → Production code inspection

---

## 💡 FAQ About the FAQ

### Why are there three documents?
- **Quick Reference:** For busy engineers needing fast answers
- **CFG Analysis:** For deep understanding of the mechanism
- **Text Flow:** For seeing the full picture (data in → computation → loss out)

### Can I just read one?
**Quick reference:** Yes, for debugging or quick facts  
**CFG Analysis:** Recommended for understanding implementation details  
**Text Flow:** Recommended for system-level understanding  
**All three:** Best for comprehensive mastery

### Which should I share with teammates?
- Sharing a specific bug? → Quick reference + relevant section
- Teaching about CFG? → Text flow (better narrative flow)
- Code review checklist? → Quick reference verification section
- Full technical understanding? → All three in order

---

**End of Documentation Index**  
Generated: 2026-05-15
