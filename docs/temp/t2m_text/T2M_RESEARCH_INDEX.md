# 🎯 HunyuanMotion T2M Official Training Code - Research Index

**Research Completion Date:** May 15, 2026  
**Status:** ✅ COMPLETE  
**Scope:** Comprehensive analysis of official HunyuanMotion T2M (Text-to-Motion) training implementation

---

## 📑 Documentation Structure

This research package contains three complementary documents organized by use case:

### 1. **T2M_ANALYSIS_FINAL_SUMMARY.md** — COMPREHENSIVE REFERENCE
   - **Best for:** In-depth understanding, implementation guidance, troubleshooting
   - **Length:** ~500 lines of detailed analysis
   - **Contents:**
     - Executive summary and key discoveries
     - Complete file locations with code organization
     - Detailed text processing pipeline explanation
     - 3-path text conditioning system documentation
     - CFG dropout implementation details
     - Training flow with step-by-step breakdowns
     - M2M vs T2M architectural comparison
     - Critical implementation details with examples
     - Key takeaways for implementation, inference, and training
     - File path reference guide
   - **Recommended for:** Developers implementing T2M systems, researchers studying architecture

### 2. **T2M_QUICK_REFERENCE.txt** — FAST LOOKUP GUIDE
   - **Best for:** Quick lookups, implementation checklist, on-the-fly reference
   - **Length:** ~100 lines of concise information
   - **Contents:**
     - Key file locations (bundle, trainer, encoder)
     - Text embedding dimensions at a glance
     - 3-path text conditioning summary
     - CFG dropout code snippet
     - Text normalization differences
     - Text padding convention
     - T2M vs M2M quick comparison
     - Implementation checklist
   - **Recommended for:** Quick reference during development, checklists

### 3. **This Document (T2M_RESEARCH_INDEX.md)** — NAVIGATION GUIDE
   - **Best for:** Understanding research scope, finding relevant information
   - **Contents:**
     - Research methodology and scope
     - Key findings summary
     - File organization
     - Topic index for easy navigation
     - FAQ and troubleshooting tips

---

## 🔍 Key Findings at a Glance

### Primary Discovery
The official HunyuanMotion T2M training code is **embedded within the hftrainer repository** as a fully integrated, production-quality implementation. It is NOT in a separate reference directory but rather represents the current local best practices.

### Text Embedding System
- **Dual-Encoder Architecture:**
  - CLIP-L (768-dim): Sentence-level embeddings, L2-normalized
  - Qwen3-8B (4096-dim): Token-level embeddings, raw (not normalized)
- **Padding:** max_text_len = 128 (standard across T2M and M2M)
- **CFG:** Bernoulli masking with ~10% unconditional probability

### Architectural Innovation
- **3-Path Text Conditioning:** Flexible system supporting pre-encoded, online, and unconditional modes
- **Null Embeddings:** nn.Parameter zero vectors for unconditional generation
- **Bundle Architecture:** All key methods centralized in HyMotionT2MBundle class

---

## 📂 Core Source Files

All files are located in the hftrainer repository:

### Bundle (Model Architecture & Forward Passes)
**Primary File:** `hftrainer/models/motion/hymotion_t2m/bundle.py` (327 lines)

Key Components:
- `HyMotionT2MBundle` class: Main model bundle
- `encode_text()`: Lazy-loads and calls HYTextModel
- `mask_text_cond()`: Implements CFG dropout via Bernoulli masking
- `predict_flow()`: Single forward pass through MMDiT
- `decode_motion_from_latent()`: FK to 3D keypoints
- `normalize_motion()` / `denormalize_motion()`: Motion normalization

### Trainer (Training Loop & Data Processing)
**Primary File:** `hftrainer/trainers/motion/hymotion_t2m_trainer.py` (193 lines)

Key Components:
- `HyMotionT2MTrainer` class: Official training loop
- `_prepare_text_encoding()`: Implements 3-path logic
  - Path 1: Pre-encoded from batch (fastest)
  - Path 2: Online encoding from captions
  - Path 3: Null embeddings for unconditional
- `_prepare_and_forward()`: Batch preparation and forward pass
- `train_step()`: Single training step

### Text Encoder (Dual-Encoder System)
**Primary File:** `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py` (271 lines)

Key Components:
- `HYTextModel` class: Dual-encoder for text embeddings
- `_encode_llm()`: Qwen3-8B encoding (4096-dim)
- `_encode_sentence_emb()`: CLIP-L encoding (768-dim)
- `encode_pooling()`: Mean pooling + L2 normalization
- `encode()`: Main interface for dual encoding

---

## 🎯 Topic-Based Navigation

### Understanding Text Conditioning
**Read:** T2M_ANALYSIS_FINAL_SUMMARY.md → "🔍 Critical Text Processing Details"
- Dual-Encoder Architecture (§1)
- Text Padding Convention (§2)
- Classifier-Free Guidance (§3)
- Three Text Conditioning Paths (§4)
- Text Embedding Normalization (§5)

### Implementing T2M System
**Read:** T2M_ANALYSIS_FINAL_SUMMARY.md → "🚀 Key Takeaways for Implementation"
- For Text-Conditioned T2M (§1)
- For Inference (§2)
- For Training (§3)

### Quick Configuration Lookup
**Read:** T2M_QUICK_REFERENCE.txt
- Text Embeddings Summary
- CFG Dropout Code
- Text Padding Convention
- Implementation Checklist

### Architectural Comparison
**Read:** T2M_ANALYSIS_FINAL_SUMMARY.md → "🆚 M2M vs T2M Comparison"
- Text Processing (IDENTICAL)
- Motion Processing (KEY DIFFERENCE: 135-dim vs 201-dim)
- Model Input (KEY DIFFERENCE: VACE vs no VACE)

---

## 📊 Text Embedding Reference

### Dimensions Summary
| Component | Shape | Type | Notes |
|-----------|-------|------|-------|
| vtxt | (B, 1, 768) | float32 | CLIP-L, L2-normalized |
| ctxt | (B, 128, 4096) | float32 | Qwen3, raw (not normalized) |
| ctxt_length | (B,) | int64 | Actual pre-padding length |
| null_vtxt_feat | (1, 1, 768) | nn.Parameter | Zero-initialized |
| null_ctxt_input | (1, 1, 4096) | nn.Parameter | Zero-initialized |

### Text Normalization
- **CLIP-L (vtxt):** ✅ L2-normalized to unit sphere (magnitude ≈ 1.0)
- **Qwen3 (ctxt):** ❌ NOT normalized (raw hidden state, arbitrary magnitude)

---

## 🔄 Data Flow Diagrams

### Text Encoding Pipeline
```
Input Text (List[str])
    ↓
    ├─→ CLIP-L Encoder → vtxt (B,1,768) [normalized]
    │
    └─→ Qwen3 Encoder → ctxt (B,Lc,4096) [not normalized]
                        ctxt_length (B,)
                            ↓
                        Pad to 128
                            ↓
                        ctxt (B,128,4096)
```

### Training Step (Simplified)
```
Batch → Text Encoding (3 paths)
        ↓
     CFG Masking (Bernoulli)
        ↓
     Transformer Forward
        ↓
     Loss Computation
        ↓
     Backward & Optimize
```

### CFG Dropout Mechanism
```
Bernoulli(p=0.1) per batch element
        ↓
mask=1 → Replace with null embeddings
        ↓
~10% of batch becomes unconditional
```

---

## ✅ Implementation Checklist

### Before Training
- [ ] Verify HYTextModel loads correctly
- [ ] Check null embeddings are initialized from pretrained checkpoint
- [ ] Verify motion normalization statistics are loaded
- [ ] Test text encoding with sample captions

### During Training
- [ ] Monitor CFG dropout probability (should see ~10% of batches with null embeddings)
- [ ] Verify text padding masks are applied correctly
- [ ] Check text embedding dimensions match expected shapes
- [ ] Save null embeddings in checkpoint (bundle-level parameters)

### For Inference
- [ ] Pre-encode text offline for speed (Path 1)
- [ ] Preserve ctxt_length for padding mask generation
- [ ] Load null embeddings from pretrained checkpoint
- [ ] Apply CFG masking during generation

---

## ⚠️ Common Pitfalls & Solutions

### Issue: Text embeddings have unexpected scale
**Cause:** Qwen3 (ctxt) not being normalized while CLIP-L (vtxt) is
**Solution:** This is correct! Qwen3 should NOT be normalized. Only CLIP-L (vtxt) should have unit norm.

### Issue: CFG dropout not working
**Cause:** Null embeddings not loaded from checkpoint
**Solution:** Ensure `_patch_zero_null_embeddings_from_pretrained()` is called during model loading. Check `bundle-level parameters` are saved in checkpoint.

### Issue: Text padding causing dimension mismatch
**Cause:** Using pre-padding length instead of padded length
**Solution:** Always pad text to 128. Use `ctxt_length` for masking, not for dimensions.

### Issue: Text encoding is slow
**Cause:** Encoding text at runtime (Path 2) instead of using pre-encoded (Path 1)
**Solution:** Pre-encode text offline and store in batch as `text_vec_raw`, `text_ctxt_raw`, `text_ctxt_raw_length` for maximum speed.

---

## 📚 Related Documentation

### In Repository
- **Main Architecture:** `CLAUDE.md` (framework overview and M2M details)
- **Motion Representation:** `CLAUDE.md` § "Motion Representation"
- **Training Patterns:** `CLAUDE.md` § "Training Configuration"
- **Text Encoder Code:** `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py` (with extensive comments)

### Generated Documents (This Package)
- **Comprehensive Analysis:** T2M_ANALYSIS_FINAL_SUMMARY.md
- **Quick Reference:** T2M_QUICK_REFERENCE.txt
- **This Index:** T2M_RESEARCH_INDEX.md

---

## 🎓 Learning Path

### For Beginners
1. Start with **T2M_QUICK_REFERENCE.txt** for overview
2. Read **T2M_ANALYSIS_FINAL_SUMMARY.md** § "Executive Summary"
3. Review "Implementation Checklist" section

### For Implementers
1. Study **T2M_ANALYSIS_FINAL_SUMMARY.md** § "Critical Text Processing Details"
2. Follow "Key Takeaways for Implementation" section
3. Reference code snippets in "CFG Dropout Masking" subsections

### For Researchers
1. Deep dive into **T2M_ANALYSIS_FINAL_SUMMARY.md** complete document
2. Cross-reference with source code files
3. Study "M2M vs T2M Comparison" section
4. Review "Training Flow (T2M)" for end-to-end understanding

---

## 🔗 Cross-References

### Text Encoding Details
- Implementation: `text_encoder.py` lines 119-200
- Analysis: T2M_ANALYSIS_FINAL_SUMMARY.md § "Critical Text Processing Details"
- Quick Ref: T2M_QUICK_REFERENCE.txt § "Text Embeddings Summary"

### CFG Dropout
- Implementation: `bundle.py` lines 193-223
- Analysis: T2M_ANALYSIS_FINAL_SUMMARY.md § "Classifier-Free Guidance"
- Code: T2M_QUICK_REFERENCE.txt § "CFG Dropout Code"

### Text Conditioning Paths
- Implementation: `hymotion_t2m_trainer.py` lines 68-127
- Analysis: T2M_ANALYSIS_FINAL_SUMMARY.md § "Three Text Conditioning Paths"
- Architecture: T2M_QUICK_REFERENCE.txt § "Text Conditioning - 3 Paths"

---

## 📋 Quick Stats

| Metric | Value |
|--------|-------|
| Total Documentation Lines | ~600 |
| Source Code Files Analyzed | 3 |
| Key Classes Documented | 4 |
| Text Embedding Dimensions | 5 |
| Implementation Paths | 3 |
| Critical Parameters | 8+ |
| Common Pitfalls | 4 |

---

## ✨ Special Features of This Research

### 1. **Complete Architecture Mapping**
   Every key component traced from source code to documentation

### 2. **3-Path Text Conditioning System**
   Unique feature of HyMotion T2M documented in detail

### 3. **Dual-Encoder Analysis**
   Side-by-side comparison of CLIP-L vs Qwen3 processing

### 4. **Implementation Checklists**
   Actionable steps for implementation and debugging

### 5. **Code-to-Documentation Traceability**
   Every claim backed by file references and line numbers

---

## 📞 Using This Documentation

### For Quick Questions
→ Use **T2M_QUICK_REFERENCE.txt**

### For Detailed Understanding
→ Use **T2M_ANALYSIS_FINAL_SUMMARY.md**

### For Navigation & Learning Path
→ Use this **T2M_RESEARCH_INDEX.md**

### For Source Truth
→ Reference actual source files in `hftrainer/` directory

---

## ✅ Research Completion Checklist

- ✅ Located all official T2M training code files
- ✅ Extracted and documented text encoding logic
- ✅ Mapped CFG dropout implementation
- ✅ Documented 3-path text conditioning
- ✅ Compared T2M vs M2M architectures
- ✅ Verified text embedding dimensions
- ✅ Documented null embedding usage
- ✅ Created implementation guidelines
- ✅ Generated quick reference guide
- ✅ Created comprehensive analysis
- ✅ Produced this research index

---

## 📝 Citation

If referencing this research, cite as:

```
HunyuanMotion T2M Official Training Code Analysis
Research Date: May 15, 2026
Repository: /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
Documentation Package: T2M_ANALYSIS_FINAL_SUMMARY.md, T2M_QUICK_REFERENCE.txt, T2M_RESEARCH_INDEX.md
```

---

**Status:** ✅ COMPLETE AND READY FOR USE  
**Last Updated:** May 15, 2026  
**Maintained By:** AI Research & Development  
**Repository:** /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
