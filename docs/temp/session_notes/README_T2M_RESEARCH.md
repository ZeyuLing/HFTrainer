# HyMotion T2M Research Package
## Complete Analysis of Official Text-to-Motion Training Implementation

**Status**: ✅ COMPLETE  
**Date**: May 15, 2026  
**Total Documentation**: 461 lines (core research) + 896 lines (supporting docs) = 1,357 lines  

---

## 📚 Documentation Overview

This research package contains four comprehensive documents providing different perspectives on the HyMotion T2M implementation:

### 1. **HUNYUAN_MOTION_T2M_FINAL_RESEARCH.md** (461 lines)
**The Master Research Document**

Contains:
- ✅ Discovery process and findings
- ✅ Official code location confirmation (commit acf4730)
- ✅ Complete file structure mapping
- ✅ Core implementation details for all three major components
- ✅ Text processing pipeline (4 stages)
- ✅ T2M vs M2M architectural comparison
- ✅ Critical implementation details
- ✅ File statistics and verification status

**Best for**: Understanding the overall architecture and how all pieces fit together

---

### 2. **T2M_ANALYSIS_FINAL_SUMMARY.md** (383 lines)
**Executive Analysis with Code Examples**

Contains:
- ✅ High-level architecture overview
- ✅ Complete file locations with line numbers
- ✅ Detailed code snippets for key functions
- ✅ 3-path text conditioning system (with diagrams)
- ✅ CFG dropout implementation (line-by-line)
- ✅ Training loop workflow
- ✅ M2M vs T2M side-by-side comparison
- ✅ Critical gotchas and edge cases

**Best for**: Deep technical understanding with concrete code references

---

### 3. **T2M_QUICK_REFERENCE.txt** (136 lines)
**Fast Lookup Guide**

Contains:
- ✅ Key file locations (quick jump table)
- ✅ Text embedding dimensions cheat sheet
- ✅ 3-path conditioning summary
- ✅ CFG dropout code snippet (copy-paste ready)
- ✅ Text normalization differences table
- ✅ Padding convention reference
- ✅ Common questions answered
- ✅ Implementation checklist

**Best for**: Quick answers and keeping during active development

---

### 4. **T2M_RESEARCH_INDEX.md** (377 lines)
**Navigation and Learning Guide**

Contains:
- ✅ Research methodology overview
- ✅ Key findings index (searchable)
- ✅ Core file descriptions with line ranges
- ✅ Topic-based navigation (8 major topics)
- ✅ Text embedding reference tables
- ✅ Data flow diagrams (ASCII)
- ✅ Implementation checklists
- ✅ Common pitfalls and solutions
- ✅ Learning paths for different skill levels

**Best for**: Finding specific topics and learning progressively

---

## 🎯 Quick Start by Role

### I'm a **New Developer** - Where do I start?
1. Read: `T2M_QUICK_REFERENCE.txt` (5 min) - Get oriented
2. Read: `T2M_RESEARCH_INDEX.md` → "Learning Paths" section (15 min)
3. Study: `T2M_ANALYSIS_FINAL_SUMMARY.md` (30 min) - Deep dive
4. Reference: Source code with documents side-by-side

### I need to **Implement Text Conditioning**
1. Check: `T2M_QUICK_REFERENCE.txt` → "3-Path Conditioning" 
2. Read: `T2M_ANALYSIS_FINAL_SUMMARY.md` → "3-Path Text Conditioning System"
3. Review: `hftrainer/trainers/motion/hymotion_t2m_trainer.py` lines 85-110
4. Reference: `HYTextModel` in `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py`

### I need to **Implement CFG Dropout**
1. Copy: Code snippet from `T2M_QUICK_REFERENCE.txt` → "CFG Dropout Code"
2. Understand: `T2M_ANALYSIS_FINAL_SUMMARY.md` → "Classifier-Free Guidance"
3. Review: `HyMotionT2MBundle.mask_text_cond()` in `bundle.py` lines 193-223

### I need to **Understand Architecture**
1. Study: `HUNYUAN_MOTION_T2M_FINAL_RESEARCH.md` → "Comparison" section
2. Review: `T2M_RESEARCH_INDEX.md` → "Data Flow Diagrams"
3. Compare: T2M (training code) vs M2M (hftrainer/trainers/motion/hymotion_m2m_trainer.py)

### I need to **Debug an Issue**
1. Search: `T2M_RESEARCH_INDEX.md` → "Common Pitfalls"
2. Check: Relevant section in `T2M_ANALYSIS_FINAL_SUMMARY.md`
3. Reference: File locations to inspect specific code

---

## 🔍 Key Findings at a Glance

### Official Code Location
```
Git Repository: /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
Git Commit: acf4730cca1591fd054c5061443e0fe9532b3adc
Date: April 26, 2026
Status: Fully integrated, ready for deployment
```

### Core Files (The Holy Trinity)
```
hftrainer/models/motion/hymotion_t2m/bundle.py               [309 lines] ⭐ MODEL BUNDLE
hftrainer/trainers/motion/hymotion_t2m_trainer.py            [193 lines] ⭐ TRAINING LOOP
hftrainer/models/motion/hymotion_m2m/network/text_encoder.py [271 lines] ⭐ TEXT ENCODING
```

### Text Encoding System
```
CLIP-L (Sentence)        → 768-dim   [L2-normalized]
+ Qwen3-8B (Tokens)      → 4096-dim  [Raw, not normalized]
= Dual-Encoder Architecture for motion description
```

### CFG Dropout Mechanism
```
Type: Bernoulli masking
Probability: ~10% per batch element
Application: Random text conditioning dropout during training
Purpose: Enable unconditional generation capability
```

### 3-Path Text Conditioning
```
Path 1: Pre-cached embeddings (fastest)        → Use if available
Path 2: Online encoding from captions (slower) → Default fallback
Path 3: Null embeddings (unconditional)        → For zero text
```

---

## 📊 Documentation Statistics

| Document | Lines | Content Type | Primary Use |
|----------|-------|--------------|------------|
| HUNYUAN_MOTION_T2M_FINAL_RESEARCH.md | 461 | Research findings + architecture | Master reference |
| T2M_ANALYSIS_FINAL_SUMMARY.md | 383 | Technical analysis + code | Deep understanding |
| T2M_QUICK_REFERENCE.txt | 136 | Lookup tables + snippets | Quick answers |
| T2M_RESEARCH_INDEX.md | 377 | Navigation + learning paths | Topic search |
| **Total** | **1,357** | **Mixed** | **Complete coverage** |

Plus complementary documentation:
- CHECKPOINT_PATHS_E2_E4_REPORT.md (374 lines) - Model checkpoints
- CFG_ANALYSIS_STATUS.md - CFG implementation status

---

## 🔗 Cross-References

### Text Encoding Implementation
- **Where to find**: `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py`
- **Key classes**: `HYTextModel`
- **Key methods**: `encode()`, `_encode_llm()`, `_encode_sentence_emb()`
- **Learn in**: `T2M_ANALYSIS_FINAL_SUMMARY.md` → "HYTextModel Details"

### Training Loop
- **Where to find**: `hftrainer/trainers/motion/hymotion_t2m_trainer.py`
- **Key classes**: `HyMotionT2MTrainer`
- **Key methods**: `train_step()`, `_prepare_text_encoding()`, `_prepare_and_forward()`
- **Learn in**: `T2M_ANALYSIS_FINAL_SUMMARY.md` → "Training Loop"

### Model Bundle
- **Where to find**: `hftrainer/models/motion/hymotion_t2m/bundle.py`
- **Key classes**: `HyMotionT2MBundle`
- **Key methods**: `encode_text()`, `mask_text_cond()`, `predict_flow()`, `decode_motion_from_latent()`
- **Learn in**: `HUNYUAN_MOTION_T2M_FINAL_RESEARCH.md` → "Core Implementation"

### Configuration
- **Where to find**: `configs/hymotion_t2m/`
- **Main config**: `hymotion_t2m_201dim_046b.py` (173 lines)
- **Light config**: `hymotion_t2m_smoke.py` (95 lines)
- **Learn in**: `T2M_QUICK_REFERENCE.txt` → "Configuration"

---

## ✅ Verification Checklist

- ✅ Official code location confirmed (commit acf4730)
- ✅ All three core files verified to exist
- ✅ File line counts verified
- ✅ Method signatures verified
- ✅ Code cross-references verified
- ✅ Git history verified
- ✅ Configuration files verified
- ✅ Supporting files verified (dataset, pipeline)
- ✅ Text embedding dimension verified (768 + 4096)
- ✅ CFG mechanism verified (Bernoulli masking)
- ✅ 3-path conditioning verified
- ✅ Mean/Std normalization verified

---

## 🎓 Learning Paths

### Path 1: Architecture Overview (30 minutes)
1. Skim: `T2M_QUICK_REFERENCE.txt` 
2. Study: `HUNYUAN_MOTION_T2M_FINAL_RESEARCH.md` → Sections 1-3
3. Review: File structure diagram

### Path 2: Text Processing Deep Dive (60 minutes)
1. Read: `T2M_ANALYSIS_FINAL_SUMMARY.md` → "Text Processing Pipeline"
2. Study: `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py`
3. Compare: `T2M_RESEARCH_INDEX.md` → "Text Embedding Reference"

### Path 3: Training Implementation (90 minutes)
1. Read: `T2M_ANALYSIS_FINAL_SUMMARY.md` → "Training Loop Workflow"
2. Study: `hftrainer/trainers/motion/hymotion_t2m_trainer.py`
3. Trace: Data flow with `T2M_RESEARCH_INDEX.md` diagrams

### Path 4: CFG Implementation (45 minutes)
1. Copy: Code from `T2M_QUICK_REFERENCE.txt`
2. Study: `T2M_ANALYSIS_FINAL_SUMMARY.md` → "Classifier-Free Guidance"
3. Review: `HyMotionT2MBundle.mask_text_cond()` source

### Path 5: Complete System (2-3 hours)
1. Follow all paths above in sequence
2. Cross-reference with source code
3. Study: Configuration files
4. Review: Common pitfalls in `T2M_RESEARCH_INDEX.md`

---

## 🚀 Using This Package

### For Documentation
```bash
# View all T2M research
ls -lh /path/to/T2M*.* HUNYUAN*.md

# Quick reference while coding
cat T2M_QUICK_REFERENCE.txt

# Search for topics
grep -n "CFG\|text conditioning\|normalization" T2M_*.md
```

### For Code Reference
```bash
# View HyMotionT2MBundle
cat hftrainer/models/motion/hymotion_t2m/bundle.py

# View training loop
cat hftrainer/trainers/motion/hymotion_t2m_trainer.py

# View text encoder
cat hftrainer/models/motion/hymotion_m2m/network/text_encoder.py
```

### For Git Investigation
```bash
# See original commit
git show acf4730 --stat

# See all T2M changes
git log --all -- hftrainer/models/motion/hymotion_t2m/
git log --all -- hftrainer/trainers/motion/hymotion_t2m_trainer.py
```

---

## 📝 Notes

### What This Package Covers
✅ Official HyMotion T2M implementation in hftrainer  
✅ Text encoding architecture (Qwen3 + CLIP-L)  
✅ CFG dropout mechanism (Bernoulli masking)  
✅ 3-path text conditioning system  
✅ Training loop implementation  
✅ Motion decoding and FK  
✅ Architectural differences from M2M  
✅ Configuration management  

### What This Package Does NOT Cover
❌ Inference pipeline optimization (covered in separate files)  
❌ Evaluation metrics and benchmarking  
❌ Custom dataset creation  
❌ Checkpoint management details  
❌ Hardware optimization and distributed training  
❌ Debugging specific training failures  

---

## 🤝 Questions Answered by This Package

**Architecture**
- "What is HyMotion T2M architecture?" → See HUNYUAN_MOTION_T2M_FINAL_RESEARCH.md
- "How is it different from M2M?" → See comparison table
- "What files contain the implementation?" → See file structure

**Text Encoding**
- "What text encoders are used?" → Qwen3-8B + CLIP-L
- "What are the output dimensions?" → 768-dim + 4096-dim
- "How is normalization handled?" → See T2M_ANALYSIS_FINAL_SUMMARY.md

**CFG Dropout**
- "How is classifier-free guidance implemented?" → Bernoulli masking
- "What's the dropout probability?" → ~10% per batch element
- "Where's the code?" → HyMotionT2MBundle.mask_text_cond() lines 193-223

**Text Conditioning**
- "What are the three paths?" → Pre-cached, online, unconditional
- "When is each used?" → See trainer implementation
- "How's it configured?" → See configs/hymotion_t2m/

**Training**
- "What's the training loop?" → See trainer.py
- "How's text handled?" → Via 3-path system
- "What loss is used?" → M2M loss (velocity or x1)

---

## 📞 Troubleshooting Quick Links

| Issue | Solution |
|-------|----------|
| "Can't find T2M code" | It's in hftrainer, not external repo |
| "Text embedding dimensions?" | 768 (CLIP) + 4096 (Qwen3) |
| "Where's CFG dropout?" | bundle.py lines 193-223 |
| "How to trace training?" | Use trainer.py + bundle.py + text_encoder.py |
| "Where to modify text handling?" | trainer.py _prepare_text_encoding() |
| "How to disable CFG?" | Set cond_mask_prob=0.0 in trainer |

---

## 📅 Research Timeline

- **April 26, 2026**: HyMotion T2M introduced (commit acf4730)
- **May 14-15, 2026**: Comprehensive research conducted
- **May 15, 2026**: Complete documentation package generated

---

**Created by**: Claude Opus 4.6  
**Purpose**: Complete analysis and documentation of HyMotion T2M training implementation  
**Status**: ✅ COMPLETE AND VERIFIED  
**Last Updated**: May 15, 2026

---

## File Manifest

```
README_T2M_RESEARCH.md                              (This file - navigation)
├── HUNYUAN_MOTION_T2M_FINAL_RESEARCH.md          (461 lines - master research)
├── T2M_ANALYSIS_FINAL_SUMMARY.md                 (383 lines - technical analysis)
├── T2M_QUICK_REFERENCE.txt                       (136 lines - lookup guide)
├── T2M_RESEARCH_INDEX.md                         (377 lines - navigation guide)
├── CHECKPOINT_PATHS_E2_E4_REPORT.md              (374 lines - model paths)
├── CFG_ANALYSIS_STATUS.md                        (Supporting analysis)
│
└── Source Code Referenced:
    ├── hftrainer/models/motion/hymotion_t2m/bundle.py
    ├── hftrainer/trainers/motion/hymotion_t2m_trainer.py
    ├── hftrainer/models/motion/hymotion_m2m/network/text_encoder.py
    ├── hftrainer/datasets/motion/hymotion_t2m_dataset.py
    ├── hftrainer/pipelines/motion/hymotion_t2m_pipeline.py
    └── configs/hymotion_t2m/*.py
```

---

**Ready for use. Start with your role's "Quick Start" section above.** 🚀

