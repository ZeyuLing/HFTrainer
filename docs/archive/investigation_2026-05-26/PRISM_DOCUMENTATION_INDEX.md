# PRISM Dataset Pipeline Documentation Index

Complete documentation for understanding `caption_path` flow in the PRISM dataset pipeline.

## 📄 Documents Generated

### 1. **PRISM_CAPTION_PATH_QUICK_REFERENCE.md** ⭐ START HERE
   - **Purpose:** Quick answer to your main question
   - **Content:** 
     - TL;DR summary
     - Key facts table
     - 4-step code flow
     - Path normalization example
     - Common issues & solutions
   - **Length:** ~1 page
   - **Best for:** Quick lookup, debugging checklist

### 2. **PRISM_CAPTION_PATH_SUMMARY.md**
   - **Purpose:** Comprehensive trace from annotation to pipeline
   - **Content:**
     - TL;DR with full answer
     - Annotation file structure
     - Dataset class explanation
     - Path resolution walkthrough
     - Configuration details
     - Debugging checklist
   - **Length:** ~3 pages
   - **Best for:** Understanding the full architecture

### 3. **PRISM_CAPTION_PATH_TRACE.md**
   - **Purpose:** Detailed step-by-step trace with examples
   - **Content:**
     - Complete flow with actual code snippets
     - Annotation entry examples
     - Dataset class implementation
     - Path resolution in prepare_data()
     - LoadPreExtractedT5Feature implementation
     - Path transformation table
     - Key architectural insights
   - **Length:** ~4 pages
   - **Best for:** Deep understanding, learning the design patterns

### 4. **PRISM_CODE_REFERENCES.md**
   - **Purpose:** Exact code locations and line numbers
   - **Content:**
     - Configuration file locations
     - Dataset class method locations
     - Annotation file structure
     - Transform pipeline code locations
     - Data flow diagram with line numbers
     - Configuration parameters
     - Refetch logic locations
     - Debug logging examples
   - **Length:** ~3 pages
   - **Best for:** Finding specific code, adding debug logging

### 5. **PRISM_CAPTION_PATH_FLOW.txt**
   - **Purpose:** Visual ASCII diagrams of the entire flow
   - **Content:**
     - Stage-by-stage visual representation
     - Path transformation at each stage
     - Key properties table
     - Refetch logic diagram
   - **Length:** ~2 pages
   - **Best for:** Understanding the big picture visually

### 6. **PRISM_DOCUMENTATION_INDEX.md** (THIS FILE)
   - **Purpose:** Navigation guide for all documentation

---

## 🎯 How to Use This Documentation

### Scenario 1: "I just want to understand the basic flow"
→ Read: **PRISM_CAPTION_PATH_QUICK_REFERENCE.md** (5 min)

### Scenario 2: "I need to fix a dataset path issue"
→ Read: **PRISM_CAPTION_PATH_QUICK_REFERENCE.md** → **Common Issues & Solutions**

### Scenario 3: "I want to understand the full architecture"
→ Read: **PRISM_CAPTION_PATH_FLOW.txt** → **PRISM_CAPTION_PATH_SUMMARY.md**

### Scenario 4: "I need to modify the dataset/transform code"
→ Read: **PRISM_CODE_REFERENCES.md** + Source files with exact line numbers

### Scenario 5: "I want to debug by adding logging"
→ Read: **PRISM_CODE_REFERENCES.md** → Section 8: Testing & Debugging

### Scenario 6: "Deep dive: understand all design decisions"
→ Read: **PRISM_CAPTION_PATH_TRACE.md** → Section: Key Architectural Insights

---

## 🔑 Key Answers Quick Lookup

| Question | Answer | Document | Section |
|----------|--------|----------|---------|
| What dataset class is used? | `MotionHubSingleAgentTextDataset` | All | - |
| Where is caption_path set? | `prepare_data()` method, line 37-39 | CODE_REFERENCES | Section 2 |
| What is the raw annotation value? | `../hymotion_data/Academic/.../file.json` | SUMMARY | Section 1 |
| What value enters the pipeline? | `data/motionhub/../hymotion_data/.../file.json` | SUMMARY | Section 3 |
| Is it normalized when passed to pipeline? | NO (contains literal `..`) | QUICK_REF | Key Facts |
| When is it normalized? | Inside `LoadPreExtractedT5Feature` | SUMMARY | Section 4 |
| What is it mapped to? | `data/t5_feature/hymotion_data/.../file.pt` | SUMMARY | Section 5 |
| What if the .pt file doesn't exist? | Returns None → triggers refetch | TRACE | Section - |
| What are the final output keys? | `t5_text_embeds`, `t5_text_mask`, `caption` | CODE_REF | Section 4 |
| Why this design? | Allows pre-extracted features to be cached | TRACE | Arch Insights |

---

## 📊 Document Comparison

| Document | Length | Depth | Visuals | Code | Best For |
|----------|--------|-------|---------|------|----------|
| QUICK_REFERENCE | 1 page | High-level | YES | Key parts | Quick answers |
| SUMMARY | 3 pages | Medium | YES | Some | Full understanding |
| TRACE | 4 pages | Deep | NO | Full | Deep learning |
| CODE_REFERENCES | 3 pages | Deep | Some | Full + line # | Finding code |
| FLOW.txt | 2 pages | High-level | ASCII only | NO | Visual learning |

---

## 🏗️ Architecture Overview

```
ANNOTATION (JSON)
    └─> RAW VALUE: ../hymotion_data/...json
        
DATASET.prepare_data()
    └─> os.path.join() [NO NORMALIZATION]
        └─> OUTPUT: data/motionhub/../hymotion_data/...json
            
PIPELINE ENTRY
    └─> results['caption_path'] = "data/motionhub/../hymotion_data/...json"
        └─> CONTAINS LITERAL ".." ⭐ KEY INSIGHT
            
LoadPreExtractedT5Feature.transform()
    ├─> NORMALIZE: os.path.normpath()
    │   └─> data/hymotion_data/...json
    ├─> STRIP PREFIXES
    │   └─> hymotion_data/...json
    ├─> CHANGE EXTENSION
    │   └─> hymotion_data/...pt
    └─> PREPEND FEATURE_DIR
        └─> data/t5_feature/hymotion_data/...pt
            
FEATURE FILE LOOKUP
    ├─> os.path.exists(pt_path)? 
    ├─> YES → Load embeddings
    └─> NO + allow_none=True → Return None → REFETCH
```

---

## 🔍 Search Guide

### By Topic

**Configuration & Setup**
- Base config: QUICK_REF (Configuration Override section)
- T5-cached config: QUICK_REF (Configuration Override section)
- Transform parameters: CODE_REFERENCES (Section 6)

**Path Handling**
- Raw annotation paths: SUMMARY (Section 1, 3)
- Path joining: CODE_REFERENCES (Section 2)
- Path normalization: TRACE (Section 5), QUICK_REF (Path Normalization)
- Feature path mapping: TRACE (Section 4)

**Data Flow**
- Full flow: FLOW.txt (Stage 1-5)
- Flow diagram: FLOW.txt (Data Flow Diagram)
- Line-by-line: CODE_REFERENCES (Section 5)

**Error Handling**
- Refetch logic: QUICK_REF (Refetch Behavior)
- Error cases: CODE_REFERENCES (Section 7)
- Debugging: CODE_REFERENCES (Section 8)

### By Keyword

- **caption_path**: All documents
- **prepare_data()**: CODE_REFERENCES, TRACE
- **LoadPreExtractedT5Feature**: TRACE, CODE_REFERENCES
- **normpath**: TRACE, QUICK_REF
- **os.path.join**: SUMMARY, TRACE
- **refetch**: FLOW.txt, CODE_REFERENCES
- **allow_none**: SUMMARY, QUICK_REF
- **feature_dir**: CODE_REFERENCES, QUICK_REF

---

## 💡 Key Insights

### Design Pattern 1: Lazy Normalization
Paths are NOT normalized early in the dataset. Normalization happens inside the transform that needs it. This allows different transforms to interpret paths differently.

**Document:** TRACE (Section: Key Architectural Insights)

### Design Pattern 2: Feature Path Mapping
The T5 feature extraction mirrors the caption file structure, allowing features to be cached and reused without re-extraction.

**Document:** TRACE (Section: Key Architectural Insights)

### Design Pattern 3: Refetch Safety
If pre-extracted features don't exist, transform returns None instead of failing, allowing the dataset to automatically fetch a different sample.

**Document:** TRACE (Section: Key Architectural Insights)

### Design Pattern 4: Relative Paths
Annotation uses relative paths (`../hymotion_data/...`) to reference shared data, allowing the dataset to be moved without breaking paths.

**Document:** TRACE (Section: Key Architectural Insights)

---

## 🐛 Debugging Workflow

1. **Check annotation file:**
   ```bash
   grep -i "hierarchical_caption_path" data/annotation/train_hq_motionhub_hymotion.json | head -5
   ```
   → Reference: QUICK_REF (Debug Commands)

2. **Verify dataset class:**
   → Reference: CODE_REFERENCES (Section 2)

3. **Check prepare_data() method:**
   → Reference: CODE_REFERENCES (Section 2, lines 31-41)

4. **Add debug logging:**
   → Reference: CODE_REFERENCES (Section 8)

5. **Verify T5 features exist:**
   ```bash
   find data/t5_feature -name "*.pt" | wc -l
   ```
   → Reference: QUICK_REF (Debug Commands)

6. **Check transform logic:**
   → Reference: CODE_REFERENCES (Section 4, lines 238-333)

---

## 📚 Source Code Files Referenced

```
hftrainer/datasets/motion/motionhub/
├── single_agent_dataset.py           [Parent dataset class]
├── single_agent_text_dataset.py       [Main dataset class] ⭐
└── transforms/
    └── load_text.py                   [LoadPreExtractedT5Feature] ⭐

configs/prism/
├── prism_1b_tp2m_1frame.py            [Base config] ⭐
└── prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py [T5-cached config] ⭐

data/annotation/
└── train_hq_motionhub_hymotion.json   [Annotation file]
```

⭐ = Files most frequently referenced in documentation

---

## 🎓 Learning Path

### Beginner
1. PRISM_CAPTION_PATH_QUICK_REFERENCE.md
2. PRISM_CAPTION_PATH_FLOW.txt

### Intermediate
3. PRISM_CAPTION_PATH_SUMMARY.md
4. PRISM_CODE_REFERENCES.md

### Advanced
5. PRISM_CAPTION_PATH_TRACE.md
6. Source code in: `hftrainer/datasets/motion/motionhub/`

---

## 📋 Checklist: Verifying Your Understanding

After reading this documentation, you should be able to answer:

- [ ] What is the dataset class name?
- [ ] In which method is caption_path set?
- [ ] What is the raw annotation value?
- [ ] What value is passed to the pipeline?
- [ ] Does the pipeline receive a normalized path?
- [ ] What transform normalizes the path?
- [ ] What is the path mapped to in LoadPreExtractedT5Feature?
- [ ] What happens if the .pt file doesn't exist?
- [ ] How does refetch work?
- [ ] Why use relative paths in annotation?
- [ ] What are the final output keys?
- [ ] How would you debug a path issue?

---

## 📞 Quick Reference Table

| What I Need | Where To Look |
|------------|----------------|
| Quick answer | QUICK_REFERENCE.md |
| Full explanation | SUMMARY.md |
| Deep dive | TRACE.md |
| Code locations | CODE_REFERENCES.md |
| Visual diagram | FLOW.txt |
| Specific file | Use grep in source files |
| Line numbers | CODE_REFERENCES.md |
| Debug logging | CODE_REFERENCES.md Section 8 |
| Common issues | QUICK_REFERENCE.md Common Issues |
| Configuration | QUICK_REFERENCE.md Configuration |
| Architecture | TRACE.md Architectural Insights |

---

## 📝 Notes

- All paths are relative to the project root (`/apdcephfs/.../hf_trainer/`)
- Annotation file is loaded via `mmengine.load()`
- Dataset uses mmengine BaseDataset as parent
- Transform uses mmcv BaseTransform as parent
- T5 features are pre-extracted to disk (not computed during training)

---

## 🔗 Cross-References

**Within Documentation:**
- All documents reference each other with "See: [DOCUMENT_NAME] Section [X]"
- Use Ctrl+F to search within documents

**In Source Code:**
- Comments in source code reference line numbers in this documentation
- See CODE_REFERENCES.md for exact line numbers

---

**Last Updated:** 2026-05-26
**Status:** Complete
**Version:** 1.0
