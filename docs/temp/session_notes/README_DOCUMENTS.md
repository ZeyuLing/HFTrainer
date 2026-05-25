# NPZ to SMPL Mesh JSON Conversion Pipeline — Documentation Index

## 📚 Document Suite Generated: 2026-05-25

This suite provides complete understanding of the motion capture to 3D web visualization conversion pipeline.

---

## 📖 Documents Overview

### 1. **UNDERSTANDING_SUMMARY.md** — START HERE
- **Purpose:** Comprehensive technical overview
- **Content:** API signatures, formats, data availability, algorithms, examples
- **Best for:** Understanding the complete pipeline
- **Length:** ~15 pages
- **Key sections:**
  - Executive summary (key facts)
  - Exact API/function signatures
  - Input format (motion_135 NPZ)
  - Output format (SMPL mesh JSON)
  - Conversion algorithm (step-by-step)
  - Usage examples
  - Troubleshooting

### 2. **NPZ_TO_SMPL_MESH_JSON_CONVERSION_PIPELINE.md** — REFERENCE
- **Purpose:** Complete specification document
- **Content:** Detailed breakdown of every aspect
- **Best for:** Reference during implementation
- **Length:** ~30 pages
- **Key sections:**
  - Script locations
  - Function signatures with context
  - NPZ format specification
  - JSON output schema
  - Data availability inventory
  - Conversion pipeline walkthrough
  - Key conversion details
  - Performance characteristics
  - Web viewer integration

### 3. **NPZ_TO_SMPL_CONVERSION_QUICK_REFERENCE.txt** — CHEAT SHEET
- **Purpose:** Quick lookup reference
- **Content:** Key facts, commands, APIs in compact form
- **Best for:** Quick facts during development
- **Length:** ~1 page (structured)
- **Includes:**
  - Function signatures
  - Input/output schemas
  - Rotation conversion pipeline (visual)
  - Data inventory with counts
  - Command examples
  - Performance metrics
  - Validation checklist

### 4. **CONVERSION_FLOW_DIAGRAM.txt** — VISUAL GUIDE
- **Purpose:** Visual representation of data flow
- **Content:** ASCII diagrams showing transformation steps
- **Best for:** Understanding the data transformation process
- **Length:** ~3 pages
- **Shows:**
  - Input file structure
  - Complete conversion pipeline (step-by-step)
  - Output file structure
  - SMPL type comparison table
  - Example command execution
  - Performance profile

### 5. **NPZ_FILES_INVENTORY.txt** — DATA LISTING
- **Purpose:** Complete list of available NPZ files
- **Content:** All 76 NPZ filenames in `output/physflow_v2_compare_iter1000/npz/`
- **Best for:** Finding specific motion files
- **Format:** Plain text list
- **Contains:**
  - All 76 filenames
  - Naming convention explanation
  - Prefix/suffix combinations

---

## 🎯 Quick Start Guide

### For First-Time Understanding
1. Read: **UNDERSTANDING_SUMMARY.md** (Sections 1-3)
2. View: **CONVERSION_FLOW_DIAGRAM.txt** (full)
3. Skim: **NPZ_TO_SMPL_CONVERSION_QUICK_REFERENCE.txt**

### For Implementation
1. Reference: **NPZ_TO_SMPL_MESH_JSON_CONVERSION_PIPELINE.md**
2. Copy: **NPZ_TO_SMPL_CONVERSION_QUICK_REFERENCE.txt** (API section)
3. Test: Command examples from **UNDERSTANDING_SUMMARY.md** (Section 8)

### For Troubleshooting
1. Check: **UNDERSTANDING_SUMMARY.md** (Section 11)
2. Reference: **NPZ_TO_SMPL_MESH_JSON_CONVERSION_PIPELINE.md** (Section 10)

### For Quick Facts
- Refer to: **NPZ_TO_SMPL_CONVERSION_QUICK_REFERENCE.txt**
- Check data: **NPZ_FILES_INVENTORY.txt**

---

## 🔑 Key Facts (TL;DR)

**Script Location:**
```
scripts/embodied/batch_npz_to_smpl_mesh_json.py
```

**Function Signature:**
```python
def convert_single_npz(
    npz_path: str,
    smpl_type: str = "smplx",    # "smpl", "smplh", "smplx"
    gender: str = "neutral"       # "neutral", "male", "female"
) -> dict
```

**Input Format:**
```
motion_135.npz
├── motion_135: (T, 135) float32  [translation(3) + rot6d(22×6)]
├── fps: int (30)
└── prompt: str
```

**Output Format:**
```
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [{
      "id": 0,
      "gender": "neutral",
      "smpl_type": "smplh",
      "Rh": [[rx, ry, rz]],
      "Th": [[tx, ty, tz]],
      "poses": [[p0, p1, ..., pN]],
      "shapes": [[0, 0, ..., 0]],
      "mocap_framerate": 30
    }],
    ...
  ]
}
```

**Data Available:**
```
output/physflow_v2_compare_iter1000/npz/
├── 76 NPZ files
├── 3.6 MB total
├── 19 motion types
├── 2 model variants (pretrained, finetuned)
└── 2 suffixes per type (raw, rl)
```

**Command:**
```bash
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
  --npz-dir output/physflow_v2_compare_iter1000/npz \
  --output-dir output/physflow_v2_compare_iter1000/smpl_mesh \
  --smpl-type smplh \
  --gender neutral
```

**Performance:**
- Per-file: ~100 ms
- Batch (76 files): ~5-10 seconds
- Output: ~425 KB per 120 frames @ 30fps

---

## 📊 Document Comparison Matrix

| Aspect | Summary | Reference | Quick Ref | Diagrams | Inventory |
|--------|---------|-----------|-----------|----------|-----------|
| APIs | ✅ | ✅✅ | ✅ | — | — |
| Formats | ✅✅ | ✅✅ | ✅ | ✅ | — |
| Examples | ✅✅ | ✅ | — | ✅ | — |
| Details | ✅ | ✅✅ | — | — | — |
| Quick Facts | ✅ | — | ✅✅ | — | — |
| Troubleshooting | ✅ | ✅ | — | — | — |
| Data List | — | — | — | — | ✅✅ |
| Visual Flow | — | — | — | ✅✅ | — |

---

## 🔗 Cross-References

### All documents discuss:
- **Input:** motion_135 NPZ format
- **Output:** SMPL mesh JSON format
- **Script:** batch_npz_to_smpl_mesh_json.py
- **Data:** output/physflow_v2_compare_iter1000/npz/ (76 files)

### Each document specializes in:
- **UNDERSTANDING_SUMMARY:** Complete walkthrough + examples
- **REFERENCE:** Specification + deep details
- **QUICK_REFERENCE:** Facts + commands
- **DIAGRAMS:** Data flow + visual representation
- **INVENTORY:** File listing

---

## 📋 Information Checklist

Use this checklist to ensure you have all needed information:

### Understanding the Pipeline
- [ ] Read UNDERSTANDING_SUMMARY.md Sections 1-3 (APIs and formats)
- [ ] Review CONVERSION_FLOW_DIAGRAM.txt (visual understanding)
- [ ] Check Quick Reference for command syntax

### Implementation Details
- [ ] Understand rot6d→axis-angle conversion (CONVERSION_FLOW_DIAGRAM)
- [ ] Review SMPL type comparison table (Quick Reference)
- [ ] Note zero-padding strategy (UNDERSTANDING_SUMMARY Section 7)

### Accessing Data
- [ ] Locate NPZ directory: `output/physflow_v2_compare_iter1000/npz/`
- [ ] Find specific motion: Check NPZ_FILES_INVENTORY.txt
- [ ] Verify file count: 76 files (3.6 MB)

### Running Conversion
- [ ] Copy command from UNDERSTANDING_SUMMARY Section 8
- [ ] Verify output directory exists or will be created
- [ ] Monitor progress output
- [ ] Validate JSON output (checklist in UNDERSTANDING_SUMMARY)

### Web Integration
- [ ] Understand JSON format (UNDERSTANDING_SUMMARY Section 3)
- [ ] Review Three.js integration (UNDERSTANDING_SUMMARY Section 10)
- [ ] Check shape coefficients (always zeros)

### Troubleshooting
- [ ] Review validation checklist (UNDERSTANDING_SUMMARY Section 11)
- [ ] Check common errors and fixes
- [ ] Verify input NPZ format matches spec

---

## 📞 Document Usage Guide

**"I need to understand what this does"**
→ Read UNDERSTANDING_SUMMARY.md Section 1-2

**"I need to run the converter"**
→ See UNDERSTANDING_SUMMARY.md Section 8, then QUICK_REFERENCE for exact syntax

**"I need the technical details"**
→ Reference NPZ_TO_SMPL_MESH_JSON_CONVERSION_PIPELINE.md

**"I need a quick reminder of the API"**
→ Check QUICK_REFERENCE.txt

**"I need to understand the data transformation"**
→ View CONVERSION_FLOW_DIAGRAM.txt

**"I need to find a specific NPZ file"**
→ Search NPZ_FILES_INVENTORY.txt

**"Something isn't working"**
→ See UNDERSTANDING_SUMMARY.md Section 11 (Troubleshooting)

**"I need to integrate with a web viewer"**
→ Read UNDERSTANDING_SUMMARY.md Section 10

---

## 📝 Document Metadata

| Document | Pages | Format | Updated | Status |
|----------|-------|--------|---------|--------|
| UNDERSTANDING_SUMMARY.md | ~15 | Markdown | 2026-05-25 | ✅ Complete |
| REFERENCE.md | ~30 | Markdown | 2026-05-25 | ✅ Complete |
| QUICK_REFERENCE.txt | 1 | Plain text | 2026-05-25 | ✅ Complete |
| FLOW_DIAGRAM.txt | 3 | ASCII | 2026-05-25 | ✅ Complete |
| INVENTORY.txt | — | Plain text | 2026-05-25 | ✅ Complete |

---

## 🎓 Learning Path

### Beginner
1. UNDERSTANDING_SUMMARY.md (Sections 1-3)
2. CONVERSION_FLOW_DIAGRAM.txt
3. QUICK_REFERENCE.txt (overview section)

### Intermediate
4. UNDERSTANDING_SUMMARY.md (Sections 6-8)
5. QUICK_REFERENCE.txt (API section)
6. NPZ_FILES_INVENTORY.txt

### Advanced
7. REFERENCE.md (complete)
8. UNDERSTANDING_SUMMARY.md (Sections 9-12)
9. Trace through actual code: scripts/embodied/batch_npz_to_smpl_mesh_json.py

---

## ✨ Key Insights

1. **Simple Input Format:** motion_135 = [translation(3) + rot6d(22×6)]
2. **Smart Rotation Conversion:** Row-major rot6d → column-major → Gram-Schmidt → axis-angle
3. **Flexible Output:** Choose SMPL type (72/156/165 dims) and gender
4. **Web-Ready:** JSON format matches Three.js SkinnedMesh expectations
5. **Zero-Padded:** Hands/face deformation stays at zero (no hand/face data in motion_135)
6. **Performance:** ~100ms per file, no GPU needed
7. **Data Available:** 76 pre-converted motions ready for conversion

---

## 🔄 Document Flow

```
README (you are here)
   ↓
Choose your path:
   ├→ "I want to understand" → UNDERSTANDING_SUMMARY.md
   ├→ "I need details" → REFERENCE.md
   ├→ "I need quick facts" → QUICK_REFERENCE.txt
   ├→ "Show me visually" → FLOW_DIAGRAM.txt
   └→ "Where's the data?" → INVENTORY.txt
   
Then proceed to:
   → scripts/embodied/batch_npz_to_smpl_mesh_json.py (actual code)
```

---

## 📞 Support Resources

**Within this suite:**
- API reference: QUICK_REFERENCE.txt (section: PRIMARY FUNCTIONS)
- Format specs: REFERENCE.md (section: SMPL MESH JSON OUTPUT FORMAT)
- Examples: UNDERSTANDING_SUMMARY.md (section: USAGE EXAMPLES)
- Errors: UNDERSTANDING_SUMMARY.md (section: TROUBLESHOOTING)

**In codebase:**
- Main script: scripts/embodied/batch_npz_to_smpl_mesh_json.py
- Joints converter: scripts/embodied/batch_npz_to_smpl_joints.py
- Related: scripts/embodied/motion135_to_smplx.py

---

**Generated:** 2026-05-25  
**Purpose:** Complete documentation of NPZ→SMPL conversion pipeline  
**Audience:** Developers, researchers, system integrators
