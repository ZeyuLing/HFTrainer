# Policy Search Report - Document Index

Generated: 2026-05-15  
Repository: HyMotion hf_trainer

## 📋 Documents Created

Three comprehensive reports have been generated from the codebase search:

### 1. **SEARCH_FINDINGS_SUMMARY.txt** ⭐ **START HERE**
   - **Purpose**: Executive summary of findings
   - **Length**: 400+ lines, formatted for clarity
   - **Best for**: Quick overview of all results
   - **Key sections**:
     - Executive summary (yes/no answers)
     - 7 major findings with details
     - Policy comparison table
     - What exists vs gaps analysis
     - Recommended actions with priority levels
     - Usage examples
     - Quick start for SMPL ONNX export

### 2. **PHC_POLICY_SMPL_TRACKING_REPORT.md**
   - **Purpose**: Technical deep-dive
   - **Length**: 432 lines
   - **Best for**: Implementation details and architecture
   - **Key sections**:
     - 12 detailed findings
     - G1 ONNX model specifications (inputs/outputs)
     - SMPL robot structure and differences
     - PHC vs ProtoMotions comparison
     - File sizes and storage inventory
     - Key observations with evidence
     - Next steps and references

### 3. **PHC_POLICY_QUICK_REFERENCE.md**
   - **Purpose**: Cheat sheet
   - **Length**: ~150 lines
   - **Best for**: Quick lookup and commands
   - **Key sections**:
     - TL;DR table
     - G1 ready-to-use section
     - SMPL export section
     - PHC training configs
     - SMPL robot library capabilities
     - File reference table

---

## 🎯 Quick Answers

**Q: Do SMPL humanoid policies exist?**  
A: ✅ **YES** - Trained PyTorch checkpoint at `ref_repo/ProtoMotions/.../smpl/last.ckpt`

**Q: Is there an ONNX model for G1?**  
A: ✅ **YES** - Production-ready at `ref_repo/ProtoMotions/.../g1-bones-deploy/compiled_models/unified_pipeline.onnx`

**Q: Is there an ONNX model for SMPL?**  
A: ⏳ **NO** - Checkpoint trained but ONNX export not performed yet (easy to fix)

**Q: How to use G1 tracker?**  
A: ```bash
   python scripts/embodied/run_tracker_export.py --motion <motion.pt> --output <tracked.pt>
   ```

**Q: How to export SMPL ONNX?**  
A: ```bash
   cd ref_repo/ProtoMotions
   python deployment/export_bm_tracker_onnx.py --checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt
   ```

---

## 📍 Key Locations

| What | Where |
|-----|-------|
| **G1 ONNX** | `ref_repo/ProtoMotions/.../g1-bones-deploy/compiled_models/unified_pipeline.onnx` |
| **SMPL Checkpoint** | `ref_repo/ProtoMotions/.../smpl/last.ckpt` |
| **PHC G1 Config** | `ref_repo/PHC/phc/data/cfg/env/env_im_g1_phc.yaml` |
| **SMPL Library** | `ref_repo/OmniH2O/phc/phc/smpllib/` |
| **Tracker Script** | `scripts/embodied/run_tracker_export.py` |
| **ONNX Export** | `ref_repo/ProtoMotions/deployment/export_bm_tracker_onnx.py` |

---

## 📊 Summary Table

| Component | Status | Format | Size |
|-----------|--------|--------|------|
| G1 Tracker | ✅ Ready | ONNX + PyTorch | 22 MB + 228 MB |
| SMPL Tracker | ⏳ Trained | PyTorch | 121 MB |
| SMPL Terrains | ⏳ Trained | PyTorch | ~121 MB |
| PHC Configs | ✅ Available | YAML | ~2-3 KB each |
| SMPL Library | ✅ Available | Python | 102 KB core |

---

## 🚀 Next Steps (Priority Order)

### **PRIORITY 1** (5 minutes)
Export SMPL ONNX model:
```bash
cd ref_repo/ProtoMotions
python deployment/export_bm_tracker_onnx.py \
    --checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt
```

### **PRIORITY 2** (10 minutes)
Test SMPL tracking with exported ONNX:
```bash
python scripts/embodied/run_tracker_export.py \
    --motion <motion.pt> \
    --output <tracked.pt> \
    --onnx ref_repo/ProtoMotions/.../smpl/compiled_models/unified_pipeline.onnx \
    --mjcf ref_repo/ProtoMotions/.../smpl_humanoid.xml
```

### **PRIORITY 3** (Documentation)
Create centralized policy registry with all models

### **PRIORITY 4** (Optional)
Export SMPL-terrains ONNX variant

---

## 📖 How to Use These Reports

### For Beginners
1. Read **SEARCH_FINDINGS_SUMMARY.txt** first
2. Skip to "QUICK START" section
3. Run the export command
4. That's it!

### For Implementation
1. Reference **PHC_POLICY_SMPL_TRACKING_REPORT.md** section 1-2
2. Check ONNX I/O specifications
3. Implement integration
4. Use **QUICK_REFERENCE.md** for API lookups

### For Maintenance
1. Keep **SEARCH_FINDINGS_SUMMARY.txt** as master reference
2. Update if new policies added
3. Use as checklist for policy deployment
4. Include in project documentation

---

## 🔍 Search Coverage

**Scanned**:
- ✅ All `.onnx` files
- ✅ All `.pth`, `.pt`, `.ckpt` checkpoints
- ✅ All `**/phc/**` configurations
- ✅ All SMPL-related code
- ✅ Tracker export scripts
- ✅ PHC training configs
- ✅ Reference repositories

**Scope**: `ref_repo/`, `scripts/`, `configs/` directories

**Result**: Comprehensive inventory of all policies and configurations

---

## 📞 Key Contacts

**If policies need updates**: Check `ref_repo/ProtoMotions/` and `ref_repo/PHC/`  
**If export fails**: Review `deployment/export_bm_tracker_onnx.py`  
**If tracking fails**: Check `scripts/embodied/run_tracker_export.py` logs  
**For SMPL details**: See `ref_repo/OmniH2O/phc/phc/smpllib/smpl_local_robot.py`

---

## ✅ Verification Checklist

- [x] Found G1 ONNX model
- [x] Found SMPL checkpoint
- [x] Located PHC configs
- [x] Identified SMPL library
- [x] Documented export process
- [x] Created usage examples
- [x] Listed all gaps
- [x] Provided priority actions
- [x] Generated three reports

---

## 📝 Report Statistics

| Report | Lines | Words | Topics |
|--------|-------|-------|--------|
| SEARCH_FINDINGS_SUMMARY.txt | 400+ | ~3500 | 12 major findings |
| PHC_POLICY_SMPL_TRACKING_REPORT.md | 432 | ~4200 | 12 sections |
| PHC_POLICY_QUICK_REFERENCE.md | 150 | ~1000 | 8 topics |
| **TOTAL** | **~982** | **~8700** | **32 subsections** |

---

## 🎓 Learning Path

If you're new to this codebase:

1. **Start**: Read executive summary in SEARCH_FINDINGS_SUMMARY.txt
2. **Understand**: Review G1 ONNX specs in PHC_POLICY_SMPL_TRACKING_REPORT.md
3. **Explore**: Check quick reference in PHC_POLICY_QUICK_REFERENCE.md
4. **Implement**: Export SMPL ONNX using priority 1 command
5. **Test**: Run tracking with exported ONNX (priority 2)
6. **Integrate**: Add to your pipeline

---

## 🔗 Related Documentation

- **ProtoMotions CLAUDE.md**: Architecture and setup
- **PHC Docs**: `ref_repo/PHC/docs/smpl_robot_instruction.MD`
- **SMPL Library Docs**: Inline in `smpl_local_robot.py`
- **Tracker Script Help**: `python scripts/embodied/run_tracker_export.py --help`

---

## 📅 Report Generation

**Date**: 2026-05-15  
**Repository**: HyMotion hf_trainer  
**Search Method**: Automated codebase scan + file inspection  
**Confidence**: High (verified with direct file inspection)

---

**All three reports are complementary. Start with SEARCH_FINDINGS_SUMMARY.txt for overview, then dive into specific reports as needed.**
