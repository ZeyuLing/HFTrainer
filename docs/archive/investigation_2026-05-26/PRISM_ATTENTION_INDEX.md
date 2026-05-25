# PRISM Transformer Attention Architecture - Document Index

This document index helps you navigate the comprehensive analysis of how the PRISM transformer model constructs its attention blocks with FP32 upcast support.

---

## 📄 Available Documents

### 1. **PRISM_QUICK_SUMMARY.txt** ⭐ START HERE
**Best for**: Quick reference and high-level understanding
**Length**: ~200 lines
**Contains**:
- Direct answers to all 5 key questions
- Parameter flow diagram
- Key file and line number references
- Verification checklist
- ASCII formatted for easy reading

**When to read this**: If you want a quick overview in 5 minutes

---

### 2. **PRISM_ATTENTION_TRACE.md**
**Best for**: Understanding the complete architecture and data flow
**Length**: ~400 lines
**Contains**:
- Full config inheritance chain with line-by-line analysis
- Model initialization details
- Block implementation code snippets
- FP32 upcast processor explanation
- Summary table linking all components
- Architecture diagram
- Answers to specific questions with evidence

**When to read this**: If you need to understand the complete flow

---

### 3. **PRISM_ATTENTION_CODE_REFS.md**
**Best for**: Detailed code references and copy-paste ready snippets
**Length**: ~350 lines
**Contains**:
- Critical file locations with exact line numbers
- Full code snippets with line annotations
- Complete config inheritance chain with references
- Data flow diagram with all component locations
- Summary table of where parameter is used
- Testing instructions
- Precision handling explanation in v3 config

**When to read this**: If you need to find specific code or understand precision handling

---

## 🎯 Quick Navigation by Question

### Q: How does PrismTransformerMotionModel construct transformer blocks?
- **Quick Answer**: See PRISM_QUICK_SUMMARY.txt, "QUESTION 1"
- **Detailed Answer**: See PRISM_ATTENTION_TRACE.md, "Part 2", lines 196-209
- **Code**: PRISM_ATTENTION_CODE_REFS.md, "Section 2", lines 195-209

### Q: Does it use WanTransformerBlockWithMask? Does it pass use_fp32_upcast_attention?
- **Quick Answer**: See PRISM_QUICK_SUMMARY.txt, "QUESTION 2"
- **Detailed Answer**: See PRISM_ATTENTION_TRACE.md, "Part 2", line 205
- **Code**: PRISM_ATTENTION_CODE_REFS.md, "Section 2"

### Q: What is the full config inheritance chain?
- **Quick Answer**: See PRISM_QUICK_SUMMARY.txt, "QUESTION 3"
- **Detailed Answer**: See PRISM_ATTENTION_TRACE.md, "Part 1"
- **Code**: PRISM_ATTENTION_CODE_REFS.md, "Complete Config Inheritance Chain"

### Q: Does config rely on default True or is it explicit?
- **Quick Answer**: See PRISM_QUICK_SUMMARY.txt, "QUESTION 4"
- **Detailed Answer**: See PRISM_ATTENTION_TRACE.md, Part 1, "prism_1b_tp2m_1frame.py"
- **Code**: PRISM_ATTENTION_CODE_REFS.md, "Section 1", line 34

### Q: How are WanTransformerBlockWithMask instances created?
- **Quick Answer**: See PRISM_QUICK_SUMMARY.txt, "QUESTION 5"
- **Detailed Answer**: See PRISM_ATTENTION_TRACE.md, "Part 2", lines 195-209
- **Code**: PRISM_ATTENTION_CODE_REFS.md, "Section 2"

---

## 🔍 Key Files Referenced

### Configuration Files
- `configs/prism/prism_1b_tp2m_1frame.py` (line 34) - **PRIMARY DEFINITION**
- `configs/prism/prism_1b_tp2m_multiframe.py` (line 9)
- `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py` (line 12)
- `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py` (line 15)
- `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_v3.py` (line 27)

### Implementation Files
- `hftrainer/models/motion/prism/network/transformer_prism.py` (lines 133-222)
- `hftrainer/models/motion/prism/network/block_with_mask.py` (lines 76-146)
- `hftrainer/models/motion/prism/network/attention_fp32_upcast.py` (lines 37-260)

---

## 📊 Critical Line Numbers at a Glance

| Component | File | Line(s) | What |
|-----------|------|---------|------|
| Config Definition | prism_1b_tp2m_1frame.py | 34 | `use_fp32_upcast_attention=True` |
| Model Parameter | transformer_prism.py | 154 | Default parameter definition |
| Block Construction | transformer_prism.py | 205 | Parameter passed to blocks |
| Block Parameter | block_with_mask.py | 85 | Default parameter definition |
| Self-Attn Processor | block_with_mask.py | 97-99 | Conditional selection |
| Cross-Attn Processor | block_with_mask.py | 113-115 | Conditional selection |
| Upcast Decision | attention_fp32_upcast.py | 104-118 | Whether to upcast |
| Upcast Execution | attention_fp32_upcast.py | 209-228 | Q, K, V conversion |

---

## 🏗️ Architecture Overview

```
Configuration Layer
    ↓
    use_fp32_upcast_attention=True (explicit in config)
    ↓
Model Layer (PrismTransformerMotionModel)
    ↓
    Creates 30 blocks with parameter passed to each
    ↓
Block Layer (WanTransformerBlockWithMask)
    ↓
    Conditional processor selection
    - With FP32 upcast: WanAttnProcessorFP32Upcast
    - Without: WanAttnProcessor
    ↓
Processor Layer (WanAttnProcessorFP32Upcast)
    ↓
    During forward pass:
    - Detect if upcast needed (fp16/bf16 or autocast context)
    - Upcast Q, K, V, mask to fp32
    - Disable autocast to preserve fp32
    - Run attention in fp32
    - Cast output back to original dtype
```

---

## ✅ Key Findings Summary

1. **Explicit Configuration**: `use_fp32_upcast_attention=True` is **explicitly set** in the base config file (line 34 of prism_1b_tp2m_1frame.py), not relying on code defaults.

2. **Direct Parameter Passing**: The parameter is **directly passed** to all 30 blocks (transformer_prism.py, line 205), not wrapped in kwargs or config dicts.

3. **Both Attention Types**: The setting affects **both self-attention and cross-attention** (block_with_mask.py, lines 97-99 and 113-115).

4. **Full Inheritance**: All configs in the chain (v3, v2, unified, multiframe, 1frame) inherit the setting because none override it.

5. **Softmax Precision**: When enabled, **all 30 blocks use fp32 softmax** (attention_fp32_upcast.py, lines 204-229) to prevent numerical overflow in mixed-precision training.

---

## 🔧 Using This Information

### To Verify the Setting
```python
model = build_model(config)
block = model.transformer.blocks[0]
processor_type = type(block.attn1.processor).__name__
print(processor_type)  # Should be "WanAttnProcessorFP32Upcast" if enabled
```

### To Modify the Setting
Edit `configs/prism/prism_1b_tp2m_1frame.py`, line 34:
```python
use_fp32_upcast_attention=False,  # Disable FP32 upcast
# or
use_fp32_upcast_attention=True,   # Enable FP32 upcast (default)
```

### To Understand Precision Handling
See: PRISM_ATTENTION_CODE_REFS.md, "Precision Handling in v3 Config" section

---

## 📚 Related Documentation

- `DTYPE_CONFIGURATION_SUMMARY.md` - Overall dtype configuration strategy
- `DTYPE_QUICK_REFERENCE.md` - Quick reference for all dtype settings
- `PRISM_V3_FIX_STATUS.txt` - Status of v3 config fixes

---

## 📝 Document Version Info

- **Created**: 2026-05-26
- **PRISM Version**: Based on transformer_prism.py from the repository
- **Config Version**: v3 (traditional AMP with DDP)
- **Completeness**: All 5 questions answered with code references

---

## 🤔 FAQ

**Q: Is use_fp32_upcast_attention enabled for my config?**
A: Check the inheritance chain starting from your config file. If it doesn't override the setting, it inherits `True` from prism_1b_tp2m_1frame.py line 34.

**Q: Does this affect performance?**
A: Minimal overhead. The upcast only happens for attention softmax computation (not all forward passes). For fp32 models, it's a no-op.

**Q: Can I disable it?**
A: Yes, set `use_fp32_upcast_attention=False` in your config. Then blocks will use standard `WanAttnProcessor` without fp32 upcast.

**Q: Why is this needed?**
A: In fp16 training, attention scores can overflow (exp x > 11.09 overflows in fp16). The fp32 upcast prevents this by running softmax in fp32.

**Q: Does it apply to both attention types?**
A: Yes, both self-attention (hidden_states only) and cross-attention (hidden_states + encoder_hidden_states).

---

**For more details, refer to the specific document that best matches your need!**
