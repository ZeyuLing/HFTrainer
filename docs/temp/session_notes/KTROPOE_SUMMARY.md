# MotionWanRotaryPosEmbed KT-RoPE Integration - Executive Summary

## Overview

I have analyzed the PRISM transformer codebase and created comprehensive documentation on how `MotionWanRotaryPosEmbed` is instantiated and how to pass new KT-RoPE parameters through the configuration system.

## Key Findings

### Current State

1. **RoPE Instantiation Location**
   - **File**: `hftrainer/models/motion/prism/network/transformer_prism.py`
   - **Lines**: 164-166 in `__init__` method
   - **Current Call**:
     ```python
     self.rope = MotionWanRotaryPosEmbed(
         attention_head_dim, patch_size, rope_max_seq_len
     )
     ```

2. **Currently Configurable Parameters**
   - `attention_head_dim`: 128 (split: 64 temporal, 64 spatial)
   - `patch_size`: (1, 1) for temporal and spatial patching
   - `rope_max_seq_len`: 1024 for pre-computing maximum sequence

3. **Critical Issue: Hardcoded `theta` Parameter**
   - Currently hardcoded to `10000.0` in `motion_rope.py` line 74
   - **NOT** configurable through config system
   - Must be made configurable for KT-RoPE support

### RoPE Implementation Details

**File**: `hftrainer/models/motion/prism/network/motion_rope.py`

- **Dimension Split** (lines 84-85):
  - `t_dim = 64` (temporal dimension)
  - `j_dim = 64` (spatial/joint dimension)
  
- **Pre-computation** (lines 96-106):
  - Uses `get_1d_rotary_pos_embed()` from diffusers library
  - Computes separate 1D RoPE for temporal and joint axes
  - Concatenates results for 2D positional encoding

- **Forward Output** (line 179):
  - Returns `(freqs_cos, freqs_sin)` with shape `(1, N, 1, 128)`
  - N = (num_frames // p_t) * (num_joints // p_j)
  - Passed to attention layers in 30 transformer blocks

### Configuration System

**Primary Config**: `configs/prism/prism_1b_tp2m_1frame.py`

- Contains transformer parameters (lines 19-42)
- Currently includes: `attention_head_dim`, `patch_size`, `rope_max_seq_len`
- **Does NOT include**: `theta`, `joint_pos_mode`, `num_spectral_modes`, `spectral_scale`

**Config Flow**:
```
Config file 
  ↓
Registry.build_from_cfg()
  ↓
PrismTransformerMotionModel.__init__(**config_dict)
  ↓
self.rope = MotionWanRotaryPosEmbed(...)
```

### Training Checkpoints

- **Main training**: `work_dirs/prism_1b_tp2m_1frame/checkpoint-iter_11000/`
- **Multi-frame**: `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000/`
- **MCM training loads from**: `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000`
- **Resume command**: `bash tools/taiji_dist_train.sh configs/prism/prism_1b_tp2m_1frame.py --auto-resume`

---

## Implementation Roadmap

### 4-Step Process to Add KT-RoPE Parameters

#### Step 1: Update Config File (1 location)
**File**: `configs/prism/prism_1b_tp2m_1frame.py`
- Add 4 parameters after line 35
- Parameters: `rope_theta`, `joint_pos_mode`, `num_spectral_modes`, `spectral_scale`

#### Step 2: Update Transformer `__init__` (2 changes)
**File**: `hftrainer/models/motion/prism/network/transformer_prism.py`

*Change 2a* - Update `__init__` signature (lines 132-149):
- Add 4 new optional parameters with defaults

*Change 2b* - Update RoPE instantiation (lines 164-166):
- Pass all 4 new parameters to `MotionWanRotaryPosEmbed()`
- Use named arguments instead of positional

#### Step 3: Update RoPE Module `__init__` (1 change)
**File**: `hftrainer/models/motion/prism/network/motion_rope.py`

- Update `__init__` signature (lines 69-75) to accept new parameters
- Already uses `theta` parameter, just needs to be accepted
- Store `joint_pos_mode`, `num_spectral_modes`, `spectral_scale` as instance variables

#### Step 4: Implement KT-RoPE Logic (1 change)
**File**: `hftrainer/models/motion/prism/network/motion_rope.py`

- In `forward()` method (after line 178)
- Add conditional logic based on `joint_pos_mode`:
  - `"sequential"`: Default (no modification)
  - `"spectral"`: Apply Laplacian eigenvector decomposition
  - `"dfs"`: Apply depth-first search traversal ordering

---

## Documentation Files Created

I have created three comprehensive documentation files in your working directory:

### 1. `ktropoe_instantiation_report.md` (13 KB)
**Comprehensive analysis covering**:
- Exact instantiation lines and parameters
- How RoPE is used in forward pass
- Config file structure and hierarchy
- Config parameter flow diagram
- Quick reference table for modifications
- Related architecture components
- Testing locations
- Summary of implementation steps

### 2. `ktropoe_config_flow.txt` (22 KB)
**Visual flow diagram showing**:
- Step-by-step parameter flow from config to RoPE
- Box diagrams for each step
- Parameter summary (existing vs. to-add)
- File-by-file modification list
- Visual representation of data transformation

### 3. `ktropoe_implementation_snippets.md` (12 KB)
**Ready-to-use code snippets for**:
- Config file changes (exact lines)
- Transformer model changes (2 distinct changes)
- RoPE module changes (4 changes)
- Testing code for unit and integration tests
- Verification script
- Troubleshooting guide
- Implementation next steps

---

## Critical Technical Details

### Parameter Propagation Path
```
transformer.rope_theta (config)
  ↓
PrismTransformerMotionModel.rope_theta (instance var)
  ↓
MotionWanRotaryPosEmbed.__init__(theta=...)
  ↓
get_1d_rotary_pos_embed(dim, max_seq_len, theta, ...)
  ↓
self.register_buffer("freqs_cos", ...)
self.register_buffer("freqs_sin", ...)
  ↓
forward(hidden_states) → (freqs_cos, freqs_sin)
```

### Head Dimension Split Logic
```python
# For attention_head_dim=128
j_dim = 128 // 2 = 64        # Spatial/joint frequencies
t_dim = 128 - 64 = 64        # Temporal frequencies
# Result: Each dimension gets 64 dims of positional encoding
```

### Sequence Length Calculation
```python
num_frames = 64, num_joints = 22, patch_size = (1, 1)
ppf = 64 // 1 = 64           # Patches per frame dimension
ppj = 22 // 1 = 22           # Patches per joint dimension
N = 64 * 22 = 1408           # Total sequence length
Output shape: (1, 1408, 1, 128)
```

---

## New Parameters to Add

| Param | Type | Default | Purpose |
|-------|------|---------|---------|
| `rope_theta` | float | 10000.0 | Base frequency for RoPE rotations |
| `joint_pos_mode` | str | "sequential" | Position encoding scheme |
| `num_spectral_modes` | int | 4 | Laplacian eigenvector count |
| `spectral_scale` | Optional[int] | None | Scaling for spectral coordinates |

---

## Integration Points

### Where RoPE is Used
- **30 transformer blocks** (line 186-199)
- Each block uses rotary embeddings in attention layers
- Passed via `torch.utils.checkpoint.checkpoint()` for memory efficiency
- Also works with `hidden_states_mask` for variable-length sequences
- Compatible with `causal_mask` for frame-level causality

### Backward Compatibility
- Default values preserve current behavior
- `rope_theta=10000.0` matches existing hardcoded value
- `joint_pos_mode="sequential"` produces standard RoPE
- Existing training can resume without config changes

---

## Verification Checklist

After implementation, verify:
- [ ] Config file loads without errors
- [ ] `PrismTransformerMotionModel` accepts new parameters
- [ ] `MotionWanRotaryPosEmbed` receives correct parameters
- [ ] Unit tests pass (motion_rope.py)
- [ ] Integration tests pass (transformer_prism.py)
- [ ] Forward pass produces correct output shapes
- [ ] Parameters stored in `model.config` object
- [ ] Parameters accessible in RoPE module via `self.rope`
- [ ] Training resumption works (`--auto-resume`)

---

## Next Steps for Implementation

1. **Read the detailed documentation**
   - Start with `ktropoe_instantiation_report.md` for understanding
   - Use `ktropoe_config_flow.txt` as reference during coding
   - Follow `ktropoe_implementation_snippets.md` for exact changes

2. **Apply changes** (following the 4-step process)
   - Each step is independent and can be verified
   - Make changes in order: config → transformer → rope module → logic

3. **Test thoroughly**
   - Run unit tests: `python motion_rope.py`
   - Run integration tests: `python transformer_prism.py`
   - Run verification: `python test_ktropoe_params.py`

4. **Resume training** with new parameters
   - Update config file with desired KT-RoPE settings
   - Use `--auto-resume` to continue from checkpoint

5. **Implement full KT-RoPE logic** (if needed)
   - Spectral mode: Implement Laplacian eigenvector computation
   - DFS mode: Implement kinematic tree traversal
   - These are optional optimizations for improved performance

---

## Files in Your Directory

Located at: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/`

- `ktropoe_instantiation_report.md` - Comprehensive analysis
- `ktropoe_config_flow.txt` - Visual flow diagrams
- `ktropoe_implementation_snippets.md` - Ready-to-use code
- `KTROPOE_SUMMARY.md` - This file

---

## Questions & Debugging

### Common Issues and Solutions

**Q: How do I know if parameters are being passed correctly?**
A: Create a test script that imports the model and checks:
```python
model.config.rope_theta  # Should show custom value
model.rope.joint_pos_mode  # Should show custom value
```

**Q: Can I keep existing training without changes?**
A: Yes! Default values match current behavior, so existing checkpoints will work.

**Q: What if I only want to change `theta`?**
A: Only modify `rope_theta` in config, leave others at defaults.

**Q: How do I test before training?**
A: Run the verification script provided in implementation_snippets.md

---

## Summary

You now have:
1. ✅ Complete understanding of RoPE instantiation
2. ✅ Identification of all 4 files to modify
3. ✅ Exact line numbers and code changes needed
4. ✅ Ready-to-use implementation code snippets
5. ✅ Testing procedures and verification steps
6. ✅ Visual flow diagrams for reference
7. ✅ Checkpoint and training information

**Estimated implementation time**: 15-30 minutes for straightforward changes
**Estimated testing time**: 10-15 minutes
**Total timeline**: 25-45 minutes from documentation review to verified implementation

