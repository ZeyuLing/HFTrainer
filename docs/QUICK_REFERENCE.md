# Quick Reference: HyMotion T2M Bug Fixes

## 🎯 The Problem
Generated motions don't match text prompts at all.

## 🔧 The Solution
Three bugs in `scripts/embodied/physflow_eval_demo.py`, function `generate_motion_from_bundle()` (lines 187-276)

---

## ⚡ Quick Fix (Copy-Paste)

### Change #1: Line 198
```diff
- motion_dim = 201
+ motion_dim = bundle.motion_transformer.output_dim
```

### Change #2: Line 208
```diff
- L_padded = TRAIN_FRAMES
+ L_padded = max(L, TRAIN_FRAMES)
```

### Change #3: Lines 211-212
```diff
- max_ctxt_len = ctxt_input.shape[1]
- ctxt_mask_temporal = torch.arange(max_ctxt_len, device=device).unsqueeze(0) >= ctxt_len.unsqueeze(1)
+ ctxt_mask_temporal = _length_to_mask(ctxt_len, ctxt_input.shape[1])
```

### Change #4: Lines 271-275
```diff
  sampled = x[:, :L, :]
  latent_denorm = bundle.denormalize_motion(sampled)
- motion_201 = latent_denorm[0].cpu().numpy()
- motion_135 = motion_201[:, :135].astype(np.float32)
+ motion_135 = latent_denorm[0].cpu().numpy().astype(np.float32)
  
  return motion_135
```

---

## 📊 Bug Summary Table

| Bug | Line | Issue | Fix |
|-----|------|-------|-----|
| #1 | 198 | `motion_dim=201` → dimension mismatch | Use `bundle.motion_transformer.output_dim` (135) |
| #2 | 208 | `L_padded=TRAIN_FRAMES` → truncates long sequences | Use `max(L, TRAIN_FRAMES)` |
| #3 | 212 | Inverted mask `>=` → destroys text conditioning | Use `_length_to_mask()` |

---

## ✅ Quick Test

After applying fixes:
```bash
python3 scripts/embodied/physflow_eval_demo.py \
    --num-prompts 1 \
    --output-dir /tmp/test
```

Generated motions should now match text prompts! ✨

---

## 📁 Files

| File | Purpose |
|------|---------|
| `scripts/embodied/physflow_eval_demo_FIXED.py` | Complete fixed version (ready to use) |
| `docs/HYMOTION_T2M_BUG_FIXES.md` | Detailed technical docs |
| `docs/BUGFIX_CODE_DIFF.md` | Full before/after code comparison |
| `docs/BUG_FIX_SUMMARY.md` | Comprehensive overview |
| `docs/QUICK_REFERENCE.md` | This file |

---

## 🚀 Deploy

**Recommended**: Copy the fixed file
```bash
cp scripts/embodied/physflow_eval_demo_FIXED.py scripts/embodied/physflow_eval_demo.py
```

**Alternative**: Manually apply 4 changes above

---

## ❓ Why Did This Happen?

1. **Bug #1**: 201-dim is from old codebase; current model uses 135-dim
2. **Bug #2**: Hardcoded for debugging; was never fixed
3. **Bug #3**: Manual mask logic error; should use helper function

---

## 📞 Need Help?

- Technical details → `HYMOTION_T2M_BUG_FIXES.md`
- Code changes → `BUGFIX_CODE_DIFF.md`
- Overview → `BUG_FIX_SUMMARY.md`
- Working code → `physflow_eval_demo_FIXED.py`

---

**Status**: ✅ Ready to deploy
**Tested**: Against official pipeline implementation
**Verified**: Dimension handling, context masking, sequence padding

