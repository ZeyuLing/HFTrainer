# Post-Extraction Integration Guide for PerMo Embeddings

**Status**: Ready to execute after extraction completes (ETA: 2026-05-14 20:30 CST)

## Phase 1: Validation (5 minutes)

### Step 1a: Count Completed Embeddings
```bash
# Should show exactly 6610 files
find data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented/train -name "*.pt" | wc -l
```

### Step 1b: Run Validation Script
```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
python3 scripts/data/validate_permo_embeddings.py \
  --permo-root data/hymotion_data/PerMo/PerMo/20260513 \
  --splits train
```

**Expected Output:**
```
Summary:
  Total Files:        6610
  Valid Files:        6610 (100.0%)
  Invalid Files:          0 (0.0%)

By Split:
    train     :  6610 /  6610 (100.0%)

✓ VALIDATION PASSED: All embeddings are valid!
```

### Step 1c: Check Total Size
```bash
du -sh data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented
# Expected: ~7.9 GB
```

---

## Phase 2: Integration (30 minutes)

### Step 2a: Locate Text Encoder Integration Points

The pre-extracted embeddings need to be integrated where `HYTextModel` is currently used:

**Files to update:**
1. `hftrainer/datasets/motion/motionhub/transforms/load_text.py`
   - Current: calls `text_encoder.encode()` at training time
   - New: loads pre-extracted .pt files

2. Training config files in `configs/motion/`
   - Add embedding source paths
   - Disable online text encoding

3. Model initialization in `hftrainer/models/motion/`
   - Skip loading text encoder weights
   - Use embedding loader instead

### Step 2b: Create LoadPreExtractedTextEmbedding Transform

```python
# hftrainer/datasets/motion/motionhub/transforms/load_text.py

class LoadPreExtractedTextEmbedding:
    """Load pre-extracted Qwen3+CLIP embeddings from .pt files."""
    
    def __init__(self, embedding_root: str, split: str = "train"):
        self.embedding_root = Path(embedding_root)
        self.split = split
        self.embedding_dir = self.embedding_root / "qwen3embedding_augmented" / split
    
    def __call__(self, results: dict) -> dict:
        # Load from {video_id}.pt
        video_id = results["video_id"]  # or extract from filename
        emb_path = self.embedding_dir / f"{video_id}.pt"
        
        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding not found: {emb_path}")
        
        emb_data = torch.load(emb_path, weights_only=False)
        emb = emb_data["result"][0]["text_embedding"]
        
        results["text_vec_raw"] = emb["text_vec_raw"]      # (1, 1, 768)
        results["text_ctxt_raw"] = emb["text_ctxt_raw"]    # (1, seq, 4096)
        results["text_ctxt_raw_length"] = emb["text_ctxt_raw_length"]  # (1,)
        
        return results
```

### Step 2c: Update Training Config

**Before:**
```yaml
data:
  train_pipeline:
    - type: LoadCaption
    - type: TextEncode
      model: HYTextModel
      device: cuda
      batch_size: 8
```

**After:**
```yaml
data:
  train_pipeline:
    - type: LoadCaption  # Still load captions for reference
    - type: LoadPreExtractedTextEmbedding
      embedding_root: data/hymotion_data/PerMo/PerMo/20260513
      split: train
```

### Step 2d: Update Model Initialization

**Before:**
```python
model = HYMotionModel(
    text_encoder=HYTextModel(
        llm_type="qwen3_embedding",
        sentence_emb_type="clipl",
    ),
    # ...
)
```

**After:**
```python
model = HYMotionModel(
    text_encoder=None,  # No text encoder needed
    use_preextracted_embeddings=True,
    # ...
)

# In model forward pass:
if self.use_preextracted_embeddings:
    text_vec = batch["text_vec_raw"]      # (B, 1, 768)
    text_ctxt = batch["text_ctxt_raw"]    # (B, seq, 4096)
    text_ctxt_len = batch["text_ctxt_raw_length"]  # (B,)
else:
    # Original path: encode text on-the-fly
    text_vec, text_ctxt, text_ctxt_len = self.text_encoder.encode(...)
```

---

## Phase 3: Training (Variable)

### Step 3a: Run Training with Pre-Extracted Embeddings

```bash
# Example training command
python3 -m hftrainer.train \
  --config configs/motion/hymotion_m2m_permo.yaml \
  --exp-name permo_with_preextracted_embeddings \
  --device cuda:0
```

### Step 3b: Monitor Training Metrics

**Expected improvements:**
- **Speed**: Training step time reduced by 10-20% (no text encoding overhead)
- **Memory**: GPU memory reduced by ~1.2-1.5 GB (no text encoder in VRAM)
- **Consistency**: Same embeddings every epoch (no re-encoding variability)
- **Loss**: Similar or slightly improved (consistent input quality)

**Sample metrics to track:**
```
Epoch 1 Step 100:
  Loss: 0.245
  Step Time: 0.85s (vs 1.02s with on-the-fly encoding)
  GPU Mem: 13.2 GB (vs 14.5 GB before)
  Text Encoding Time: 0.00s (pre-computed)
```

### Step 3c: Validate Results

```python
# Check that embeddings are being used correctly
import torch

batch = train_loader.get_batch()
print(f"text_vec_raw shape: {batch['text_vec_raw'].shape}")
print(f"text_ctxt_raw shape: {batch['text_ctxt_raw'].shape}")

# Verify shapes match model expectations
assert batch['text_vec_raw'].shape == (batch_size, 1, 768)
assert batch['text_ctxt_raw'].shape[0] == batch_size
assert batch['text_ctxt_raw'].shape[2] == 4096
```

---

## Rollback Plan (If Issues Occur)

If training with pre-extracted embeddings shows problems:

1. **Quick Rollback**: Switch config back to `TextEncode` with `HYTextModel`
   - No data loss (original embeddings still saved)
   - Revert to 1 command change
   - Lost time: just this training run

2. **Investigate Issues**:
   ```bash
   # Check individual embedding format
   python3 -c "
   import torch
   data = torch.load('data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented/train/sample.pt')
   print(data['result'][0]['text_embedding'])
   "
   ```

3. **Regenerate if Needed**:
   ```bash
   # Re-run extraction (will skip existing, add missing)
   python3 scripts/data/prepare_permo_embeddings_optimized.py \
     --permo-root data/hymotion_data/PerMo/PerMo/20260513 \
     --splits train \
     --device cpu \
     --overwrite
   ```

---

## Success Criteria Checklist

- [ ] Validation script reports 6,610/6,610 valid embeddings
- [ ] Integration changes applied and tested
- [ ] Training runs without errors
- [ ] Step time reduced by 10-20%
- [ ] GPU memory reduced by 1-2 GB
- [ ] Loss curves match or improve previous training
- [ ] Model produces valid motion outputs

---

## Timeline

| Phase | Duration | When | What |
|-------|----------|------|------|
| Extraction | ~18 hours | Now - 20:30 | CPU text encoding |
| Validation | 5 min | 20:30 | Verify files |
| Integration | 30 min | 20:35 | Update configs/code |
| Training | Variable | 21:00+ | Run training |
| Analysis | Variable | 21:00+ | Compare metrics |

**Total time to ready-to-train: ~18.5 hours**

---

## Contact & Support

If extraction fails:
- Check `PERMO_EXTRACTION_MONITOR.md` for status
- Review error logs in process output
- Ensure sufficient disk space (need ~8 GB)
- Verify all 6,610 caption files exist

If integration fails:
- Check data pipeline compatibility
- Verify embedding file format with validation script
- Review tensor shape mismatches in error logs

If training fails:
- Check config syntax and paths
- Verify batch loading works with LoadPreExtractedTextEmbedding
- Review model code for HYTextModel removal conflicts

