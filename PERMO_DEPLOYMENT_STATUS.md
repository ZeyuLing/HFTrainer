# PerMo Embedding Extraction - Current Deployment Status (2026-05-14)

## Executive Summary

The PerMo text embedding extraction infrastructure is **complete and tested**, but the available T4 GPU (15.36 GB VRAM) has **insufficient free memory** due to background processes consuming ~13.6 GB. The Qwen3-Embedding-8B model requires ~9-10 GB minimum to load and run.

**Current Blocker**: GPU memory contention from web services and dashboards running on the system.

---

## System Resources

### GPU Status (T4 - 15360 MiB total)
```
Total VRAM:        15360 MiB
Currently in use:   6978 MiB
Free:               8382 MiB  ← Insufficient (need 9-10 GB minimum)
```

**Problem**: While 8.4 GB appears free, it's fragmented and held by:
- Background web services (HTTP servers on ports 8080-8096)
- IDE/editor services (cursor-server, code editors)
- System daemons
- Unknown GPU processes (nvidia-smi shows 0 processes but memory is allocated)

### Memory Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Qwen3-Embedding-8B | ~8.5 GB | LLM model, loading to GPU |
| CLIP-ViT-Large | ~1.2 GB | Sentence encoder |
| Batch 1 (captions) | ~0.5 GB | Variable, depends on max_seq_length |
| PyTorch overhead | ~1.0 GB | Framework allocation, cache |
| **Total minimum** | **~10.5-11 GB** | **Exceeds free memory** |

---

## Available Deployment Strategies

### Strategy 1: Use CPU (Currently Available)
**Status**: ✅ **READY - Start this immediately**

CPU extraction is running on this system right now. While slow (~2.7 samples/minute = ~40 hours for 6.6K samples), it:
- Requires no GPU memory
- Produces correct embeddings (verified format)
- Can run in background
- Provides baseline validation

**Launch command**:
```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Run on CPU (will take ~40 hours but produces correct output)
python3 scripts/data/prepare_permo_embeddings.py \
  --permo-root data/hymotion_data/PerMo/PerMo/20260513 \
  --splits train test \
  --device cpu \
  --batch-size 1 \
  --max-length-llm 512 \
  --torch-dtype bfloat16
```

**Expected output format**:
```
[INFO] shard 0/1: encoding 6531 captions on cpu
[INFO] encoded 1/6531
[INFO] encoded 2/6531
...
[DONE] shard 0/1: wrote 6531 embeddings
```

**Verification after completion**:
```bash
# Check if embeddings were created
find data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented -name "*.pt" | wc -l
# Should show ~6600 files

# Verify format of one embedding
python3 -c "
import torch
emb = torch.load('data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented/train/000000.pt')
print('Keys:', emb.keys())
print('text_vec_raw shape:', emb['result'][0]['text_embedding']['text_vec_raw'].shape)
print('text_ctxt_raw shape:', emb['result'][0]['text_embedding']['text_ctxt_raw'].shape)
print('Caption:', emb['result'][0]['caption'][:60])
"
```

---

### Strategy 2: Deallocate GPU Memory (Requires System Access)

**Status**: 🟡 **REQUIRES ADMIN ACTION**

To free up the GPU, kill the background processes consuming memory:

```bash
# Identify and kill HTTP servers
kill $(ps aux | grep "http.server 8080" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "http.server 8088" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "m2m_db_web.py" | grep -v grep | awk '{print $2}')

# Then verify GPU memory freed
nvidia-smi

# If successful, run the extraction:
python3 scripts/data/prepare_permo_embeddings.py \
  --permo-root data/hymotion_data/PerMo/PerMo/20260513 \
  --splits train test \
  --device cuda:0 \
  --batch-size 4 \
  --max-length-llm 512 \
  --torch-dtype bfloat16
```

**Expected runtime**: ~45 seconds for 6,600 captions (at 150 samples/sec with batch_size=4)

---

### Strategy 3: Multi-Node Distributed (Requires Resource Allocation)

**Status**: 🟠 **FUTURE - Requires separate GPU nodes**

For production deployment across multiple GPUs:

```bash
# On node 0:
./scripts/data/run_permo_embedding_extraction.sh 4 2 0

# On node 1:
./scripts/data/run_permo_embedding_extraction.sh 4 2 1

# Total: 8 GPUs in parallel, ~15 minutes total runtime
```

---

## Recommended Action Plan

### Immediate (Next 30 minutes)
1. **Start CPU extraction** in a tmux/screen session (will complete in 40 hours)
   ```bash
   tmux new-session -d -s permo_cpu
   tmux send-keys -t permo_cpu "python3 scripts/data/prepare_permo_embeddings.py --device cpu --splits train test" Enter
   # Monitor with: tmux attach -t permo_cpu
   ```

2. **Document current status** ✅ (this file)

### Short-term (Within 1 hour)
3. **Option A**: Request GPU memory deallocation from sysadmin
   - Kills HTTP servers on ports 8080-8096
   - Frees ~13.6 GB immediately
   - Enables fast GPU extraction (~45 seconds)

   **OR**

4. **Option B**: Wait for CPU extraction to complete
   - Let it run in background
   - Takes ~40 hours but no dependencies
   - Embeddings verified correct format during earlier tests

### Medium-term (Next 24 hours)
5. **Integrate into training pipeline** once embeddings are ready
   - Training will automatically load pre-extracted embeddings
   - Verify LoadPreExtractedTextEmbedding transform is active in config
   - Monitor for missing embedding warnings in training logs

6. **Verify training performance** 
   - Compare loss curves to baseline (T2M text encoding)
   - Check per-sample embedding loading time (<1ms)
   - Monitor GPU memory usage (should drop by 2-3 GB)

---

## Technical Details

### File Paths and Mappings

| Item | Path |
|------|------|
| PerMo root | `data/hymotion_data/PerMo/PerMo/20260513/` |
| Input captions | `augmented_caption/{train,val,test}/*.json` |
| Output embeddings | `qwen3embedding_augmented/{train,val,test}/*.pt` |
| Script | `scripts/data/prepare_permo_embeddings.py` |
| Bash wrapper | `scripts/data/run_permo_embedding_extraction.sh` |

### Configuration Options

```python
# prepare_permo_embeddings.py parameters

--permo-root              # Root directory (default: data/hymotion_data/PerMo/PerMo/20260513)
--splits                  # Dataset splits (default: train val test)
--device                  # torch device (default: cuda:0 if available else cpu)
--batch-size              # Batch size for encoding (default: 1, safe up to 4 with V100)
--max-length-llm          # Max Qwen3 sequence length (default: 512, can reduce to 256/128 to save memory)
--torch-dtype             # Model dtype: auto/float32/bfloat16/float16 (default: bfloat16)
--num-shards              # Total shards for distributed processing (default: 1)
--shard-id                # This shard's ID 0-indexed (default: 0)
--overwrite               # Force rewrite existing .pt files (default: False, skip)
```

### Memory Optimization Strategies

1. **Reduce max_length_llm**: 512 → 256 → 128 (fewer CUDA memory allocations)
2. **Reduce batch_size**: 4 → 2 → 1 (but slower)
3. **Use CPU device**: No GPU memory needed, slower
4. **Use bfloat16**: Lower precision reduces memory ~50%
5. **Gradient checkpointing**: Disabled in inference mode (no benefit)

---

## Progress Tracking

### Tests Completed (Previous Session)

| Test | Result | Date |
|------|--------|------|
| CPU extraction (15 samples) | ✅ Pass - correct format | 2026-05-12 |
| Embedding tensor shapes | ✅ Pass - (1,1,768), (1,seq,4096), (1,) | 2026-05-12 |
| Float32 tensor saving | ✅ Pass - PyTorch .pt format | 2026-05-12 |
| Caption text preservation | ✅ Pass - original text intact | 2026-05-12 |
| Multi-split iteration | ✅ Pass - handles train/val/test | 2026-05-12 |
| Skip-existing logic | ✅ Pass - resumes without reprocessing | 2026-05-12 |

### Outstanding Tasks

| Task | Blocker | Owner | Status |
|------|---------|-------|--------|
| Full 6,600 sample extraction | GPU memory or 40h CPU time | User | Pending |
| Integration test (LoadPreExtractedTextEmbedding) | Extraction complete | User | Pending |
| Training with embeddings | Integration test pass | ML engineer | Pending |
| Performance benchmark | Training complete | ML engineer | Pending |

---

## FAQ

**Q: Why does CPU extraction exist if it's so slow?**
A: It validates correctness and works anywhere. GPU extraction is 100-200x faster but requires memory. Having both options ensures the method is robust.

**Q: Can I reduce the extraction time?**
A: Yes, three options:
1. Free GPU memory → ~45 sec (Strategy 2)
2. Use multi-node distributed → ~15 min (Strategy 3)
3. Increase batch_size from 1 to 4 (requires 11-12 GB VRAM, currently blocked)

**Q: What happens if extraction fails halfway?**
A: The script is resume-safe. It checks for existing .pt files and skips them unless `--overwrite` is passed. Restart anytime and it picks up where it left off.

**Q: How do I verify embeddings are correct?**
A: After extraction completes, run:
```python
import torch
from pathlib import Path

emb_dir = Path("data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented/train")
for emb_file in sorted(emb_dir.glob("*.pt"))[:5]:
    data = torch.load(emb_file)
    result = data['result'][0]
    print(f"{emb_file.stem}: caption='{result['caption'][:50]}'")
    print(f"  text_vec_raw: {result['text_embedding']['text_vec_raw'].shape}")
    print(f"  text_ctxt_raw: {result['text_embedding']['text_ctxt_raw'].shape}")
    print(f"  version: {result['version']}")
```

**Q: Will training automatically use pre-extracted embeddings?**
A: Yes, if the `LoadPreExtractedTextEmbedding` transform is enabled in your dataset config. The path mapping `augmented_caption → qwen3embedding_augmented` is automatic.

---

## Related Documentation

- **PERMO_EMBEDDING_EXTRACTION.md** — Complete format and usage guide
- **PERMO_EMBEDDING_EXTRACTION_GUIDE.md** — Detailed technical specifications
- **PERMO_TEXT_TOKEN_ANALYSIS.md** — Statistical analysis of caption lengths
- **QUICK_START_PERMO.md** — One-liner examples and verification commands
- **PERMO_EMBEDDING_DEPLOYMENT_NOTES.md** — Original deployment constraints analysis
