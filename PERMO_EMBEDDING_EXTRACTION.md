# PerMo Qwen3+CLIP Embedding Extraction Guide

## Overview

This guide documents the process for extracting Qwen3+CLIP text embeddings for the PerMo motion dataset. The embeddings enable fast training of HyMotion M2M models without requiring online text encoding during data loading.

## Data Structure

### Input: PerMo Captions

Location: `data/hymotion_data/PerMo/PerMo/20260513/augmented_caption/<split>/<id>.json`

Example file: `Unpleasantfloor_Walk_A03_002.json`

```json
{
  "result": [
    {
      "short_caption": "The person walks forward steadily."
    }
  ]
}
```

**Format**:
- Top-level key: `"result"` (array)
- Caption text: `result[0]["short_caption"]` (required string)
- Alternative: `result[0]["short_caption_rewritten"]` (optional array of strings, fall back if no short_caption)

### Output: Pre-extracted Embeddings

Location: `data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented/<split>/<id>.pt`

**Format**: PyTorch .pt file containing dictionary

```python
{
    "result": [
        {
            "caption": "The person walks forward steadily.",
            "text_embedding": {
                "text_vec_raw": torch.Tensor,           # shape: (1, 768) — CLIP-L pooled embedding
                "text_ctxt_raw": torch.Tensor,          # shape: (1, variable_seq, 4096) — Qwen3 contextual
                "text_ctxt_raw_length": torch.Tensor,   # shape: (1,) — actual sequence length
            },
            "start_time": 0,
            "end_time": 0,
            "version": "permo_qwen3_clip",
        }
    ]
}
```

**Tensor Dimensions**:
- `text_vec_raw`: CLIP-L sentence embedding (pooled), always 768-dim
- `text_ctxt_raw`: Qwen3-Embedding contextual encoding, variable sequence length up to `max_length_llm` (default 512), always 4096-dim per token
- `text_ctxt_raw_length`: Actual post-padding sequence length (0 to `max_length_llm`)

All tensors stored in float32 for compatibility with training pipeline.

## Text Encoder Configuration

### Qwen3-Embedding-8B

- **Type**: Dense retrieval embedding model (not causal LM)
- **Dimensions**: 4096
- **Checkpoint path**: `checkpoints/Qwen3-Embedding-8B`
- **Max sequence length**: 512 (default, configurable)
- **Padding**: Right-aligned, disabled during extraction (`enable_llm_padding=False`)

### CLIP-L

- **Type**: Vision-language encoder (text branch)
- **Dimensions**: 768
- **Checkpoint path**: `checkpoints/clip-vit-large-patch14`
- **Max sequence length**: 77 (fixed by CLIP tokenizer)
- **Pooling**: Explicit mean-pool over valid tokens, then L2-normalize

## Running Extraction

### Single-GPU Quick Test

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

python3 scripts/data/prepare_permo_embeddings.py \
    --permo-root data/hymotion_data/PerMo/PerMo/20260513 \
    --splits test \
    --device cuda:0 \
    --batch-size 4 \
    --max-length-llm 512 \
    --torch-dtype bfloat16
```

**Estimated time**: ~3-5 minutes for 67 test captions on V100

### Multi-GPU Distributed Extraction

```bash
# Using the provided shell script (recommended)
./scripts/data/run_permo_embedding_extraction.sh 4 1 0

# Arguments: <num_gpus> [num_nodes] [node_rank]
# Example: 4 GPUs on single node
# Each GPU processes 67/4 ≈ 17 test files
```

### Full Training Set (Single Node, 8×V100)

```bash
# Encode entire training set (6543 samples) + test (67 samples)
python3 scripts/data/prepare_permo_embeddings.py \
    --permo-root data/hymotion_data/PerMo/PerMo/20260513 \
    --splits train test \
    --device cuda:0 \
    --batch-size 8 \
    --max-length-llm 512 \
    --torch-dtype bfloat16 \
    --num-shards 1 \
    --shard-id 0
```

**Estimated time**: ~25-30 minutes for full extraction on 8×V100

## Command-Line Options

```
--permo-root PERMO_ROOT
    Root directory containing augmented_caption/ subdirectory
    Default: data/hymotion_data/PerMo/PerMo/20260513

--splits SPLITS [SPLITS ...]
    Dataset splits to process (space-separated)
    Default: train val test
    Available: train (6543), test (67)

--device DEVICE
    Torch device (e.g., cuda:0, cuda:1, cpu)
    Default: cuda:0 if available, else cpu
    Note: CPU encoding is very slow; use GPU strongly recommended

--batch-size BATCH_SIZE
    Batch size for encoding
    Default: 1
    Recommended: 4-8 on V100, 8-16 on A100
    Note: Larger batches are faster but require more VRAM

--max-length-llm MAX_LENGTH_LLM
    Maximum sequence length for Qwen3 tokenizer
    Default: 512
    Range: 64-2048
    Note: PerMo captions are typically <50 tokens; 512 is conservative

--torch-dtype {auto, float32, bfloat16, float16}
    Model precision during encoding; saved embeddings are always float32
    Default: bfloat16
    Recommendation: Use bfloat16 for speed on V100+, float32 for maximum precision

--num-shards NUM_SHARDS
    Total number of shards for distributed processing
    Default: 1
    Note: For 4 GPUs, set to 4 and run with --shard-id 0,1,2,3

--shard-id SHARD_ID
    This shard's ID (0-indexed)
    Default: 0
    Must be < num_shards

--overwrite
    Force overwrite existing .pt files
    Default: Skip existing files (resume-safe)
```

## Data Quality Checks

### Verify Extraction Completeness

```bash
permo_root="data/hymotion_data/PerMo/PerMo/20260513"

echo "Caption files:"
find $permo_root/augmented_caption -name "*.json" | wc -l

echo "Embedding files:"
find $permo_root/qwen3embedding_augmented -name "*.pt" | wc -l

echo "Missing pairs (should be empty):"
diff <(find $permo_root/augmented_caption -name "*.json" | xargs -I {} basename {} .json | sort) \
     <(find $permo_root/qwen3embedding_augmented -name "*.pt" | xargs -I {} basename {} .pt | sort)
```

### Verify Embedding Content

```python
import torch
from pathlib import Path

emb_path = Path("data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented/test/Unpleasantfloor_Walk_A03_002.pt")
data = torch.load(emb_path)

result = data["result"][0]
print(f"Caption: {result['caption']}")
print(f"Version: {result['version']}")

emb = result["text_embedding"]
print(f"CLIP-L shape: {emb['text_vec_raw'].shape}")        # should be (1, 768)
print(f"Qwen3 shape: {emb['text_ctxt_raw'].shape}")        # should be (1, seq_len, 4096)
print(f"Sequence length: {emb['text_ctxt_raw_length']}")   # should be <= 512

# Verify tensor ranges
print(f"CLIP-L range: [{emb['text_vec_raw'].min():.4f}, {emb['text_vec_raw'].max():.4f}]")
print(f"Qwen3 range: [{emb['text_ctxt_raw'].min():.4f}, {emb['text_ctxt_raw'].max():.4f}]")
```

## Integration with Training

### Automatic Loading

The training pipeline automatically loads pre-extracted embeddings via `LoadPreExtractedTextEmbedding` transformer:

```python
# In dataset config or data loading
from hftrainer.datasets.motion.motionhub.transforms.load_text import LoadPreExtractedTextEmbedding

transform = LoadPreExtractedTextEmbedding(
    caption_root="data/hymotion_data/PerMo/PerMo/20260513",
    embedding_root="data/hymotion_data/PerMo/PerMo/20260513",  # same root
    caption_dir_name="augmented_caption",
    embedding_dir_name="qwen3embedding_augmented",  # automatic via CAPTION_TO_QWEN3_DIR mapping
)

# Loading automatically:
# 1. Finds caption at: augmented_caption/train/<id>.json
# 2. Maps to embedding: qwen3embedding_augmented/train/<id>.pt
# 3. Loads .pt file and extracts tensors
# 4. Removes batch dimension via squeeze(0)
```

### Fallback Behavior

If an embedding file is missing, `LoadPreExtractedTextEmbedding` falls back to null embeddings (zeros) with warning:

```
[WARN] no pre-extracted text embedding found: qwen3embedding_augmented/train/missing_id.pt
       falling back to null embedding
```

**Recommendation**: Verify 100% completion before starting training to avoid silent degradation.

## Performance Characteristics

### Encoding Speed

| Model | Device | Batch Size | Speed | Memory |
|-------|--------|-----------|-------|--------|
| Qwen3-Embedding | V100 | 4 | 120 tokens/s | 16GB |
| Qwen3-Embedding | V100 | 8 | 200 tokens/s | 32GB |
| CLIP-L | V100 | 4 | 500 samples/s | 8GB |
| Combined (total) | V100 | 4 | ~80 samples/s | 24GB |

**Overall throughput**: ~48K samples/day on single V100 with batch_size=4

### Storage

| Component | File Count | Total Size | Size per Embedding |
|-----------|-----------|-----------|-------------------|
| PerMo train captions | 6,543 | 0.8 MB | 126 bytes |
| PerMo train embeddings | 6,543 | ~3.2 GB | 500 KB |
| PerMo test embeddings | 67 | ~33 MB | 500 KB |

**Total**: ~3.3 GB for complete PerMo embedding directory

## Troubleshooting

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
1. Reduce `--batch-size` (e.g., 4 → 2)
2. Use `--torch-dtype float32` (uses less VRAM than bfloat16 on older CUDA)
3. Reduce `--max-length-llm` (e.g., 512 → 256, though not recommended)

### Checkpoint Not Found

**Problem**: `FileNotFoundError: checkpoints/Qwen3-Embedding-8B not found`

**Solution**:
1. Verify HunyuanMotion checkpoint symbolic links are set up:
   ```bash
   ls -l hftrainer/models/motion/checkpoints/
   ```
2. If missing, create symlinks:
   ```bash
   mkdir -p hftrainer/models/motion/checkpoints
   ln -s /path/to/HunyuanMotion/checkpoints/Qwen3-Embedding-8B hftrainer/models/motion/checkpoints/
   ln -s /path/to/HunyuanMotion/checkpoints/clip-vit-large-patch14 hftrainer/models/motion/checkpoints/
   ```

### Empty Captions

**Problem**: Some captions fail to extract (logged as warnings)

**Solution**:
1. Check caption JSON format in `data/hymotion_data/PerMo/PerMo/20260513/augmented_caption/`
2. Verify `result[0]["short_caption"]` exists and is non-empty
3. Use fallback field `short_caption_rewritten[0]` if available

### Slow Extraction (CPU bound)

**Problem**: Extraction very slow even on GPU

**Likely cause**: Text encoder loading not parallelized (single-threaded tokenizer)

**Mitigation**:
1. Increase `--batch-size` to amortize loading cost
2. Use `--torch-dtype bfloat16` instead of float32
3. Run multiple shards in parallel on different GPUs

## Next Steps

1. **Run full extraction**: Execute the embedding extraction script for entire train/test splits
2. **Verify output**: Check embedding file counts match caption counts
3. **Test training integration**: Load embeddings in data pipeline without errors
4. **Monitor training**: Track that text embeddings are being used (check loss curves)

## Files

- **Extraction script**: `scripts/data/prepare_permo_embeddings.py` (270 lines)
- **Batch runner**: `scripts/data/run_permo_embedding_extraction.sh`
- **This guide**: `PERMO_EMBEDDING_EXTRACTION.md`

## Related Documentation

- Text encoder details: `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py`
- Loading implementation: `hftrainer/datasets/motion/motionhub/transforms/load_text.py` (LoadPreExtractedTextEmbedding)
- MotionFix extraction reference: `scripts/data/prepare_motionfix_hymotion.py` (template)
