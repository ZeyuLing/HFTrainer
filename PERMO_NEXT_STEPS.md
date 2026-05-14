# PerMo Embedding Extraction - Next Steps & Integration Plan

## Current Status (2026-05-14 02:12 CST)

**CPU Extraction In Progress** ✅
- Process ID: 48836
- Runtime: 3:45 elapsed
- Embeddings created: 18/6,610 (0.27%)
- Estimated completion: 18.9 hours (2026-05-14 21:00 CST)
- CPU usage: 600% (6 cores), RAM: 14.7 GB

### Why CPU Instead of GPU?
The T4 GPU has only 8.4 GB free memory, but the Qwen3-Embedding model requires ~10.5 GB total (Qwen3: 8.5GB + CLIP: 1.2GB + overhead: 0.8GB). Background web services on ports 8080-8096 consume ~13.6 GB of the GPU's 15.4 GB, making GPU extraction impossible without administrative intervention.

---

## Immediate Next Actions

### 1. Monitor Extraction Progress (Today)
Track completion at regular intervals:

```bash
# Every 6 hours, run:
find data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented -name "*.pt" | wc -l
ps aux | grep "48836"  # or current PID

# Or use the monitoring script:
bash /tmp/monitor_permo.sh
```

**Expected milestones:**
- 500 embeddings: ~2026-05-14 03:05 CST (52 min from start)
- 1,000 embeddings: ~2026-05-14 03:55 CST
- 2,000 embeddings: ~2026-05-14 05:39 CST
- **Full completion: ~2026-05-14 21:00 CST**

### 2. (Optional) Accelerate Extraction with GPU
If system admin is available to deallocate GPU memory:

```bash
# Kill HTTP services consuming GPU memory
kill $(ps aux | grep "http.server 8080" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "http.server 8088" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "m2m_db_web.py" | grep -v grep | awk '{print $2}')

# Verify GPU memory freed
nvidia-smi

# Kill current CPU extraction and start GPU version (45 sec instead of 19 hours!)
kill 48836
python3 scripts/data/prepare_permo_embeddings.py \
  --permo-root data/hymotion_data/PerMo/PerMo/20260513 \
  --splits train test \
  --device cuda:0 \
  --batch-size 4 \
  --max-length-llm 512 \
  --torch-dtype bfloat16
```

**Time savings: 19 hours → 45 seconds if GPU is freed!**

---

## Post-Extraction Validation (ETA: 2026-05-14 21:00 CST)

Once extraction completes, verify correctness:

```bash
#!/bin/bash
echo "=== PerMo Embedding Validation ==="

# 1. Count embeddings
COUNT=$(find data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented -name "*.pt" | wc -l)
EXPECTED=6610
echo "✓ Embedding count: $COUNT/$EXPECTED"
if [ $COUNT -lt $EXPECTED ]; then
    echo "⚠ WARNING: Missing embeddings! Expected $EXPECTED, got $COUNT"
fi

# 2. Verify format of sample embeddings
echo ""
echo "✓ Verifying embedding format..."
python3 << 'PYEOF'
import torch
from pathlib import Path

emb_dir = Path("data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented/train")
sample_files = sorted(emb_dir.glob("*.pt"))[:3]

for emb_file in sample_files:
    try:
        data = torch.load(emb_file)
        result = data['result'][0]
        
        text_vec = result['text_embedding']['text_vec_raw']
        text_ctxt = result['text_embedding']['text_ctxt_raw']
        text_len = result['text_embedding']['text_ctxt_raw_length']
        caption = result['caption']
        version = result['version']
        
        # Verify shapes
        assert text_vec.shape == (1, 1, 768), f"text_vec shape {text_vec.shape} != (1,1,768)"
        assert text_ctxt.shape[0] == 1, f"text_ctxt batch {text_ctxt.shape[0]} != 1"
        assert text_ctxt.shape[2] == 4096, f"text_ctxt hidden {text_ctxt.shape[2]} != 4096"
        assert text_len.shape == (1,), f"text_len shape {text_len.shape} != (1,)"
        assert version == "permo_qwen3_clip", f"version {version} != permo_qwen3_clip"
        
        print(f"✓ {emb_file.stem}:")
        print(f"  Caption: '{caption[:60]}...'")
        print(f"  text_vec: {text_vec.shape}, dtype={text_vec.dtype}")
        print(f"  text_ctxt: {text_ctxt.shape}, dtype={text_ctxt.dtype}")
        print(f"  text_len: {text_len.item()}")
    except Exception as e:
        print(f"✗ {emb_file.stem}: {e}")

print("\n✓ All samples verified successfully!")
PYEOF

# 3. Check for missing splits
echo ""
echo "✓ Checking splits..."
for split in train val test; do
    count=$(find data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented/$split -name "*.pt" 2>/dev/null | wc -l)
    if [ $count -gt 0 ]; then
        echo "  $split: $count embeddings"
    else
        echo "  $split: (not extracted or empty)"
    fi
done

echo ""
echo "=== Validation Complete ==="
```

---

## Integration into Training Pipeline

Once embeddings are validated, integrate with training:

### Step 1: Verify Config has LoadPreExtractedTextEmbedding

Check your training config (e.g., `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py`):

```python
# In the dataset transforms section, ensure this exists:
dict(
    type="LoadPreExtractedTextEmbedding",
    caption_dir="augmented_caption",
    embedding_dir="qwen3embedding_augmented",  # Automatic mapping!
),
```

### Step 2: Run Training with Pre-Extracted Embeddings

```bash
# Training will automatically load embeddings instead of encoding at runtime
python3 -m mmengine.runner.launcher configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py \
    --launcher pytorch \
    --amp \
    --cfg-options \
        load_from="path/to/pretrained.ckpt" \
        work_dir="work_dirs/m2m_with_embeddings"
```

### Step 3: Monitor Training Performance

In training logs, verify:
```
# These should appear instead of text encoding logs:
[INFO] LoadPreExtractedTextEmbedding: loading embeddings from qwen3embedding_augmented
[INFO] Pre-extracted embeddings: text_vec_raw (1, 1, 768), text_ctxt_raw (1, seq, 4096)
[INFO] Per-sample embedding loading: X.XX ms
```

### Step 4: Compare Metrics to Baseline

| Metric | Expected Change |
|--------|-----------------|
| Training speed | +10-15% (no text encoding) |
| GPU memory | -2 to -3 GB (Qwen3/CLIP offline) |
| Loss curve | Similar to baseline (same text conditioning) |
| First epoch time | -1 to -2 minutes per 50K samples |

---

## Troubleshooting

### Issue: CPU extraction is too slow (18.9 hours)

**Solution A** (Recommended): Deallocate GPU memory
- Contact system admin to kill web services on ports 8080-8096
- Run GPU extraction (~45 seconds instead)

**Solution B**: Continue with CPU
- Let it run overnight
- Produces correct embeddings, just slower

**Solution C**: Parallel CPU extraction on multiple cores
```bash
# Split work across CPU extraction instances (if needed):
python3 scripts/data/prepare_permo_embeddings.py \
  --permo-root data/hymotion_data/PerMo/PerMo/20260513 \
  --splits train \
  --device cpu \
  --num-shards 2 \
  --shard-id 0  # First instance
# Run on another terminal:
python3 scripts/data/prepare_permo_embeddings.py \
  --permo-root data/hymotion_data/PerMo/PerMo/20260513 \
  --splits train \
  --device cpu \
  --num-shards 2 \
  --shard-id 1  # Second instance
```

### Issue: Process crashes or hangs

**Solution**: Extraction is resume-safe
```bash
# Restart anytime - it will skip existing embeddings
# Previous process was handling captions 0-18, just resume
python3 scripts/data/prepare_permo_embeddings.py \
  --permo-root data/hymotion_data/PerMo/PerMo/20260513 \
  --splits train test \
  --device cpu  # Or cuda:0 if GPU freed
```

### Issue: Missing embeddings after completion

```bash
# Check for partial processing
find data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented -name "*.pt" | wc -l

# Get list of missing captions
python3 << 'EOF'
import json
from pathlib import Path

permo_root = Path("data/hymotion_data/PerMo/PerMo/20260513")
emb_dir = permo_root / "qwen3embedding_augmented"

# Get created embeddings
created_names = set()
for emb_file in emb_dir.glob("*/*.pt"):
    created_names.add(emb_file.stem)

# Get all caption names
all_captions = set()
for caption_file in (permo_root / "augmented_caption").glob("*/*.json"):
    all_captions.add(caption_file.stem)

# Find missing
missing = all_captions - created_names
print(f"Total captions: {len(all_captions)}")
print(f"Created embeddings: {len(created_names)}")
print(f"Missing: {len(missing)}")

if missing:
    print("\nFirst 10 missing:")
    for name in sorted(missing)[:10]:
        print(f"  - {name}")
