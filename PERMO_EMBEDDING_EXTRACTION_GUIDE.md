# PerMo Qwen3+CLIP Text Embedding Extraction Guide

Complete findings for extracting text embeddings using Qwen3-Embedding-8B + CLIP-L models.

---

## 1. EXACT .PT FILE FORMAT EXPECTED BY LoadPreExtractedTextEmbedding

**Location:** `hftrainer/datasets/motion/motionhub/transforms/load_text.py` (lines 71-184)

### Saved Format

```python
{
    'result': [
        {
            'caption': str,                          # Original text
            'text_embedding': {
                'text_vec_raw': Tensor[1, 1, 768],   # CLIP-L (batch=1, seq=1, dim=768)
                'text_ctxt_raw': Tensor[1, seq, 4096],  # Qwen3-Embedding (batch=1, variable seq, dim=4096)
                'text_ctxt_raw_length': Tensor[1],   # Actual token count (batch=1 scalar)
            },
            'start_time': 0,
            'end_time': 0,
            'version': 'motionfix_hymotion',  # or 'permo_hymotion'
        }
    ]
}
```

### Loading Operation

```python
# From LoadPreExtractedTextEmbedding.transform() (lines 138-184)
data = torch.load(pt_path, map_location='cpu', weights_only=False)
result_list = data.get('result', [])
idx = random.randint(0, len(result_list) - 1)  # Random variant for augmentation
item = result_list[idx]
emb = item.get('text_embedding')

# Key: squeeze out the batch dimension [1, ...] → [...]
text_vec_raw = emb['text_vec_raw'].squeeze(0)           # [1, 768]
text_ctxt_raw = emb['text_ctxt_raw'].squeeze(0)         # [seq, 4096]
text_ctxt_raw_length = emb['text_ctxt_raw_length'].squeeze(0)  # scalar
```

**Critical Point:** Tensors are saved WITH batch dim `[1, ...]` from single-sample encoding, then squeezed during loading.

---

## 2. CAPTION-TO-EMBEDDING PATH MAPPING

**Location:** `hftrainer/datasets/motion/motionhub/transforms/load_text.py` (lines 18-67)

### Mapping Dictionary (CAPTION_TO_QWEN3_DIR)

```python
CAPTION_TO_QWEN3_DIR = {
    # Academic / AcademicRetarget / Game / Taobao
    'human_checked_augmented_caption': 'qwen3_augmented',
    'human_checked_augmented_caption_deprecated_mirror_251215': 'qwen3_augmented',
    'human_checked_augmented_caption_mirror': 'qwen3_augmented',
    'human_checked_caption': 'qwen3_human_checked_short',
    'human_checked_caption_deprecated_mirror_251215': 'qwen3_human_checked_short',
    'human_checked_caption_mirror': 'qwen3_human_checked_short',
    'improved_simple_augmented_caption': 'qwen3_improved_simple_short',
    'improved_simple_augmented_caption_deprecated_mirror_251215': 'qwen3_improved_simple_short',
    'improved_simple_caption': 'qwen3_improved_simple_short',
    'improved_simple_caption_deprecated_mirror_251215': 'qwen3_improved_simple_short',
    # Older directory names (used by MotionFix and PerMo)
    'augmented_caption': 'qwen3embedding_augmented',
    'augmented_caption_deprecated_250905': 'qwen3embedding_augmented',
    'augmented_caption_deprecated_250926': 'qwen3embedding_augmented',
}
```

### PerMo Path Mapping (Works Automatically!)

**PerMo already uses the `augmented_caption` directory name**, which is already mapped:

```
Caption:  data/hymotion_data/PerMo/PerMo/20260513/augmented_caption/train/Motion1.json
         └─ part='augmented_caption' matches CAPTION_TO_QWEN3_DIR key
         └─ maps to 'qwen3embedding_augmented'

Embedding: data/hymotion_data/PerMo/PerMo/20260513/qwen3embedding_augmented/train/Motion1.pt
```

**No code changes needed for PerMo!** The mapping works out-of-the-box.

---

## 3. HYTextModel INSTANTIATION FOR EXTRACTION

**Location:** `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py` (lines 73-118)

### Model Configuration

```python
LLM_ENCODER_LAYOUT = {
    "qwen3_embedding": {
        "module_path": "checkpoints/Qwen3-Embedding-8B",  # Path relative to repo root
        "template": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}\n{{}}",
        "crop_start": 0,
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModel,  # NOT AutoModelForCausalLM
    },
}

SENTENCE_EMB_LAYOUT = {
    "clipl": {
        "module_path": "checkpoints/clip-vit-large-patch14",
        "tokenizer_class": CLIPTokenizer,
        "text_encoder_class": CLIPTextModel,
    },
}
```

### Instantiation Code (from prepare_motionfix_hymotion.py, lines 323-331)

```python
from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel

dtype = {
    "auto": None,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[torch_dtype]  # "bfloat16" recommended for memory efficiency

text_encoder = HYTextModel(
    llm_type="qwen3_embedding",           # ✓ Qwen3-Embedding-8B (not causal)
    sentence_emb_type="clipl",             # ✓ CLIP-L for sentence embeddings
    max_length_llm=512,                    # Max tokens for Qwen3 context
    enable_llm_padding=False,              # No padding during extraction
    torch_dtype=dtype,                     # Memory optimization
)
text_encoder.to(device)
text_encoder.eval()
```

### Encoding API (lines 197-200)

```python
def encode(self, text: List[str]) -> Tuple[Tensor, Tensor, Tensor]:
    """Encode list of texts to embeddings.
    
    Returns:
        vtxt_raw: [batch_size, 1, 768]      # CLIP-L embeddings
        ctxt_raw: [batch_size, seq, 4096]   # Qwen3 context (variable seq per text)
        ctxt_len: [batch_size]               # Actual token counts
    """
    ctxt_raw, ctxt_length = self._encode_llm(text=text)     # Qwen3
    vtxt_raw = self._encode_sentence_emb(text=text)         # CLIP-L
    return vtxt_raw, ctxt_raw, ctxt_length
```

---

## 4. PERMO CAPTION FILE FORMAT

**File:** `data/hymotion_data/PerMo/PerMo/20260513/augmented_caption/train/Unpleasantfloor_Walk_A03_002.json`

### Actual Format

```json
{
  "result": [
    {
      "short_caption": "The person walks forward steadily."
    }
  ]
}
```

### Structure Details

- **Root key:** `"result"` — Array of caption variants (for data augmentation)
- **Each variant:**
  - `"short_caption"` (string) — Main caption text (always present)
  - `"short_caption_rewritten"` (optional array of strings) — Alternative captions

### Text Extraction Logic (from LoadHYMotionCaption, lines 232-269)

```python
result_list: List[Dict] = hierarchical_caption.get("result", [])
caption_list = []

for item in result_list:
    # Try rewritten variants first (if they exist)
    if "short_caption_rewritten" in item and isinstance(item["short_caption_rewritten"], list):
        for rewritten_caption in item["short_caption_rewritten"]:
            if isinstance(rewritten_caption, str) and len(rewritten_caption.strip()) > 0:
                caption_list.append(rewritten_caption.strip())
    
    # Fallback to short_caption
    elif "short_caption" in item and isinstance(item["short_caption"], str):
        short_caption = item["short_caption"].strip()
        if len(short_caption) > 0:
            caption_list.append(short_caption)

# At training time, randomly select one variant
select_idx = random.randint(0, len(caption_list) - 1)
selected_caption = caption_list[select_idx]
```

---

## 5. CHECKPOINT LOCATIONS

**Actual locations:**

```
checkpoints/Qwen3-Embedding-8B → /apdcephfs_cq10/share_1467498/home/rexwen/code/HunyuanMotion_T2M/ckpts/Qwen3-Embedding-8B/
checkpoints/clip-vit-large-patch14 → /apdcephfs_cq10/share_1467498/home/rexwen/code/HunyuanMotion_T2M/ckpts/clip-vit-large-patch14/
```

Both are symlinks. Models are loaded via HuggingFace `from_pretrained()`.

---

## 6. KEY TAKEAWAYS

| Item | Value |
|------|-------|
| **LLM Model** | Qwen3-Embedding-8B (NOT causal LM) |
| **Sentence Embedding** | CLIP-L (768-dim) |
| **LLM Output Dim** | 4096 |
| **Max Context Length** | 512 tokens |
| **Caption Dir Name** | `augmented_caption` (already mapped!) |
| **Embedding Dir Name** | `qwen3embedding_augmented` |
| **File Format** | PyTorch dict with `result[0]['text_embedding']` |
| **Tensor Shapes** | vec: [1, 1, 768], ctx: [1, seq, 4096], len: [1] |
| **Recommended dtype** | bfloat16 (memory efficient) |

