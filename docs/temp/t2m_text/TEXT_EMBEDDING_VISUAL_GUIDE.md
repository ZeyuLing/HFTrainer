# Text Embedding Data Flow - Visual Guide

## 1. LoadPreExtractedTextEmbedding State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│  LoadPreExtractedTextEmbedding.transform(results)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  caption_path = results.get('caption_path')                    │
│         │                                                       │
│         ├─ None ──────────────────┬─ allow_none=True           │
│         │                         │  └─ _fill_null_embedding() │
│         │                         │     Return immediately      │
│         │                         │                            │
│         │                         └─ allow_none=False          │
│         │                            └─ Raise ValueError       │
│         │                                                       │
│         └─ Path exists                                         │
│                │                                                │
│                ├─ Derive .pt path                              │
│                │  (via CAPTION_TO_QWEN3_DIR mapping)           │
│                │         │                                      │
│                │         ├─ .pt exists                         │
│                │         │  └─ torch.load(pt_path)             │
│                │         │     │                                │
│                │         │     ├─ Success → Unpack embeddings  │
│                │         │     │  ├─ text_vec_raw            │
│                │         │     │  ├─ text_ctxt_raw           │
│                │         │     │  ├─ text_ctxt_raw_length    │
│                │         │     │  └─ caption                  │
│                │         │     │                                │
│                │         │     └─ Error → _fill_null_embedding()
│                │         │                                      │
│                │         └─ .pt missing → _fill_null_embedding()
│                │                                                │
│                └─ Invalid .json → _fill_null_embedding()       │
│                                                                 │
│  _fill_null_embedding():                                       │
│  ├─ text_vec_raw = zeros(1, 768)                              │
│  ├─ text_ctxt_raw = zeros(1, 4096)                            │
│  ├─ text_ctxt_raw_length = 0                                  │
│  └─ _text_is_null = True                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. LoadCompatibleCaption Decision Logic

```
┌─────────────────────────────────────────────────────────────────┐
│  LoadCompatibleCaption.transform(results)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  caption_path = results.get('caption_path')                    │
│         │                                                       │
│         ├─ None ──────────┬─ allow_none=True                  │
│         │                 │  └─ Return results unchanged       │
│         │                 │     (No keys added)                │
│         │                 │                                    │
│         │                 └─ allow_none=False                 │
│         │                    └─ Raise ValueError               │
│         │                                                       │
│         └─ Load JSON                                            │
│            │                                                    │
│            ├─ Format detection:                                │
│            │  ├─ Has ["macro", "meso", "micro"]               │
│            │  │  └─ Hierarchical format                        │
│            │  │     ├─ caption = random pick from all levels  │
│            │  │     ├─ granularity: "macro"|"meso"|"micro"   │
│            │  │     ├─ caption_list: all captions              │
│            │  │     └─ granularity_list: matching levels       │
│            │  │                                                │
│            │  └─ Has ["result"] array                          │
│            │     └─ HYMotion format                            │
│            │        ├─ caption = random pick from result       │
│            │        └─ caption_list: all captions              │
│            │                                                    │
│            └─ Neither format matches                           │
│               └─ Raise ValueError (if allow_none=False)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. PackInputs Filtering Logic

```
┌──────────────────────────────────────────────────────────────────┐
│  PackInputs.transform(results)                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  packed = {}                                                     │
│                                                                  │
│  for k in self.keys:                                            │
│      value = results.get(k, None)                              │
│      │                                                          │
│      ├─ value is not None:                                     │
│      │  ├─ isinstance(value, np.ndarray)                       │
│      │  │  └─ packed[k] = torch.from_numpy(value)            │
│      │  └─ else:                                               │
│      │     └─ packed[k] = value                                │
│      │                                                          │
│      └─ value is None:                                         │
│         ├─ set_dummy_value=True                               │
│         │  └─ packed[k] = self.dummy_value (e.g., None)      │
│         └─ set_dummy_value=False (DEFAULT)                    │
│            └─ [SILENTLY OMIT] (don't add to packed)           │
│                                                                  │
│  for k in self.meta_keys + self.data_keys:                    │
│      if k in results:                                          │
│          [ADD TO PACKED]                                       │
│                                                                  │
│  return packed                                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Trainer Text Embedding Fallback Waterfall

```
┌────────────────────────────────────────────────────────────────────────┐
│  HyMotionM2MTrainer._prepare_and_forward(batch)                        │
│  TEXT CONDITIONING PREPARATION                                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  LEVEL 1: Check pre-extracted embeddings                        │ │
│  │  ─────────────────────────────────────────────────────────────  │ │
│  │  if batch.get('text_vec_raw') is not None:                     │ │
│  │    │                                                            │ │
│  │    ├─ vtxt_input = batch['text_vec_raw'].to(device)           │ │
│  │    ├─ ctxt_raw = batch['text_ctxt_raw']                        │ │
│  │    ├─ Handle variable-length:                                 │ │
│  │    │  ├─ If list of tensors: Pad all to max_text_len=128      │ │
│  │    │  └─ If stacked tensor: Pad/truncate to 128               │ │
│  │    ├─ Build attention mask: ctxt_mask_temporal (B, 128)       │ │
│  │    │                                                            │ │
│  │    ├─ ┌──────────────────────────────────────────────────────┐ │ │
│  │    │  │  LEVEL 1.1: Replace null samples                     │ │ │
│  │    │  │  ──────────────────────────────────────────────────  │ │ │
│  │    │  │  null_mask = (ctxt_length == 0)                     │ │ │
│  │    │  │  if null_mask.any():                                │ │ │
│  │    │  │    vtxt_input[null_mask] = null_vtxt_feat          │ │ │
│  │    │  │    ctxt_input[null_mask] = null_ctxt_input         │ │ │
│  │    │  │                                                      │ │ │
│  │    │  │  WHY: Zeros ≠ valid conditioning. Learned nulls    │ │ │
│  │    │  │       enable CFG via gradient signal.               │ │ │
│  │    │  └──────────────────────────────────────────────────────┘ │ │
│  │    │                                                            │ │
│  │    ├─ ┌──────────────────────────────────────────────────────┐ │ │
│  │    │  │  LEVEL 1.2: Apply CFG dropout                       │ │ │
│  │    │  │  ──────────────────────────────────────────────────  │ │ │
│  │    │  │  vtxt_input, ctxt_input, text_available =          │ │ │
│  │    │  │      mask_text_cond(                               │ │ │
│  │    │  │          vtxt_input, ctxt_input,                   │ │ │
│  │    │  │          cond_mask_prob=0.1  # 10% dropout        │ │ │
│  │    │  │      )                                              │ │ │
│  │    │  │                                                      │ │ │
│  │    │  │  text_available: (B,) bool mask                    │ │ │
│  │    │  │    = True where text kept, False where dropped    │ │ │
│  │    │  └──────────────────────────────────────────────────────┘ │ │
│  │    │                                                            │ │
│  │    └─ ┌──────────────────────────────────────────────────────┐ │ │
│  │       │  LEVEL 1.3: Update mask for dropped samples        │ │ │
│  │       │  ──────────────────────────────────────────────────  │ │ │
│  │       │  if not text_available.all():                      │ │ │
│  │       │    dropped = ~text_available                       │ │ │
│  │       │    ctxt_mask_temporal[dropped] = False             │ │ │
│  │       │    ctxt_mask_temporal[dropped, 0] = True           │ │ │
│  │       │                                                      │ │ │
│  │       │  WHY: Inference CFG narrows to 1 position.         │ │ │
│  │       │       Must match training for consistency.         │ │ │
│  │       └──────────────────────────────────────────────────────┘ │ │
│  │       RESULT: → Use for forward pass                          │ │
│  │                                                                │ │
│  │  else:  # text_vec_raw is None                               │ │
│  │    │                                                           │ │
│  │    ├─ LEVEL 2: Check online caption                          │ │
│  │    │  ─────────────────────────────────────────────────────  │ │
│  │    │  elif 'caption' in batch and batch['caption'] is not None: │ │
│  │    │    │                                                      │ │
│  │    │    ├─ text_feats = bundle.encode_text(captions)         │ │
│  │    │    │  ├─ vtxt: (B, 1, 768) CLIP-L                      │ │
│  │    │    │  ├─ ctxt: (B, max_seq, 4096) Qwen3                │ │
│  │    │    │  └─ ctxt_len: (B,) actual lengths                  │ │
│  │    │    │                                                      │ │
│  │    │    ├─ [BUILD MASK & APPLY CFG (same as LEVEL 1.2)]     │ │
│  │    │    └─ RESULT: → Use for forward pass                    │ │
│  │    │                                                           │ │
│  │    └─ LEVEL 3: Full null embedding fallback                 │ │
│  │       ──────────────────────────────────────────────────────  │ │
│  │       else:  # No embeddings, no caption                     │ │
│  │         │                                                      │ │
│  │         ├─ vtxt_input = null_vtxt_feat.expand(B, 1, -1)     │ │
│  │         ├─ ctxt_input = null_ctxt_input.expand(B, 1, -1)    │ │
│  │         ├─ ctxt_mask_temporal = [[True, False, ...]]        │ │
│  │         └─ text_available = [False, False, ...] (all false) │ │
│  │            RESULT: → Fully unconditioned, use for forward    │ │
│  │                                                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Model Forward Pass Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  HyMotionM2MBundle.predict_flow()                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Inputs (ALWAYS tensors, NEVER None):                           │
│  ├─ x_input: (B, L, D+3*D)      motion + VACE context          │
│  ├─ ctxt_input: (B, L_c, 4096)  token embeddings               │
│  ├─ vtxt_input: (B, 1, 768)     sentence embedding              │
│  ├─ timesteps: (B,)              diffusion timesteps            │
│  ├─ x_mask_temporal: (B, L)      motion validity mask           │
│  └─ ctxt_mask_temporal: (B, L_c) text validity mask             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  When CONDITIONED (real text):                             │ │
│  │  ├─ ctxt_mask_temporal[b] = [T, T, T, F, F, ...]        │ │
│  │  │                            └─ 15 real tokens          │ │
│  │  ├─ Cross-attention attends to all valid positions       │ │
│  │  └─ Model learns text-conditioned motion                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  When UNCONDITIONED (null embeddings):                     │ │
│  │  ├─ vtxt_input[b] = null_vtxt_feat (learned param)       │ │
│  │  ├─ ctxt_input[b] = null_ctxt_input (learned param)      │ │
│  │  ├─ ctxt_mask_temporal[b] = [T, F, F, ...]              │ │
│  │  │                             └─ Only 1 position        │ │
│  │  ├─ Cross-attention ignores padding positions            │ │
│  │  └─ Model learns unconditioned motion (for CFG)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  MMDiT Transformer:                                             │
│  ├─ Self-attention on motion: (B, L, D+3*D)                   │
│  ├─ Cross-attention to text: key/value from (ctxt, vtxt)      │
│  │  └─ Queries: motion frame representations                 │
│  │  └─ Keys/Values: text token + sentence embeddings         │
│  ├─ Time embedding: for diffusion timestep                    │
│  └─ Output: (B, L, D_motion) predictions                      │
│                                                                  │
│  Output: pred (B, L, 135)  or (B, L, 198)                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Batch State Transitions

```
                    Dataset
                      ↓
    ┌─────────────────────────────────────┐
    │   PrepareM2Mv2Condition:            │
    │   src_motion, tgt_motion, src_mask  │
    └─────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────┐
    │   LoadCompatibleCaption:            │
    │   caption (str)                     │
    └─────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────────────────┐
    │   LoadPreExtractedTextEmbedding:                │
    │                                                 │
    │   SUCCESS CASE:                                │
    │   ├─ text_vec_raw:         (1, 768)           │
    │   ├─ text_ctxt_raw:        (seq, 4096)        │
    │   ├─ text_ctxt_raw_length: scalar (e.g., 15) │
    │   └─ _text_is_null:        False              │
    │                                                 │
    │   FALLBACK CASE:                              │
    │   ├─ text_vec_raw:         (1, 768) zeros     │
    │   ├─ text_ctxt_raw:        (1, 4096) zeros    │
    │   ├─ text_ctxt_raw_length: 0                  │
    │   └─ _text_is_null:        True               │
    └─────────────────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────┐
    │   PackInputs:                       │
    │   (May omit text_* keys if None)    │
    └─────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────────────────────┐
    │   Trainer Receives Batch:                          │
    │                                                     │
    │   Case 1: text_vec_raw in batch                   │
    │   ├─ Pad & replace nulls with learned nulls       │
    │   ├─ Apply CFG dropout                            │
    │   └─ → Model.predict_flow()                        │
    │                                                     │
    │   Case 2: text_vec_raw missing, caption present   │
    │   ├─ Online encode: bundle.encode_text(caption)   │
    │   ├─ Apply CFG dropout                            │
    │   └─ → Model.predict_flow()                        │
    │                                                     │
    │   Case 3: Both missing                            │
    │   ├─ Use null_vtxt_feat, null_ctxt_input         │
    │   └─ → Model.predict_flow()                        │
    └─────────────────────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────┐
    │   Model Forward:                    │
    │   ├─ predict_flow()                │
    │   ├─ Cross-attention with text     │
    │   └─ Output: motion predictions    │
    └─────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────┐
    │   Loss Computation                  │
    │   & Backprop                        │
    └─────────────────────────────────────┘
```

---

## 7. Text Embedding Shape Evolution

```
Original Caption File (JSON):
  caption: "A person walks forward, swinging arms"

    ↓ LoadCompatibleCaption

Intermediate:
  caption: str (randomly selected variant)

    ↓ LoadPreExtractedTextEmbedding

Dataset Sample (pre-extracted path):
  text_vec_raw: (1, 768)              [CLIP-L sentence]
  text_ctxt_raw: (15, 4096)           [15 Qwen3 tokens]
  text_ctxt_raw_length: 15            [scalar]

    ↓ Collate (flexible_collate)

Batch (variable length):
  text_vec_raw: (B, 1, 768)           [B samples, stacked]
  text_ctxt_raw: List[(seq_i, 4096)]  [B lists, different lengths!]
  text_ctxt_raw_length: (B,)          [e.g., [15, 12, 8, 10]]

    ↓ Trainer padding

Training Batch (fixed length):
  text_vec_raw: (B, 1, 768)           [unchanged]
  text_ctxt_raw: (B, 128, 4096)       [padded to max_text_len=128]
  text_ctxt_raw_length: (B,)          [actual lengths kept]
  ctxt_mask_temporal: (B, 128)        [True for valid, False for padded]

    ↓ Model forward

Model Input:
  ctxt_input: (B, 128, 4096)          [with attention mask]
  vtxt_input: (B, 1, 768)             [sentence embedding]
  ctxt_mask_temporal: (B, 128)        [validity mask]

    ↓ Cross-attention

Output:
  pred: (B, L, 135 or 198)            [motion predictions]
```

---

## 8. Three Data Paths Through System

### Path A: Pre-Extracted Embeddings Available ✅ (FAST)
```
.json caption → .pt embedding file found
     ↓
text_vec_raw, text_ctxt_raw loaded
     ↓
Trainer: Use directly (just pad & replace nulls)
     ↓
Model gets real embeddings
     ↓
No online encoding needed (FAST!)
```

### Path B: No Pre-Extracted, Caption Available 🟡 (MEDIUM)
```
.json caption found → .pt embedding file NOT found
     ↓
LoadPreExtractedTextEmbedding: Fills zeros + length=0
     ↓
Trainer: Detects length==0
     ↓
Falls back to bundle.encode_text(caption)
     ↓
Qwen3-8B online encoding (runs on CPU)
     ↓
Model gets freshly encoded embeddings
     ↓
Slower but flexible (can handle new captions)
```

### Path C: No Data Available 🔴 (UNCONDITIONED)
```
No caption file found / PackInputs omitted key
     ↓
LoadPreExtractedTextEmbedding: Fills zeros
     ↓
Trainer: No text_vec_raw AND no caption
     ↓
Uses bundle.null_vtxt_feat, bundle.null_ctxt_input
     ↓
Model gets learned null embeddings
     ↓
Fully unconditioned (for CFG null branch)
```

