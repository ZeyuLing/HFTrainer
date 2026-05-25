# PRISM Training-Inference Mismatch Analysis

## Summary of Key Findings

I've traced the full text embedding flow in PRISM through training and inference. Here are the critical differences and potential mismatches:

---

## 1. Text Embedding Pipeline: Training vs Inference

### TRAINING PATH (`prism_trainer.py`, line 56-61):
```python
text_states = self.bundle.encode_prompt(
    captions,
    max_sequence_length=self.max_text_length,  # ← DEFAULT: 128 (trainer init line 24)
    prompt_drop_rate=self.prompt_drop_rate,   # ← 0.1 by default
    dtype=next(self.bundle.transformer.parameters()).dtype,
)
```

### INFERENCE PATH A - `prism_backend.py` (line 826-862):
```python
prompt_embeds = self._get_t5_prompt_embeds(
    prompt=prompt,
    num_motion_per_prompt=num_motion_per_prompt,
    max_sequence_length=max_sequence_length,  # ← DEFAULT: 512 (line 334)
    device=device,
    dtype=dtype,
)
```

### INFERENCE PATH B - Simple `prism_pipeline.py`:
Simply wraps the backend pipeline, no direct text encoding.

---

## 2. MAX SEQUENCE LENGTH MISMATCH ⚠️ CRITICAL

| Component | Training | Inference (Backend) | Difference |
|-----------|----------|-------------------|-----------|
| max_sequence_length | **128** | **512** | **4x difference** |
| Source | `PrismTrainer.__init__(max_text_length=128)` | `PrismARPipeline.generate_single_segment()` default | |

**This is a PRIMARY mismatch source!**

- **During training**: Text sequences are truncated/padded to 128 tokens
- **During inference**: Text sequences use 512 tokens by default
- The transformer NEVER saw sequences longer than 128 during training

---

## 3. TEXT ENCODER MODEL

**BOTH paths use the SAME encoder:** `UMT5EncoderModel` (T5-XXL based)

From `bundle.py` line 102-105:
```python
'text_encoder': {
    'type': text_encoder_type,  # Default: 'UMT5EncoderModel'
    'from_pretrained': {...}
}
```

Both `encode_prompt()` in bundle and `_get_t5_prompt_embeds()` in backend use identical tokenizer/encoder setup.

---

## 4. TEXT EMBEDDING PRE-COMPUTATION

**ANSWER: No pre-computation - embeddings computed on-the-fly in both paths**

### Training (bundle.py, line 156-193):
```python
@torch.no_grad()
def encode_prompt(self, prompt, max_sequence_length=128, ...):
    # On-the-fly encoding:
    text_inputs = self.tokenizer(
        prompt,
        padding='max_length',
        max_length=max_sequence_length,  # ← Truncate to 128
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    seq_lens = attention_mask.gt(0).sum(dim=1).long()  # ← Count actual tokens

    outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    prompt_embeds = outputs.last_hidden_state.to(device=device, dtype=dtype)
    
    # ← CRITICAL: Trim to actual sequence length, then pad to max_sequence_length
    prompt_embeds = [emb[:seq_len] for emb, seq_len in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([emb, emb.new_zeros(max_sequence_length - emb.size(0), emb.size(1))])
            for emb in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds  # Shape: [B, 128, hidden_dim]
```

### Inference (prism_backend.py, line 866-909):
```python
@torch.no_grad()
def _get_t5_prompt_embeds(self, prompt, max_sequence_length=512, ...):
    # On-the-fly encoding (IDENTICAL to training):
    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,  # ← Truncate to 512
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()  # ← Count actual tokens

    prompt_embeds = self.text_encoder(
        text_input_ids.to(device), mask.to(device)
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
    # ← IDENTICAL trimming and padding logic
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )

    # ← Additional repeat for num_motion_per_prompt (line 904-907)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_motion_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(
        batch_size * num_motion_per_prompt, seq_len, -1
    )

    return prompt_embeds
```

**Key difference in inference**: `num_motion_per_prompt` repeat logic (lines 904-907) not present in training.

---

## 5. ATTENTION MASK HANDLING

### Training (prism_trainer.py, line 87-93):
```python
model_pred = self.bundle.transformer(
    hidden_states=noisy_latents,
    encoder_hidden_states=text_states,      # Shape: [B, 128, text_dim]
    timestep=timesteps,
    hidden_states_mask=padding_mask if num_frames is not None else None,
    encoder_hidden_states_mask=None,        # ← ALWAYS NONE during training!
).float()
```

### Inference (prism_backend.py, line 420-427):
```python
noise_pred = current_model(
    hidden_states=latent_model_input,
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,    # Shape: [B, 512, text_dim]
    attention_kwargs=attention_kwargs,
    is_causal=self.config.is_causal,
    hidden_states_mask=motion_mask,
    # NOTE: encoder_hidden_states_mask is NOT passed here either
)
```

**KEY FINDING**: `encoder_hidden_states_mask` is **NEVER used** in either training or inference!

The transformer expects this optional parameter but it's not utilized. The text attention masks are NOT applied.

---

## 6. TEXT EMBEDDING SHAPE TRANSFORMATION

### Training:
```
Input caption: string (e.g., "a person walks forward")
                 ↓
Tokenize + pad to 128 tokens
                 ↓
UMT5Encoder (frozen)
                 ↓
Output shape: [batch_size, 128, hidden_dim]  (e.g., [2, 128, 768 or 4096])
                 ↓
Passed to transformer as encoder_hidden_states
```

### Inference (512-token default):
```
Input prompt: string
                 ↓
Tokenize + pad to 512 tokens
                 ↓
UMT5Encoder (frozen)
                 ↓
Output shape: [batch_size, 512, hidden_dim]
                 ↓
Repeat by num_motion_per_prompt
                 ↓
Passed to transformer as encoder_hidden_states
```

**MISMATCH**: Transformer sees 128-token sequences during training, but 512-token sequences during inference!

---

## 7. PROMPT DROPOUT

### Training:
```python
def encode_prompt(self, prompt, prompt_drop_rate=0.0, ...):
    if prompt_drop_rate > 0:
        prompt = ['' if torch.rand(1).item() < prompt_drop_rate else p for p in prompt]
```

Default in trainer: `prompt_drop_rate=0.1` (line 23)

- 10% of prompts during training are replaced with empty strings
- This creates "" → tokenized to [CLS, PAD, PAD, ...] → zero embeddings

### Inference:
No prompt dropout applied. All text is preserved.

**Impact**: Classifier-free guidance learns from training with 10% empty prompts, but inference always uses full text.

---

## 8. PADDING & TRUNCATION STRATEGY

Both paths use identical tokenization logic:

```python
text_inputs = self.tokenizer(
    prompt,
    padding='max_length',      # Pad to max_length (not to longest in batch)
    max_length=max_sequence_length,
    truncation=True,           # Truncate long sequences
    add_special_tokens=True,   # Add [CLS], [SEP], etc.
    return_attention_mask=True,
    return_tensors='pt',
)
```

But then BOTH apply the same trimming:
```python
seq_lens = attention_mask.gt(0).sum(dim=1).long()  # Count actual tokens (excluding [PAD])
prompt_embeds = [emb[:seq_len] for emb, seq_len in zip(prompt_embeds, seq_lens)]
```

This means:
- Embeddings are trimmed to actual token length
- Then padded back to `max_sequence_length` with zero vectors

**Important**: Both use zero-padding, not learnable padding tokens.

---

## 9. TRANSFORMER FORWARD PASS

From `transformer_prism.py`:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    hidden_states_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states_mask: Optional[torch.Tensor] = None,  # ← Accepted but not used in training!
    attention_kwargs: Optional[Dict[str, Any]] = None,
    is_causal: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Args:
        encoder_hidden_states (torch.Tensor): Text encoder hidden states.
            Shape: [B, N_ctx, text_dim] where N_ctx is typically 512.
        encoder_hidden_states_mask (torch.Tensor, optional): Attention mask for text tokens.
            Shape: [B, N_ctx]. Values: 1 = visible/valid, 0 = masked/padding.
    """
```

The model supports masking text tokens (line 362-374), but:
- **Training never passes this mask** (always `None`)
- **Inference also doesn't pass this mask**

This means the transformer attends to ALL text positions including padding zeros!

---

## 10. DATASET FLOW

From `random_motion_text_dataset.py`:

```python
def get_data_info(self, idx) -> Dict:
    return {
        'motion': self.motion[idx].clone(),
        'num_frames': torch.tensor(self.num_frames, dtype=torch.long),
        'caption': self.captions[idx % len(self.captions)],  # ← Raw string caption
    }
```

From `single_agent_text_dataset.py`:
```python
def prepare_data(self, idx: int) -> dict:
    return {
        "motion_path": ...,
        "caption_path": ...,  # ← Path to caption file
    }
```

Pipeline processes:
1. Load caption from file or use provided string
2. Pass to trainer's `__getitem__`
3. Batch via `flexible_collate`
4. Passed to `train_step(batch: Dict)` where `batch['caption']` is a list of strings

---

## ROOT CAUSES OF TRAINING-INFERENCE MISMATCH

### 🔴 PRIMARY ISSUE: Max Sequence Length (128 vs 512)
- **Training**: Truncates text to 128 tokens
- **Inference**: Uses 512 tokens by default
- **Effect**: Transformer extrapolates to longer sequences than seen in training
- **Symptom**: Model may struggle with variable-length text encoding in the 256-512 range

### 🟠 SECONDARY ISSUE: No Encoder Attention Mask
- **Training**: `encoder_hidden_states_mask=None` always
- **Inference**: Also doesn't pass this mask
- **Effect**: Model attends to padding zeros, which might dilute text signal
- **Could fix**: Pass proper text attention masks derived from token sequences

### 🟡 TERTIARY ISSUE: Prompt Dropout
- **Training**: 10% of text → empty (for CFG training)
- **Inference**: Never empty
- **Effect**: Model sees different distribution of text embeddings
- **Mitigation**: This is intentional for classifier-free guidance

### 🟡 MINOR ISSUE: Num Motion Per Prompt Repeat
- **Training**: No repeat logic
- **Inference**: Repeats embeddings by `num_motion_per_prompt` (default 1)
- **Effect**: Minimal impact if num_motion_per_prompt=1

---

## RECOMMENDATIONS

### 1. **IMMEDIATE FIX: Match Max Sequence Length**
```python
# In prism_backend.py, set consistent default
max_sequence_length: int = 128,  # Match training!
```

Or in trainer config, increase to 256:
```python
PrismTrainer(
    max_text_length=256,  # Increase from default 128
)
```

### 2. **ENABLE TEXT ATTENTION MASKING**
```python
# In prism_trainer.py
text_states_mask = ... # Compute from seq_lens
model_pred = self.bundle.transformer(
    hidden_states=noisy_latents,
    encoder_hidden_states=text_states,
    encoder_hidden_states_mask=text_states_mask,  # ← ADD THIS
    ...
)
```

### 3. **ADD OPTIONAL DROPOUT AT INFERENCE**
```python
# In prism_backend.py, add parameter
def __call__(
    self,
    ...
    use_cfg_dropout: bool = False,  # Apply CFG dropout at inference
    ...
):
    if use_cfg_dropout and negative_prompt is not None:
        # Randomly drop prompt tokens for both positive and negative
        pass
```

### 4. **VERIFY TEXT EMBEDDING DIMENSIONS**
Check that `hidden_dim` of text encoder matches transformer's `text_dim` parameter.
If mismatch, there's a projection layer that could introduce additional variation.

### 5. **LOG TEXT SHAPES DURING INFERENCE**
Add debugging:
```python
print(f"Text embeddings shape: {prompt_embeds.shape}")
print(f"Text embeddings mean: {prompt_embeds.mean():.6f}, std: {prompt_embeds.std():.6f}")
```

Compare mean/std between training batches and inference runs.

---

## VERIFICATION CHECKLIST

- [ ] Check training config for `max_text_length` actual value
- [ ] Check inference default `max_sequence_length` actual value
- [ ] Add print statements to log text shape in both paths
- [ ] Verify text encoder is frozen (no gradients) in training
- [ ] Check if `encoder_hidden_states_mask` support is needed
- [ ] Run inference with `max_sequence_length=128` and compare outputs
- [ ] Verify hidden_dim of UMT5 matches transformer text_dim

---

## FILES INVOLVED

**Training:**
- `/hftrainer/trainers/motion/prism_trainer.py` (line 56-61)
- `/hftrainer/models/motion/prism/bundle.py` (line 156-193, encode_prompt)
- `/hftrainer/datasets/motion/random_motion_text_dataset.py`

**Inference:**
- `/hftrainer/pipelines/motion/prism_backend.py` (line 809-909, encode_prompt)
- `/hftrainer/pipelines/motion/prism_pipeline.py` (wrapper)
- `/hftrainer/models/motion/prism/network/transformer_prism.py` (forward)

**Transformer:**
- `/hftrainer/models/motion/prism/network/transformer_prism.py` (forward pass, mask handling)
- `/hftrainer/models/motion/prism/network/block_with_mask.py` (attention implementation)
