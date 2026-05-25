# PRISM Text Embedding - Code Location Reference

## Quick Navigation Guide

### 🔴 PRIMARY ISSUE: MAX SEQUENCE LENGTH

#### Training Side (128 tokens)
```
FILE: hftrainer/trainers/motion/prism_trainer.py
LINE: 24
CODE: max_text_length: int = 128,

LINE: 56-61
CODE: 
    text_states = self.bundle.encode_prompt(
        captions,
        max_sequence_length=self.max_text_length,  # ← 128
        prompt_drop_rate=self.prompt_drop_rate,
        dtype=next(self.bundle.transformer.parameters()).dtype,
    )
```

#### Encoding Implementation (Training)
```
FILE: hftrainer/models/motion/prism/bundle.py
LINE: 156-193
FUNCTION: def encode_prompt(self, prompt, max_sequence_length=128, ...)

KEY LINES:
  170-178: Tokenizer call with max_length=max_sequence_length
  179-180: Pass to text_encoder
  181:     seq_lens = attention_mask.gt(0).sum(dim=1).long()  # Count real tokens
  185:     prompt_embeds = [emb[:seq_len] for ...]            # Trim to actual
  186-192: Pad back to max_sequence_length with zeros
```

#### Inference Side (512 tokens) ⚠️
```
FILE: hftrainer/pipelines/motion/prism_backend.py
LINE: 334
CODE: max_sequence_length: int = 512,

LINE: 826-862
CODE: prompt_embeds, negative_prompt_embeds = self.encode_prompt(
          prompt=prompt,
          negative_prompt=negative_prompt,
          do_classifier_free_guidance=do_cfg,
          num_motion_per_prompt=1,
          max_sequence_length=max_sequence_length,  # ← 512
          device=device,
      )
```

#### Encoding Implementation (Inference)
```
FILE: hftrainer/pipelines/motion/prism_backend.py
LINE: 866-909
FUNCTION: def _get_t5_prompt_embeds(self, prompt, max_sequence_length=512, ...)

KEY LINES:
  877-885: Tokenizer call with max_length=max_sequence_length (512)
  889-891: Pass to text_encoder
  887:     seq_lens = mask.gt(0).sum(dim=1).long()  # Count real tokens
  893:     prompt_embeds = [u[:v] for ...]         # Trim to actual
  894-900: Pad back to max_sequence_length with zeros
  904-907: Repeat for num_motion_per_prompt (only in inference!)
```

---

### 🟠 SECONDARY ISSUE: ATTENTION MASK HANDLING

#### Training - No Encoder Mask
```
FILE: hftrainer/trainers/motion/prism_trainer.py
LINE: 87-93
CODE:
    model_pred = self.bundle.transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=text_states,
        timestep=timesteps,
        hidden_states_mask=padding_mask if num_frames is not None else None,
        encoder_hidden_states_mask=None,  # ← ALWAYS NONE!
    ).float()
```

#### Inference - No Encoder Mask
```
FILE: hftrainer/pipelines/motion/prism_backend.py
LINE: 420-427
CODE:
    noise_pred = current_model(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        attention_kwargs=attention_kwargs,
        is_causal=self.config.is_causal,
        hidden_states_mask=motion_mask,
        # NOTE: encoder_hidden_states_mask is NOT passed!
    )
```

#### Transformer Support for Masking
```
FILE: hftrainer/models/motion/prism/network/transformer_prism.py
LINE: 232-241
FUNCTION: def forward(self, hidden_states, timestep, encoder_hidden_states,
                       hidden_states_mask=None, encoder_hidden_states_mask=None, ...)

DOC (line 252-260): Explains both mask types

LINE: 362-374: Processing of encoder_hidden_states_mask
CODE:
    # Convert text mask to attention bias format for cross-attention
    if encoder_hidden_states_mask is not None:
        encoder_hidden_states_mask = (
            (
                (1.0 - encoder_hidden_states_mask.float())
                * torch.finfo(hidden_states.dtype).min
            )
            .unsqueeze(1)
            .unsqueeze(2)
        )
```

**Status**: Mask support exists in transformer but is NOT used in either training or inference!

---

### 🟡 TERTIARY ISSUE: PROMPT DROPOUT

#### Training with Dropout (10% by default)
```
FILE: hftrainer/trainers/motion/prism_trainer.py
LINE: 23
CODE: prompt_drop_rate: float = 0.1,

LINE: 59
CODE: prompt_drop_rate=self.prompt_drop_rate,  # ← 0.1 = 10%
```

#### Dropout Implementation
```
FILE: hftrainer/models/motion/prism/bundle.py
LINE: 167-168
CODE:
    if prompt_drop_rate > 0:
        prompt = ['' if torch.rand(1).item() < prompt_drop_rate else p for p in prompt]
```

#### Inference - No Dropout
```
FILE: hftrainer/pipelines/motion/prism_backend.py
LINE: 809-862
FUNCTION: def encode_prompt(...)
NOTE: No prompt_drop_rate parameter or logic
```

---

### 🟢 MINOR ISSUE: NUM_MOTION_PER_PROMPT

#### Training - No Repeat
```
FILE: hftrainer/trainers/motion/prism_trainer.py
NOTE: Direct use of text_states, no repeat logic
```

#### Inference - Repeat Logic
```
FILE: hftrainer/pipelines/motion/prism_backend.py
LINE: 904-907
CODE:
    # duplicate text embeddings for each generation per prompt
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_motion_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(
        batch_size * num_motion_per_prompt, seq_len, -1
    )
```

**Impact**: Minimal (default num_motion_per_prompt=1)

---

## Transformer Forward Pass Details

### Input Shape Transformation

#### Training receives:
```
encoder_hidden_states: [B, 128, text_dim]
  where:
    - B = batch_size
    - 128 = max_text_length from trainer config
    - text_dim = usually 768 or 4096 (UMT5 hidden size)
```

#### Inference receives (default):
```
encoder_hidden_states: [B, 512, text_dim]
  where:
    - B = batch_size
    - 512 = max_sequence_length from inference config
    - text_dim = same as training
```

### Cross-Attention Implementation

```
FILE: hftrainer/models/motion/prism/network/block_with_mask.py
NOTE: Actual cross-attention logic (need to check file for details)
```

---

## Dataset Flow

### Training Dataset
```
FILE: hftrainer/datasets/motion/random_motion_text_dataset.py
LINE: 43-48
FUNCTION: def get_data_info(self, idx) -> Dict

RETURNS:
{
    'motion': torch.Tensor,
    'num_frames': torch.tensor,
    'caption': string,  # ← Raw caption string
}
```

### More Complex Dataset
```
FILE: hftrainer/datasets/motion/motionhub/single_agent_text_dataset.py
LINE: 31-41
FUNCTION: def prepare_data(self, idx: int) -> dict

RETURNS:
{
    "motion_path": str,
    "caption_path": str,  # ← Path to caption file
}
```

---

## Configuration Points

### Training Config (need to check actual config file)
```
Look for: PrismTrainer(
    max_text_length=128,  # ← KEY PARAMETER
    prompt_drop_rate=0.1,
    ...
)
```

### Inference Config (default)
```
FILE: hftrainer/pipelines/motion/prism_backend.py
LINE: 334
    max_sequence_length: int = 512,
```

Also check: `main()` function at line 912-929 for hardcoded defaults

---

## Debugging Commands

### Check actual training max_text_length
```bash
grep -r "max_text_length" /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/configs/
grep -r "max_text_length" /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/
```

### Find all encode_prompt calls
```bash
grep -r "encode_prompt" /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer \
    --include="*.py" | grep -E "(def|\.)"
```

### Check text_dim in transformer config
```bash
grep -r "text_dim" /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/models/motion/prism/
```

---

## Files to Modify (Priority Order)

### 1. HIGHEST PRIORITY: Fix Inference Default
```
FILE: hftrainer/pipelines/motion/prism_backend.py
LINE: 334
CHANGE FROM: max_sequence_length: int = 512,
CHANGE TO:   max_sequence_length: int = 128,

RATIONALE: Match training configuration
```

### 2. MEDIUM PRIORITY: Add Text Masking
```
FILE: hftrainer/trainers/motion/prism_trainer.py
LINE: 56-61
ADD: encoder_hidden_states_mask=... (compute from seq_lens)

FILE: hftrainer/trainers/motion/prism_trainer.py  
LINE: 92
CHANGE FROM: encoder_hidden_states_mask=None,
CHANGE TO:   encoder_hidden_states_mask=<computed mask>
```

### 3. LOW PRIORITY: Document CFG Dropout
```
FILE: hftrainer/pipelines/motion/prism_backend.py
LOCATION: encode_prompt() docstring
ADD: Note about CFG dropout during training (10%) vs inference (0%)
```

---

## Testing Checklist

- [ ] Log text embedding shapes during training
  - Print shape of `text_states` in `train_step()`
  
- [ ] Log text embedding shapes during inference
  - Print shape of `prompt_embeds` in `generate_single_segment()`
  
- [ ] Compare mean/std of embeddings
  - Training: mean/std of real text tokens
  - Inference: compare with 128 vs 512 default
  
- [ ] Run inference with modified max_sequence_length=128
  - Compare output quality
  - Measure generation consistency
  
- [ ] Verify text encoder frozen in training
  - Check `.requires_grad` for text_encoder parameters
  
- [ ] Check if attention masks are computed
  - Add debug print in transformer forward()
  - Verify encoder_hidden_states_mask is None
