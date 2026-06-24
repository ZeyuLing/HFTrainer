# Go to Zero (MotionMillion)

Million-scale, 7B-parameter autoregressive text-to-motion model ("Go to Zero",
ICCV 2025 **Highlight**) integrated into the hftrainer Model Zoo. Our
reproduction is **fully self-contained and independent of the original
repository** at runtime: the HumanVQVAE (FSQ tokenizer) and the LLaMA
autoregressive transformer live in
`hftrainer.models.motion.motionmillion.network`. Only the T2M inference path is
exercised.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M), zero-shot |
| **Bundle / Pipeline** | `MotionMillionBundle` / `MotionMillionPipeline` |
| **Processed HF artifact** | [`ZeyuLing/hftrainer-gotozero-7b-humanml272`](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-humanml272) |
| **Motion representation** | **humanml3d_272** (272-dim, 30 fps) — *identical layout to MotionStreamer-272* |
| **Tokenizer** | HumanVQVAE + **FSQ** (levels `[8,8,8,5,5,5]`, codebook 64000) |
| **AR model** | LLaMA 7B (n_layer=36, n_head=32, n_embd=4096, RoPE, length-causal text cross-attn) |
| **Text encoder** | Flan-T5-XL (`google/flan-t5-xl`, frozen, hidden 2048) |
| **Paper** | *Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data*, ICCV 2025 — [arXiv:2507.07095](https://arxiv.org/abs/2507.07095) |
| **Project page** | https://vankouf.github.io/MotionMillion/ |
| **Original code** | https://github.com/VankouF/MotionMillion-Codes |
| **Dataset** | [InternRobotics/MotionMillion](https://huggingface.co/datasets/InternRobotics/MotionMillion) |

---

## Weights

Current hftrainer artifact (diffusers-style `from_pretrained`):

| Artifact | Location | Contents | Status |
|---|---|---|---|
| Go-to-Zero 7B HumanML3D-272 | [`ZeyuLing/hftrainer-gotozero-7b-humanml272`](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-humanml272) | `fsq.safetensors` + `ar.safetensors` + `mm_config.json` + `model_index.json` + `mean.npy` / `std.npy` + `text_encoder/` | public Hub artifact; Flan-T5-XL is packaged as safetensors |
| local mirror | `checkpoints/gotozero/hftrainer_7b_humanml272` | same layout | optional local cache |

**Use directly from the Hub (recommended):**

```python
from hftrainer.pipelines.motionmillion import MotionMillionPipeline

pipe = MotionMillionPipeline.from_pretrained(
    "ZeyuLing/hftrainer-gotozero-7b-humanml272",
    device="cuda",
)
# cast the 7B AR to bf16 for a 32 GB GPU:
import torch
pipe.bundle.ar.to(dtype=torch.bfloat16)
motions = pipe.infer_t2m(["a person swings a golf club"])  # list of (T, 272)
```

Converter/debug code can also load explicit released upstream checkpoints
(`fsq.zip`, `t2m_7B_all.zip`) directly:

```python
bundle = MotionMillionBundle(
    fsq_path="checkpoints/motionmillion/pretrained_models/fsq.zip",
    ar_path="checkpoints/motionmillion/pretrained_models/t2m_7B_all.zip",
    text_model_name="checkpoints/flan-t5-xl",
)
```

Package the hftrainer artifact from local upstream weights:

```bash
python3 scripts/eval/convert_motionmillion_checkpoint.py \
  --out_dir checkpoints/gotozero/hftrainer_7b_humanml272 \
  --text_model_source checkpoints/flan-t5-xl \
  --verify
```

---

## Motion representation

**humanml3d_272**, the 272-dim global motion representation at 30 fps
(see the [272-dim representation repo](https://github.com/Li-xingXiao/272-dim-Motion-Representation)).
This is the **same layout** used by MotionStreamer-272, so after de-normalising
with the MotionMillion `vector_272` mean/std the raw 272 vectors feed *directly*
into the `MotionStreamer272Evaluator` — no rotation re-encoding required.
Generation path:

```
text -> Flan-T5-XL -> LLaMA 7B AR (greedy, EOS-stopped, ≤50 tokens)
     -> FSQ de-quantize -> HumanVQVAE decoder (×2 upsample) -> 272-dim motion
```

The released sampler emits at most 50 motion tokens (≈100 frames @ 30 fps); we
keep this faithful behaviour and truncate each prediction to the GT length for
evaluation (`max_sample_steps` is configurable on the pipeline).

Convert to HumanML3D-263 with `hftrainer.motion.representation.convert`
(`motion272_to_hml263`).

---

## Evaluation

Generation pairs mirror `MotionStreamer272Evaluator.load_test_pairs()` (per
`(name, caption)` on the released `humanml3d_272` test split, 7412 pairs).
Reproduce with:

```bash
# 1) generate + score (dedicated 8-GPU Taiji job, bf16)
python3 tools/taiji_submit.py mm_t2m_h3d272 --host_num 1 --host_gpu_num 8 --gpu_name V100 \
    --start-cmd "cd $PWD && bash scripts/eval/_run_motionmillion_h3d272_taiji.sh"
# or locally:
python3 scripts/eval/motionmillion_h3d272.py --out_dir outputs/evaluation/motionmillion_h3d272/mm_272 --device cuda --dtype bf16
python3 scripts/eval/eval_ms_h3d272.py --pred_dir outputs/evaluation/motionmillion_h3d272/mm_272
```

### MotionStreamer-272 evaluator (native space)

Full HumanML3D test set (7411 / 7412 pairs scored, 8×V100, `n_repeats=20`):

| Metric | hftrainer (Go-to-Zero 7B) | GT (real) |
|---|---|---|
| FID ↓ | **3.029** | 0.0 |
| R-Precision Top-1 / 2 / 3 ↑ | 0.696 / 0.848 / 0.903 | 0.704 / 0.856 / 0.911 |
| MM-Dist ↓ | 15.183 | 15.006 |
| Diversity → | 27.222 | 27.328 |

Go-to-Zero 7B gives the **lowest FID** of the Model-Zoo T2M baselines on this
evaluator (vs MotionStreamer 11.79 / HY-Motion-1.0 12.96), and its generated
motions are **almost as retrievable as the ground truth** (R@1 0.696 vs GT 0.704)
with matching MM-Dist (15.18 vs 15.01) and Diversity (27.22 vs 27.33).

#### Sampler / evaluation protocol (important reproduction notes)

- **Sampling length**: the released greedy sampler hard-codes a 50-token
  (`for k in range(51)`, ~100 frames @ 30 fps) cap that truncates long motions.
  The model itself (block size 301) generates the full length — we sample up to
  **150 tokens (~300 frames)**, which covers the entire HumanML3D length range.
  Using the released 50-token cap instead drops FID to ~22.9 purely from the
  length mismatch against the 60–300-frame GT.
- **KV-cache**: we add a cached decoder (`LLaMAHF.sample_cached`) that is verified
  **token-for-token identical** to the un-cached sampler while running ~7× faster
  (≈5 s vs ≈40 s per sample at 150 tokens), making full-set evaluation tractable.
- **Length alignment**: GT is encoded at its full `m_length`; predictions at their
  own generated length (mirroring `evaluation_transformer_motionmillion`). A short
  AR sample must **not** truncate the GT reference.

> Note: the official "Go to Zero" paper reports on **MotionMillion-Eval** (its own
> zero-shot benchmark), not the HumanML3D test set, so there is no directly
> comparable HumanML3D paper row. The numbers above place Go-to-Zero on the
> *same* in-house HumanML3D-272 evaluator as MDM / MotionStreamer for an
> apples-to-apples Model-Zoo comparison.

---

## Implementation notes

- **hftrainer-native runtime**: `hftrainer/models/motion/motionmillion/network/` holds
  `fsq.py` (Finite Scalar Quantization), `resnet.py` / `modules.py` (Haar
  patch/unpatch) / `encdec.py` / `vqvae.py` (HumanVQVAE tokenizer) and
  `llama.py` (LLaMA AR). The `args` namespace was refactored into explicit
  keyword arguments.
- **FSQ codebook**: `nb_code=65536` selects levels `[8,8,8,5,5,5]`, whose product
  is 64000 — the real codebook size; AR vocab = 64000 + 2 (PAD/EOS).
- **Checkpoint loader**: the released `t2m_*_all.zip` bundles a DeepSpeed optimizer
  state next to the `trans` weights; a tolerant unpickler stubs the unimportable
  classes so only the tensor `state_dict` is read (repo-independent).
- **Complete text encoder artifact**: Flan-T5-XL is frozen and stored under
  `text_encoder/` in the hftrainer artifact. `MotionMillionBundle.from_pretrained`
  resolves that artifact-local directory automatically.
- **Representation parity verified**: on smoke pairs the de-normalised prediction
  matches GT per-block scale almost exactly (e.g. position-block std 0.514 vs
  0.514; rot6d-block std 0.495 vs 0.489), confirming MotionMillion-272 ≡ MS-272.
