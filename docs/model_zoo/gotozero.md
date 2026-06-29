# Go to Zero (MotionMillion)

Million-scale, 3B/7B-parameter autoregressive text-to-motion model ("Go to Zero",
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
| **Processed HF artifact** | [`7B-train`](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-train-humanml272), [`3B-train`](https://huggingface.co/ZeyuLing/hftrainer-gotozero-3b-train-humanml272) |
| **Motion representation** | **humanml3d_272** (272-dim, 30 fps) — *identical layout to MotionStreamer-272* |
| **Tokenizer** | HumanVQVAE + **FSQ** (levels `[8,8,8,5,5,5]`, codebook 64000) |
| **AR model** | LLaMA-style 3B / 7B AR transformer (RoPE, length-causal text cross-attn) |
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
| Go-to-Zero 7B train-only HumanML3D-272 | [`ZeyuLing/hftrainer-gotozero-7b-train-humanml272`](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-train-humanml272) | `fsq.safetensors` + `ar.safetensors` + `mm_config.json` + `model_index.json` + `mean.npy` / `std.npy` + `text_encoder/` | public Hub artifact; Flan-T5-XL is packaged as safetensors |
| Go-to-Zero 3B train-only HumanML3D-272 | [`ZeyuLing/hftrainer-gotozero-3b-train-humanml272`](https://huggingface.co/ZeyuLing/hftrainer-gotozero-3b-train-humanml272) | same layout | public Hub artifact; Flan-T5-XL is packaged as safetensors |
| local mirror, 7B | `checkpoints/gotozero/hftrainer_7b_train_humanml272` | same layout | optional local cache |
| local mirror, 3B | `checkpoints/gotozero/hftrainer_3b_train_humanml272` | same layout | optional local cache |

The public leaderboard uses the **train-only** checkpoints for both 7B and 3B.
The older `t2m_*_all.zip` checkpoints were trained with all HumanML3D splits and
should not be used for HumanML3D test-set leaderboard rows.

**Use directly from the Hub (recommended):**

```python
from hftrainer.pipelines.motionmillion import MotionMillionPipeline

pipe = MotionMillionPipeline.from_pretrained(
    "ZeyuLing/hftrainer-gotozero-7b-train-humanml272",
    device="cuda",
)
# Use "ZeyuLing/hftrainer-gotozero-3b-train-humanml272" for the 3B train-only artifact.
# Cast the AR to bf16 for memory-efficient inference:
import torch
pipe.bundle.ar.to(dtype=torch.bfloat16)
motions = pipe.infer_t2m(["a person swings a golf club"])  # list of (T, 272)
```

Converter/debug code can also load explicit released upstream checkpoints
(`fsq.zip`, `motionmillion_7B.pth`) directly:

```python
bundle = MotionMillionBundle(
    fsq_path="checkpoints/motionmillion/pretrained_models/fsq.zip",
    ar_path="checkpoints/motionmillion/pretrained_models/motionmillion_7B.pth",
    text_model_name="checkpoints/flan-t5-xl",
)
```

Package the hftrainer artifact from local upstream weights:

```bash
python3 scripts/eval/convert_motionmillion_checkpoint.py \
  --ar checkpoints/motionmillion/pretrained_models/motionmillion_7B.pth \
  --out_dir checkpoints/gotozero/hftrainer_7b_train_humanml272 \
  --repo_id ZeyuLing/hftrainer-gotozero-7b-train-humanml272 \
  --text_model_source checkpoints/flan-t5-xl \
  --verify
```

Use the same converter with the 3B train-only checkpoint and
`--out_dir checkpoints/gotozero/hftrainer_3b_train_humanml272 --repo_id
ZeyuLing/hftrainer-gotozero-3b-train-humanml272` for the 3B artifact.

---

## Motion representation

**humanml3d_272**, the 272-dim global motion representation at 30 fps
(see the [272-dim representation repo](https://github.com/Li-xingXiao/272-dim-Motion-Representation)).
This is the **same layout** used by MotionStreamer-272, so after de-normalising
with the MotionMillion `vector_272` mean/std the raw 272 vectors feed *directly*
into the `MotionStreamer272Evaluator` — no rotation re-encoding required.
Generation path:

```
text -> Flan-T5-XL -> LLaMA-style AR (greedy, EOS-stopped, ≤150 tokens)
     -> FSQ de-quantize -> HumanVQVAE decoder (×2 upsample) -> 272-dim motion
```

The upstream demo sampler caps generation at 50 motion tokens (about 100 frames
at 30 fps), which truncates many HumanML3D test clips. The model block size
supports the full benchmark range, so the hftrainer reproduction uses
`max_sample_steps=150` and writes exact-GT-length MS272 predictions.

Convert to HumanML3D-263 with `hftrainer.motion.representation.convert`
(`motion272_to_hml263`).

---

## Evaluation

Generation uses the shared corrected HumanML3D official-test caption protocol
(`4042` clips, one verified caption per motion) and writes canonical-id MS272
files to:

```
outputs/evaluation/t2m/humanml3d_official_test/ms272/gotozero_7b_train
outputs/evaluation/t2m/humanml3d_official_test/ms272/gotozero_3b_train
```

Reproduce generation with the packaged artifact:

```bash
NGPU=6 TOTAL_SHARDS=6 LIMIT=0 \
OUT=outputs/evaluation/t2m/humanml3d_official_test/ms272/gotozero_7b_train \
ARTIFACT=checkpoints/gotozero/hftrainer_7b_train_humanml272 \
STEPS=150 DTYPE=bf16 \
bash scripts/eval/run_motionmillion_official272_exactlen_genonly.sh
```

Swap `OUT` and `ARTIFACT` to the `gotozero_3b_train` / `hftrainer_3b_train_humanml272`
paths for the 3B train-only row.

### MotionStreamer-272 evaluator (native space)

HumanML3D official test, corrected official captions, `n=4042` files (`nb=4032` after
32-way R-Precision batching):

| Metric | 7B train-only | 3B train-only | GT (real) |
|---|---:|---:|---:|
| FID ↓ | **3.081** | **3.066** | 0.0 |
| FID-refk ↓ | 4.957 | 4.981 | - |
| R-Precision Top-1 / 2 / 3 ↑ | 0.740 / 0.878 / 0.924 | 0.740 / 0.877 / 0.923 | 0.778 / 0.906 / 0.946 |
| MM-Dist ↓ | 15.371 | 15.381 | 14.820 |
| Diversity → | 27.575 | 27.560 | 27.853 |

Both train-only Go-to-Zero checkpoints remain very close to GT in the native
MS272 evaluator: the semantic retrieval gap is small, MM-Dist is near-real, and
the motion activation distribution has low FID.

### MotionCLIP and physical metrics

MotionCLIP metrics use the current leaderboard protocol: MS272 predictions are
converted to MotionCLIP-135 and compared against the HML3D-roundtrip GT reference
with **raw, non-L2-normalized** MotionCLIP projection embeddings. This metric is
more sensitive to representation conversion than the native MS272 evaluator, so
use it as a secondary diagnostic.

| Metric | 7B train-only | 3B train-only |
|---|---:|---:|
| MotionCLIP R-Precision Top-1 / 2 / 3 ↑ | 0.689 / 0.827 / 0.881 | 0.689 / 0.827 / 0.881 |
| MotionCLIP FID ↓ | 299.617 | 298.933 |
| MotionCLIP MM-Dist ↓ | 42.537 | 42.528 |
| MotionCLIP Diversity → | 23.092 | 23.117 |
| Slide ↓ | 4.441 | 4.440 |
| Float ↓ | 20.060 | 20.093 |
| Jitter ↓ | 9.906 | 9.886 |
| Dynamic → | 21.324 | 21.327 |

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
- **Length alignment**: the wrapper writes one `<HumanML3D id>.npy` file per
  corrected official-test motion, exactly cropped/padded to the GT
  `num_frames`.
- **Caption source**: generation and evaluation use
  `outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected/`.
  Do not fall back to the first raw HumanML3D caption, because several raw first
  captions are known mismatches.

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
