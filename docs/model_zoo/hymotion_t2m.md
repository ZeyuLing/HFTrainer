# HY-Motion T2M 1.0

Tencent Hunyuan open-source text-to-motion model integrated into the hftrainer
Model Zoo as a first-class `HyMotionT2MBundle` / `HyMotionT2MPipeline`.

The release ships two variants that share representation, text encoder, training
recipe, and inference protocol, and differ only in MMDiT size.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M) |
| **Bundle / Pipeline** | `HyMotionT2MBundle` / `HyMotionT2MPipeline` |
| **Processed HF artifact** | [`full`](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0), [`lite`](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0-lite) |
| **Architecture** | `HunyuanMotionMMDiT` flow matching, Euler ODE, 50 steps, CFG scale 5.0 |
| **Native representation** | 201-dim HY-Motion feature at 30 fps; hftrainer renders/scores from the `motion_135` slice |
| **Text encoder** | Qwen3-8B token context + CLIP-L sentence embedding, frozen and stored in the hftrainer artifact |
| **Original weights** | [`tencent/HY-Motion-1.0`](https://huggingface.co/tencent/HY-Motion-1.0), mirrored locally under `checkpoints/HY-Motion-1.0/` |

## Weights

Self-contained hftrainer artifacts are stored locally and reload with
`HyMotionT2MBundle.from_pretrained`. They include the motion transformer,
classifier-free null embeddings, 201-dim Mean/Std stats, and frozen Qwen3-8B /
CLIP-L text encoder directories. The artifact writes `model_index.json`
metadata for hftrainer discovery, but it is not a native diffusers
`DiffusionPipeline` repo.

| Variant | Local artifact | Processed Hugging Face artifact | Contents |
|---|---|---|---|
| HY-Motion T2M 1.0 | `checkpoints/hymotion_t2m/1.0b` | [`ZeyuLing/hftrainer-hymotion-t2m-1.0`](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0) | `motion_transformer.safetensors`, `hymotion_t2m_config.json`, `model_index.json`, `Mean.npy`, `Std.npy`, `text_encoder/llm/`, `text_encoder/sentence/` |
| HY-Motion T2M 1.0-Lite | `checkpoints/hymotion_t2m/0.46b` | [`ZeyuLing/hftrainer-hymotion-t2m-1.0-lite`](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0-lite) | same layout |

Use a local artifact:

```python
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline

pipe = HyMotionT2MPipeline.from_pretrained(
    "checkpoints/hymotion_t2m/1.0b",
    device="cuda",
    text_dtype="bf16",
    num_steps=50,
    text_guidance_scale=5.0,
    should_apply_smoothing=True,
)
out = pipe({"caption": ["a person walks forward."], "num_frames": [196]})
rot6d = out["rot6d"]      # (B, T, 22, 6)
transl = out["transl"]    # (B, T, 3)
```

For the HF artifact:

```python
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline

pipe = HyMotionT2MPipeline.from_pretrained(
    "ZeyuLing/hftrainer-hymotion-t2m-1.0",
    device="cuda",
    text_dtype="bf16",
    num_steps=50,
    text_guidance_scale=5.0,
    should_apply_smoothing=True,
)
out = pipe({"caption": ["a person walks forward."], "num_frames": [196]})
rot6d = out["rot6d"]      # (B, T, 22, 6)
transl = out["transl"]    # (B, T, 3)
```

The same config can be reconstructed without loading weights:

```python
cfg_bundle = HyMotionT2MBundle.from_config(
    "checkpoints/hymotion_t2m/1.0b/hymotion_t2m_config.json"
)
assert not cfg_bundle.text_encoder_requires_external_weights()
```

Artifacts are produced with:

```bash
python3 scripts/eval/convert_hymotion_checkpoint.py \
    --out_dir checkpoints/hymotion_t2m/1.0b --variant 1.0b --verify

python3 scripts/eval/convert_hymotion_checkpoint.py \
    --config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --out_dir checkpoints/hymotion_t2m/0.46b --variant 0.46b --verify
```

## Variants

| | HY-Motion T2M 1.0 | HY-Motion T2M 1.0-Lite |
|---|---:|---:|
| `feat_dim` | 1280 | 1024 |
| `num_layers` | 27 | 18 |
| `num_heads` | 20 | 16 |
| `input_dim` / `output_dim` | 201 / 201 | 201 / 201 |
| config | `configs/hymotion_t2m/hymotion_t2m_201dim_full.py` | `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` |
| upstream checkpoint | `checkpoints/HY-Motion-1.0/HY-Motion-1.0/latest.ckpt` | `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` |

## Evaluation Protocol

Published Model-Zoo metrics use the official HY-Motion inference path:

- CFG scale 5.0
- 50 Euler ODE steps
- MMDiT / ODE / null embeddings / Mean-Std in fp32
- text encoder in bf16, with text features upcast to fp32 before MMDiT
- decode smoothing enabled: SLERP on rot6d and Savitzky-Golay on root translation

HY-Motion outputs SMPL `motion_135`; for MotionStreamer comparison it is encoded
to MotionStreamer-272 and scored with `MotionStreamer272Evaluator`. HumanML3D-263
cross-eval converts the same indexed MS272 predictions and paired MS272 GT clips
through `motion272_to_hml263`, then scores with `HumanML263Evaluator`.

### MotionStreamer-272 Evaluator

HY-Motion T2M 1.0 smooth full HumanML3D test run:
`outputs/evaluation/hymotion_h3d272/metrics_smooth.json`.

| Metric | HY-Motion T2M 1.0 | MS272 GT/Real |
|---|---:|---:|
| **FID** ↓ | **16.021** | 0.000 |
| R-Precision Top-1 ↑ | **0.737** | 0.706 |
| R-Precision Top-2 ↑ | **0.881** | 0.857 |
| R-Precision Top-3 ↑ | **0.929** | 0.911 |
| **MM-Dist** ↓ | **14.789** | 15.007 |
| **Diversity** → | **27.187** | 27.367 |

HY-Motion T2M 1.0-Lite MS272 metrics are pending
`outputs/evaluation/hymotion_h3d272/metrics_lite_smooth.json`.

### HumanML3D-263 Cross-Eval

HY-Motion T2M 1.0 smooth cross-eval:
`outputs/evaluation/hymotion_h3d272/metrics_smooth_h3d263.json`.

This is not a native HumanML3D-263 generation run. It converts the indexed
MotionStreamer-272 predictions and their paired MS272 GT clips to HML263 and
scores the aligned population.

| Metric | HY-Motion T2M 1.0 | Converted GT/Real |
|---|---:|---:|
| **FID** ↓ | **0.103** | 0.000 |
| R-Precision Top-1 ↑ | **0.561** | 0.522 |
| R-Precision Top-2 ↑ | **0.761** | 0.725 |
| R-Precision Top-3 ↑ | **0.853** | 0.823 |
| **MM-Dist** ↓ | **2.532** | 2.691 |
| **Diversity** → | **10.031** | 9.876 |

Run details: `n_samples = 7340`, `n_repeats = 20`, `caption_selection = first`,
`drop_last = true`.

HY-Motion T2M 1.0-Lite HML263 metrics are pending
`outputs/evaluation/hymotion_h3d272/metrics_lite_smooth_h3d263.json`.

Reproduce the full-variant cross-eval:

```bash
python3 scripts/eval/eval_272dir_h3d263.py \
    --pred_dir outputs/evaluation/hymotion_h3d272/hy_272_smooth \
    --out_json outputs/evaluation/hymotion_h3d272/metrics_smooth_h3d263.json \
    --with_fid --workers 16 --caption_selection first
```

## Implementation Notes

- `HyMotionT2MBundle.save_pretrained` writes a self-contained hftrainer artifact
  with `motion_transformer.safetensors`, null CFG embeddings, Mean/Std,
  `text_encoder/llm/`, `text_encoder/sentence/`, and `model_index.json`
  metadata. Passing `include_text_encoder=False` is a legacy lightweight export
  mode and is not used for Model-Zoo publishing.
- `HyMotionT2MBundle.from_pretrained` accepts a local path or HF Hub id and keeps
  text encoder loading lazy; new artifacts resolve Qwen3-8B and CLIP-L from the
  artifact-local `text_encoder/` directories.
- `HyMotionT2MBundle.from_config` accepts either the raw bundle config or the
  saved `hymotion_t2m_config.json`, matching the hftrainer ModelBundle API.
- Raw/no-smoothing outputs are diagnostic only. The previous bf16-ODE / wrong-CFG
  metrics are deprecated and must not be used for Model-Zoo reporting.
- The `h3d272` suffix in scripts and output paths is historical; the evaluator
  space is MotionStreamer-272 on the HumanML3D test split.

## Training Text Data And Features

HY-Motion T2M / PRISM-style training uses
`data/annotation/train_hq_motionhub_hymotion.json`, a merged high-quality source
containing HYMotion data plus the MotionHub HQ subset. HYMotion M2M caption
experiments that target T2M evaluation should use this same annotation when the
goal is to avoid a caption-style gap against HumanML3D.

Text feature extraction must match the training text encoder:
`HYTextModel(llm_type="qwen3", sentence_emb_type="clipl")`, not
`qwen3_embedding`. `LoadPreExtractedTextEmbedding` resolves caption JSON paths to
precomputed `.pt` files as follows:

| Caption directory | Qwen3 feature directory |
|---|---|
| `human_checked_augmented_caption*` | `qwen3_human_checked_short` |
| `human_checked_caption*` | `qwen3_human_checked_short` |
| `improved_simple_augmented_caption*` | `qwen3_improved_simple_short` |
| `improved_simple_caption*` | `qwen3_improved_simple_short` |
| `augmented_caption*` | `qwen3_augmented` |
| `editing_caption` | `qwen3_editing` |
| `hierarchical_caption` | `data/motionhub_qwen3/.../qwen3_hierarchical/...` |

The MotionHub HQ `hierarchical_caption` files are not written back into the
source MotionHub tree. They are extracted into `data/motionhub_qwen3/` while
preserving the relative MotionHub layout:

```text
data/motionhub/<subset>/hierarchical_caption/<rel>.json
data/motionhub_qwen3/<subset>/qwen3_hierarchical/<rel>.pt
```

Use `scripts/data/extract_motionhub_hierarchical_qwen3.py` to build those files.
Each `.pt` stores `result[*].caption`, `result[*].granularity`, and
`result[*].text_embedding.{text_vec_raw,text_ctxt_raw,text_ctxt_raw_length}` in
the same schema as HYMotion caption features.
