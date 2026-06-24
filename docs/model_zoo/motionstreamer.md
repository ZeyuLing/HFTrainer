# MotionStreamer

Streaming/autoregressive text-to-motion baseline integrated into the hftrainer
Model Zoo. Our reproduction is **fully self-contained and independent of
`ref_repo`** at runtime: the causal TAE, the LLaMA autoregressive transformer,
the per-token diffusion head and the Gaussian-diffusion sampler live in
`hftrainer.models.motion.motionstreamer.network`. The `save_pretrained` /
`from_pretrained` round-trip is **bit-identical** (`max-abs-diff = 0.0` for both
the TAE and the AR weights).

| | |
|---|---|
| **Task** | Text-to-Motion (T2M) |
| **Bundle / Pipeline** | `MotionStreamerBundle` / `MotionStreamerPipeline` |
| **Processed HF artifact** | [`ZeyuLing/hftrainer-motionstreamer-humanml272`](https://huggingface.co/ZeyuLing/hftrainer-motionstreamer-humanml272) |
| **Motion representation** | **MotionStreamer-272** (272-dim, 30 fps) |
| **Text encoder** | SentenceT5-XXL (`sentence-transformers/sentence-t5-xxl`, frozen) |
| **Paper** | *MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model*, 2025 — [arXiv:2503.15451](https://arxiv.org/abs/2503.15451) |
| **Original code** | https://github.com/zju3dv/MotionStreamer |

---

## Weights

Current hftrainer artifact (diffusers-style `from_pretrained`):

| Artifact | Location | Contents | Status |
|---|---|---|---|
| MotionStreamer HumanML3D-272 | [`ZeyuLing/hftrainer-motionstreamer-humanml272`](https://huggingface.co/ZeyuLing/hftrainer-motionstreamer-humanml272) | `tae.safetensors` + `ar.safetensors` + `ms_config.json` + `Mean.npy` / `Std.npy` | public Hub artifact; complete SentenceT5 packaging pending |
| local mirror | `checkpoints/motionstreamer/t2m_humanml272` | same layout | optional local cache |

**Use directly from the Hub:**

```python
from hftrainer.pipelines.motionstreamer import MotionStreamerPipeline

pipe = MotionStreamerPipeline.from_pretrained(
    "ZeyuLing/hftrainer-motionstreamer-humanml272",
    device="cuda",
)
motions = pipe.infer_t2m(["a person walks forward then turns around"], [120])  # list of (T, 272)
```

Complete text-encoder packaging is still pending for the current public
MotionStreamer artifact: the TAE/AR weights reload through
`MotionStreamerPipeline.from_pretrained`, but SentenceT5-XXL is currently
resolved by name rather than stored inside the repo.

**Or download to disk first:**

```bash
huggingface-cli download ZeyuLing/hftrainer-motionstreamer-humanml272 \
    --local-dir checkpoints/motionstreamer/t2m_humanml272
```

---

## Motion representation

**MotionStreamer-272**, a 272-dim global motion representation at 30 fps
(see the [272-dim representation repo](https://github.com/Li-xingXiao/272-dim-Motion-Representation)).
Generation path:

```
text -> SentenceT5-XXL -> LLaMA AR (CFG, per-token diffusion sampling)
     -> latent tokens (dim 16) -> causal TAE decoder (×4 upsample) -> 272-dim motion
```

Convert to/from HumanML3D-263 with `hftrainer.motion.representation.convert`
(`hml263_to_motion272`, etc.).

---

## Evaluation

Generation pairs mirror `MotionStreamer272Evaluator.load_test_pairs()` (per
`(name, caption)` on the released `humanml3d_272` test split); each prediction is
scored against its GT/caption with the persisted MS-272 evaluator. Reproduce
with:

```bash
# 1) generate (8-GPU sharded)
bash scripts/eval/_run_ms_h3d272_shards.sh
# 2) score
python3 scripts/eval/eval_ms_h3d272.py --pred_dir outputs/evaluation/ms_h3d272/ms_272
```

### MotionStreamer-272 evaluator (native space)

The hftrainer `MotionStreamer272Evaluator` is the same TMR-style evaluator used
in the paper (matching feature scale: MM-Dist ≈ 15, Diversity ≈ 27). Paper
numbers below are from the ICCV 2025 HumanML3D test-set table.

> _Full-set generation (7412 pairs, 8 GPUs) is in progress; the `hftrainer`
> column is filled in once scoring completes._

| Metric | hftrainer | MotionStreamer paper (ICCV'25) |
|---|---|---|
| FID ↓ | _pending_ | 11.790 |
| R-Precision Top-1 / 2 / 3 ↑ | _pending_ | 0.631 / 0.802 / 0.859 |
| MM-Dist ↓ | _pending_ | 16.081 |
| Diversity → | _pending_ | 27.284 |
| **GT(real)** FID / R@1 / R@3 / MM / Div | 0.0 / 0.706 / 0.911 / 15.01 / 27.36 | 0.002 / 0.702 / 0.914 / 15.151 / 27.492 |

The GT(real) row already reproduces the MotionStreamer paper *Real motion* row,
confirming the evaluator; the model row follows once generation finishes.

---

## Implementation notes

- **hftrainer-native runtime**: `hftrainer/models/motion/motionstreamer/network/` holds
  `tae.py` / `causal_cnn.py` / `resnet.py` (causal TAE), `llama_model.py` (LLaMA
  AR), `diffloss.py` + `diffusion/` (per-token diffusion head). Only relative
  imports are package-local.
- **Text encoder reloaded by name**: SentenceT5-XXL is frozen and not duplicated
  into the artifact (like CLIP for MDM).
- **Guidance**: classifier-free, default scale `4.0`, token unit length `4`.
