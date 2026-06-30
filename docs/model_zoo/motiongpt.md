# MotionGPT - Human Motion as a Foreign Language

Text-to-motion baseline integrated into the hftrainer Model Zoo. The runtime is
self-contained under `hftrainer.models.motion.motiongpt.network` and does not
import the original repository at inference time.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M), motion-language generation |
| **Bundle / Pipeline** | `MotionGPTBundle` / `MotionGPTPipeline` |
| **Processed HF artifact** | [`ZeyuLing/hftrainer-motiongpt-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motiongpt-humanml3d) |
| **Motion representation** | **HumanML3D-263** (263-dim, 20 fps, 22 joints) |
| **Architecture** | Motion tokenizer VQ-VAE + FLAN-T5-base-style language model with motion tokens |
| **Paper** | *MotionGPT: Human Motion as a Foreign Language*, Jiang et al., NeurIPS 2023 - [arXiv:2306.14795](https://arxiv.org/abs/2306.14795) |
| **Original code** | https://github.com/OpenMotionLab/MotionGPT |

---

## Weights

Self-contained hftrainer artifact:

| Artifact | Location | Contents | Status |
|---|---|---|---|
| MotionGPT HumanML3D | [`ZeyuLing/hftrainer-motiongpt-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motiongpt-humanml3d) | `motiongpt_s3_h3d.tar` + `assets/meta/{mean,std}.npy` + `deps/flan-t5-base/` + `model_index.json` | public Hub artifact |
| local mirror | `checkpoints/baselines/motiongpt` | same layout | optional local cache |

Use directly from the Hub:

```python
from hftrainer.pipelines.motiongpt import MotionGPTPipeline

pipe = MotionGPTPipeline.from_pretrained(
    "ZeyuLing/hftrainer-motiongpt-humanml3d",
    bundle_kwargs={"local_files_only": False},
    device="cuda",
)
motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
)  # list of (T, 263)
```

For a local mirror:

```python
pipe = MotionGPTPipeline.from_pretrained(
    "checkpoints/baselines/motiongpt",
    bundle_kwargs={"local_files_only": True},
    device="cuda",
)
```

## Motion Representation

MotionGPT natively generates **HumanML3D-263** at 20 fps. For shared SMPL and
MotionStreamer-272 evaluation, use the validated bridge:

```text
HumanML3D-263 -> SMPL motion_135 via IK refine-80 -> MotionStreamer-272
```

The artifact packages the released MotionGPT checkpoint, HumanML3D statistics,
and the local FLAN-T5-base tokenizer/config files required to instantiate the
language model without a separate upstream checkout.

## HumanML3D Leaderboard Metrics

The row below uses the shared HumanML3D official-test caption protocol and the
HML263 round-trip GT reference for SMPL-based evaluators. MotionCLIP metrics use
raw projection embeddings without L2 normalization.

| Evaluator | R1 up | R2 up | R3 up | FID down | MM down | Div up |
|---|---:|---:|---:|---:|---:|---:|
| MotionStreamer-272 | 0.4940 | 0.6352 | 0.6944 | 23.6811 | 19.6781 | 25.5410 |
| MotionCLIP-135 no-L2 | 0.3688 | 0.5049 | 0.5828 | 84.8756 | 42.8579 | 23.2174 |

Physical metrics:

| Slide down | Float down | Jitter down | Dynamic down |
|---:|---:|---:|---:|
| 3.8783 | 10.8835 | 5.1680 | 21.0609 |

## Implementation Notes

- Artifact inference imports only `hftrainer.models.motion.motiongpt.network`.
- The released checkpoint has FLAN-T5-base / T5-v1.1 FFN shapes rather than
  ordinary `t5-base` FFN shapes.
- The checkpoint stores a distinct LM head while sharing the encoder and
  decoder input embeddings; the bundle keeps `shared_encoder_decoder_untied_lm_head`.
- The validated HumanML3D setting uses the official no-length prompt mode
  (`official_nolen`) and the corrected official-test caption annotation.
