# MotionGPT3 - Human Motion as a Second Modality

Text-to-motion baseline integrated into the hftrainer Model Zoo. The runtime is
self-contained under `hftrainer.models.motion.motiongpt3.network` and does not
import the original repository at inference time.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M), motion-language generation |
| **Bundle / Pipeline** | `MotionGPT3Bundle` / `MotionGPT3Pipeline` |
| **Processed HF artifact** | [`ZeyuLing/hftrainer-motiongpt3-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motiongpt3-humanml3d) |
| **Motion representation** | **HumanML3D-263** (263-dim, 20 fps, 22 joints) |
| **Architecture** | Motion-language model with MotionGPT3 VAE and MoT-GPT2 adapter |
| **Paper** | *MotionGPT3: Human Motion as a Second Modality*, OpenMotionLab - [arXiv:2506.24086](https://arxiv.org/abs/2506.24086) |
| **Original code** | https://github.com/OpenMotionLab/MotionGPT3 |

---

## Weights

Self-contained hftrainer artifact:

| Artifact | Location | Contents | Status |
|---|---|---|---|
| MotionGPT3 HumanML3D | [`ZeyuLing/hftrainer-motiongpt3-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motiongpt3-humanml3d) | `motiongpt3.ckpt` + `configs/` + `assets/meta/{mean,std}.npy` + `deps/mot-gpt2/` + `model_index.json` | public Hub artifact |
| local mirror | `checkpoints/baselines/motiongpt3` | same layout | optional local cache |

Use directly from the Hub:

```python
from hftrainer.pipelines.motiongpt3 import MotionGPT3Pipeline

pipe = MotionGPT3Pipeline.from_pretrained(
    "ZeyuLing/hftrainer-motiongpt3-humanml3d",
    device="cuda",
)
motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
)  # list of (T, 263)
```

For a local mirror:

```python
pipe = MotionGPT3Pipeline.from_pretrained("checkpoints/baselines/motiongpt3", device="cuda")
```

## Motion Representation

MotionGPT3 natively generates **HumanML3D-263** at 20 fps. For shared SMPL and
MotionStreamer-272 evaluation, use the validated bridge:

```text
HumanML3D-263 -> SMPL motion_135 via IK refine-80 -> MotionStreamer-272
```

The artifact includes the local `mot-gpt2` adapter and HumanML3D statistics so
the published pipeline can be restored without an external runtime checkout.

## HumanML3D Leaderboard Metrics

The row below uses the shared HumanML3D official-test caption protocol and the
HML263 round-trip GT reference for SMPL-based evaluators.

| Evaluator | R1 up | R2 up | R3 up | FID down | MM down | Div up |
|---|---:|---:|---:|---:|---:|---:|
| MotionStreamer-272 | 0.6709 | 0.8242 | 0.8817 | 20.9913 | 17.5664 | 25.6889 |
| MotionCLIP-135 no-L2 | 0.4894 | 0.6570 | 0.7455 | 91.0385 | 41.5060 | 23.0747 |

Physical metrics:

| Slide down | Float down | Jitter down | Dynamic down |
|---:|---:|---:|---:|
| 3.8137 | 9.6933 | 4.7599 | 23.1948 |

## Implementation Notes

- Artifact inference imports only `hftrainer.models.motion.motiongpt3.network`.
- The bundle patches the small transformer compatibility fields required by
  newer `transformers` versions before loading the released checkpoint.
- The validated HumanML3D setting uses the `test` generation stage and
  temperature `1.0`.
