# ViMoGen

Text-to-motion baseline integrated into the hftrainer Model Zoo. The runtime is
hftrainer-native and does not import the upstream repository at inference time:
the released ViMoGen transformer, scheduler, smoothing step, and DART276 motion
representation bridge live under `hftrainer.models.motion.vimogen` and
`hftrainer.motion.representation.dart276`.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M) |
| **Bundle / Pipeline** | `ViMoGenBundle` / `ViMoGenPipeline` |
| **Processed HF artifact** | [`ZeyuLing/hftrainer-vimogen-1.3b-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-vimogen-1.3b-humanml3d) |
| **Motion representation** | **DART276** (276-dim, 20 fps), decoded to SMPL `motion_135` for mesh visualization and evaluator bridges |
| **Backbone** | WanVideoTM2M 1.3B flow-matching DiT, 50 inference steps |
| **Text encoder** | Wan2.1 T2V-1.3B UMT5-XXL encoder |
| **Paper** | *ViMoGen: Scaling Full-Body Human Motion Generation through Visual Generative Priors* |
| **Original code** | https://github.com/MotrixLab/ViMoGen |

---

## Weights

Current hftrainer artifact:

| Artifact | Location | Contents | Status |
|---|---|---|---|
| ViMoGen-DiT 1.3B HumanML3D | [`ZeyuLing/hftrainer-vimogen-1.3b-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-vimogen-1.3b-humanml3d) | `model.pt` + `model_index.json` + `assets/meta/{mean,std}.npy` | public Hub artifact |

Load through the same `from_pretrained` surface as the other reproduced
baselines:

```python
from hftrainer.pipelines.vimogen import ViMoGenPipeline

pipe = ViMoGenPipeline.from_pretrained(
    "ZeyuLing/hftrainer-vimogen-1.3b-humanml3d",
    device="cuda",
)

motions_276 = pipe.infer_t2m(
    ["Full-body shot, stable camera. A person walks forward at an average pace."],
    [200],
    seed=0,
)
```

`ViMoGenBundle.from_pretrained` reads `model_index.json`. If the Wan2.1 base
assets are not already available locally, the bundle resolves the public
`Wan-AI/Wan2.1-T2V-1.3B` Hub repo declared by `wan_repo_id`.

---

## Motion Representation

ViMoGen emits **DART276**, the global DART-style representation:

```
text -> UMT5-XXL embeddings -> WanVideoTM2M DiT -> denormalized DART276
     -> dart276_to_motion135(...) for SMPL mesh / MotionCLIP / physics
     -> motion135_to_motion272(...) for MotionStreamer-272 evaluator
```

The public conversion API is:

```python
from hftrainer.motion.representation.dart276 import dart276_to_motion135

motion_135 = dart276_to_motion135(motion_276, rotation_convention="row")
```

See `docs/motion/representations.md` for the DART276 channel layout and the
root / coordinate-system convention.

---

## Evaluation

The leaderboard row uses the HumanML3D official-test split (`n=4042`) and the
shared corrected caption set. ViMoGen is sensitive to terse HumanML3D captions,
so generation uses a ViMoGen-style prompt rewrite derived from the corrected
caption. The rewrite adds presentation/context details such as camera, floor,
and motion-capture clothing while preserving the original action content. The
semantic evaluators are still computed against the same corrected HumanML3D
caption protocol used by the other methods.

### MotionStreamer-272 and MotionCLIP

| Evaluator | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
|---|---:|---:|---:|---:|---:|---:|
| MotionStreamer-272 (HML round-trip GT) | 0.4291 | 0.5687 | 0.6518 | 152.2095 | 21.0737 | 24.1803 |
| MotionCLIP-135 no-L2 (HML round-trip GT) | 0.3572 | 0.4992 | 0.5893 | 457.5443 | 44.4103 | 21.6806 |

### Physical Diagnostics

| Slide | Float | Jitter | Dynamic | Penet |
|---:|---:|---:|---:|---:|
| 6.9485 | 23.7270 | 4.4370 | 16.3838 | 0.0000 |

---

## Implementation Notes

- **hftrainer-native runtime**: `hftrainer.models.motion.vimogen.network` vendors
  the required ViMoGen transformer modules and scheduler.
- **No `ref_repo` dependency**: full-set HumanML3D inference uses
  `scripts/eval/vimogen_t2m_humanml3d.py` with `ViMoGenBundle` /
  `ViMoGenPipeline`.
- **Prompt sensitivity**: for leaderboard-quality generation, use the
  ViMoGen-style prompt rewrite workflow before inference. The plain corrected
  HumanML3D captions produce substantially weaker text following.
- **Evaluator bridge**: DART276 outputs are converted to repository
  `motion_135`, then to MotionStreamer-272 or MotionCLIP-135 for the shared
  cross-model leaderboard protocol.
