# FlowMDM - Seamless Human Motion Composition with Blended Positional Encodings

Text-to-motion and multi-prompt motion-composition baseline integrated into the
hftrainer Model Zoo. The runtime is self-contained under
`hftrainer.models.motion.flowmdm.network` and does not import the original
repository at inference time.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M), sequential / multi-prompt T2M |
| **Bundle / Pipeline** | `FlowMDMBundle` / `FlowMDMPipeline` |
| **Processed HF artifact** | [`ZeyuLing/hftrainer-flowmdm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-flowmdm-humanml3d) |
| **Motion representation** | **HumanML3D-263** (263-dim, 20 fps, 22 joints) |
| **Model family** | MDM-style diffusion with blended positional encodings |
| **Paper** | *Seamless Human Motion Composition with Blended Positional Encodings*, Barquero et al., CVPR 2024 - [arXiv:2402.15509](https://arxiv.org/abs/2402.15509) |
| **Original code** | https://github.com/BarqueroGerman/FlowMDM |

---

## Weights

Self-contained hftrainer artifact:

| Artifact | Location | Contents | Status |
|---|---|---|---|
| FlowMDM HumanML3D | [`ZeyuLing/hftrainer-flowmdm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-flowmdm-humanml3d) | `model000500000.pt` + `args.json` + `Mean.npy` / `Std.npy` + `model_index.json` | public Hub artifact |
| local mirror | `checkpoints/baselines/flowmdm` | same layout | optional local cache |

Use directly from the Hub:

```python
from hftrainer.pipelines.flowmdm import FlowMDMPipeline

pipe = FlowMDMPipeline.from_pretrained(
    "ZeyuLing/hftrainer-flowmdm-humanml3d",
    device="cuda",
)
motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
)  # list of (T, 263)
```

For a local mirror:

```python
pipe = FlowMDMPipeline.from_pretrained("checkpoints/baselines/flowmdm", device="cuda")
```

Sequential multi-prompt generation is exposed as:

```python
motions = pipe.infer_sequential_t2m(
    [["a person walks forward", "then turns around"]],
    [[80, 80]],
)
```

## Motion Representation

FlowMDM natively generates **HumanML3D-263** at 20 fps. For shared SMPL and
MotionStreamer-272 evaluation, use the validated bridge:

```text
HumanML3D-263 -> SMPL motion_135 via IK refine-80 -> MotionStreamer-272
```

The bridge is a representation-conversion diagnostic. Native HumanML3D quality
should be assessed in the 263-dim evaluator when paper-comparable numbers are
needed.

## HumanML3D Leaderboard Metrics

The row below uses the shared HumanML3D official-test caption protocol and the
HML263 round-trip GT reference for SMPL-based evaluators.

| Evaluator | R1 up | R2 up | R3 up | FID down | MM down | Div up |
|---|---:|---:|---:|---:|---:|---:|
| MotionStreamer-272 | 0.4737 | 0.6496 | 0.7312 | 36.3767 | 20.0018 | 25.1783 |
| MotionCLIP-135 no-L2 | 0.3317 | 0.4795 | 0.5737 | 131.9653 | 43.0012 | 22.9482 |

Physical metrics:

| Slide down | Float down | Jitter down | Dynamic down |
|---:|---:|---:|---:|
| 3.0452 | 7.4055 | 5.0130 | 22.3205 |

## TP2M HumanML3D Metrics

These rows score FlowMDM on the canonical HumanML3D TP2M protocols
`humanml3d_official_test_c1/c5/c9` with the MotionStreamer-272 evaluator
and the selected-caption text directory used by the T2M leaderboard. Each
condition has `ids_with_required_files=4042`; `nb` is the evaluator-consumed
count after the standard min/max motion-length filter.

| Cond frames | nb | FID native | FID refk | R@1 | R@2 | R@3 | MM-Dist | Diversity | Metric JSON |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3968 | 83.7731 | 86.6019 | 0.4493 | 0.6300 | 0.7061 | 19.8721 | 26.3653 | `outputs/evaluation/tp2m/humanml3d_official_test_c1/ms272/flowmdm/metrics/motionstreamer.json` |
| 5 | 3968 | 75.8526 | 78.1347 | 0.4806 | 0.6540 | 0.7291 | 19.4557 | 26.4668 | `outputs/evaluation/tp2m/humanml3d_official_test_c5/ms272/flowmdm/metrics/motionstreamer.json` |
| 9 | 3968 | 71.3384 | 73.3075 | 0.4902 | 0.6636 | 0.7424 | 19.2618 | 26.6253 | `outputs/evaluation/tp2m/humanml3d_official_test_c9/ms272/flowmdm/metrics/motionstreamer.json` |

Recompute command:

```bash
env RUN_ROOT=outputs/evaluation/tp2m/_runs/ms272_metrics_20260629 \
    GPU_LIST=0,1,2,3,4,5,6,7 SKIP_CACHE=1 \
    TEXT_DIR=outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/texts \
    bash scripts/eval/run_tp2m_ms272_metrics_remote.sh
```

Latest Taiji recompute: `tp2m_ms272_metrics_eval272_v100_0629_2124` plus
`tp2m_ms272_metrics_fill3-V100-1x8-2139` for the no-cache fill run.

## Implementation Notes

- Artifact inference imports only `hftrainer.models.motion.flowmdm.network`.
- The SMPL visualizer path from the original implementation is stubbed for T2M
  inference because the released HumanML3D checkpoint predicts HML263 features.
- `Mean.npy` and `Std.npy` are packaged with the artifact to avoid the recurring
  wrong-statistics failure mode.
