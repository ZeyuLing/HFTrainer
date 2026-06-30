# MotionLab - Unified Human Motion Generation and Editing

Text-to-motion baseline integrated into the hftrainer Model Zoo. The runtime is
self-contained under `hftrainer.models.motion.motionlab.network` and does not
import the original repository at inference time.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M), motion generation / editing research stack |
| **Bundle / Pipeline** | `MotionLabBundle` / `MotionLabPipeline` |
| **Processed HF artifact** | [`ZeyuLing/hftrainer-motionlab-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motionlab-humanml3d) |
| **Motion representation** | **HumanML3D-263** (263-dim, 20 fps, 22 joints) |
| **Architecture** | RFMotion / MotionFlow Transformer with CLIP text conditioning |
| **Paper** | *MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm*, Guo et al., ICCV 2025 - [arXiv:2502.02358](https://arxiv.org/abs/2502.02358) |
| **Original code** | https://github.com/Diouo/MotionLab |

---

## Weights

Self-contained hftrainer artifact:

| Artifact | Location | Contents | Status |
|---|---|---|---|
| MotionLab HumanML3D | [`ZeyuLing/hftrainer-motionlab-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motionlab-humanml3d) | `motionflow.ckpt` + `configs/` + `Mean.npy` / `Std.npy` + `mean_motion.npy` / `std_motion.npy` + `model_index.json` | public Hub artifact |
| local mirror | `checkpoints/baselines/motionlab` | same layout | optional local cache |

Use directly from the Hub:

```python
from hftrainer.pipelines.motionlab import MotionLabPipeline

pipe = MotionLabPipeline.from_pretrained(
    "ZeyuLing/hftrainer-motionlab-humanml3d",
    device="cuda",
)
motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
)  # list of (T, 263)
```

For a local mirror:

```python
pipe = MotionLabPipeline.from_pretrained("checkpoints/baselines/motionlab", device="cuda")
```

## Motion Representation

MotionLab natively generates **HumanML3D-263** at 20 fps. For shared SMPL and
MotionStreamer-272 evaluation, use the validated bridge:

```text
HumanML3D-263 -> SMPL motion_135 via IK refine-80 -> MotionStreamer-272
```

The artifact contains both the HumanML3D denormalization statistics and
MotionLab's internal motion statistics so the published pipeline does not depend
on a separate dataset checkout.

## HumanML3D Leaderboard Metrics

The row below uses the shared HumanML3D official-test caption protocol and the
HML263 round-trip GT reference for SMPL-based evaluators.

| Evaluator | R1 up | R2 up | R3 up | FID down | MM down | Div up |
|---|---:|---:|---:|---:|---:|---:|
| MotionStreamer-272 | 0.6367 | 0.7882 | 0.8529 | 25.4469 | 17.9756 | 25.5355 |
| MotionCLIP-135 no-L2 | 0.4807 | 0.6457 | 0.7353 | 102.7770 | 41.5472 | 23.0179 |

Physical metrics:

| Slide down | Float down | Jitter down | Dynamic down |
|---:|---:|---:|---:|
| 2.4231 | 4.0795 | 5.8493 | 24.3519 |

## TP2M HumanML3D Metrics

These rows score MotionLab on the canonical HumanML3D TP2M protocols
`humanml3d_official_test_c1/c5/c9` with the MotionStreamer-272 evaluator
and the selected-caption text directory used by the T2M leaderboard. Each
condition has `ids_with_required_files=4042`; `nb` is the evaluator-consumed
count after the standard min/max motion-length filter.

| Cond frames | nb | FID native | FID refk | R@1 | R@2 | R@3 | MM-Dist | Diversity | Metric JSON |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3968 | 49.9744 | 50.0184 | 0.5262 | 0.6706 | 0.7437 | 18.7170 | 26.5698 | `outputs/evaluation/tp2m/humanml3d_official_test_c1/ms272/motionlab/metrics/motionstreamer.json` |
| 5 | 3968 | 57.4464 | 57.2512 | 0.5015 | 0.6394 | 0.7198 | 19.1391 | 26.6071 | `outputs/evaluation/tp2m/humanml3d_official_test_c5/ms272/motionlab/metrics/motionstreamer.json` |
| 9 | 3968 | 59.4081 | 59.2748 | 0.5020 | 0.6494 | 0.7220 | 19.1731 | 26.5502 | `outputs/evaluation/tp2m/humanml3d_official_test_c9/ms272/motionlab/metrics/motionstreamer.json` |

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

- Artifact inference imports only `hftrainer.models.motion.motionlab.network`.
- Config targets are rewritten from the original `rfmotion.*` namespace into the
  vendored hftrainer namespace before model construction.
- The default inference stage is `demo`, matching the validated qualitative
  HumanML3D T2M setting.
