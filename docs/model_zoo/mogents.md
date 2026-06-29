# MoGenTS - Motion Generation Based on Spatial-Temporal Joint Modeling

Text-to-motion baseline integrated into the hftrainer Model Zoo. The runtime is
self-contained under `hftrainer.models.motion.mogents.network` and does not
import the original repository at inference time.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M) |
| **Bundle / Pipeline** | `MoGenTSBundle` / `MoGenTSPipeline` |
| **Processed HF artifact** | [`ZeyuLing/hftrainer-mogents-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mogents-humanml3d) |
| **Motion representation** | **HumanML3D-263** (263-dim, 20 fps, 22 joints) |
| **Tokenizer** | dual-stream RVQ-VAE: 1D auxiliary tokens + 2D spatial-temporal tokens |
| **Generator** | 1D/2D MaskTransformers + 1D/2D ResidualTransformers |
| **Text encoder** | CLIP ViT-B/32 (frozen) |
| **Paper** | *MoGenTS: Motion Generation based on Spatial-Temporal Joint Modeling*, Yuan et al., NeurIPS 2024 - [arXiv:2409.17686](https://arxiv.org/abs/2409.17686) |
| **Original code** | https://github.com/weihaosky/mogents |

---

## Weights

Self-contained hftrainer artifact:

| Artifact | Location | Contents | Status |
|---|---|---|---|
| MoGenTS HumanML3D | [`ZeyuLing/hftrainer-mogents-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mogents-humanml3d) | `vq.safetensors` + `mask_aux.safetensors` + `mask_ts.safetensors` + `res_aux.safetensors` + `res_ts.safetensors` + `length_est.safetensors` + `clip.safetensors` + `mogents_config.json` + `Mean.npy` / `Std.npy` | public Hub artifact |
| local mirror | `checkpoints/mogents/humanml3d` | same layout | optional local cache |

Convert the official checkpoints into a self-contained hftrainer artifact:

```bash
python3 scripts/eval/convert_mogents_checkpoint.py \
    --weights_root logs \
    --length_root checkpoints \
    --out_dir checkpoints/mogents/humanml3d \
    --verify
```

Expected artifact layout:

```text
checkpoints/mogents/humanml3d/
  mogents_config.json
  model_index.json
  vq.safetensors
  mask_aux.safetensors
  mask_ts.safetensors
  res_aux.safetensors
  res_ts.safetensors
  length_est.safetensors
  clip.safetensors
  Mean.npy
  Std.npy
```

## Use

```python
from hftrainer.pipelines.mogents import MoGenTSPipeline

pipe = MoGenTSPipeline.from_pretrained(
    "ZeyuLing/hftrainer-mogents-humanml3d",
    device="cuda",
)
motions = pipe.infer_t2m(
    ["a person walks forward then turns around"],
    [120],
)  # list of (T, 263)
```

For a local mirror:

```python
pipe = MoGenTSPipeline.from_pretrained("checkpoints/mogents/humanml3d", device="cuda")
```

## Motion Representation

MoGenTS natively generates **HumanML3D-263** at 20 fps. For cross-model
comparison with SMPL or MotionStreamer-272 methods, first generate the native
263-dim outputs and then use the validated bridge:

```text
HumanML3D-263 -> SMPL motion_135 via IK refine-80 -> MotionStreamer-272
```

The bridge is a representation-conversion diagnostic. It should not be treated
as the native MoGenTS paper metric space.

---

## Evaluation

Generate under the official HumanML3D test protocol and score with the
HumanML3D-263 evaluator:

```bash
python3 scripts/eval/mogents_t2m_h3d263.py \
    --model_path checkpoints/mogents/humanml3d \
    --out_dir outputs/evaluation/t2m/humanml3d_official_test/hml263/mogents_ts10_cfg4_rescfg5_seed0

python3 scripts/eval/verify_evaluators.py --which hml263 \
    --hml263-pred outputs/evaluation/t2m/humanml3d_official_test/hml263/mogents_ts10_cfg4_rescfg5_seed0 \
    --out-dir outputs/evaluation/t2m/humanml3d_official_test/hml263/mogents_ts10_cfg4_rescfg5_seed0/metrics
```

### HumanML3D-263 evaluator (native space, n=3970)

Metric JSON:
`outputs/evaluation/t2m/humanml3d_official_test/hml263/mogents_ts10_cfg4_rescfg5_seed0/metrics/verify_hml263.json`.

| Metric | hftrainer MoGenTS |
|---|---:|
| FID down | 0.0806 |
| R-Precision Top-1 / 2 / 3 up | 0.5219 / 0.7128 / 0.8056 |
| Diversity -> | 9.4063 |
| MM-Dist down | 2.9290 |
| GT(real) R-Precision Top-1 / 2 / 3 | 0.5135 / 0.7108 / 0.8069 |
| GT(real) Diversity / MM-Dist | 9.4527 / 2.9323 |

### SMPL motion_135 + MotionStreamer-272 evaluator

Convert the same HumanML3D test predictions to SMPL `motion_135` and then to
MotionStreamer-272:

```bash
METHOD=mogents RUN_ID=table1_mogents_exact_20260624 \
NUM_SHARDS=8 GPU_LIST=0,1,2,3,4,5,6,7 WORKERS=32 \
REFINE_ITERS=80 REFINE_LR=0.02 \
    bash scripts/eval/run_hml263_exact_smplfix_20260622.sh
```

The restartable script runs the following stages:

```bash
python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir outputs/evaluation/t2m/humanml3d_official_test/hml263/mogents \
    --out-dir outputs/evaluation/t2m/humanml3d_official_test/motion135/mogents \
    --source-fps 20 --target-fps 30 \
    --floor-align --refine-iters 80 --refine-lr 0.02 \
    --device cuda

python3 scripts/data/convert_motion135_to_h3d272.py \
    --in-dir outputs/evaluation/t2m/humanml3d_official_test/motion135/mogents \
    --out-dir outputs/evaluation/t2m/humanml3d_official_test/ms272/mogents \
    --workers 32

python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir outputs/evaluation/t2m/humanml3d_official_test/ms272/mogents \
    --gt-272-dir outputs/evaluation/t2m/humanml3d_official_test/_runs/noncanonical_legacy_20260623/ms272/gt_hml263_roundtrip_20260623_rootfix/predictions/ms272 \
    --text-dir outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/texts \
    --tag mogents_hmlroundtrip_fix_20260629 \
    --min-motion-len 1 \
    --out-json outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/hmlroundtrip_fix_20260629/results/mogents_motionstreamer272_hmlroundtrip.json
```

Metric JSON:
`outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/hmlroundtrip_fix_20260629/results/mogents_motionstreamer272_hmlroundtrip.json`.

Run summary:
`outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/table1_mogents_exact_20260624/logs/mogents.run.log`.

| Metric | MoGenTS HML263 -> SMPL135 -> MS272 | MS272 GT(real) |
|---|---:|---:|
| FID down | 20.1861 | 0.0 |
| R-Precision Top-1 / 2 / 3 up | 0.4993 / 0.6520 / 0.7354 | 0.7173 / 0.8720 / 0.9219 |
| Diversity -> | 25.6972 | 26.6252 |
| MM-Dist down | 19.5354 | 16.7867 |
| Samples present | 4042 | 4042 |
| Eval batches / nb | 126 / 4032 | 126 / 4032 |

Canonical bridge outputs contain 4042 HML263 predictions, 4042 SMPL
`motion_135` files, and 4042 MotionStreamer-272 files. The SMPL IK shard
summaries report zero conversion failures and mean joint-fit MPJPE around
15.2-15.4 mm.

## Implementation Notes

- **Architecture**: MoGenTS generates a 1D auxiliary token stream and a 2D
  spatial-temporal token grid, then decodes both streams together with the
  dual RVQ-VAE.
- **Runtime package**: `hftrainer/models/motion/mogents/network/` contains only
  the inference-time model components from the MIT-licensed upstream code.
- **Artifact loading**: `MoGenTSBundle.from_pretrained` consumes local/HF-style
  artifacts and stores CLIP once as `clip.safetensors`; raw upstream `.tar`
  checkpoints are supported only through explicit converter/debug paths.
- **Native representation**: generated outputs are HumanML3D-263. Any
  MotionStreamer-272 or SMPL `motion_135` comparison should be produced by the
  existing representation-conversion pipeline after generation.
