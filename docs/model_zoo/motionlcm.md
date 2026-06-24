# MotionLCM - Latent Consistency Model for Human Motion Generation

Text-to-motion baseline integrated into the hftrainer Model Zoo. Our
reproduction keeps the MLD motion VAE, latent consistency denoiser, LCM
scheduler wiring, and SentenceT5 text wrapper in the native
`hftrainer.models.motion.motionlcm.network` package, so inference no longer
imports the upstream repository at runtime.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M) |
| **Bundle / Pipeline** | `MotionLCMBundle` / `MotionLCMPipeline` |
| **Motion representation** | **HumanML3D-263** (263-dim, 20 fps, 22 joints) |
| **Backbone** | MLD VAE + latent consistency denoiser, default 1 LCM step |
| **Text encoder** | SentenceT5-Large (`sentence-transformers/sentence-t5-large`, frozen) |
| **Paper** | *MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model*, Dai et al., ECCV 2024 |
| **Original code** | https://github.com/Dai-Wenxun/MotionLCM |

---

## Weights

Current hftrainer artifact:

| Artifact | Location | Contents | Status |
|---|---|---|---|
| MotionLCM HumanML3D | [`ZeyuLing/hftrainer-motionlcm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motionlcm-humanml3d) / `checkpoints/motionlcm/humanml3d` | `vae.safetensors` + `denoiser.safetensors` + `motionlcm_config.json` + `Mean.npy` / `Std.npy` | uploaded hftrainer artifact; v1 benchmark checkpoint |

The local artifact reloads through the same `from_pretrained` surface as the
published model-zoo checkpoints:

```python
from hftrainer.pipelines.motionlcm import MotionLCMPipeline

pipe = MotionLCMPipeline.from_pretrained(
    "ZeyuLing/hftrainer-motionlcm-humanml3d",
    device="cuda",
)
motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
    num_inference_steps=1,
)
```

Package the artifact from upstream checkpoints:

```bash
python3 scripts/eval/convert_motionlcm_checkpoint.py \
    --vae_ckpt ref_repo/MotionLCM/experiments_t2m/mld_humanml/mld_humanml_v1.ckpt \
    --denoiser_ckpt ref_repo/MotionLCM/experiments_t2m/motionlcm_humanml/motionlcm_humanml_v1.ckpt \
    --out_dir checkpoints/motionlcm/humanml3d \
    --verify
```

The frozen SentenceT5-Large encoder is resolved by name rather than duplicated
inside the artifact. For fully offline use, snapshot the text encoder into the
local Hugging Face cache before calling `from_pretrained`.

The published artifact uses the upstream v1 benchmark checkpoints:
`mld_humanml_v1.ckpt` and `motionlcm_humanml_v1.ckpt`. The non-v1 files in the
same upstream folder are a different latent-shape family and produced collapsed
dynamic features in the hftrainer evaluator (`FID=44.24`, `Diversity=5.70` on
HML3D-263); do not use those numbers as model-card metrics.

---

## Motion representation

**HumanML3D-263**, the standard redundant T2M feature (Guo et al.), 20 fps,
22-joint SMPL skeleton. Per frame (263 dims):

| Slice | Dim | Meaning |
|---|---|---|
| `root_rot_vel` | 1 | root angular velocity (about Y) |
| `root_lin_vel` | 2 | root linear velocity (XZ plane) |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions (21x3) |
| `rot_data` | 126 | local joint rotations (21x6, cont. 6D) |
| `local_vel` | 66 | local joint velocities (22x3) |
| `foot_contact` | 4 | binary foot-contact labels |

MotionLCM samples in the MLD latent space and decodes directly back to
HumanML3D-263. Convert to SMPL or MotionStreamer-272 with
`hftrainer.motion.representation.convert` when cross-model comparison requires
another evaluator space.

---

## Evaluation

Generation follows the official HumanML3D protocol: standard test split,
native 263-dim @ 20 fps, first caption, and one prediction per test id.

```bash
# 1) generate
python3 scripts/eval/motionlcm_t2m_h3d263.py \
    --data_root ref_repo/CondMDI/dataset/HumanML3D \
    --model_path checkpoints/motionlcm/humanml3d \
    --num_inference_steps 1 \
    --out_dir outputs/evaluation/motionlcm_h3d263_official/motionlcm_263

# 2) score with the HumanML3D-263 evaluator
python3 scripts/eval/verify_evaluators.py --which hml263 \
    --hml263-pred outputs/evaluation/motionlcm_h3d263_official/motionlcm_263
```

Report the LCM step count (`--num_inference_steps`) alongside any metrics. The
model-zoo table should use metrics copied from the generated evaluator JSON,
not handwritten values.

### HumanML3D-263 evaluator (fixed v1 artifact, n=3970)

Metric JSON:
`outputs/evaluation/motionlcm_hml3d_v1_fixed_20260617/metrics/verify_hml263.json`.

| Metric | hftrainer |
|---|---:|
| FID ↓ | 0.2921 |
| R-Precision Top-1 / 2 / 3 ↑ | 0.4958 / 0.6906 / 0.7883 |
| Diversity → | 9.5662 |
| MM-Dist ↓ | 3.0813 |
| GT(real) R-Precision Top-1 / 2 / 3 | 0.5135 / 0.7108 / 0.8069 |
| GT(real) Diversity / MM-Dist | 9.4527 / 2.9323 |

The debug sanity check that made the previous metrics untrusted was feature
distribution collapse: the old non-v1 artifact produced root/velocity features
with order-of-magnitude smaller variance and almost-always-on foot contacts.
The fixed v1 artifact restores root velocity, local velocity, root height and
foot-contact statistics close to the HumanML3D test distribution before metric
evaluation.

### MotionStreamer-272 evaluator (cross-representation, n=7392)

Metric JSON:
`outputs/evaluation/motionlcm_hml3d_v1_fixed_ms272_ik8_20260617/metrics/verify_ms272.json`.

| Metric | MotionLCM HML263→SMPL135→MS272 | MS272 GT(real) |
|---|---:|---:|
| FID ↓ | 149.9622 | 0.0 |
| R-Precision Top-1 / 2 / 3 ↑ | 0.4428 / 0.6059 / 0.6904 | 0.7059 / 0.8569 / 0.9106 |
| Diversity → | 24.7223 | 27.2813 |
| MM-Dist ↓ | 20.3028 | 15.0066 |

As with other native HML263 baselines, the MS272 row includes a representation
bridge (`HML263 -> SMPL motion_135 via IK refine-80 -> MotionStreamer-272`) and
should be interpreted as a cross-representation diagnostic, not a native
MotionLCM paper number.

---

## Implementation notes

- **hftrainer-native runtime**: `hftrainer/models/motion/motionlcm/network/` holds
  the MLD VAE, latent denoiser, text wrapper, scheduler config, and generation
  helper with package-local imports.
- **Checkpoint architecture is inferred from raw weights**: upstream releases
  include both one-token v1 and sixteen-token checkpoint families; raw loading
  reads `vae.global_motion_token` / `vae.latent_pre.weight` so the artifact is
  built with the matching latent shape.
- **Sub-modules**: `vae` + `denoiser` + `scheduler`; the default generation path
  uses distilled classifier-free guidance folded into the timestep conditioning.
- **Normalization travels with the checkpoint**: `Mean.npy` / `Std.npy` are the
  HumanML3D training stats embedded in the artifact.
- **Text encoder**: SentenceT5-Large is frozen and currently resolved by name;
  keep this explicit in any published Hub card.
