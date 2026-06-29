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
`mld_humanml_v1.ckpt` and `motionlcm_humanml_v1.ckpt`. These are the one-token
latent checkpoints (`latent_dim=[1, 256]`) compatible with the official T2M
test config. The non-v1 files in the same upstream folder are a different
sixteen-token latent family and should not be treated as the model-card
benchmark artifact.

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

Generation follows the shared HumanML3D official-test protocol used by the
leaderboard: 4042 official test ids, corrected official captions under
`outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected/`,
native 263-dim at 20 fps, and one prediction per test id.

```bash
python3 scripts/eval/motionlcm_t2m_h3d263.py \
    --anno_file outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json \
    --anno_data_dir . \
    --model_path checkpoints/motionlcm/humanml3d \
    --num_inference_steps 1 \
    --out_dir outputs/evaluation/t2m/humanml3d_official_test/hml263/motionlcm
```

The full reproduction pipeline writes the canonical outputs:

| Representation | Canonical path |
|---|---|
| HML263 | `outputs/evaluation/t2m/humanml3d_official_test/hml263/motionlcm` |
| SMPL motion_135 | `outputs/evaluation/t2m/humanml3d_official_test/motion135/motionlcm` |
| MotionStreamer-272 | `outputs/evaluation/t2m/humanml3d_official_test/ms272/motionlcm` |

Run the Taiji wrapper for full generation, conversion, and evaluators:

```bash
python3 scripts/submit/submit_motionlcm_hml3d_full_taiji.py \
    --gpu V100 \
    --num-gpus 8 \
    --num-inference-steps 1 \
    --elastic
```

Report the LCM step count (`--num_inference_steps`) alongside any metrics. The
model-zoo table should use metrics copied from the generated evaluator JSONs
under `outputs/evaluation/t2m/humanml3d_official_test/_runs/<run>/metrics/`,
not handwritten values.
For HumanML3D-263 semantic metrics, the evaluator `texts_dir` must match the
captions used for generation. The corrected official-test run is scored
with
`outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected/texts`;
scoring these outputs against the older CondMDI text files produces mismatched
R-Precision / MM-Dist.

Current HumanML3D official-test metrics (4042 generated motions, corrected
official caption protocol, NFE=1):

| Evaluator | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
|---|---:|---:|---:|---:|---:|---:|
| HumanML3D-263 (corrected official captions) | 0.5093 | 0.7080 | 0.8108 | 0.3396 | 2.9694 | 9.6407 |
| MotionStreamer-272 (HML roundtrip GT) | 0.5657 | 0.7346 | 0.8075 | 44.0549 | 19.4543 | 24.6395 |
| MotionCLIP-135 no-L2 (HML roundtrip GT) | 0.3620 | 0.5157 | 0.6078 | 146.7212 | 42.6430 | 22.9160 |

Physical diagnostics on SMPL motion_135: Slide 4.2898, Float 19.1150, Jitter
3.2493, Dynamic 19.8250.

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
