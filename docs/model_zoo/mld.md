# MLD - Motion Latent Diffusion

Text-to-motion baseline integrated into the hftrainer Model Zoo. The
reproduction keeps the MLD motion VAE, latent denoiser, DDIM scheduler wiring,
and SentenceT5 text wrapper in the native `hftrainer` runtime, so inference no
longer imports the upstream repository.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M) |
| **Bundle / Pipeline** | `MLDBundle` / `MLDPipeline` |
| **Motion representation** | **HumanML3D-263** (263-dim, 20 fps, 22 joints) |
| **Backbone** | MLD VAE + latent diffusion denoiser, default 50 DDIM steps |
| **Text encoder** | SentenceT5-Large (`sentence-transformers/sentence-t5-large`, frozen) |
| **Paper** | *Executing Your Commands via Motion Diffusion in Latent Space*, Chen et al., CVPR 2023 |
| **Original code** | https://github.com/ChenFengYe/motion-latent-diffusion |

---

## Weights

Current hftrainer artifact:

| Artifact | Location | Contents | Status |
|---|---|---|---|
| MLD HumanML3D | [`ZeyuLing/hftrainer-mld-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mld-humanml3d) / `checkpoints/mld/humanml3d` | `vae.safetensors` + `denoiser.safetensors` + `mld_config.json` + `Mean.npy` / `Std.npy` | hftrainer artifact |

Load through the same `from_pretrained` surface as the other reproduced
baselines:

```python
from hftrainer.pipelines.mld import MLDPipeline

pipe = MLDPipeline.from_pretrained(
    "ZeyuLing/hftrainer-mld-humanml3d",
    device="cuda",
)
motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
    num_inference_steps=50,
)
```

Package the artifact from the upstream Lightning checkpoint:

```bash
python3 scripts/eval/convert_mld_checkpoint.py \
    --model_ckpt ref_repo/MotionLCM/experiments_t2m/mld_humanml/mld_humanml_v1.ckpt \
    --out_dir checkpoints/mld/humanml3d
```

The frozen SentenceT5-Large encoder is resolved by name rather than duplicated
inside the artifact. For fully offline use, snapshot the text encoder into the
local Hugging Face cache before calling `from_pretrained`.

---

## Motion Representation

**HumanML3D-263**, the standard redundant T2M feature (Guo et al.), 20 fps,
22-joint SMPL skeleton. Per frame (263 dims):

| Slice | Dim | Meaning |
|---|---:|---|
| `root_rot_vel` | 1 | root angular velocity (about Y) |
| `root_lin_vel` | 2 | root linear velocity (XZ plane) |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions (21x3) |
| `rot_data` | 126 | local joint rotations (21x6, continuous 6D) |
| `local_vel` | 66 | local joint velocities (22x3) |
| `foot_contact` | 4 | binary foot-contact labels |

MLD samples in latent space and decodes directly back to HumanML3D-263. Convert
to SMPL or MotionStreamer-272 only when a cross-representation evaluator needs
that space.

---

## Evaluation

Generation follows the shared HumanML3D official-test protocol used by the
leaderboard: 4042 official test ids, corrected selected captions under
`outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/`,
native 263-dim at 20 fps, and one prediction per test id.

```bash
python3 scripts/eval/mld_t2m_h3d263.py \
    --anno_file outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/test_hml3d_official272_gtlen_motionclip_selected_caption.json \
    --anno_data_dir . \
    --model_path checkpoints/mld/humanml3d \
    --num_inference_steps 50 \
    --out_dir outputs/evaluation/t2m/humanml3d_official_test/hml263/mld
```

The full reproduction pipeline writes the canonical outputs:

| Representation | Canonical path |
|---|---|
| HML263 | `outputs/evaluation/t2m/humanml3d_official_test/hml263/mld` |
| SMPL motion_135 | `outputs/evaluation/t2m/humanml3d_official_test/motion135/mld` |
| MotionStreamer-272 | `outputs/evaluation/t2m/humanml3d_official_test/ms272/mld` |

Run the Taiji wrapper for full generation, conversion, and evaluators:

```bash
python3 scripts/submit/submit_mld_standard_pipeline_taiji.py \
    --gpu V100 \
    --num-gpus 8 \
    --elastic
```

Report current metrics from the generated evaluator JSONs under
`outputs/evaluation/t2m/humanml3d_official_test/_runs/<run>/metrics/`.
For HumanML3D-263 semantic metrics, the evaluator `texts_dir` must match the
captions used for generation. The selected-caption official-test run is scored
with
`outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/texts`;
scoring these outputs against the older CondMDI text files produces mismatched
R-Precision / MM-Dist.

Current HumanML3D official-test metrics (4042 generated motions, selected
caption protocol):

| Evaluator | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
|---|---:|---:|---:|---:|---:|---:|
| HumanML3D-263 (selected captions) | 0.5176 | 0.7161 | 0.8159 | 0.2969 | 2.9498 | 9.6283 |
| MotionStreamer-272 (HML roundtrip GT) | 0.5660 | 0.7326 | 0.8095 | 39.7437 | 19.3374 | 24.9017 |
| MotionCLIP-135 no-L2 (HML roundtrip GT) | 0.3831 | 0.5380 | 0.6319 | 134.6484 | 42.4679 | 22.9470 |

Physical diagnostics on SMPL motion_135: Slide 4.2199, Float 16.7402, Jitter
3.2692, Dynamic 20.1758.

---

## Implementation Notes

- **hftrainer-native runtime**: `hftrainer.models.motion.mld` wraps the shared
  native MLD VAE / denoiser / SentenceT5 components and does not import
  `ref_repo` at inference time.
- **Scheduler**: MLD uses `diffusers.DDIMScheduler` with 50 inference steps by
  default (`eta=0.0`, `steps_offset=1`), matching the official inference config.
- **Classifier-free guidance**: the denoiser has no LCM `time_cond_proj`, so
  guidance uses the standard unconditional/conditional two-pass batch.
- **Normalization travels with the checkpoint**: `Mean.npy` / `Std.npy` are
  embedded in the artifact to avoid evaluator drift caused by mismatched
  HumanML3D statistics.
