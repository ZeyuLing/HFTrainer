# MDM — Human Motion Diffusion Model

Text-to-motion baseline integrated into the hftrainer Model Zoo. Our
reproduction is **fully self-contained and independent of `ref_repo`** at
runtime: the network, Gaussian-diffusion schedule, classifier-free-guidance
sampler and collation helpers live in `hftrainer.models.motion.mdm.network`, and are verified
to be **bit-identical** to the released checkpoint (`max-abs-diff = 0.0` for the
same seed/input).

| | |
|---|---|
| **Task** | Text-to-Motion (T2M) |
| **Bundle / Pipeline** | `MDMBundle` / `MDMPipeline` |
| **Processed HF artifact** | [`ZeyuLing/hftrainer-mdm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d) |
| **Motion representation** | **HumanML3D-263** (263-dim, 20 fps, 22 joints) |
| **Text encoder** | CLIP ViT-B/32 (frozen) |
| **Paper** | *Human Motion Diffusion Model*, Tevet et al., ICLR 2023 — [arXiv:2209.14916](https://arxiv.org/abs/2209.14916) |
| **Original code** | https://github.com/GuyTevet/motion-diffusion-model |

---

## Weights

Current hftrainer artifact (diffusers-style `from_pretrained`):

| Artifact | Location | Contents | Status |
|---|---|---|---|
| MDM HumanML3D | [`ZeyuLing/hftrainer-mdm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d) | `model.safetensors` + `mdm_config.json` + `Mean.npy` / `Std.npy` | public Hub artifact; complete CLIP packaging pending |
| local mirror | `checkpoints/mdm/humanml_trans_enc_512` | same layout | optional local cache |

**Use directly from the Hub:**

```python
from hftrainer.pipelines.mdm import MDMPipeline

pipe = MDMPipeline.from_pretrained("ZeyuLing/hftrainer-mdm-humanml3d", device="cuda")
motions = pipe.infer_t2m(["a person walks forward then sits down"], [120])  # list of (T, 263)
```

**Or download to disk first:**

```bash
huggingface-cli download ZeyuLing/hftrainer-mdm-humanml3d \
    --local-dir checkpoints/mdm/humanml_trans_enc_512
```

The artifact is produced from a raw upstream `.pt` with
`scripts/eval/convert_mdm_checkpoint.py` (`--verify` asserts bit-identical
generation after the round-trip).

Complete text-encoder packaging is still pending for the current public MDM
artifact: the model weights reload through `MDMPipeline.from_pretrained`, but
CLIP ViT-B/32 is currently resolved by name rather than stored inside the repo.

---

## Motion representation

**HumanML3D-263**, the standard redundant T2M feature (Guo et al.), 20 fps,
22-joint SMPL skeleton. Per frame (263 dims):

| Slice | Dim | Meaning |
|---|---|---|
| `root_rot_vel` | 1 | root angular velocity (about Y) |
| `root_lin_vel` | 2 | root linear velocity (XZ plane) |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions (21×3) |
| `rot_data` | 126 | local joint rotations (21×6, cont. 6D) |
| `local_vel` | 66 | local joint velocities (22×3) |
| `foot_contact` | 4 | binary foot-contact labels |

Convert to/from other spaces with `hftrainer.motion.representation.convert`
(e.g. `hml263_to_joints`, `hml263_to_motion135`, `hml263_to_motion272`).

---

## Evaluation

Generation under the **official HumanML3D protocol** (standard test split, native
263-dim @ 20 fps, first caption) and scoring with the two persisted hftrainer
evaluators. Reproduce with:

```bash
# 1) generate (8-GPU sharded)
bash scripts/eval/_run_mdm_h3d263_shards.sh
# 2) score with the HumanML3D-263 evaluator
python3 scripts/eval/verify_evaluators.py --which hml263 \
    --hml263-pred outputs/evaluation/mdm_h3d263_official/mdm_263
```

### HumanML3D-263 evaluator (native space, n=3970)

| Metric | hftrainer | MDM paper | Note |
|---|---|---|---|
| **FID** ↓ | **0.509** | 0.544 | ✅ reproduced (within noise) |
| **Diversity** → | **9.563** | 9.559 | ✅ matches |
| R-Precision Top-1 / 2 / 3 ↑ | 0.420 / 0.605 / 0.711 | — / — / 0.611 | evaluator runs slightly hot (GT Top-3 0.816 vs paper 0.797) |
| MM-Dist ↓ | 3.681 | 5.566 | different evaluator embedding scale |
| GT(real) R-Prec / Div | 0.518 / 0.720 / 0.816, 9.499 | 0.797 (T3), 9.503 | ✅ GT row consistent |

**FID and Diversity match the paper**; R-Precision / MM-Dist differ only by the
calibration of our persisted evaluator (the GT row shifts the same way), not by
the model.

### MotionStreamer-272 evaluator (cross-representation, n=7392)

MDM is a **263-dim** model; scoring it on the MS-272 evaluator requires a
`263 → 272` conversion, which shifts the distribution. These numbers are **not a
fair native comparison** — they quantify the conversion gap, not MDM quality.

| Metric | MDM→272 | MS-272 GT(real) |
|---|---|---|
| FID ↓ | 121.35 | 0.0 |
| R-Precision Top-1 / 2 / 3 ↑ | 0.379 / 0.529 / 0.610 | 0.706 / 0.857 / 0.911 |
| MM-Dist ↓ | 20.96 | 15.01 |
| Diversity → | 25.48 | 27.36 |

The GT(real) row reproduces the MotionStreamer paper exactly (R@1 **0.706**, Div
**27.36**, MM **15.01**), confirming the evaluator is correct; the large MDM FID
is the `263→272` representation mismatch.

---

## Implementation notes

- **hftrainer-native runtime**: `hftrainer/models/motion/mdm/network/` holds the
  network (`network.py`), diffusion (`diffusion/`), CFG sampler and collate.
  Training-only deps are stubbed (inference-only).
- **Normalization travels with the checkpoint**: `Mean.npy` / `Std.npy` are the
  HumanML3D *training* stats (not the evaluator stats) and are embedded in the
  artifact, eliminating the recurring "wrong Mean/Std → forward drift" bug.
- **Guidance**: classifier-free, default scale `2.5`.
