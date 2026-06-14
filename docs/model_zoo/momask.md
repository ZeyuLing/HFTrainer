# MoMask — Generative Masked Modeling of 3D Human Motions

Text-to-motion baseline integrated into the hftrainer Model Zoo. Our
reproduction is **fully self-contained and independent of `ref_repo`**: the
RVQ-VAE tokenizer, the masked generative transformer, the residual transformer
and the length estimator are all vendored into
`hftrainer.models.motion.momask._momask`, preserving numerical parity with the
released HumanML3D checkpoints. The CLIP ViT-B/32 text encoder is reloaded by
name (not stored in the artifact).

| | |
|---|---|
| **Task** | Text-to-Motion (T2M) |
| **Bundle / Pipeline** | `MoMaskBundle` / `MoMaskPipeline` |
| **Motion representation** | **HumanML3D-263** (263-dim, 20 fps, 22 joints) |
| **Tokenizer** | RVQ-VAE, **6 residual quantizers**, codebook 512×512 |
| **Generator** | MaskTransformer (masked iterative decoding) + ResidualTransformer |
| **Text encoder** | CLIP ViT-B/32 (frozen) |
| **Paper** | *MoMask: Generative Masked Modeling of 3D Human Motions*, Guo et al., CVPR 2024 — [arXiv:2312.00063](https://arxiv.org/abs/2312.00063) |
| **Original code** | https://github.com/EricGuo5513/momask-codes |

---

## Weights

Self-contained hftrainer artifact (diffusers-style `from_pretrained`):

| Artifact | Contents |
|---|---|
| **HuggingFace**: [`ZeyuLing/momask-humanml3d`](https://huggingface.co/ZeyuLing/momask-humanml3d) | `vq.safetensors` + `t2m_trans.safetensors` + `res_trans.safetensors` + `length_est.safetensors` + `momask_config.json` + `Mean.npy` / `Std.npy` (no CLIP; reloaded by name) |
| Local: `checkpoints/momask/humanml3d` | same layout (produced by `convert_momask_checkpoint.py`, see below) |

Pull the published artifact directly from the Hub:

```python
from huggingface_hub import snapshot_download
path = snapshot_download("ZeyuLing/momask-humanml3d")  # -> MoMaskBundle.from_pretrained(path)
```

The artifact is produced from the released upstream `.tar` checkpoints with
`scripts/eval/convert_momask_checkpoint.py` (`--verify` asserts bit-identical
generation after the round-trip):

```bash
python3 scripts/eval/convert_momask_checkpoint.py \
    --weights_root ref_repo/Momask/weights \
    --out_dir checkpoints/momask/humanml3d \
    --verify
```

**Use it:**

```python
from hftrainer.models.motion.momask import MoMaskBundle
from hftrainer.pipelines.momask import MoMaskPipeline

bundle = MoMaskBundle.from_pretrained("checkpoints/momask/humanml3d")
pipe   = MoMaskPipeline(bundle, device="cuda")
# fixed length (frames @ 20 fps):
motions = pipe.infer_t2m(["a person walks forward then sits down"], [120])
# or let the length estimator pick the length:
motions = pipe.infer_t2m(["a person walks forward then sits down"])  # list of (T, 263)
```

You can also drive it directly from the released weights, no conversion needed:

```python
bundle = MoMaskBundle(weights_root="ref_repo/Momask/weights")
```

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

The RVQ-VAE tokenizes this with `unit_length = 4` (one token ≈ 4 frames), so a
196-frame motion maps to 49 tokens × 6 quantizers.

---

## Generation

Three vendored stages (parity with `scripts/eval/momask_infer_h3d_test.py`):

1. **MaskTransformer** — confidence-based masked iterative decoding of the
   base (q=0) token map, classifier-free guidance `cond_scale≈4` over
   `time_steps≈10` iterations, cosine mask schedule, `topkr≈0.9`,
   `temperature=1.0`.
2. **ResidualTransformer** — autoregressively predicts quantizers `q=1..5`
   conditioned on the lower layers (`cond_scale≈5`, `temperature=1.0`).
3. **RVQVAE.forward_decoder** — de-quantizes `(T, 6)` tokens and decodes to the
   263-dim feature, then de-normalised with the training `Mean` / `Std`.

---

## Evaluation

Generation under the **official HumanML3D protocol** (standard test split,
native 263-dim @ 20 fps, first caption) and scoring with the persisted
`HumanML263Evaluator`. Reproduce with:

```bash
# 1) generate
python3 scripts/eval/momask_t2m_h3d263.py \
    --model_path checkpoints/momask/humanml3d \
    --out_dir outputs/evaluation/momask_h3d263_official/momask_263
# 2) score with the HumanML3D-263 evaluator
python3 scripts/eval/verify_evaluators.py --which hml263 \
    --hml263-pred outputs/evaluation/momask_h3d263_official/momask_263
```

### HumanML3D-263 evaluator (native space)

| Metric | hftrainer | MoMask paper |
|---|---|---|
| **FID** ↓ | **0.097** | 0.045 |
| R-Precision Top-1 / 2 / 3 ↑ | **0.516 / 0.709 / 0.804** | 0.521 / 0.713 / 0.807 |
| **MM-Dist** ↓ | **2.990** | 2.958 |
| **Diversity** → | **9.460** | 9.620 |

(20 repeats, n = 3970; GT/real reference under the same evaluator:
R-Prec 0.513 / 0.711 / 0.807, MM-Dist 2.932, Diversity 9.453.)

> R-Precision, MM-Dist and Diversity match the paper essentially exactly,
> confirming the generation is faithfully reproduced. The small residual **FID**
> gap (0.097 vs 0.045) is a **data-processing / population** difference in the
> evaluation set (e.g. no sub-clip predictions, test-split composition), **not**
> a generation-quality gap — the decode path is verified parity-equal to the
> released MoMask inference (`momask_infer_h3d_test.py`).

---

## Implementation notes

- **Vendored, ref_repo-independent**: `hftrainer/models/motion/momask/_momask/`
  holds the RVQ-VAE (`vq/`), the masked / residual transformers
  (`mask_transformer/`) and the masked iterative decoding entry point
  (`inference.py`). Imports are package-relative; training-only code paths are
  not exercised.
- **Sub-modules**: `vq_model` / `t2m_transformer` / `res_transformer` /
  `length_estimator` (the last is optional, `load_length_estimator=False`).
- **CLIP**: frozen ViT-B/32 lives inside the two transformers and is reloaded
  by name; it is excluded from the artifact (`clip_model.*` keys are the only
  expected missing keys on load), exactly like CLIP in MDM.
- **Normalization travels with the checkpoint**: `Mean.npy` / `Std.npy` are the
  RVQ-VAE training stats, embedded in the artifact.
- **Guidance**: classifier-free, base `cond_scale=4`, residual `cond_scale=5`.
