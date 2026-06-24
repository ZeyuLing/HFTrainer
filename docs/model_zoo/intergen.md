# InterGen — Diffusion-based Multi-human Motion Generation

Two-person text-to-motion baseline integrated into the hftrainer Model Zoo. The
reproduction is **self-contained and independent of external source trees**:
InterGen's inference runtime lives in
`hftrainer.models.motion.intergen.network`, and the checkpoint plus
normalization stats live in `checkpoints/intergen/hftrainer_interhuman`.

| | |
|---|---|
| **Task** | Two-person Text-to-Motion |
| **Bundle** | `InterGenBundle` |
| **Processed HF artifact** | [`ZeyuLing/hftrainer-intergen-interhuman`](https://huggingface.co/ZeyuLing/hftrainer-intergen-interhuman) |
| **Local artifact** | `checkpoints/intergen/hftrainer_interhuman` |
| **Motion representation** | **InterHuman native-262** per person, 30 fps |
| **Text encoder** | CLIP ViT-L/14@336px (frozen) |
| **Paper** | *InterGen: Diffusion-based Multi-human Motion Generation under Complex Interactions*, CVPR 2024 — [arXiv:2304.05684](https://arxiv.org/abs/2304.05684) |
| **Original code** | https://github.com/tr3e/intergen |

## Weights

| Artifact | Location | Contents | Status |
|---|---|---|---|
| InterGen InterHuman | `checkpoints/intergen/hftrainer_interhuman` / [`ZeyuLing/hftrainer-intergen-interhuman`](https://huggingface.co/ZeyuLing/hftrainer-intergen-interhuman) | `intergen.ckpt`, `global_mean.npy`, `global_std.npy`, `intergen_config.json`, generated `README.md` | hftrainer inference artifact |

Load from Hugging Face:

```python
from hftrainer.models.motion.intergen import InterGenBundle

bundle = InterGenBundle.from_pretrained(
    "ZeyuLing/hftrainer-intergen-interhuman",
    device="cuda",
)
motion = bundle.generate(
    ["two people shake hands and then walk apart"],
    motion_len=120,
    seed=1234,
)  # (1, 120, 2, 262), denormalized InterHuman native-262
```

Use the local artifact in the same way:

```python
from hftrainer.models.motion.intergen import InterGenBundle

bundle = InterGenBundle.from_pretrained(
    "checkpoints/intergen/hftrainer_interhuman",
    device="cuda",
)
motion = bundle.generate(
    ["one person walks toward another person"],
    motion_len=210,
    seed=123,
)  # (B, T, 2, 262), denormalized InterHuman native-262
```

The runtime never imports `third_party/intergen`, `ref_repo`, `_vendor`, or a
copied upstream package path. `InterGenBundle` loads the native hftrainer
network modules and the artifact stats directly.

The Hub checkpoint is an inference-only hftrainer artifact: `intergen.ckpt`
contains the model `state_dict` and lightweight metadata, while optimizer,
callback, scheduler, and PyTorch-Lightning loop states are removed.

## Motion Representation

InterGen uses the **InterHuman native-262** feature per person:

| Slice | Dim | Meaning |
|---|---:|---|
| joint positions | 66 | 22 joints x xyz |
| joint velocities | 66 | 22 joints x xyz velocity |
| local rotations | 126 | 21 joints x 6D rotation |
| foot contacts | 4 | binary contact labels |

`InterGenBundle.generate` returns `(B, T, 2, 262)` after de-normalization with
the packaged InterGen training stats. Use
`hftrainer.motion.representation.interhuman262` for SMPL-X / joints conversion
when needed.

For visualization, the model-zoo web viewer fits the 262 joint-position block
to a body-only SMPL mesh. That mesh bridge is a viewer convenience; the
canonical model output and evaluator input remain native InterHuman-262.

## Evaluation

The official InterGen evaluation path is InterCLIP over native InterHuman-262
packs. In hftrainer this is:

```bash
python3 tools/eval_interclip_2p_native262.py \
  --gt outputs/evaluation/interhuman_gt_native262.npz \
  --pred InterGen=outputs/evaluation/intergen_native262.npz \
  --out-json outputs/evaluation/intergen_interclip262_metrics.json
```

Input packs are `.npz` files with `m1`, `m2`, `lens`, and `texts` arrays:

```python
np.savez(path, m1=m1, m2=m2, lens=lens, texts=texts)
```

## Verification

Parity with the original source tree was checked on a short deterministic
sample:

| Check | Result |
|---|---:|
| checkpoint load missing / unexpected | 0 / 0 |
| source vs hftrainer output max abs diff | 0.0 |
| source vs hftrainer output mean abs diff | 0.0 |
| artifact reload smoke | `Bundle.from_pretrained(...)` + one text prompt |

The parity run used the same checkpoint, prompt, seed, and the default
`ddim50` sampling strategy on a short sequence. The current viewer smoke was
also checked on `"two people shake hands and then walk apart"` with
`motion_len=120`, `seed=1234`; the generated InterHuman-262 output was fitted
to SMPL for inspection.
