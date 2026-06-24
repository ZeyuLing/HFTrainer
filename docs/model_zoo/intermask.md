# InterMask — Collaborative Masked Modeling for Human Interaction

Two-person / human-object interaction generation baseline integrated into the
hftrainer Model Zoo. The reproduction is **self-contained and independent of
external source trees**: InterMask's VQ-VAE and MaskTransformer runtime is
implemented under `hftrainer.models.motion.intermask.network`, while
released checkpoints and stats live in `checkpoints/intermask`.

| | |
|---|---|
| **Task** | Two-person Text-to-Motion / InterX generation |
| **Bundle** | `InterMaskBundle` |
| **Processed HF artifacts** | **Not published**. Current InterHuman and InterX visual QA failed; Hub release is blocked until the generation / representation bridge is debugged. |
| **Local artifacts** | `checkpoints/intermask/hftrainer_interhuman`, `checkpoints/intermask/hftrainer_interx` |
| **Motion representation** | InterHuman native-262 per person, or InterX `(T, 56, 12)` |
| **Tokenizer** | RVQ-VAE |
| **Generator** | MaskTransformer, collaborative masked token decoding |
| **Text encoder** | CLIP ViT-L/14@336px (frozen) |
| **Paper** | *3D Human Interaction Generation via Collaborative Masked Modeling*, 2025 — [project](https://gohar-malik.github.io/intermask/) |
| **Original code** | https://github.com/gohar-malik/intermask |

## Weights

> **Current status:** these artifacts reproduce the current hftrainer runtime
> numerically, but both InterHuman and InterX samples show visible quality
> problems in the SMPL viewer. Treat the checkpoints as **debug-only** and do
> not use them for model-zoo release, Hugging Face publishing, or headline
> evaluation until the issue is isolated.

| Artifact | Location | Contents | Status |
|---|---|---|---|
| InterMask InterHuman | `checkpoints/intermask/hftrainer_interhuman` | `vq_default/`, `trans_default/`, `stats/global_mean.npy`, `stats/global_std.npy`, generated `README.md` | debug-only; visual QA failed |
| InterMask InterX | `checkpoints/intermask/hftrainer_interx` | `vq_default/`, `trans_default/`, `stats/interx_mean.npy`, `stats/interx_std.npy`, generated `README.md` | debug-only; visual QA failed |

Use the InterHuman checkpoint:

```python
from hftrainer.models.motion.intermask import InterMaskBundle

bundle = InterMaskBundle.from_pretrained(
    "checkpoints/intermask/hftrainer_interhuman",
    dataset_name="interhuman",
    device="cuda",
)
motion = bundle.generate(
    ["one person walks toward another person"],
    motion_len=90,
    seed=123,
)  # (B, T, 2, 262), denormalized InterHuman native-262
```

Use the InterX checkpoint:

```python
from hftrainer.models.motion.intermask import InterMaskBundle

bundle = InterMaskBundle.from_pretrained(
    "checkpoints/intermask/hftrainer_interx",
    dataset_name="interx",
    device="cuda",
)
motion = bundle.generate(
    ["one person passes an object to another person"],
    motion_len=90,
    seed=123,
)  # (B, T, 56, 12), official InterX evaluator layout
```

## Motion Representations

For `dataset_name="interhuman"`, `generate` returns denormalized
`(B, T, 2, 262)` native InterHuman features. This path is evaluated with
`InterHuman262Evaluator`.

For `dataset_name="interx"`, `generate` returns `(B, T, 56, 12)`, matching the
official Inter-X text-to-motion evaluator input layout.

## Evaluation

InterHuman uses the InterGen/InterCLIP evaluator:

```bash
python3 tools/eval_interclip_2p_native262.py \
  --gt outputs/evaluation/interhuman_gt_native262.npz \
  --pred InterMask=outputs/evaluation/intermask_native262.npz \
  --out-json outputs/evaluation/intermask_interclip262_metrics.json
```

InterX does **not** use InterCLIP. The official Inter-X repository points its
text-to-motion benchmark at `evaluation/text2motion/final_evaluation.py`, with
HHI evaluator weights laid out as:

```text
checkpoints/hhi/text_mot_match/model/finest.tar
```

hftrainer exposes that path as `InterXText2MotionEvaluator`. Place the official
Inter-X evaluator checkpoint under:

```text
checkpoints/evaluators/interx_text2motion/checkpoints/hhi/text_mot_match/model/finest.tar
```

Then call:

```python
from hftrainer.evaluation.evaluators import InterXText2MotionEvaluator

evaluator = InterXText2MotionEvaluator(device="cuda")
metrics = evaluator.evaluate(samples, mode="pred")
```

Each sample should provide `motion_gt`, `motion_pred`, `length`, and either
tokenized text (`tokens`) with HHI glove files or precomputed
`word_emb`/`pos_ohot`/`sent_len`.

## Verification

Parity with the original source tree was checked on short deterministic samples:

| Dataset | Output shape | max abs diff | mean abs diff |
|---|---|---:|---:|
| InterHuman | `(1, 16, 2, 262)` | 0.0 | 0.0 |
| InterX | `(1, 16, 56, 12)` | 0.0 | 0.0 |

Both checks used the same checkpoint, prompt, seed, `time_steps=2`, and
`motion_len=16`. This verifies the rewritten hftrainer runtime path against the
current implementation, but **does not validate visual quality**. The current
viewer results for both InterHuman and InterX are known to be wrong, so the next
debug step is to separate model inference correctness from normalization,
InterHuman/InterX decoding, and SMPL retargeting.
