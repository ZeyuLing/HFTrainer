# T2M-GPT — Generating Human Motion from Textual Descriptions with Discrete Representations

Text-to-motion baseline integrated into the hftrainer Model Zoo. Our
reproduction is **fully self-contained and independent of `ref_repo`** at
runtime: the VQ-VAE motion tokenizer (`HumanVQVAE`) and the GPT-style
autoregressive generator (`Text2Motion_Transformer`) live in the native
`hftrainer.models.motion.t2mgpt.network` package, preserving numerical parity
with the released HumanML3D checkpoints. New hftrainer artifacts include the frozen CLIP ViT-B/32
text encoder as `clip.safetensors`; legacy lightweight artifacts still fall
back to loading CLIP by name.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M) |
| **Bundle / Pipeline** | `T2MGPTBundle` / `T2MGPTPipeline` |
| **Processed HF artifact** | [`ZeyuLing/hftrainer-t2mgpt-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-t2mgpt-humanml3d) |
| **Motion representation** | **HumanML3D-263** (263-dim, 20 fps, 22 joints) |
| **Tokenizer** | `HumanVQVAE` (VQ-VAE, EMA-reset quantizer, codebook 512×512, `down_t=2`) |
| **Generator** | `Text2Motion_Transformer` (GPT, 9 layers, 16 heads, embed 1024) |
| **Text encoder** | CLIP ViT-B/32 (frozen) |
| **Paper** | *T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations*, Zhang et al., CVPR 2023 — [arXiv:2301.06052](https://arxiv.org/abs/2301.06052) |
| **Original code** | https://github.com/Mael-zys/T2M-GPT |

---

## Weights

Self-contained hftrainer artifact (diffusers-style `from_pretrained`):

| Artifact | Location | Contents | Status |
|---|---|---|---|
| T2M-GPT HumanML3D | [`ZeyuLing/hftrainer-t2mgpt-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-t2mgpt-humanml3d) | `vq.safetensors` + `gpt.safetensors` + `clip.safetensors` + `t2mgpt_config.json` + `Mean.npy` / `Std.npy` | public Hub artifact |
| local mirror | `checkpoints/t2mgpt/humanml3d` | same layout | optional local cache |

```python
from hftrainer.pipelines.t2mgpt import T2MGPTPipeline

pipe = T2MGPTPipeline.from_pretrained(
    "ZeyuLing/hftrainer-t2mgpt-humanml3d",
    device="cuda",
)
motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
)  # list of (T, 263)
```

The artifact is produced from the released upstream `.pth` checkpoints (a VQ-VAE
`net_last.pth` with a `net` key and a GPT `net_best_fid.pth` with a `trans` key)
with `scripts/eval/convert_t2mgpt_checkpoint.py`. The `--verify` flag reloads the
artifact and asserts **bit-identical** generation after the round-trip
(max-abs-diff = 0):

```bash
python3 scripts/eval/convert_t2mgpt_checkpoint.py \
    --vq_path ref_repo/T2M-GPT/pretrained/VQVAE/net_last.pth \
    --gpt_path ref_repo/T2M-GPT/pretrained/VQTransformer_corruption05/net_best_fid.pth \
    --out_dir checkpoints/t2mgpt/humanml3d --verify
# -> [verify] OK: artifact is bit-identical to the raw checkpoint.
```

**Use it:**

```python
from hftrainer.pipelines.t2mgpt import T2MGPTPipeline

pipe = T2MGPTPipeline.from_pretrained("checkpoints/t2mgpt/humanml3d", device="cuda")
# let the GPT decide the length via its EOS token:
motions = pipe.infer_t2m(["a person walks forward then sits down"])  # list of (T, 263)
# or clip to a fixed length (frames @ 20 fps):
motions = pipe.infer_t2m(["a person walks forward then sits down"], [120])
```

Converter/debug code can also drive it directly from explicit released `.pth`
weights, no default `ref_repo` lookup involved:

```python
bundle = T2MGPTBundle(
    vq_path="ref_repo/T2M-GPT/pretrained/VQVAE/net_last.pth",
    gpt_path="ref_repo/T2M-GPT/pretrained/VQTransformer_corruption05/net_best_fid.pth",
)
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

The VQ-VAE tokenizes this with a temporal downsampling of `down_t=2`
(`stride_t=2`), i.e. one token ≈ 4 frames, so a 196-frame motion maps to ≤ 49
discrete tokens drawn from a 512-entry codebook.

---

## Generation

Two runtime stages (parity with the gold-standard `t2mgpt_infer_hml3d263.py`):

1. **`Text2Motion_Transformer` (GPT)** — autoregressively samples motion token
   indices conditioned on the CLIP ViT-B/32 text embedding. Unlike a diffusion
   model, T2M-GPT determines the sequence length itself via a learned **EOS
   token**, so no target length is fed by default.
2. **`HumanVQVAE.forward_decoder`** — de-quantizes the predicted `(T,)` token
   sequence and decodes to the 263-dim feature, then de-normalises with the
   training `Mean` / `Std`.

---

## Evaluation

Generation under the **official HumanML3D protocol** (standard test split,
native 263-dim @ 20 fps, first caption, model-chosen length — *not* truncated to
GT) and scoring with the persisted `HumanML263Evaluator`. Reproduce with:

```bash
# 1) generate (whole test split, GPT picks the length)
python3 scripts/eval/t2mgpt_t2m_h3d263.py \
    --model_path checkpoints/t2mgpt/humanml3d \
    --out_dir outputs/evaluation/t2mgpt_h3d263_official/t2mgpt_263 \
    --progress
# 2) score with the HumanML3D-263 evaluator
python3 scripts/eval/verify_evaluators.py --which hml263 \
    --hml263-pred outputs/evaluation/t2mgpt_h3d263_official/t2mgpt_263
```

### HumanML3D-263 evaluator (native space)

`HumanML263Evaluator` now follows the canonical Guo/MoMask
`Text2MotionDatasetEval` protocol (time-tagged captions spawn sub-clip samples,
random caption + shuffle + `coin2` length jitter + `drop_last`), so its **GT/Real
reference row matches the published GT row**. Two measurements:

- **T2M-GPT predictions** (`n_samples = 3940` full-clip captions, `caption = first`
  to match generation, `n_repeats = 20`).
- **GT/Real reference** (full canonical set incl. sub-clips, `n_samples = 4402`,
  random caption) — validates the evaluator against the paper's GT row.

| Metric | hftrainer T2M-GPT | T2M-GPT paper | hftrainer GT/Real | paper GT/Real |
|---|---|---|---|---|
| **FID** ↓ | **0.176** | 0.116 | — | 0.002 |
| R-Precision Top-1 ↑ | **0.470** | 0.417 | 0.508 | 0.511 |
| R-Precision Top-2 ↑ | **0.660** | — | 0.698 | 0.703 |
| R-Precision Top-3 ↑ | **0.761** | 0.745 | 0.796 | 0.797 |
| **MM-Dist** ↓ | **3.238** | 3.118 | 2.970 | 2.974 |
| **Diversity** → | **9.563** | 9.761 | 9.398 | 9.503 |

The **GT/Real row now lands on the paper's GT** (R@3 0.796 vs 0.797, MM-Dist 2.970
vs 2.974), confirming the evaluator is faithful to the canonical protocol. The
generation path is verified bit-identical to the released T2M-GPT checkpoint
(`convert_t2mgpt_checkpoint.py --verify`, max-abs-diff = 0); R-Precision even
edges above the reported values (Top-3 0.761 vs 0.745).

The residual **FID gap (0.176 vs 0.116) is a population difference, not a model or
evaluator bug**: the canonical eval set contains 4402 samples (incl. sub-clips),
but the T2M-GPT predictions here cover only the 3940 full clips — sub-clips have
no matching prediction and are dropped, so pred-vs-GT runs on the full-clip
subset. A strictly paper-comparable FID requires generating one prediction per
canonical sample (incl. `<id>__sub<k>`); see the Model Zoo TODO.

### MotionStreamer-272 evaluator (SMPL retarget path)

For cross-model comparison with the MotionStreamer / HYMotion-M2M evaluator,
the native HumanML3D-263 predictions are retargeted through the validated
MDM-style chain:

```bash
# Stage A: HML263 -> SMPL motion_135 (IK refine-80, 20 -> 30 fps)
# Stage B: motion_135 -> MotionStreamer-272
# Stage C: MotionStreamer272Evaluator
N=4 bash scripts/eval/_run_263_to_ms272_taiji.sh
```

The IK stage uses `hml263_to_smpl_ik.py --refine-iters 80 --floor-align
--rot6d-convention row`, matching the MDM reproduction path; using the
non-refined analytic IK substantially inflates MS-272 FID.

| Metric | hftrainer T2M-GPT | MS-272 GT/Real |
|---|---:|---:|
| **FID** ↓ | **113.316** | 0.000 |
| R-Precision Top-1 ↑ | **0.446** | 0.706 |
| R-Precision Top-2 ↑ | **0.600** | 0.857 |
| R-Precision Top-3 ↑ | **0.678** | 0.911 |
| **MM-Dist** ↓ | **19.787** | 15.007 |
| **Diversity** → | **25.405** | 27.300 |

Run details: `n_repeats = 20`, `n_samples_used = 7328`,
`skipped_no_pred = 66`, outputs under
`outputs/evaluation/ms272_from263/t2mgpt_272`, metrics in
`outputs/evaluation/ms272_from263/metrics_t2mgpt.json`.

---

## Implementation notes

- **hftrainer-native runtime**: `hftrainer/models/motion/t2mgpt/network/` holds
  the VQ-VAE and the GPT transformer with package-relative imports; the artifact
  loads with zero dependency on `ref_repo` or the original `.pth` format.
- **Sub-modules**: `vqvae` (`HumanVQVAE`) + `gpt` (`Text2Motion_Transformer`),
  configured by `t2mgpt_config.json` (`vqvae`: `nb_code=512`, `code_dim=512`,
  `down_t=2`, `quantizer=ema_reset`; `gpt`: `embed_dim_gpt=1024`,
  `num_layers=9`, `n_head_gpt=16`, `block_size=51`).
- **CLIP**: frozen ViT-B/32 is stored once as `clip.safetensors` in new
  artifacts and restored by `T2MGPTBundle.from_pretrained`. Legacy lightweight
  artifacts without `clip.safetensors` still fall back to `clip_name`.
- **Normalization travels with the checkpoint**: `Mean.npy` / `Std.npy` are the
  263-dim training stats, embedded in the artifact (self-contained).
- **Length**: the GPT emits an EOS token, so the model selects its own sequence
  length; pass `--truncate_to_gt` to `t2mgpt_t2m_h3d263.py` to clip to GT length
  instead.
