# PRISM 1.0 - Iter15000 HumanML3D Checkpoint

PRISM 1.0 is the original PRISM text-to-motion checkpoint used for the
no-KT-RoPE / no-KAFS baseline. It is the sequential-RoPE `iter_15000`
checkpoint from `work_dirs/prism_1b_tp2m_multiframe`, evaluated on the
HumanML3D official test split through the hftrainer PRISM inference stack.

| | |
|---|---|
| **Task** | Text-to-Motion (T2M) |
| **Bundle / Pipeline** | `PrismBundle` / `PrismPipeline` |
| **Checkpoint artifact** | [`ZeyuLing/hftrainer-prism-1.0-humanml3d-iter15000`](https://huggingface.co/ZeyuLing/hftrainer-prism-1.0-humanml3d-iter15000) |
| **Checkpoint type** | raw hftrainer checkpoint package (`model.pt` + `meta.pt` + reproduction configs) |
| **Motion representation** | PRISM / VerMo `motion_138`: `[transl_abs(3), transl_rel(3), 22xrot6d]`, column-major, 30 fps |
| **Text encoder** | Wan2.1 VACE UMT5 text encoder (resolved from shared checkpoints) |
| **VAE** | `wanmo_vae2d_aug` |
| **Variant** | sequential RoPE; no KT-RoPE; no KAFS at inference |

---

## Weights

This artifact is a raw checkpoint package rather than a fully self-contained
`PrismPipeline.from_pretrained` directory. It intentionally stores the released
train checkpoint and the exact inference config needed to replay it.

| Artifact | Location | Contents | Status |
|---|---|---|---|
| PRISM 1.0 HumanML3D iter15000 | [`ZeyuLing/hftrainer-prism-1.0-humanml3d-iter15000`](https://huggingface.co/ZeyuLing/hftrainer-prism-1.0-humanml3d-iter15000) | `model.pt`, `meta.pt`, `model_index.json`, and `config/*.py` | public Hub checkpoint package |
| local mirror | `checkpoints/prism/prism_1_0_humanml3d_iter15000` | same layout | optional local cache |

Download:

```bash
huggingface-cli download ZeyuLing/hftrainer-prism-1.0-humanml3d-iter15000 \
    --local-dir checkpoints/prism/prism_1_0_humanml3d_iter15000
```

The original local checkpoint source is:

```text
work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000
```

`meta.pt` records `global_step=15000` and `current_epoch=8`.

## Inference

Use the packaged config `config/prism_1b_tp2m_multiframe_iter15k.py`. The key
compatibility detail is the VAE: this checkpoint was trained with
`wanmo_vae2d_aug`, not the newer `vermo_vae` default. Decoding it with the
wrong VAE gives invalid motions.

Example single-node generation:

```bash
CONFIG=checkpoints/prism/prism_1_0_humanml3d_iter15000/config/prism_1b_tp2m_multiframe_iter15k.py \
CKPT=checkpoints/prism/prism_1_0_humanml3d_iter15000 \
MODE=none \
ANNO=data/annotation/test_hml3d_official272_gtlen.json \
DATA=. \
OUT=outputs/evaluation/t2m/humanml3d_official_test/_runs/prism_1_0_h3d \
NSHARDS=8 SHARD_START=0 NGPU=8 NUM_INFER=50 \
    bash scripts/eval/run_gen_node.sh
```

For the full HumanML3D test run used in the model card, the repo used the
strict official-272 PRISM ablation runner:

```bash
VARIANT=iter15k_no_kt_no_kafs \
RUN_ROOT=outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_kafs_kt_compare_20260621 \
    bash scripts/eval/run_prism_kafs_kt_compare_gen_20260621.sh
```

## Representation

PRISM natively uses `motion_138`:

```text
motion_138 = [transl_abs(3), transl_rel(3), 22 x rot6d_column(132)]
```

The HumanML3D leaderboard viewer and SMPL mesh comparisons use repacked
`motion_135`:

```text
motion_135 = [transl_abs(3), 22 x rot6d_row(132)]
```

For MotionStreamer-272 evaluation, PRISM outputs are converted through:

```text
PRISM raw SMPL params -> motion_135 -> MotionStreamer-272
```

## HumanML3D Evaluation

PRISM 1.0 was evaluated on the HumanML3D official test split with the corrected
official-length generation protocol and selected GT captions.

| Metric set | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|
| MotionStreamer-272 evaluator | 3968 | 0.7104 | - | 0.8805 | 22.4874 | 16.2994 | 27.2828 |
| MotionCLIP-135, L2-normalized legacy protocol | 3972 | 0.7709 | 0.8820 | 0.9223 | 0.4096 | 1.0690 | 21.9964 |
| MotionCLIP-135, raw/no-L2 protocol | 4042 | 0.5915 | 0.7187 | 0.7764 | 541.4167 | 41.5765 | 22.0501 |

The raw/no-L2 MotionCLIP FID is currently treated as a diagnostic number rather
than a final quality ranking. The same PRISM outputs have good retrieval under
the L2-normalized protocol, and the raw protocol shows a large embedding-mean
shift for several methods.

### Physical Metrics

| Metric | PRISM 1.0 |
|---|---:|
| Foot slide | 4.8457 |
| Float | 21.4024 |
| Jitter | 7.4677 |
| Dynamic penetration | 28.4771 |

## Implementation Notes

- **Checkpoint identity**: `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000`,
  `global_step=15000`, `current_epoch=8`.
- **Version name**: `PRISM 1.0` means sequential RoPE, no KT-RoPE, and KAFS
  disabled at inference.
- **VAE compatibility**: use `checkpoints/wanmo_vae2d_aug`. The newer
  `vermo_vae` latent space is incompatible with this checkpoint.
- **Runtime dependencies**: the artifact expects the hftrainer PRISM code,
  Wan2.1 VACE tokenizer/text encoder, `wanmo_vae2d_aug`, and SMPL assets to be
  available locally.
- **Future packaging**: a full `PrismPipeline.from_pretrained` artifact should
  split transformer, VAE, tokenizer, text encoder, scheduler, and
  `smpl_pose_processor.json` into a self-contained directory. This checkpoint
  package preserves the exact raw training checkpoint first.
