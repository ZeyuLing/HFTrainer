# Public-data training evidence

The current release uses four downloadable public datasets and eight model recipes. Every recipe targets 20 optimizer steps from a processed pretrained checkpoint, with seed 42. Curves are raw and unsmoothed; short runs do not establish convergence.

| Model | Dataset | Raw measurements | Loss trace |
| :--- | :--- | :--- | :--- |
| ViT | Beans | [JSON](vit.json) | [Curve](vit_loss.png) |
| TinyLlama | WikiText-2 | [JSON](llama.json) | [Curve](llama_loss.png) |
| SD1.5 | Smithsonian butterflies | [JSON](sd15.json) | [Curve](sd15_loss.png) |
| DMD | Butterfly captions + online teacher | [JSON](dmd.json) | [Curve](dmd_loss.png) |
| StyleGAN2 | Smithsonian butterflies | [JSON](stylegan2.json) | [Curve](stylegan2_loss.png) |
| Wan | CREMA-D video | [JSON](wan.json) | [Curve](wan_loss.png) |
| LTX 2.5 | CREMA-D audio-video | [JSON](ltx_video.json) | [Curve](ltx_video_loss.png) |
| H3 | CREMA-D audio-video | [JSON](minimax_h3.json) | [Curve](minimax_h3_loss.png) |

[Checkpoint-only execution and media verification](execution_verification.json) records the produced files and LTX final-weight check.

## Protocol

Dataset IDs, immutable revisions, split counts and verification hashes are recorded in each model JSON. [Anonymous dataset-access checks](datasets_verified.json) confirm that the datasets are public and ungated. Model weights have their own separate access agreements.

All runs use one NVIDIA H200 and PyTorch 2.8.0+cu128. H3 and LTX optimize rank-2 LoRAs using locally prepared text, video and audio features; TinyLlama uses rank-16 LoRA. Other recipes update the full trainable components named in their model cards. Saved optimizer state is separate from the public pretrained bases.

ViT reports accuracy on all 133 held-out validation images. TinyLlama reports token-weighted NLL/perplexity on the first 128 nonempty validation lines, independently truncated to 256 tokens. H3 reports stochastic flow loss on eight actor-disjoint clips. No FID, FVD or cross-model perceptual benchmark is claimed.

## Reproduce the reports

Run the recipes from the [model catalog](../../index.md), retain their logs, then generate the figures with:

```bash
python -m pip install matplotlib
python tools/validation/report_public_training.py --logs path/to/logs
```

The report script expects `train_{model}.log` files and final checkpoints in the recipe work directories. LTX's full-precision local metrics are retained as JSONL; other loss values reflect the logger's four-decimal display precision. H3 validation resamples noise/timesteps, so its endpoint is not a fixed-noise convergence comparison.

## Validation corrections

The release checks exposed and corrected real-data error masking, inactive experiment seeding, ViT preprocessing mismatch, duplicate-final-save retention in LTX, and a missing negative-conditioning/guidance branch in Wan inference. The published training traces correspond to the corrected recipes. The LTX retention regression is covered by a test that saves the same final step twice with `keep_last_n=1` and checks the latest tensor values remain loadable.

Base-model demo images/videos on the cards are separate from these short fine-tuning runs. [Earlier private-video experiments](../index.md) remain historical records, not the inputs required for current reproduction.

The initial Wan sample lacked classifier-free guidance and showed severe visual artifacts. The corrected pipeline uses both conditioning branches, flow shift 3 and FP32 latent accumulation; its replacement sample uses 480 × 832 and 50 Euler steps. This does not establish parity with the official UniPC/DPM++ sampler.
