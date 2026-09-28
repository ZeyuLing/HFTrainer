# Public demo datasets

Four small public datasets are hosted under **ZeyuLing/hftrainer_{dataset name}**. The examples use pinned revisions, fixed splits and verified file checksums. Downloaded data lives under `data/`; it is not stored in Git.

| Dataset | Train / validation / test | Used by | License |
| :--- | :--- | :--- | :--- |
| [beans](https://huggingface.co/datasets/ZeyuLing/hftrainer_beans) | 1034 / 133 / 128 | ViT classification | MIT |
| [wikitext2](https://huggingface.co/datasets/ZeyuLing/hftrainer_wikitext2) | 23767 / 2461 / 2891 | TinyLlama plain-text causal LM | Upstream CC BY-SA / GFDL notices |
| [smithsonian_butterflies](https://huggingface.co/datasets/ZeyuLing/hftrainer_smithsonian_butterflies) | 900 / 100 / — | SD1.5, DMD caption-conditioned distillation, StyleGAN2 unconditional images | CC0-1.0 (Smithsonian Open Access) |
| [crema_d_mini](https://huggingface.co/datasets/ZeyuLing/hftrainer_crema_d_mini) | 24 / 8 / — | H3/LTX synchronized audio-video; Wan video-only | ODbL database + DbCL individual contents |

## Download and train

After installing HFTrainer and any model-specific extras, one command downloads the pinned dataset and checkpoint, verifies the data, prepares video caches when needed, and launches the normal trainer:

```bash
python tools/run_public_demo.py vit
# Other choices: llama sd15 dmd stylegan2 wan ltx_video minimax_h3
```

Use `--prepare-only` to stop before training, or `--checkpoint path/to/complete_bundle` to reuse local weights. Video-model access agreements and CUDA hardware remain prerequisites. All recipes run 20 optimizer steps and are execution checks, not convergence benchmarks.

To download data without weights or preprocessing:

```bash
hf download ZeyuLing/hftrainer_beans --repo-type dataset \
  --revision 712f72df3166cf40eacae68df26330f7096b0685 \
  --local-dir data/hftrainer_beans
```

```bash
hf download ZeyuLing/hftrainer_wikitext2 --repo-type dataset \
  --revision e9aaa6354f59bf18cb8d0fe53cd2e0dce1dd9aa4 \
  --local-dir data/hftrainer_wikitext2
```

```bash
hf download ZeyuLing/hftrainer_smithsonian_butterflies --repo-type dataset \
  --revision c34c2176cf5d0d2a1d64f034a83f989dddbf71d2 \
  --local-dir data/hftrainer_smithsonian_butterflies
```

```bash
hf download ZeyuLing/hftrainer_crema_d_mini --repo-type dataset \
  --revision 1188c5d5fac3e69ea014831ab4409f729f8eda81 \
  --local-dir data/hftrainer_crema_d_mini
```

## Provenance and transformations

**Beans:** all 1,295 original images and the upstream 1,034/133/128 splits are retained. The three folder labels are angular leaf spot, bean rust and healthy. Original encoded image bytes are preserved.

**WikiText-2 raw:** original splits are retained; empty/whitespace-only lines are removed, with source row indices preserved. The plain-text loader adds token boundaries and masks padding, without instruction/response formatting. The demo truncates each line to 256 tokens. Any reported line-wise perplexity is explicitly different from the standard concatenated-corpus benchmark. Upstream metadata lists CC BY-SA 3.0 and GFDL while its prose mentions 4.0; the original notice is retained rather than silently replacing it.

**Smithsonian butterflies:** all 1,000 images from the upstream cropped RGB subset are retained without recompression. Sort by SHA256(source ID) and hold out the first 100. Captions are species-name templates, not human annotations. Manifests retain original image URLs, scientific names and attribution. DMD uses these captions with online teacher targets; StyleGAN2 ignores captions. The source subset points to the original CC0 Smithsonian collection.

**CREMA-D mini:** 24 training clips from actors 1001–1004 and eight validation clips from actors 1005–1006. Within each actor, candidates are ordered by SHA256(path); retain the first six/four clips with audio and at least 2.5 seconds. Extract the first 2.5 seconds at 24 FPS, preserve source resolution, and encode H.264 plus stereo AAC at 32 kHz. Emotion annotations generate descriptive captions; they are not transcripts. No actor appears in both splits. H3 uses 56 frames at 128 × 128; LTX uses 49 at 256 × 256; Wan samples 17 frames at 128 × 128 and discards audio.

Each dataset contains `provenance.json`, `SHA256SUMS.json` and upstream notices. All published files were re-downloaded at the recorded revision and checked against their hashes. [Pinned dataset registry](https://github.com/ZeyuLing/HFTrainer/blob/main/configs/public_data/datasets.json).

## Rebuild from original sources

```bash
python -m pip install pyarrow requests huggingface_hub
# ffmpeg and ffprobe must be available on PATH
python tools/prepare_public_datasets.py
```

Source revisions, split algorithms and transformations are fixed in the preparation script. Re-encoding media may differ bytewise across FFmpeg versions; the published hashes identify the measured release. Local cached model features are excluded from dataset publication.

## Historical private-video experiments

Earlier H3/LTX validation used two private dance clips and one held-out dance clip. Those media and caches were never published. Their [historical loss traces](evidence/index.md) remain separate from the public CREMA-D runs; they are not prerequisites for the current recipes.

[Model catalog](index.md) · [Model-card standard](standard.md)
