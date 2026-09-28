# Demo data availability

**No HFTrainer demo dataset has been published to Hugging Face as of 2026-09-28.** The authenticated project account listing and Hub search were checked. There is no public dataset ID or pinned dataset revision to download for the H3/LTX measurements.

| Consumer | Current data | Public reproduction status |
| :--- | :--- | :--- |
| H3 / LTX 2.5 | Three local dance clips | Source clips and feature caches are not distributed |
| ViT / StyleGAN2 | Image-folder demo | Supply compatible images; no curated HFTrainer release |
| LLaMA | Ten Alpaca-format records | Supply records; no curated HFTrainer release |
| SD1.5 / DMD | Captioned image demo | Supply image/text pairs; no curated HFTrainer release |
| Wan | Synthetic tensors selected by config | Does not require a real-video download |

## Video measurement provenance

Training: `Ballet427.mp4`, `Popular-Jazz110.mp4`. Held out: `Classic-dunhuang8.mp4`. Initial excerpts: first six seconds, 256 × 256, 24 FPS, 32 kHz stereo. Shared caption: “A dancer performs a dance routine with music.” H3 caches: 128 × 128 × 124 frames. LTX caches: 256 × 256 × 49 frames.

The original clips came from machine-local storage. Redistribution permission has not been established, so they are not offered as a public dataset. Model weights and generated feature caches are also not bundled. The H3/LTX training commands require prepared caches and cannot currently reproduce these exact measurements from a clean public checkout alone.

## Existing downloader

The legacy local `download_demo_data.py` helper (not part of the published package) is a convenience script using third-party Hub datasets for images and text. It does not publish an HFTrainer dataset, pin a reproducible dataset release, or download the real H3/LTX clips. Its text-to-video branch only creates synthetic metadata. Download failure messages must not be interpreted as successful data preparation.

## Public release requirements

A downloadable demo release needs authorized media, an English dataset card, license and source attribution, checksums, fixed train/validation splits, captions and media metadata, a pinned Hub revision, and a download command checked in a clean environment. This release remains pending; no placeholder Hub URL is presented as a working download.
