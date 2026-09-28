# Pretrained artifact release checks

Each model card links its processed Hugging Face bundle, immutable revision,
download command and model-specific JSON record in this directory. Tensor
weights originate from real pretrained checkpoints; no initialization-only
artifact is presented as a trained model.

## What is checked

- ViT, TinyLlama, SD1.5, DMD2 initialization, StyleGAN2 and Wan: export through
  the native bundle, reload and compare recorded outputs; download the complete
  Hub snapshot and match every file's SHA256 to the export.
- LTX: verify the complete local role manifest, match each file to its original
  upstream LFS object, copy those immutable objects into our own bundle repository,
  verify destination LFS hashes and fetch all root metadata. A second full tensor
  download is not claimed.
- H3: export the full native component graph and manifest, compare exported
  SHA256 values with independently uploaded Hub objects, fetch metadata and
  compare a real VAE encoder output before and after export. A new full MP4
  inference run from the exported directory is not claimed.

## Training checks

The published base bundles initialize both Trainer and Pipeline through
`cfg.model.from_pretrained`. ViT, LLaMA, SD1.5, DMD, StyleGAN2 and Wan received
one-step full-weight or LoRA execution checks on one H200. Image checks used
four generated colored squares; text checks used four repeated synthetic
instruction records; Wan used synthetic video tensors. These are not demo
quality results or loss-decrease curves. H3/LTX retain the separately recorded
real-video 20-step traces and official inference comparisons.

## Reproducibility boundary

A one-command checkpoint download does not install CUDA dependencies, authorize
a model agreement, or prepare training data. Read the matching model card.
DMD2 is an initialization for the DMD implementation, not an original-DMD
paper reproduction. NVIDIA StyleGAN2 remains under its non-commercial license.
LTX requires agreement acceptance; H3 requires manually approved access under
its territorial terms. The artifacts use HFTrainer-owned schemas rather than
claiming compatibility with another framework's automatic model loader.
