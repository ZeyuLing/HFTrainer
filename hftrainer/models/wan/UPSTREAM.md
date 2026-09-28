# Wan 2.1 provenance

The checkpoint-compatible video VAE is adapted from Hugging Face Diffusers
revision `c1bf18c92c6285334adcaac7e75ef8946a227f49`,
`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`.
Original Wan/Hugging Face copyright notices are retained; Apache-2.0 applies.
See `LICENSE.apache-2.0`.

HFTrainer replaces external configuration, artifact loading and output classes
with repository-owned implementations. The earlier compact VAE did not match
real Wan checkpoints and has been replaced rather than partially loading it.
UMT5 uses a relative attention bias per layer and tied shared embeddings;
the transformer uses the official zero-shift timestep frequency denominator.

Real checkpoint source: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, revision
`0fad780a534b6463e45facd96134c9f345acfa5b`.
Independent reduced-size comparisons cover encode/decode, text encoding and
denoising. Full-weight release provenance is provided with the Hub artifact.
