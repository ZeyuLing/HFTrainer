# Wan local implementation: provenance and compatibility scope

The code in this directory is a repository-owned PyTorch implementation. It is
not presented as an original invention or an official Wan release. Modified
upstream source retains its original attribution and license.

Its architecture and public configuration/API contracts were informed by:

- Alibaba's Wan 2.1 model family and published model configuration conventions.
- Hugging Face Transformers' T5/UMT5 encoder organization.
- Hugging Face Diffusers' Wan VAE, Wan 3D transformer, and flow-matching Euler
  scheduler APIs.

Those upstream projects publish relevant implementation work under Apache-2.0
licenses. The complete causal VAE is adapted from pinned Diffusers source;
see UPSTREAM.md and LICENSE.apache-2.0. Other components retain the local
bundle contract. It uses compatible state-dictionary names
to make checkpoint conversion auditable, but it is not bit-for-bit equivalent
to every upstream release. Foreign checkpoints are shape-checked and produce a
coverage report; low-coverage loads fail unless the caller explicitly opts into
partial initialization. Artifacts saved by this implementation include hashes
and a state schema and load strictly by default.

`WanTokenizer` includes a small standard-library parser and unigram segmenter
for `spiece.model`. Its normalization is an NFKC approximation and does not
execute the serialized precompiled normalization map. The deterministic byte
backend is for locally trained/tiny models only and must not be described as
token-compatible with an upstream UMT5 checkpoint.

No external model implementation is imported or dynamically selected by the
Wan execution path. PyTorch is required; `safetensors` is used when available
for artifact I/O, with PyTorch tensor serialization as the local fallback.
