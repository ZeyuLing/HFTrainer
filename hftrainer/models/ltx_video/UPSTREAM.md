# LTX-Video 2.5 upstream record

HFTrainer includes a modified source snapshot of Lightricks' LTX-2 code so
that the model, training loop, and inference graph are available from one
repository and do not import a separately installed LTX implementation.

- Upstream repository: https://github.com/Lightricks/LTX-2
- Pinned commit: `400fd31054597515f47125691032c04b1c3ee24e`
- Snapshot date: 2026-08-29
- Upstream license: `LICENSE.ltx-2.x` in this directory
- Local namespace: `hftrainer.models.ltx_video.network`
- Local inference namespace: `hftrainer.pipelines.ltx_video.backend`
- Local training namespace: `hftrainer.trainers.ltx_video.native`

HFTrainer changed the snapshot by relocating modules to the framework's
model/trainer/pipeline boundaries, rewriting internal imports, replacing
external model-framework construction with repository-local implementations,
and integrating HFTrainer's local LoRA and artifact APIs. Modified Python
files carry a notice at the top of the file as required by section 3.3 of the
LTX-2.x Community License Agreement.

The LTX license contains use restrictions and commercial-license conditions.
Redistributors and users must read the complete agreement, including its
attachments, before using this implementation or LTX-derived checkpoints.

The local Gemma text-only runtime in
`network/text_encoders/gemma/local_model.py` is an HFTrainer rewrite informed
by the Apache-2.0 reference implementation in Transformers v5.14.1:

- Reference repository: https://github.com/huggingface/transformers
- Reference commit: `a08ace4bbd97e721c98751deec37d87b026acadc`
- Relevant family: Gemma 4 Unified

That runtime exists only to execute the text path required by LTX-2.5. It does
not import or execute Transformers code at runtime.
