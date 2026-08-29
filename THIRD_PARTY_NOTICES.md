# Third-party notices

## MiniMax-H3

`hftrainer/models/minimax_h3` contains a repository-local, modified
implementation of the public MiniMax-H3 Base 768p execution path. The code was
adapted from Apache-2.0 Hugging Face Diffusers and Transformers references;
the exact commits and modification record are documented in
`hftrainer/models/minimax_h3/UPSTREAM.md`.
The Apache License text shipped with those reference implementations is
included at `hftrainer/models/minimax_h3/LICENSE.apache-2.0`.

MiniMax model materials are governed by the MiniMax H3 Community License
Agreement. The complete agreement and required notice are included at
`hftrainer/models/minimax_h3/LICENSE.minimax-h3` and
`hftrainer/models/minimax_h3/NOTICE.minimax-h3`. The agreement contains
territorial exclusions, use restrictions, redistribution obligations, and
additional commercial terms. HFTrainer does not bundle pretrained weights or
tokenizer/config artifacts; downloading or using them requires accepting and
complying with the upstream agreement.

## LTX-2.x

The directories below contain a modified, pinned snapshot of Lightricks'
LTX-2 implementation:

- `hftrainer/models/ltx_video/network`
- `hftrainer/pipelines/ltx_video/backend`
- `hftrainer/trainers/ltx_video/native`
- `hftrainer/trainers/ltx_video/preprocess_scripts`

Source: https://github.com/Lightricks/LTX-2 at commit
`400fd31054597515f47125691032c04b1c3ee24e`.

These files and LTX-derived artifacts are governed by the LTX-2.x Community
License Agreement, not by an implied permissive license. The complete license
is included at `hftrainer/models/ltx_video/LICENSE.ltx-2.x`; it contains use
restrictions and commercial-license requirements. See
`hftrainer/models/ltx_video/UPSTREAM.md` for the modification record.

## Gemma reference implementation

The repository-local Gemma text runtime is a rewrite informed by the
Apache-2.0 Transformers implementation at commit
`a08ace4bbd97e721c98751deec37d87b026acadc`. It does not require that package
at runtime. Copyright and license details for the reference project are at
https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/LICENSE
