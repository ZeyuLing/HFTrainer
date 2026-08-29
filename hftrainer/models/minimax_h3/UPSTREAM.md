# MiniMax-H3 upstream and modification record

HFTrainer provides a repository-local implementation of the released
MiniMax-H3 Base 768p execution path. It does not import Diffusers,
Transformers, Tokenizers, PEFT, or the MiniMax repository at runtime.

The implementation was frozen against these public sources on 2026-08-30:

- MiniMax model repository: https://github.com/MiniMax-AI/MiniMax-H3
- MiniMax repository commit: `d21241f0a4b3acbb34c97dae47fa417b7065e438`
- MiniMax checkpoint repository: https://huggingface.co/MiniMaxAI/MiniMax-H3
- MiniMax checkpoint revision: `42ed227ee7df40d41602854ae760620d6eb651fe`
- Diffusers reference repository: https://github.com/huggingface/diffusers
- Diffusers reference commit: `c1bf18c92c6285334adcaac7e75ef8946a227f49`
- Transformers Qwen3-VL reference repository: https://github.com/huggingface/transformers
- Transformers reference tag: `v5.14.1`
- Transformers reference commit: `a08ace4bbd97e721c98751deec37d87b026acadc`

The repository-local transformer, schedulers, video/audio autoencoders, and
pipeline arithmetic are modified adaptations of the Apache-2.0 Diffusers
reference files carrying the MiniMax and Hugging Face copyright header. The
repository-local Qwen3-VL conditioner and Qwen2 byte-level BPE runtime are
modified adaptations of the Apache-2.0 Transformers reference. HFTrainer
replaced the two external frameworks' configuration, registry, artifact,
attention-processor, output-container, preprocessing, and orchestration
machinery with local equivalents and organized the result into HFTrainer's
model/bundle/trainer/pipeline boundaries. Modified source files carry a
prominent modification notice.

No pretrained weights, tokenizer vocabulary, merge table, or model config is
vendored in the HFTrainer source tree. Users obtain those artifacts directly
from the frozen MiniMax checkpoint revision after accepting its terms.

## License boundary

MiniMax-H3 model materials and their use are governed by the MiniMax H3
Community License Agreement, not an open-source permissive license. The full
agreement is included as `LICENSE.minimax-h3`, and its required redistribution
notice is included as `NOTICE.minimax-h3`. Among other provisions, the
agreement defines excluded territories, use restrictions, redistribution
requirements, and additional commercial terms. Read the agreement itself;
this summary is not legal advice.

Qwen3-VL and the Hugging Face reference implementation code are licensed
under Apache-2.0. A copy is included as `LICENSE.apache-2.0`. Including those
permissively licensed reference adaptations does not change the terms that
apply to downloaded MiniMax model materials.

## Supported public scope

The local integration owns the open H3-Base 768p paths:

- text-to-synchronized-audio/video (`t2va`);
- first-frame, last-frame, and first/last-frame conditioning (`fl2va`);
- ordered image/video/audio reference conditioning (`ref2va`);
- the released full-attention 50-layer Omni Transformer;
- Qwen3-VL-32B conditioning through hidden state 50;
- the released visual VAE, audio VAE, and video/audio flow schedulers;
- experimental cached-feature fine-tuning of the released transformer.

Hosted-only H3-Context-IR, Regenerate-2K, 2K post-processing, and unreleased
sparse-attention kernels are not represented as local capabilities.
