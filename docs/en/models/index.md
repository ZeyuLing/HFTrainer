# Model cards

Eight local implementations share a common release contract: a processed Hugging Face checkpoint, a public dataset, one-command training/inference, and auditable evidence. Start with a model below.

| Model / released setting | Task | Public dataset | Evidence |
| :--- | :--- | :--- | :--- |
| [ViT-B/16 · 224](vit.md) | Classification | Beans | 20 steps; held-out accuracy |
| [TinyLlama · 1.1B Chat](llama.md) | Plain-text causal LM | WikiText-2 | LoRA loss; held-out line-wise perplexity |
| [Stable Diffusion · 1.5](sd15.md) | Text to image | Smithsonian butterflies | UNet loss; base image |
| [DMD · DMD2 SD1.5 initialization](dmd.md) | One-step image distillation | Butterfly captions | Generator/fake-score loss; base image |
| [StyleGAN2-ADA · FFHQ 1024](stylegan2.md) | Unconditional images | Smithsonian butterflies | G/D losses; official base parity |
| [Wan 2.1 · T2V 1.3B](wan.md) | Text to video | CREMA-D video | Reduced-resolution loss; 480p base preview |
| [LTX-Video · 2.5 22B Dev](ltx_video_2_5.md) | Video + audio | CREMA-D audio-video | LoRA loss; official base comparison |
| [MiniMax-H3 · Base FL2VA](minimax_h3.md) | Video + audio | CREMA-D audio-video | LoRA/held-out loss; official T2VA comparison |

## Read the evidence precisely

**Short training** verifies real inputs, optimizer updates and saved checkpoints. All public recipes run 20 steps; this does not establish convergence. Raw traces remain unsmoothed, including fluctuations.

**Base demos** come from the published pretrained checkpoints, unless explicitly labeled as fine-tuned. They are separate from the public-data training experiments. Official numerical parity applies only to the settings and comparison protocol named on each card.

**Checkpoint-only inference** reads the root `config.json` and uses an allowlisted repository-owned implementation. Infrastructure libraries and checkpoint assets remain dependencies; the original code repository is not a runtime requirement.

[Public datasets](demo-data.md) · [Public training records](evidence/public_data/index.md) · [Historical video evidence](evidence/index.md) · [Model-card standard](standard.md)
