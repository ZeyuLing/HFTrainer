# Model cards

Choose an implementation, check its verification scope, then prepare weights and data before launching a recipe.

| Model | Task | Published verification |
| :--- | :--- | :--- |
| [ViT](vit.md) | Classification | Full pretrained export/reload + tiny numerical tests |
| [LLaMA](llama.md) | Text generation | Full pretrained export/reload + tiny numerical tests |
| [Stable Diffusion 1.5](sd15.md) | Text to image | Full pretrained export/reload + tiny numerical tests |
| [DMD](dmd.md) | One-step image generation | DMD2 trained initialization + exact export/reload |
| [StyleGAN2](stylegan2.md) | Unconditional images | Official FFHQ 1024 output matches exactly |
| [Wan T2V](wan.md) | Text to video | Full 1.3B components + exact export/reload |
| [LTX-Video 2.5](ltx_video_2_5.md) | Video + audio | Full dev samples + short LoRA runs |
| [MiniMax-H3](minimax_h3.md) | Video + audio | Full T2VA samples + short LoRA runs |

## Read the status precisely

- **Tiny tests:** reduced configurations verify contracts; they are not full-weight quality evaluations.
- **Full samples:** the named modes and settings ran with real weights; this does not validate every supported mode.
- **Short training:** optimizer, checkpoint and resume paths ran; convergence requires separate evidence.

[Demo data availability](demo-data.md) · [Video loss evidence](evidence/index.md) · [Model-card standard](standard.md)
