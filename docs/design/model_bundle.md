# ModelBundle Design / ModelBundle 设计

## Overview / 概述

ModelBundle is the shared core between Trainer and Pipeline. It holds all sub-modules (text_encoder, vae, transformer, scheduler, etc.) and implements all atomic forward functions. Both Trainer and Pipeline call into the same ModelBundle, eliminating code duplication.

ModelBundle 是 Trainer 和 Pipeline 之间的共同核心。它持有所有子模块（text_encoder、vae、transformer、scheduler 等），并实现所有原子前向函数。Trainer 和 Pipeline 都调用同一个 ModelBundle，消除了代码重复。

This page covers two related design issues:

本页涵盖两个相关的设计问题：

1. **Per-Module Trainable & Checkpoint Control** -- each sub-module independently declares its training and checkpoint behavior / 每个子模块独立声明其训练和 checkpoint 行为
2. **Trainer / Pipeline Code Sharing** -- atomic forward functions are written once in the bundle / 原子前向函数在 bundle 中只写一次

---

## Per-Module Trainable & Checkpoint Control / 按模块控制训练和 Checkpoint

Each sub-module declares in config:

每个子模块在 config 中独立声明：

- **Loading method / 加载方式**: `from_pretrained` / `from_config` / `from_single_file` (via `HF_MODELS` registry)
- **Whether it participates in training / 是否参与训练**: `trainable=True/False` or `trainable='lora'` (auto-applies peft)
- **Whether it participates in checkpoint save/resume / 是否参与 checkpoint save/resume**: `save_ckpt=True/False`

```python
# configs/text2video/wan_finetune.py
model = dict(
    type='WanModelBundle',
    text_encoder=dict(
        type='UMT5EncoderModel',
        from_pretrained=dict(pretrained_model_name_or_path='Wan-AI/Wan2.1-T2V-14B'),
        trainable=False,      # frozen, no gradients / 冻结，不计算梯度
        save_ckpt=False,      # skip during save/resume / resume/save 时完全跳过
    ),
    vae=dict(
        type='AutoencoderKLWan',
        from_pretrained=dict(pretrained_model_name_or_path='Wan-AI/Wan2.1-T2V-14B'),
        trainable=False,
        save_ckpt=False,
    ),
    transformer=dict(
        type='WanTransformer3DModel',
        from_pretrained=dict(pretrained_model_name_or_path='Wan-AI/Wan2.1-T2V-14B'),
        trainable=True,       # only this participates in training / 只有这个参与训练
        save_ckpt=True,       # only save/restore this module / 只保存/恢复这个模块
    ),
    scheduler=dict(
        type='FlowMatchEulerDiscreteScheduler',
        from_pretrained=dict(pretrained_model_name_or_path='Wan-AI/Wan2.1-T2V-14B'),
        trainable=False,
        save_ckpt=False,
    ),
)
```

### ModelBundle Base Class Implementation / ModelBundle 基类实现

The base class handles all these declarations uniformly:

基类负责统一处理这些声明：

```python
class ModelBundle(nn.Module):
    """
    Holds all sub-modules needed for a task. Shared core for Trainer and Pipeline.
    Responsible for: module instantiation, freeze/unfreeze, checkpoint save/load filtering.

    持有一个任务所需的所有子模块，是 Trainer 和 Pipeline 共同的核心。
    负责：模块实例化、freeze/unfreeze、checkpoint save/load 筛选。
    """

    def _build_modules(self, modules_cfg: dict):
        """Iterate modules_cfg, instantiate each sub-module, set requires_grad, record save_ckpt flags."""
        self._save_ckpt_modules = []
        self._trainable_modules = []
        for name, sub_cfg in modules_cfg.items():
            sub_cfg = copy.deepcopy(sub_cfg)
            trainable = sub_cfg.pop('trainable', True)
            save_ckpt = sub_cfg.pop('save_ckpt', True if trainable else False)
            module = HF_MODELS.build(sub_cfg)
            if trainable == 'lora':
                module = apply_lora(module)
                trainable = True
            if not trainable:
                module.requires_grad_(False)
            setattr(self, name, module)
            if save_ckpt:
                self._save_ckpt_modules.append(name)
            if trainable:
                self._trainable_modules.append(name)

    def trainable_parameters(self):
        """Only return parameters of trainable modules, for the optimizer."""
        params = []
        for name in self._trainable_modules:
            params.extend(getattr(self, name).parameters())
        return params

    def state_dict_to_save(self) -> dict:
        """Only return state_dict of save_ckpt=True modules."""
        sd = {}
        for name in self._save_ckpt_modules:
            sd[name] = getattr(self, name).state_dict()
        return sd

    def load_state_dict_selective(self, state_dict: dict):
        """Only load modules present in state_dict, leave others unchanged."""
        for name, sd in state_dict.items():
            if hasattr(self, name):
                getattr(self, name).load_state_dict(sd)
```

---

## Trainer / Pipeline Code Sharing / Trainer / Pipeline 代码共享

### The Problem / 问题根源

Taking WAN text-to-video as an example:

以 WAN text-to-video 为例：

- **Training forward**: `encode_text -> encode_video_to_latent -> add_noise -> transformer_forward -> loss`
- **Inference forward**: `encode_text -> init_random_latent -> [for t: transformer_forward -> scheduler_step] -> decode_latent`

- **训练前向**：`encode_text → encode_video_to_latent → add_noise → transformer_forward → loss`
- **推理前向**：`encode_text → init_random_latent → [for t: transformer_forward → scheduler_step] → decode_latent`

Functions like `encode_text` and `transformer_forward` are **identical** in training and inference, but the traditional approach duplicates them in both Trainer and Pipeline.

`encode_text`、`transformer_forward` 等函数在训练和推理中**完全相同**，但传统做法在 Trainer 和 Pipeline 中各写一遍。

### Solution: Three-Layer Separation / 解决方案：三层分离

```
ModelBundle          <-- holds all sub-modules + all atomic forward functions
    |                    Trainer and Pipeline both hold ModelBundle, call the same functions
    +-- Trainer      <-- only: assemble training forward graph, compute loss
    +-- Pipeline     <-- only: assemble inference forward graph (denoising loop etc.)
```

```
ModelBundle          ← 持有所有子模块 + 所有原子前向函数
    ↑                   Trainer 和 Pipeline 都持有 ModelBundle，调用同一套函数
    ├── Trainer      ← 只负责：组装训练前向图、计算 loss、梯度更新
    └── Pipeline     ← 只负责：组装推理前向图（denoising loop 等）、后处理输出
```

### Concrete Example: WAN / 具体示例：WAN

```python
# hftrainer/models/wan/wan_bundle.py
class WanModelBundle(ModelBundle):
    def __init__(self, cfg):
        self._build_modules(cfg)

    # --- Atomic forward functions (shared by training and inference) ---

    def encode_text(self, text: list[str]) -> Tensor:
        tokens = self.tokenizer(text, ...)
        return self.text_encoder(**tokens).last_hidden_state

    def encode_video(self, video: Tensor) -> Tensor:
        return self.vae.encode(video).latent_dist.sample() * self.vae.config.scaling_factor

    def decode_latent(self, latent: Tensor) -> Tensor:
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample

    def predict_noise(self, noisy_latent, timestep, text_emb, **kwargs) -> Tensor:
        return self.transformer(noisy_latent, timestep, encoder_hidden_states=text_emb, **kwargs).sample


# hftrainer/trainers/text2video_trainer.py
class WanTrainer(BaseTrainer):
    def train_step(self, batch) -> dict:
        text_emb = self.bundle.encode_text(batch['text'])           # reuse / 复用
        latent   = self.bundle.encode_video(batch['video'])         # reuse / 复用
        noise    = torch.randn_like(latent)
        t        = torch.randint(0, 1000, (latent.shape[0],))
        noisy    = self.bundle.scheduler.add_noise(latent, noise, t)
        pred     = self.bundle.predict_noise(noisy, t, text_emb)    # reuse / 复用
        loss     = F.mse_loss(pred, noise)
        return {'loss': loss}


# hftrainer/pipelines/text2video_pipeline.py
class WanPipeline(BasePipeline):
    @torch.no_grad()
    def __call__(self, text: str, num_steps: int = 50, **kwargs):
        text_emb = self.bundle.encode_text([text])                   # same call / 完全相同
        latent   = torch.randn(1, ...)
        for t in self.bundle.scheduler.timesteps:
            pred   = self.bundle.predict_noise(latent, t, text_emb)  # same call / 完全相同
            latent = self.bundle.scheduler.step(pred, t, latent).prev_sample
        return self.bundle.decode_latent(latent)
```

### Key Benefits / 关键收益

- Functions like `encode_text` and `predict_noise` are **written once** in `WanModelBundle` / `encode_text`、`predict_noise` 等函数**只写一次**
- Changing the model (e.g., modifying transformer forward args) only requires editing `bundle.predict_noise`; Trainer and Pipeline stay in sync automatically / 改动模型只需改 `bundle.predict_noise`，Trainer 和 Pipeline 自动同步
- Pipeline loads from a trained checkpoint: `bundle.load_state_dict_selective(ckpt)`, then `WanPipeline(model=bundle)` / Pipeline 从训练好的 checkpoint 加载后直接使用

### Switching from Training to Inference / 从训练切换到推理

```python
bundle = WanModelBundle.from_cfg(cfg.model)
bundle.load_state_dict_selective(torch.load('work_dirs/wan_exp/checkpoint.pth'))
pipeline = WanPipeline(model=bundle)
video = pipeline("a cat running in the park")
```
