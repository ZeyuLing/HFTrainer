# HF-Trainer: A Unified Training Framework Built on HuggingFace Ecosystem

## Motivation

MMLab (MMDet, MMPose, MMEngine) 定义了一套非常优雅的训练框架范式：通过 `.py` config 文件以 dict 形式声明式地实例化所有模块，通过 `dist_train.sh` 一键启动训练。这种设计在模块化和可复用性上做得很好。

但在实际使用中（尤其是在 versatilemotion 项目中），暴露出了若干严重问题：

1. **基建陈旧，与 HuggingFace 生态脱节**
   - FSDP / DeepSpeed (ZeRO) 等分布式训练的实现质量差、Bug 多
   - 不能通过 `accelerate` 统一管理训练过程（device placement, gradient accumulation, mixed precision 等）
   - Checkpoint 存储/加载不遵循 `accelerator.save_state()` / `accelerator.load_state()`，导致大模型 checkpoint 读写极慢

2. **过度封装，学习曲线陡峭**
   - `DataSample`, `BaseModel.forward(mode='loss'|'predict'|'tensor')` 等抽象让初学者困惑
   - Dataset 需要继承 `BaseDataset` 并遵循特定的 `__getitem__` 返回格式
   - 评估流程绑定在 `BaseMetric` 上，与 HuggingFace `evaluate` 库不兼容

3. **与 diffusers / transformers 不互通**
   - 无法直接使用 `diffusers.StableDiffusionPipeline`, `transformers.AutoModelForCausalLM` 等
   - Scheduler, Tokenizer, Processor 等组件需要额外包装才能在 MMEngine 体系中使用

4. **Trainer 与 Pipeline 代码重复，极易出错**
   - 训练写完 Trainer 后，改写推理 Pipeline 时需要重复大量前向计算逻辑（encode text、
     denoise loop 等），两处容易出现不一致

**HF-Trainer 的目标**：保留 MMLab config-driven 的声明式设计范式，但将运行时完全迁移到
HuggingFace Accelerate 生态上，并通过 `ModelBundle` 机制彻底解决 Trainer/Pipeline 代码重复问题。

---

## Design Principles

1. **Config-Driven, Registry-Based**: 保留 MMEngine 的 `Config` + `Registry` 机制
2. **Accelerate-Native**: 所有分布式、混合精度、gradient accumulation 等逻辑均由 `Accelerator` 管理
3. **HuggingFace-First**: 直接使用 diffusers / transformers 原生类，不做多余包装
4. **Per-Module Control**: 每个子模块（text_encoder, vae, transformer 等）独立控制
   trainable、checkpoint load/save，不依赖上层 Trainer 硬编码
5. **ModelBundle = Shared Core**: Trainer 和 Pipeline 共享同一个 `ModelBundle`，
   所有前向计算逻辑只写一次，训练和推理都调用它

## Current Public API Contract

当前代码实现上的约定，和上面的设计目标一样重要：

1. **`ModelBundle.from_config(...)` 是统一父类入口**
   - 所有 bundle 都应该支持
   - 自研模型默认走这条路径

2. **`ModelBundle.from_pretrained(...)` 也是统一父类入口**
   - 但 HF-native bundle 需要实现 `_bundle_config_from_pretrained(...)`
   - 子类负责定义“一个 HuggingFace / diffusers artifact 如何映射成 bundle 的多个子模块”
   - 不要把 `from_pretrained(...)` 直接做成完全独立、互不兼容的子类 API

3. **`save_pretrained(...)` 是任务相关导出能力**
   - 只有当 bundle 能导出官方 `transformers` / `diffusers` 能直接读取的 artifact 时才实现
   - 例如 `CausalLMBundle`、`ViTBundle`、`SD15Bundle`、`WanBundle`
   - 对自研任务，如果没有稳定导出格式，可以只提供 `from_config(...)` + checkpoint save/load

4. **文档必须明确区分两类接入路径**
   - 已有 `transformers` / `diffusers` 模型：保留官方模型类和 `from_pretrained(...)` 语义，只补 bundle / trainer / export
   - 自研模型：实现 `nn.Module` + bundle，默认走 `from_config(...)`

5. **不要为了统一而发明新的推理语义**
   - 推理侧尽量贴近官方 `transformers` / `diffusers`
   - 如果某个 bundle 实现了 `save_pretrained(...)`，README 和 docs 里必须写清楚对应的官方加载方式

6. **显存与精度控制也必须是配置驱动的**
   - 全局 runtime 通过 `accelerator.mixed_precision` 和 `accelerator.gradient_accumulation_steps`
   - 子模块级控制通过 `from_pretrained.torch_dtype` / `dtype`、`module_dtype`、`gradient_checkpointing`
   - 如果文档宣传“可以把某些模块保持在 fp32”，必须同时写清楚全局 AMP 对实际计算 dtype 的影响

## Smoke Test Policy

仓库里现在有一套真实走 CLI 的 startup smoke：

- 入口：`python3 -m pytest -m smoke tests/smoke/test_task_startup.py`
- 覆盖：`classification`、`gan`、`llm_sft`、`llm_lora`、`sd15`、`dmd`，以及带硬件门槛的 `wan`
- 语义：每个 case 都会生成临时 config，走 `tools/train.py` 训练 1 step，再走 `tools/infer.py`
- 目标：验证 config 解析、registry、runner、checkpoint、推理入口都能正常启动

约定：

1. 新增任务栈时，必须补一个对应的 startup smoke case。
2. smoke 可以收紧 batch size / image size / max length / optimizer，只要主链路仍然是真实训练和真实推理。
3. 如果某个任务对硬件有明确门槛，应该在测试里显式 skip，而不是让它在不满足条件的机器上随机 OOM。

---

## Design Detail: Six Key Issues

### Issue 1: Per-Module Trainable, Memory, and Checkpoint Control

每个子模块在 config 中独立声明：
- **加载方式**：`from_pretrained` / `from_config` / `from_single_file`（通过 `HF_MODELS` registry）
- **模块精度**：`from_pretrained.torch_dtype` / `dtype`，以及 HF-Trainer 的 `module_dtype`
- **activation 显存**：`gradient_checkpointing=True` 或 kwargs dict
- **是否参与训练**：`trainable=True/False` 或 `trainable='lora'`（自动调用 peft）
- **是否参与 checkpoint save/resume**：`save_ckpt=True/False`
- **checkpoint 保存格式**：`checkpoint_format='full'|'lora'`（LoRA 模块默认 `lora`，即 adapter-only）

```python
# configs/text2video/wan_finetune.py
model = dict(
    type='WanBundle',
    text_encoder=dict(
        type='UMT5EncoderModel',
        from_pretrained=dict(
            pretrained_model_name_or_path='Wan-AI/Wan2.1-T2V-14B',
            torch_dtype='bf16',
        ),
        trainable=False,      # 冻结，不计算梯度
        save_ckpt=False,      # resume/save 时完全跳过，节省 IO
    ),
    vae=dict(
        type='AutoencoderKLWan',
        from_pretrained=dict(pretrained_model_name_or_path='Wan-AI/Wan2.1-T2V-14B'),
        module_dtype='fp32',  # 强制 VAE 保持 fp32 参数
        trainable=False,
        save_ckpt=False,
    ),
    transformer=dict(
        type='WanTransformer3DModel',
        from_pretrained=dict(
            pretrained_model_name_or_path='Wan-AI/Wan2.1-T2V-14B',
            torch_dtype='bf16',
        ),
        gradient_checkpointing=True,
        trainable=True,       # 只有这个参与训练
        save_ckpt=True,       # 只保存/恢复这个模块的权重
    ),
    scheduler=dict(
        type='FlowMatchEulerDiscreteScheduler',
        from_pretrained=dict(pretrained_model_name_or_path='Wan-AI/Wan2.1-T2V-14B'),
        trainable=False,
        save_ckpt=False,
    ),
)
```

`ModelBundle` 的基类负责统一处理这些声明：

```python
class ModelBundle(nn.Module):
    """
    持有一个任务所需的所有子模块，是 Trainer 和 Pipeline 共同的核心。
    负责：模块实例化、freeze/unfreeze、checkpoint save/load 筛选。
    """
    # 子类在 __init__ 中调用 self._build_modules(cfg) 完成实例化+冻结
    # BaseTrainer 和 BasePipeline 都持有同一类型的 ModelBundle

    def _build_modules(self, modules_cfg: dict):
        """遍历 modules_cfg，实例化每个子模块，设置 requires_grad，记录 save_ckpt 标记。"""
        self._save_ckpt_modules = []   # 需要保存/恢复的模块名列表
        self._trainable_modules = []   # 需要传给 optimizer 的模块名列表
        for name, sub_cfg in modules_cfg.items():
            sub_cfg = copy.deepcopy(sub_cfg)
            trainable = sub_cfg.pop('trainable', True)
            save_ckpt = sub_cfg.pop('save_ckpt', True if trainable else False)
            checkpoint_format = sub_cfg.pop(
                'checkpoint_format',
                'lora' if trainable == 'lora' else 'full',
            )
            # 实例化（via HF_MODELS registry，支持 from_pretrained 等）
            module = HF_MODELS.build(sub_cfg)
            if trainable == 'lora':
                module = apply_lora(module)  # peft LoRA 注入
                trainable = True
            if not trainable:
                module.requires_grad_(False)
            setattr(self, name, module)
            if save_ckpt:
                self._save_ckpt_modules.append(name)
            if trainable:
                self._trainable_modules.append(name)
            self._module_checkpoint_formats[name] = checkpoint_format

    def trainable_parameters(self):
        """只返回 trainable 模块的参数，供 Runner 构建 optimizer。"""
        params = []
        for name in self._trainable_modules:
            params.extend(getattr(self, name).parameters())
        return params

    def state_dict_to_save(self) -> dict:
        """只返回 save_ckpt=True 的模块状态；LoRA 模块默认只存 adapter 权重。"""
        sd = {}
        for name in self._save_ckpt_modules:
            if self._module_checkpoint_formats[name] == 'lora':
                sd[name] = get_lora_state_dict(getattr(self, name))
            else:
                sd[name] = getattr(self, name).state_dict()
        return sd

    def load_state_dict_selective(self, state_dict: dict):
        """只加载 state_dict 中存在的模块，其余保持不变。"""
        for name, sd in state_dict.items():
            if hasattr(self, name):
                getattr(self, name).load_state_dict(sd)
```

---

### Issue 2: Dataset 目录结构

不同数据集即使任务相同，格式也各异，因此按 `{task}/{dataset_name}_dataset.py` 组织：

```
hftrainer/datasets/
├── __init__.py
├── text2image/
│   ├── __init__.py
│   ├── base_text2image_dataset.py   # 定义接口：__getitem__ 返回 {'image': Tensor, 'text': str}
│   ├── laion_dataset.py             # LAION 格式
│   ├── webdataset_t2i.py            # WebDataset tar 格式
│   └── hf_imagefolder_dataset.py    # HuggingFace ImageFolder 格式
├── text2video/
│   ├── __init__.py
│   ├── base_text2video_dataset.py   # 接口：{'video': Tensor[T,C,H,W], 'text': str}
│   ├── webvid_dataset.py
│   └── internvid_dataset.py
├── llm/
│   ├── __init__.py
│   ├── base_llm_dataset.py          # 接口：{'input_ids': Tensor, 'labels': Tensor, 'attention_mask': Tensor}
│   ├── alpaca_dataset.py
│   ├── sharegpt_dataset.py
│   └── pretrain_jsonl_dataset.py
├── classification/
│   ├── __init__.py
│   ├── base_classification_dataset.py  # 接口：{'image': Tensor, 'label': int}
│   ├── imagenet_dataset.py
│   └── hf_image_classification_dataset.py
```

每个 `base_{task}_dataset.py` 中用 ABC 定义接口，同时提供公共的 transform 逻辑和
collate_fn，具体数据集子类只需实现 `_load_meta()` 和 `__getitem__` 的数据解析部分。

---

### Issue 3: Trainer / Pipeline 代码共享设计

**问题根源**：以 WAN text-to-video 为例：

- 训练前向：`encode_text → encode_video_to_latent → add_noise → transformer_forward → loss`
- 推理前向：`encode_text → init_random_latent → [for t: transformer_forward → scheduler_step] → decode_latent`

`encode_text`、`transformer_forward` 等函数在训练和推理中**完全相同**，但传统做法是
在 Trainer 里写一遍、在 Pipeline 里再写一遍，导致容易不一致出错。

**解决方案：三层分离**

```
ModelBundle          ← 持有所有子模块 + 所有原子前向函数（encode_text, decode_latent 等）
    ↑                   Trainer 和 Pipeline 都持有 ModelBundle，调用同一套函数
    ├── Trainer      ← 只负责：组装训练前向图、计算 loss、梯度更新
    └── Pipeline     ← 只负责：组装推理前向图（denoising loop 等）、后处理输出
```

具体到 WAN 为例：

```python
# hftrainer/models/text2video/wan_bundle.py
class WanBundle(ModelBundle):
    """
    持有 WAN 的所有子模块，并实现所有原子前向函数。
    Trainer 和 Pipeline 都使用这个类。
    """
    def __init__(self, cfg):
        self._build_modules(cfg)  # 实例化 text_encoder, vae, transformer, scheduler

    # --- 原子前向函数（训练和推理共享）---

    def encode_text(self, text: list[str]) -> Tensor:
        """文本编码，训练和推理都调用这个。"""
        tokens = self.tokenizer(text, ...)
        return self.text_encoder(**tokens).last_hidden_state

    def encode_video(self, video: Tensor) -> Tensor:
        """视频编码为 latent，训练时用。"""
        return self.vae.encode(video).latent_dist.sample() * self.vae.config.scaling_factor

    def decode_latent(self, latent: Tensor) -> Tensor:
        """latent 解码为视频，推理时用。"""
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample

    def predict_noise(self, noisy_latent, timestep, text_emb, **kwargs) -> Tensor:
        """单步去噪前向，训练和推理都调用这个（最核心的共享函数）。"""
        return self.transformer(noisy_latent, timestep, encoder_hidden_states=text_emb, **kwargs).sample


# hftrainer/trainers/text2video_trainer.py
class WanTrainer(BaseTrainer):
    """
    只负责训练逻辑：组装训练前向图、计算 loss。
    所有前向函数从 bundle 调用，不重复实现。
    """
    def __init__(self, model: WanBundle, loss_cfg, **kwargs):
        self.bundle = model

    def train_step(self, batch) -> dict:
        text_emb = self.bundle.encode_text(batch['text'])           # 复用
        latent   = self.bundle.encode_video(batch['video'])         # 复用
        noise    = torch.randn_like(latent)
        t        = torch.randint(0, 1000, (latent.shape[0],))
        noisy    = self.bundle.scheduler.add_noise(latent, noise, t)
        pred     = self.bundle.predict_noise(noisy, t, text_emb)    # 复用
        loss     = F.mse_loss(pred, noise)
        return {'loss': loss}


# hftrainer/pipelines/text2video_pipeline.py
class WanPipeline(BasePipeline):
    """
    只负责推理逻辑：denoising loop、后处理。
    所有前向函数从 bundle 调用，不重复实现。
    """
    def __init__(self, model: WanBundle, **kwargs):
        self.bundle = model

    @torch.no_grad()
    def __call__(self, text: str, num_steps: int = 50, **kwargs):
        text_emb = self.bundle.encode_text([text])                   # 复用，完全相同
        latent   = torch.randn(1, ...)
        for t in self.bundle.scheduler.timesteps:
            pred   = self.bundle.predict_noise(latent, t, text_emb) # 复用，完全相同
            latent = self.bundle.scheduler.step(pred, t, latent).prev_sample
        return self.bundle.decode_latent(latent)                     # 复用
```

**关键收益**：
- `encode_text`、`predict_noise` 等函数**只写一次**，在 `WanBundle` 里
- 改动模型（比如修改 transformer 的 forward 参数）只需改 `bundle.predict_noise`，
  Trainer 和 Pipeline 自动同步
- Pipeline 从训练好的 checkpoint 加载：`bundle.load_state_dict_selective(ckpt)`，
  然后传给 `WanPipeline(model=bundle)` 即可，无需额外适配

**从训练切换到推理的典型流程**：

```python
# 加载训练 config 中的 ModelBundle，切换到推理
bundle = WanBundle.from_config(cfg.model)
bundle.load_state_dict_selective(torch.load('work_dirs/wan_exp/checkpoint.pth'))
pipeline = WanPipeline(model=bundle)
video = pipeline("a cat running in the park")
```

---

### Issue 4: Resume vs Load Checkpoint — Unified `load_from`

resume 和 load_checkpoint 本质上是"加载哪些东西"的程度差异，而非两套独立逻辑，
统一为一个 `load_from` 字段，用 `load_scope` 控制加载范围：

```python
# configs/text2video/wan_finetune.py

# 场景 A：只加载 model weights（迁移学习 / 从 pretrained 开始 finetune）
load_from = dict(
    path='work_dirs/wan_exp/checkpoint-iter_10000/',
    load_scope='model',        # 只加载模型权重，optimizer/scheduler/meta 全部重置
)

# 场景 B：完整 resume（中断后续训练）
load_from = dict(
    path='work_dirs/wan_exp/checkpoint-iter_10000/',
    load_scope='full',         # 加载 model + optimizer + scheduler + training meta (epoch/iter)
)

# 场景 C：从已有 checkpoint 加载部分模块（例如只加载 transformer，跳过 text_encoder）
# 不需要额外字段 —— 这由 ModelBundle.load_state_dict_selective() 自动处理：
# state_dict 里有哪个模块的 key 就加载哪个，没有的跳过
```

**`AccelerateRunner` 的处理逻辑**：

```python
# runner 启动时
if cfg.get('auto_resume'):
    # 自动检测 work_dir 中最新的 checkpoint，full resume
    last_ckpt = find_latest_checkpoint(cfg.work_dir)
    if last_ckpt:
        runner.load(last_ckpt, load_scope='full')

elif cfg.get('load_from'):
    runner.load(cfg.load_from.path, load_scope=cfg.load_from.get('load_scope', 'model'))
```

**`load_scope` 的含义**：

| `load_scope` | model weights | optimizer states | scheduler states | training meta (epoch/iter) |
|---|---|---|---|---|
| `'model'` | ✓（selective） | ✗ reset | ✗ reset | ✗ reset（从 0 开始） |
| `'full'` | ✓ | ✓ | ✓ | ✓（接续 epoch/iter） |

`AccelerateRunner` 在 `load_scope='full'` 时调用 `accelerator.load_state(path)`，
自动处理 FSDP / DeepSpeed 的 optimizer state 格式；`'model'` 时只调用
`bundle.load_state_dict_selective()`。

`auto_resume=True` 是推荐的默认行为，不需要手动指定 `load_from`，Runner 自动
检测 work_dir 中最新 checkpoint 并 full resume，适合集群任务被抢占后重启的场景。

---

### Issue 5: 多 Optimizer 支持（GAN / Distillation 等）

单 optimizer 设计无法支持 GAN（G/D 各一个 optimizer，每步交替更新）、知识蒸馏（student/discriminator）等场景。
设计通过 `trainer_controls_optimization` 标志和 `params` 显式参数映射来解决。

> 详细文档和 recipes 见 `docs/design/multi_optimizer.md`

**核心机制：`trainer_controls_optimization` 标志**

`BaseTrainer` 有一个类属性 `trainer_controls_optimization = False`（默认）。
当 trainer 子类设为 `True` 时：
- Runner 把构建好的 optimizers/schedulers 注入 trainer（调用 `trainer.set_optimizers()`）
- Runner 训练循环中**跳过** backward/step/zero_grad
- Trainer 在 `train_step()` 中自主管理所有优化步骤
- `accelerator.accumulate()` 上下文仍然包裹 `train_step`（控制 DDP 梯度同步）

所有现有 trainer（ViT、SD15、WAN、LLM）默认 `False`，完全不受影响。

**Config 层**：`optimizer` 字段支持单个 dict（常规情况）或命名 dict（多 optimizer）：

```python
# 常规：单 optimizer
optimizer = dict(type='AdamW', lr=1e-5)

# GAN：多 optimizer，key 匹配 bundle 模块名
optimizer = dict(
    generator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
)

# DMD 蒸馏：optimizer 名与模块名不同时，用 params 显式指定
# 'student' optimizer -> 管理 'student_unet' 模块的参数
optimizer = dict(
    student=dict(type='AdamW', lr=1e-5, params=['student_unet']),
    discriminator=dict(type='AdamW', lr=4e-5),
)

# 同理，lr_scheduler 也支持多个
lr_scheduler = dict(
    generator=dict(type='cosine_with_warmup', num_warmup_steps=500),
    discriminator=dict(type='constant'),
)
```

**Runner 层**：`_build_optimizers` 检测命名 dict，支持 `params` key：

```python
# 多 optimizer 检测逻辑
if is_multi:
    for name, opt_cfg in optimizer_cfg.items():
        param_names = opt_cfg.pop('params', None)
        if param_names is not None:
            # 显式指定：从 bundle 中按名字收集参数
            params = bundle.get_module_parameters(*param_names)
        else:
            # 默认：optimizer 名 == bundle 模块名
            params = list(getattr(bundle, name).parameters())
        optimizers[name] = build_optimizer(opt_cfg, params)

# trainer_controls_optimization=True 时注入 trainer
if trainer.trainer_controls_optimization:
    trainer.set_optimizers(optimizers, lr_schedulers)
```

**Trainer 层**：通过 `self.get_optimizer(name)` / `self.get_lr_scheduler(name)` 访问，
`train_step` 完全自主控制更新顺序：

```python
@TRAINERS.register_module()
class GANTrainer(BaseTrainer):
    trainer_controls_optimization = True   # 关键标志

    def train_step(self, batch) -> dict:
        opt_d = self.get_optimizer('discriminator')
        opt_g = self.get_optimizer('generator')

        # === Phase 1: Train Discriminator ===
        opt_d.zero_grad()
        # torch.no_grad() 阻止梯度流入 G，节省显存
        with torch.no_grad():
            fake_data = self.bundle.generator(noise)
        loss_d = discriminator_loss(real_data, fake_data)
        self.accelerator.backward(loss_d)  # 必须用 accelerator.backward()
        opt_d.step()

        # === Phase 2: Train Generator ===
        opt_g.zero_grad()
        fake_data = self.bundle.generator(noise)       # WITH gradient
        fake_score = self.bundle.discriminator(fake_data)  # 梯度流经 D
        loss_g = generator_loss(fake_score)
        self.accelerator.backward(loss_g)
        opt_g.step()

        return {'loss': None, 'loss_d': loss_d.detach(), 'loss_g': loss_g.detach()}
        #        ^^^^^^^^^^
        #        loss=None 告诉 Runner 不要再执行 backward/step
```

**关键注意事项**：
- 始终使用 `self.accelerator.backward(loss)`，不要用 `loss.backward()`
- 用 `torch.no_grad()` 隔离梯度（比 `.detach()` 更省显存）
- 返回 `'loss': None` 防止 Runner 重复 backward
- 返回 dict 中的 loss 值要 `.detach()` 避免内存泄漏

**LR 日志**：多 optimizer 时日志自动显示 `lr_{name}=`：
```
step [100/10000]  lr_generator=2.00e-04  lr_discriminator=4.00e-04  loss_d=0.45  loss_g=1.23
```

**Demo 实现**：
- `hftrainer/trainers/gan/gan_trainer.py` — GAN 对抗训练（支持 StyleGAN2-style loss / regularization）
- `hftrainer/trainers/distillation/dmd_trainer.py` — DMD 蒸馏（distribution matching + fake score）
- `configs/gan/gan_demo.py` — GAN 配置示例
- `configs/distillation/dmd_demo.py` — DMD 蒸馏配置示例（带 `params` key）

**Checkpoint** 时 `accelerator.save_state()` 会自动保存所有 optimizer 的 state，
`load_scope='full'` 时也自动恢复，无需特殊处理。

---

### Issue 6: Demo Data & Evaluator/Visualizer 接口设计

#### 6a. Demo Data

每个任务在 `data/{task}/demo/` 下放少量测试数据，并提供下载脚本：

```
data/
├── text2image/demo/
│   ├── images/          # 几张测试图片
│   └── metadata.jsonl   # {"image": "images/001.jpg", "text": "a cat"}
├── text2video/demo/
│   ├── videos/
│   └── metadata.jsonl
├── llm/demo/
│   └── alpaca_sample.json   # 10 条 instruction-tuning 样本
├── classification/demo/
│   ├── images/
│   └── labels.txt
```

`tools/download_demo_data.py` 用 `datasets` 库从 HuggingFace Hub 下载：

```python
# tools/download_demo_data.py
# 用法: python tools/download_demo_data.py --task text2image
from datasets import load_dataset
DEMO_SOURCES = {
    'text2image':    ('lambdalabs/pokemon-blip-captions', 10),
    'text2video':    ('BestWishYsh/ConsisID-preview', 5),
    'llm':           ('tatsu-lab/alpaca', 10),
    'classification':('zh-plus/tiny-imagenet', 20),
}
```

每个任务的 `base_{task}_dataset.py` 提供一个 `demo()` 类方法直接返回 demo DataLoader：

```python
# 一行命令验证整个 pipeline 是否跑通
dataloader = Text2ImageDataset.demo()
```

#### 6b. Evaluator / Visualizer 接口——纯 dict，无特殊容器

**MMEngine 的痛点**：`DataSample` 是一个重量级容器，evaluator 和 visualizer 都要
知道它的内部字段（`pred_instances`, `gt_instances`, `metainfo` 等），coupling 极深，
初学者难以理解，跨任务复用几乎不可能。

**本框架的设计原则**：`val_step` 返回**纯 Python dict**，约定好 key 即可，
无任何特殊容器类。Evaluator 和 Visualizer 都消费这个 dict。

**`val_step` 的返回值约定**（各任务的 Base Trainer 中文档注释定义）：

```python
# text-to-image val_step 返回
{
    'preds':   Tensor[B, C, H, W],   # 生成的图像，float, [0,1]
    'gts':     Tensor[B, C, H, W],   # 真实图像（如有），用于 FID/LPIPS
    'prompts': list[str],             # 对应的文本 prompt
    'metas':   list[dict],            # 可选：原始文件名、分辨率等
}

# classification val_step 返回
{
    'preds':   Tensor[B],             # 预测类别 id
    'scores':  Tensor[B, num_cls],    # logits / softmax scores
    'gts':     Tensor[B],             # 真实类别 id
    'metas':   list[dict],            # 可选：图片路径等
}

# llm val_step 返回
{
    'preds':         list[str],   # 生成的文本
    'gts':           list[str],   # 参考文本
    'input_prompts': list[str],   # 输入 prompt
}
```

**Evaluator 设计**：接收整个 epoch 的 `list[dict]`（每个 dict 是一个 batch 的输出），
合并后计算指标：

```python
class BaseEvaluator:
    def process(self, output: dict) -> None:
        """每个 val batch 调用一次，缓存结果。output 就是 val_step 返回的 dict。"""
        self._results.append(output)

    def compute(self) -> dict:
        """整个 val epoch 结束后调用，返回指标 dict。"""
        raise NotImplementedError

# 示例：分类任务
class AccuracyEvaluator(BaseEvaluator):
    def compute(self) -> dict:
        all_preds = torch.cat([r['preds'] for r in self._results])
        all_gts   = torch.cat([r['gts']   for r in self._results])
        top1 = (all_preds == all_gts).float().mean().item()
        return {'top1_acc': top1}
```

**Visualizer 设计**：同样接收 `val_step` 返回的 dict，不需要知道任何容器类：

```python
class BaseVisualizer:
    def visualize(self, output: dict, step: int) -> None:
        """从 output dict 中取需要的字段，记录到 wandb/tensorboard。"""
        raise NotImplementedError

# 示例：文生图任务
class Text2ImageVisualizer(BaseVisualizer):
    def visualize(self, output: dict, step: int) -> None:
        images  = output['preds']        # Tensor[B, C, H, W]
        prompts = output['prompts']      # list[str]
        self.logger.log_images('val_samples', images, captions=prompts, step=step)
```

**完整的 val loop 流程**（在 Runner 中）：

```python
# AccelerateRunner.val_epoch()
all_outputs = []
for batch in val_dataloader:
    output = trainer.val_step(batch)          # 返回纯 dict
    output = accelerator.gather(output)       # 多卡聚合
    all_outputs.append(output)

# 计算指标
metrics = evaluator.compute_from_outputs(all_outputs)   # {'top1_acc': 0.82, ...}
self.log(metrics)

# 可视化（只取最后一个 batch）
visualizer.visualize(all_outputs[-1], step=self.global_step)
```

---

## Repository Structure

```
hf_trainer/
├── CLAUDE.md                               # This file
├── README.md
├── setup.py
│
├── configs/                                # Runnable demo configs (.py, config-driven)
│   ├── classification/vit_base_demo.py
│   ├── distillation/dmd_demo.py
│   ├── gan/gan_demo.py
│   ├── llm/llama_lora_demo.py
│   ├── llm/llama_sft_demo.py
│   ├── text2image/sd15_demo.py
│   └── text2video/wan_demo.py
│
├── hftrainer/                              # Core framework package
│   ├── registry.py                         # Registries + HF model construction helpers
│   ├── runner/                             # AccelerateRunner + loop abstractions
│   ├── models/                             # ModelBundle base class + task bundles
│   ├── trainers/                           # Training logic only
│   ├── pipelines/                          # Inference logic only
│   ├── datasets/                           # Task-specific dataset implementations
│   ├── hooks/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
│
├── tools/
│   ├── train.py
│   ├── infer.py
│   ├── dist_train.sh
│   ├── download_demo_data.py
│   ├── download_checkpoints.sh
│   └── analysis_tools/
│
├── docs/
│   ├── en/                                 # Public English docs, source of truth
│   ├── zh-cn/                              # Public Simplified Chinese docs, source of truth
│   ├── design/                             # Root-level compatibility pages for design docs
│   └── *.md                                # Root-level landing / compatibility pages
│
└── data/                                   # Demo data used by smoke tests and examples
```

**文档约定**：
- `docs/en/` 和 `docs/zh-cn/` 是对外文档的 source of truth
- 根目录 `docs/*.md` 和 `docs/design/*.md` 只保留轻量入口与兼容页，不再继续堆混合语言正文
- 公共 API 文档集中放在 `docs/en/api_reference.md` 和 `docs/zh-cn/api_reference.md`，优先覆盖 runner / bundle / trainer / pipeline / hook / CLI 这些用户入口
- 需要解释控制流时，优先在 `architecture`、`experiment_dir`、`lora` 这些高频入口页用 Mermaid 流程图

---

## Core Architecture

### 1. AccelerateRunner

```
Config (.py) ──► AccelerateRunner.from_cfg(cfg)
                     │
                     ├─ Create work_dir + timestamped run_dir (work_dir/YYYYMMDD_HHMMSS/)
                     │   - run_dir: logs (train.log), config dump, TensorBoard events
                     │   - work_dir: checkpoints (shared across runs for auto_resume)
                     ├─ Build ModelBundle  (via HF_MODELS + _build_modules: 实例化+freeze+lora)
                     ├─ Build Trainer      (via TRAINERS registry, 注入 bundle)
                     ├─ Build DataLoader   (via DATASETS registry)
                     ├─ Build Optimizer(s) (单/多 optimizer，从 bundle.trainable_parameters() 或按模块名)
                     ├─ Build LR Scheduler(s)
                     ├─ Build Hooks + Evaluator(s) + Visualizer(s)
                     ├─ Create Accelerator
                     ├─ accelerator.prepare(trainable_modules, optimizer(s), dataloader, scheduler(s))
                     │   注意：只 prepare trainable 的模块，frozen 模块不经过 accelerator
                     ├─ Handle load_from / auto_resume (model-only 或 full resume)
                     └─ Run loop: for batch in dataloader: trainer.train_step(batch)
```

**Hook system boundary**：
- Hook 是 runner 持有的回调，只负责 logging / checkpoint / EMA 这类副作用
- Hook 不负责任务前向、loss 组装、优化顺序；这些仍然属于 trainer
- Evaluator / Visualizer 是 validation 组件，不属于 hook
- Hook 按 `priority` 升序执行；当前内置顺序是 `LoggerHook(10)` → `EMAHook(15)` → `LRSchedulerHook(20, compatibility no-op)` → `CheckpointHook(80)`

**LoggerHook 日志格式**：

LoggerHook 的 `by_epoch` 默认为 `None`，自动从 `train_cfg.by_epoch` 继承。

Iter-based (`by_epoch=False`，每 N iter 打印一次):
```
step [5/10]  lr=2.00e-05  loss=1.45  data_time=0.01s  train_time=0.12s  eta=0:00:01
```

Epoch-based (`by_epoch=True`，每 N epoch 结束打印 summary):
```
epoch [1/100]  lr=2.00e-05  loss=1.45  epoch_time=120.3s  avg_iter_time=0.60s  eta=2:30:00
```

**Experiment Directory Layout**：

Checkpoint 命名基于训练模式：iter-based 用 `checkpoint-iter_N`，epoch-based 用 `checkpoint-epoch_N`。

```
work_dirs/{experiment}/
├── 20260309_142500/           # run_dir: timestamped, per-run isolation
│   ├── config.py
│   ├── train.log
│   └── training/              # TensorBoard events
├── 20260310_091200/           # second run (separate logs)
│   └── ...
├── checkpoint-iter_5000/      # iter-based checkpoint
├── checkpoint-iter_10000/
└── vis/                       # FileVisualizer output (if configured)
    └── step_5/
```

### 2. ModelBundle（核心：Trainer + Pipeline 的共同基础）

```
ModelBundle
  ├─ _build_modules(cfg)           ← 实例化所有子模块，设置 freeze/lora，记录 save_ckpt
  ├─ trainable_parameters()        ← 只返回 trainable 模块的参数
  ├─ state_dict_to_save()          ← 只返回 save_ckpt=True 的模块权重
  ├─ load_state_dict_selective()   ← 只加载 state_dict 中存在的模块
  └─ [任务相关的原子前向函数]        ← 在子类中实现，Trainer 和 Pipeline 均调用
       encode_text(texts) → Tensor
       encode_image/video(x) → latent
       predict_noise(latent, t, cond) → Tensor
       decode_latent(latent) → image/video
       ...
```

### 3. Trainer（只负责训练逻辑）

```python
class WanTrainer(BaseTrainer):
    def train_step(self, batch) -> dict:
        # 组装训练前向图，计算 loss
        # 全部通过 self.bundle.xxx() 调用，不重复实现前向函数
        text_emb = self.bundle.encode_text(batch['text'])
        latent   = self.bundle.encode_video(batch['video'])
        ...
        return {'loss': loss, 'loss_mse': ..., ...}
```

### 4. Pipeline（只负责推理逻辑）

```python
class WanPipeline(BasePipeline):
    @classmethod
    def from_checkpoint(cls, bundle_cfg, ckpt_path):
        bundle = WanBundle(bundle_cfg)
        bundle.load_state_dict_selective(torch.load(ckpt_path))
        return cls(bundle)

    @torch.no_grad()
    def __call__(self, text, num_steps=50) -> Tensor:
        # 组装推理前向图
        # 全部通过 self.bundle.xxx() 调用，与 Trainer 完全共享
        text_emb = self.bundle.encode_text([text])
        ...
        return self.bundle.decode_latent(latent)
```

### 5. Config 示例（WAN T2V finetune）

```python
# configs/text2video/wan_t2v_finetune.py
_base_ = ['../_base_/default_runtime.py']

# ModelBundle 声明：每个子模块独立控制 trainable / save_ckpt
model = dict(
    type='WanBundle',
    text_encoder=dict(
        type='UMT5EncoderModel',
        from_pretrained=dict(pretrained_model_name_or_path='/path/to/Wan2.1-T2V-14B'),
        trainable=False,
        save_ckpt=False,   # 不参与 IO，省时省空间
    ),
    vae=dict(
        type='AutoencoderKLWan',
        from_pretrained=dict(pretrained_model_name_or_path='/path/to/Wan2.1-T2V-14B'),
        trainable=False,
        save_ckpt=False,
    ),
    transformer=dict(
        type='WanTransformer3DModel',
        from_pretrained=dict(pretrained_model_name_or_path='/path/to/Wan2.1-T2V-14B'),
        trainable=True,    # 只训练这个
        save_ckpt=True,    # 只保存/恢复这个
    ),
    scheduler=dict(
        type='FlowMatchEulerDiscreteScheduler',
        from_pretrained=dict(pretrained_model_name_or_path='/path/to/Wan2.1-T2V-14B'),
        trainable=False,
        save_ckpt=False,
    ),
)

trainer = dict(type='WanTrainer', prediction_type='flow_matching', snr_gamma=None)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type='WebVidDataset',
        data_root='data/webvid/',
        resolution=(480, 832),
        num_frames=81,
    ),
)

optimizer = dict(type='AdamW', lr=1e-5, weight_decay=1e-2)
lr_scheduler = dict(type='cosine_with_warmup', num_warmup_steps=1000)

train_cfg = dict(by_epoch=False, max_iters=50000, val_interval=2000)

accelerator = dict(
    mixed_precision='bf16',
    gradient_accumulation_steps=4,
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,
        max_keep_ckpts=3,
    ),
    logger=dict(type='LoggerHook', interval=10),
)

val_evaluator = dict(type='FIDEvaluator', real_stats_path='data/text2image/fid_stats.npz')
val_visualizer = dict(
    type='Text2ImageVisualizer',
    backend='wandb',
    project='wan_t2v',
    interval=500,
    prompts=['a cat running in the park', 'ocean waves at sunset'],
)

# 完整 resume（推荐默认开启，集群抢占重启时自动接续）
auto_resume = True
```

### 6. Registry System

```python
# hftrainer/registry.py

# 实例化 HuggingFace 原生类（from_pretrained / from_config / from_single_file）
HF_MODELS = Registry('hf_model', build_fn=build_hf_model_from_cfg)

# 实例化 ModelBundle（内部调用 HF_MODELS 实例化子模块）
MODEL_BUNDLES = Registry('model_bundle')

# 实例化 Trainer（注入已构建的 ModelBundle）
TRAINERS = Registry('trainer')

# 实例化 Pipeline（注入已构建的 ModelBundle）
PIPELINES = Registry('pipeline')

DATASETS    = Registry('dataset')
TRANSFORMS  = Registry('transform')
HOOKS       = Registry('hook')
EVALUATORS  = Registry('evaluator')
VISUALIZERS = Registry('visualizer')
```

### 7. Checkpoint 策略

统一通过 `load_from` + `load_scope` 控制，`auto_resume=True` 为推荐默认值：

```python
# 完整 resume（集群抢占后重启，接续 epoch/iter/optimizer state）
auto_resume = True    # 自动检测 work_dir 最新 ckpt，full resume

# 只加载模型权重（迁移学习 / 切换数据集重新训练）
load_from = dict(path='work_dirs/wan_exp/checkpoint-iter_10000/', load_scope='model')

# full resume（等同于 auto_resume，但指定具体路径）
load_from = dict(path='work_dirs/wan_exp/checkpoint-iter_10000/', load_scope='full')
```

**Checkpoint 命名**：
- Iter-based (`by_epoch=False`): `checkpoint-iter_5000`
- Epoch-based (`by_epoch=True`): `checkpoint-epoch_3`
- 旧格式 `checkpoint-N` 在加载时仍然兼容

**Save**：`CheckpointHook` 调用 `bundle.state_dict_to_save()`，只写 `save_ckpt=True` 的模块。
`accelerator.save_state()` 同时写 optimizer / scheduler states（用于 full resume）。
Checkpoint 保存统一由 `CheckpointHook` 控制，`train_cfg` 中无需 `save_interval`。LoRA 模块默认走
adapter-only 保存，模块级 checkpoint format 元信息记录在 `model.pt` 的 `__hftrainer_meta__` 中。

**`by_epoch` 自动继承**：`CheckpointHook` 和 `LoggerHook` 的 `by_epoch` 默认为 `None`，
自动从 `train_cfg.by_epoch` 继承。用户无需手动在每个 hook 上写 `by_epoch=True`。

**`max_keep_ckpts`**：通过 `CheckpointHook` 的 `max_keep_ckpts` 参数控制磁盘上保留的
checkpoint 数量。每次保存新 checkpoint 后，自动删除最旧的 checkpoint 直到不超过上限。
`max_keep_ckpts=None`（默认）表示保留所有 checkpoint。

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,         # 按 train_cfg.by_epoch 自动解释为 iter 或 epoch
        max_keep_ckpts=3,      # 只保留最近 3 个 checkpoint
    ),
)
```

**Resume**（`load_scope='full'`）：调用 `accelerator.load_state(path)`，
自动处理 FSDP / DeepSpeed ZeRO optimizer state 格式，同时恢复 global_step / epoch。
Resume 时会输出清晰的日志：

```
============================================================
Resuming from checkpoint: work_dirs/wan_exp/checkpoint-iter_5000
Resumed: global_step=5000, epoch=0. Training will continue from step 5001.
============================================================
```

**Model-only load**（`load_scope='model'`）：只调用 `bundle.load_state_dict_selective()`，
state_dict 里有哪个模块的 key 就加载哪个，没有的跳过，optimizer 和 meta 全部重置。

**Inference**：`WanPipeline.from_checkpoint(bundle_cfg, ckpt_path)` —— 先完整构建
ModelBundle（text_encoder / vae 等从 pretrained 加载），再用 `load_state_dict_selective`
覆盖 transformer 的 finetune 权重，无需特殊适配。对于 LoRA 模块，推理阶段还可以做 merge
（例如 `tools/infer.py --merge-lora`）。

### 8. Val Loop & Evaluator/Visualizer 流程

```python
# AccelerateRunner.val_epoch() 伪代码
all_outputs = []
for batch in val_dataloader:
    output = trainer.val_step(batch)           # 返回纯 dict（preds/gts/metas/...）
    output = accelerator.gather_for_metrics(output)  # 多卡聚合
    for ev in evaluators:
        ev.process(output)                     # 缓存结果
    all_outputs.append(output)

metrics = {}
for ev in evaluators:
    metrics.update(ev.compute())               # {'top1_acc': 0.82, 'fid': 12.3, ...}
runner.log(metrics)

for vis in visualizers:
    vis.visualize(all_outputs[-1], step=runner.global_step)   # 最后一个 batch 可视化
```

`val_step` 返回纯 Python dict，key 由各任务 base trainer 的文档约定：

| Task | 必须包含的 key |
|------|--------------|
| text2image | `preds` (Tensor), `gts` (Tensor, 可选), `prompts` (list[str]) |
| text2video | `preds` (Tensor[B,T,C,H,W]), `prompts` (list[str]) |
| llm | `preds` (list[str]), `gts` (list[str]), `input_prompts` (list[str]) |
| classification | `preds` (Tensor[B]), `scores` (Tensor[B,C]), `gts` (Tensor[B]) |

---

## Supported Tasks (Target)

| Task | ModelBundle | Trainer | Pipeline | Example Models |
|------|-------------|---------|----------|----------------|
| Text-to-Image | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` | SD1.5, SDXL, SD3, Flux |
| Text-to-Video | `WanBundle` | `WanTrainer` | `WanPipeline` | WAN, CogVideoX |
| LLM | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | LLaMA, Qwen, Mistral |
| Classification | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` | ViT, DeiT, Swin |

---

## Key Dependencies

- `torch >= 2.0`
- `accelerate` — distributed training, mixed precision, gradient accumulation
- `transformers` — LLM, ViT, DETR, tokenizers, processors
- `diffusers` — SD, WAN, noise schedulers
- `peft` — LoRA, QLoRA
- `mmengine` — Config (.py parsing, `_base_` inheritance), Registry（仅用这两个，不用其训练循环）
- `safetensors` — fast checkpoint I/O
- `wandb` / `tensorboard` — logging

---

## Difference from Existing Frameworks

| | MMEngine (MMLab) | HuggingFace Trainer | HF-Trainer (ours) |
|---|---|---|---|
| Config system | .py dict + registry | TrainingArguments dataclass | .py dict + registry (MMEngine) |
| Distributed | Custom FSDP (buggy) | Accelerate | Accelerate (native) |
| Per-module freeze/ckpt | Hardcoded in Trainer | Not supported | Config-driven per-module `trainable`/`save_ckpt` |
| Multi-optimizer (GAN) | Limited | Not supported | Named optimizer dict，Trainer 自主控制 |
| Resume vs load_ckpt | Two separate APIs | `resume_from_checkpoint` | 统一 `load_from` + `load_scope` |
| Model support | BaseModel only | transformers only | Any nn.Module / HF models |
| Diffusion | No | No | First-class via DiffusionTrainer |
| Trainer↔Pipeline share | No (重复实现) | No pipeline concept | ModelBundle 共享原子前向函数 |
| Evaluator/Visualizer input | DataSample（复杂容器） | dict（但与 pipeline 耦合） | 纯 dict，key 约定即可 |
| Demo data | No | No | `data/{task}/demo/` + download script |
| Checkpoint IO | Custom, slow | Accelerate | Accelerate + selective save (跳过 frozen 模块) |

---

## Development Notes

- Package name: `hftrainer` (previously `mmhug`)
- Config parsing: `mmengine.Config` for `.py` parsing + `_base_` inheritance
- Registry: `mmengine.Registry` but our own root registries (not inheriting MMEngine's tree)
- Training loop: own implementation, not using `mmengine.runner.FlexibleRunner`
- Entry: `tools/train.py` → `AccelerateRunner.from_cfg(cfg)` → `runner.train()`
- Launch: `tools/dist_train.sh` wraps `accelerate launch`
