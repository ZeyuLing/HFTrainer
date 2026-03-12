# 模型接入

这一页回答两个直接的问题：

1. 如果 `diffusers` 或 `transformers` 已经实现了某个模型，我们应该怎么改造成 HF-Trainer 的训练实现？
2. 如果 `diffusers` / `transformers` 还没有这个模型，或者这是用户自研模型，我们应该怎么接入？

## 统一约定

`ModelBundle` 现在暴露两类公共构造 API：

- `ModelBundle.from_config(cfg, **kwargs)`：通用父类方法，负责 config 驱动构造
- `ModelBundle.from_pretrained(pretrained_model_name_or_path, **kwargs)`：通用父类方法，负责 HuggingFace 风格加载

职责划分是：

- 父类负责统一 public API 形状
- 普通 HF-native bundle 优先通过 `HF_PRETRAINED_SPEC` 和 `HF_SAVE_PRETRAINED_SPEC` 完成接入
- 只有在 artifact 结构特殊到声明式 spec 表达不了时，才需要覆盖 `_bundle_config_from_pretrained(...)` 或 `save_pretrained(...)`

## 路径 1：从现有 HuggingFace 模型开始

当核心模型类已经存在于 `transformers` 或 `diffusers` 中时，走这条路径。

### 保留什么

- 官方模型类，例如 `AutoModelForCausalLM`、`UNet2DConditionModel`、`WanTransformer3DModel`
- 官方 tokenizer / processor / scheduler
- 组件级别的官方 `from_pretrained(...)` 语义

### 你需要补什么

1. 一个 `ModelBundle` 子类，用来组织任务所需组件。
2. 一组任务原子前向函数，例如 `encode_text`、`predict_noise`、`generate`、`classify`。
3. 一个 task trainer，定义 loss 和优化顺序。
4. 通常只要声明导出 spec；只有导出格式特殊时才需要手写 `save_pretrained(...)`。

显存和精度控制也应该放在 bundle config 这一层完成，而不是去改写官方模型类。常见 override 包括 `trainable`、`save_ckpt`、`checkpoint_format`、`lora_cfg`、`gradient_checkpointing`、`from_pretrained.torch_dtype` 和 `module_dtype`。

### 最小模式

```python
@MODEL_BUNDLES.register_module()
class MyLMBundle(ModelBundle):
    HF_PRETRAINED_SPEC = {
        'components': {
            'model': {
                'default_type': 'AutoModelForCausalLM',
                'pretrained_kwargs_arg': 'model_kwargs',
                'overrides_arg': 'model_overrides',
            },
        },
        'init_args': {
            'tokenizer_path': {
                'default': ModelBundle._PRETRAINED_PATH_SENTINEL,
            },
        },
    }
    HF_SAVE_PRETRAINED_SPEC = {
        'kind': 'module',
        'module': 'model',
        'extra_artifacts': ['tokenizer'],
    }

    def __init__(self, model, tokenizer_path=None):
        super().__init__()
        self._build_modules({'model': model})
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
```

### 如何把官方推理模型改成可训练

通过 `config_overrides` 或子模块 override，在官方加载路径上附加训练行为：

```python
bundle = CausalLMBundle.from_pretrained(
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    config_overrides=dict(
        model=dict(
            trainable='lora',
            checkpoint_format='lora',
            lora_cfg=dict(
                task_type='CAUSAL_LM',
                target_modules='all-linear',
                r=16,
                lora_alpha=32,
            ),
        ),
    ),
    max_length=1024,
)
```

对 diffusion 类任务也是同样模式，只是一个 pretrained 目录会在 bundle 中展开成多个子模块：

```python
bundle = SD15Bundle.from_pretrained(
    '/path/to/stable-diffusion-v1-5',
    unet_overrides=dict(trainable=True, save_ckpt=True),
    text_encoder_overrides=dict(trainable=False, save_ckpt=False),
)
```

关于全局 AMP 和按模块 dtype 的区别，以及推荐的配置写法，见 [显存与精度](memory.md)。

### 如何导出回官方推理 API

HF-native bundle 通常应该声明 `HF_SAVE_PRETRAINED_SPEC`。只有少数导出格式特殊的任务才需要手写 `save_pretrained(...)`。

当前已实现：

- `CausalLMBundle.save_pretrained(...)`
- `ViTBundle.save_pretrained(...)`
- `SD15Bundle.save_pretrained(...)`
- `WanBundle.save_pretrained(...)`

示例：

```python
bundle.save_pretrained('exports/tinyllama', merge_lora=True)
model = AutoModelForCausalLM.from_pretrained('exports/tinyllama')
```

```python
bundle.save_pretrained('exports/sd15')
pipe = StableDiffusionPipeline.from_pretrained('exports/sd15')
```

如果 LoRA bundle 用 `merge_lora=False` 导出，产物仍然是 adapter 语义，这时应该走 PEFT 的加载 API，而不是普通 `AutoModel...`。

## 路径 2：从自研模型开始

当 HuggingFace 还没有提供核心模型类时，走这条路径。

### 你需要实现什么

1. 自己的 `nn.Module` 或模块组。
2. 在 `HF_MODELS` 中注册，或者直接在 config 里传类对象。
3. 一个调用 `_build_modules(...)` 的 `ModelBundle` 子类。
4. 任务原子前向函数。
5. 一个 task trainer。

### 最小模式

```python
class MyBackbone(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        ...

@MODEL_BUNDLES.register_module()
class MyCustomBundle(ModelBundle):
    def __init__(self, backbone, head):
        super().__init__()
        self._build_modules({
            'backbone': backbone,
            'head': head,
        })

    def forward_features(self, x):
        x = self.backbone(x)
        return self.head(x)
```

通过通用父类 API 构造：

```python
bundle = MyCustomBundle.from_config(
    dict(
        backbone=dict(type=MyBackbone, hidden_size=1024),
        head=dict(type=MyHead, num_classes=1000),
    )
)
```

### 什么时候需要自己实现 `from_pretrained(...)`

只有当你需要下面这些能力时，才有必要给 bundle 增加专门的 pretrained 路径：

- 用户可以直接调用 `MyCustomBundle.from_pretrained(...)`
- 你的导出产物有稳定的磁盘格式
- 你希望把这个自定义任务桥接到第三方推理 API

这时需要：

1. 先尝试 `HF_PRETRAINED_SPEC` / `HF_SAVE_PRETRAINED_SPEC`
2. 如果声明式 spec 表达不了，再实现 `_bundle_config_from_pretrained(...)`
3. 只有当导出还需要任务相关逻辑时，再实现 `save_pretrained(...)`
4. 文档里明确导出 artifact 的格式

如果不需要这些保证，只用 `from_config(...)` 加 checkpoint save/load 就够了。

## 设计原则

保持 public API 简单：

- 已有 HuggingFace artifact 的任务，用 `from_pretrained(...)`
- 自定义模型或不适合映射到单一官方 artifact 的任务，用 `from_config(...)`
- 不要为了统一表面形式，再包一层改变官方类的语义
