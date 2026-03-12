# ModelBundle

`ModelBundle` 是训练和推理共享的任务核心。

## 职责

- 从 config 构建子模块
- 对每个子模块应用 `trainable` / `save_ckpt`
- 暴露任务级原子前向函数
- 提供选择性 checkpoint save/load
- 暴露统一的 `from_config(...)` 父类入口
- 暴露统一的 `from_pretrained(...)` 父类入口

## 为什么需要它

如果没有共享 bundle，同一套任务逻辑通常会在 trainer 和 inference pipeline 中各写一遍。HF-Trainer 把共享逻辑收敛到 bundle 中，让 trainer 和 pipeline 只负责各自的控制流。

## 当前示例

- `ViTBundle`
- `SD15Bundle`
- `CausalLMBundle`
- `WanBundle`

## 构造约定

`ModelBundle` 现在有一套统一的父类构造约定：

- `from_config(...)` 是通用方法，所有 bundle 都能用
- `from_pretrained(...)` 也定义在父类上
- 大多数 HF-native bundle 应该通过声明 `HF_PRETRAINED_SPEC` 来完成加载映射，而不是手写 `_bundle_config_from_pretrained(...)`
- 大多数 HF-native bundle 应该通过声明 `HF_SAVE_PRETRAINED_SPEC` 来完成导出，而不是手写 `save_pretrained(...)`
- 只有在 artifact 结构很特殊，或者导出还有额外副作用时，才需要自己覆盖方法

这样用户只需要记一套 public API，普通 bundle 作者通常也只需要写声明式 spec。

## 两条接入路径

### 路径 A：HuggingFace 原生模型

如果 `transformers` 或 `diffusers` 已经有这个模型类：

- 在 bundle 内继续使用官方类
- 对外入口用 `Bundle.from_pretrained(...)`
- 通过声明 `HF_PRETRAINED_SPEC` 把官方 artifact 映射成 bundle 组件
- 通过 `trainable`、`save_ckpt`、`checkpoint_format`、`lora_cfg` 这些 config 覆盖项附加训练行为
- 如果希望训练产物能回到官方推理 API，就声明 `HF_SAVE_PRETRAINED_SPEC`

### 路径 B：自研模型

如果 HuggingFace 生态里还没有这个模型：

- 自己实现 `nn.Module`
- 通过 `Bundle.from_config(...)` 实例化
- 只有在你需要稳定导出 artifact，且声明式 spec 不足以表达时，再补自定义 `from_pretrained(...)` / `save_pretrained(...)`

具体例子见 [模型接入](../integration.md)。

## 声明式 HF Spec

对大多数 HF-native bundle，父类已经把模板代码准备好了。

### `HF_PRETRAINED_SPEC`

用它描述“一个 pretrained artifact 如何变成 bundle config”：

```python
class MyBundle(ModelBundle):
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
            'max_length': 1024,
        },
    }
```

### `HF_SAVE_PRETRAINED_SPEC`

用它描述“bundle 如何导出回推理 artifact”：

```python
class MyBundle(ModelBundle):
    HF_SAVE_PRETRAINED_SPEC = {
        'kind': 'module',
        'module': 'model',
        'merge_lora_modules': ['model'],
        'extra_artifacts': ['tokenizer'],
    }
```

这样普通 bundle 一般只需要写 `__init__` 和任务原子方法。

## 按模块控制

每个子模块都可以声明：

- `trainable=True/False/'lora'`
- `save_ckpt=True/False`
- `checkpoint_format='full'|'lora'`
- `gradient_checkpointing=True` 或 kwargs dict
- `module_dtype='fp32'|'fp16'|'bf16'`（或 `torch.dtype`）
- `from_pretrained` / `from_config` / `from_single_file`

这样 optimizer 只会看到需要训练的参数，checkpoint 也可以跳过冻结模块。

如果底层 HF loader 已经支持目标 dtype，优先用 `from_pretrained.torch_dtype`；如果你希望在 load 之后由 HF-Trainer 统一 cast，就用 `module_dtype`。具体示例和 AMP 注意事项见 [显存与精度](../memory.md)。

## LoRA

推荐配置方式：

```python
model = dict(
    type='CausalLMBundle',
    model=dict(
        type='AutoModelForCausalLM',
        from_pretrained=dict(pretrained_model_name_or_path=CKPT_PATH),
        trainable='lora',
        checkpoint_format='lora',
        lora_cfg=dict(
            task_type='CAUSAL_LM',
            target_modules='all-linear',
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        ),
    ),
)
```

LoRA 模块默认使用 `checkpoint_format='lora'`，也就是 adapter-only save/load。如果你明确希望保存完整的 LoRA 包装模块 state dict，再改成 `checkpoint_format='full'`。
