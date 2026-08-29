# ModelBundle

`ModelBundle` 是仓库本地模型数学与训练/推理编排之间的边界。

## Bundle 应负责什么

- 明确构造本地组件；
- 管理组件的冻结、训练、LoRA 和 checkpoint 策略；
- 提供 trainer 与 pipeline 共用的原子操作；
- 严格读写 artifact；
- 校验模型族的配置、角色和 shape 不变量。

## Bundle 不应负责什么

- import 另一套模型实现；
- 任意点分类路径解析；
- 完整训练循环或 optimizer step；
- CLI、结果编码或可视化；
- 再维护一份只给 pipeline 使用的去噪实现。

## 本地组件

`_build_modules()` 只通过 `MODEL_COMPONENTS` 解析组件名。未知名字会直接报错并显示当前已注册的本地名称；点分类路径不会被当成 import 指令。

组件 config 可以使用：

```python
transformer=dict(
    type='MyTransformer',
    from_pretrained=dict(
        pretrained_model_name_or_path='checkpoints/my-method/transformer',
    ),
    trainable='lora',       # True、False 或 'lora'
    lora_cfg=dict(rank=16, alpha=16, target_modules=['to_q', 'to_v']),
    save_ckpt=True,
    checkpoint_format='lora',
    module_dtype='bf16',
    gradient_checkpointing=True,
)
```

这里的 `from_pretrained` 表示“把受支持的 artifact 加载到本地类”，不表示选择外部类。

## `PRETRAINED_SPEC`

简单 bundle 可以声明单个 artifact root 如何映射到本地组件：

```python
class MyBundle(ModelBundle):
    PRETRAINED_SPEC = {
        'components': {
            'model': {
                'default_type': 'MyLocalModel',
                'subfolder': 'model',
                'overrides_arg': 'model_overrides',
            },
        },
        'init_args': {
            'tokenizer_path': {
                'default': ModelBundle._PRETRAINED_PATH_SENTINEL,
            },
        },
    }
```

最终类仍然只能来自 `MODEL_COMPONENTS`。需要角色校验或格式转换的复杂模型族，可以覆盖 `_bundle_config_from_pretrained()`。

## 导出

每个具体 bundle 都要实现 `save_pretrained()`。框架不提供“动态 import 一个 pipeline 再让它保存”的通用路径。具体实现必须定义：

- 配置 schema；
- tensor 文件与分片索引；
- tokenizer/processor 资源；
- 共享权重别名；
- manifest、hash 和版本；
- 严格恢复校验。

这样训练产物就不会依赖推理环境中恰好安装了哪套模型包。

## 原子操作

扩散模型 bundle 可能提供：

```python
encode_text(prompts)
encode_image(images)
add_noise(latents, noise, timesteps)
predict_noise(noisy_latents, timesteps, conditioning)
decode_latent(latents)
```

Trainer 用这些操作组成 loss；pipeline 用同一组操作组成采样。两者都不应绕过 bundle 重写组件内部行为。

## Checkpoint 范围

HFTrainer checkpoint 可以保存选定组件和 bundle 直属参数。组件格式必须明确：

- `full`：完整组件 state dict；
- `lora`：本地 adapter-only state dict。

当冻结组件能从基础 artifact 确定性恢复时，可以不写入训练 checkpoint。Bundle artifact 导出是另一条完整推理产物路径，应遵循该实现自己的 schema。

## 测试不变量

- model 层源码没有动态 import 逃逸路径；
- 内置 registry 组件都定义在 `hftrainer.models` 下；
- frozen module 在训练时仍保持 eval；
- LoRA state 不会静默丢 key；
- artifact round-trip 保持输出一致；
- trainer 与 pipeline config 能独立解析。
