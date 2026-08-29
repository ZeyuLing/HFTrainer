# 模型接入规范

HFTrainer 的“接入”是源码级接入，不是在外部模型框架外面再包一层。
只有当可执行的模型核心代码由当前仓库维护，并遵守统一目录边界时，才算完成接入。

## 不可绕过的边界

HFTrainer 发布的模型代码不能 import、动态查找或把执行委托给另一套模型实现包。
这条规则覆盖会影响数值行为的模型类、tokenizer、adapter、scheduler 和 pipeline。

PyTorch、Accelerate、MMEngine、safetensors、NumPy、Pillow、torchvision 以及媒体/科学计算工具仍然可以作为基础依赖。这里限制的是“模型执行权”必须在仓库内，而不是要求重写张量和系统基础设施。

测试会在新进程中主动拦截禁用模型包。把 import 移进 `bundle.py`、藏进
`importlib` 或通过点分类路径动态加载，都不符合规范。

## 标准纵向切片

同一个具体实现，在各层使用同一个稳定的 `implementation_id`：

```text
hftrainer/models/my_method/
  __init__.py
  bundle.py
  checkpoint.py       # artifact 较复杂时使用
  network/
    __init__.py
    configuration.py
    modeling.py
    tokenization.py   # 模型专用时使用

hftrainer/trainers/my_method/
  trainer.py

hftrainer/pipelines/my_method/
  pipeline.py

configs/my_method/
  train.py
  infer.py
```

只有当训练器或推理接口确实能被多个实现复用时，才放到
`hftrainer/tasks/<task_contract>`。例如 ViT 可以复用 `image_classification`；一个方法专用的扩散算法不应被塞进泛化的 `models/text_to_image` 目录。

## 分层职责

### `network/`

负责模型数学和模型专用基础组件：

- 网络层与 forward；
- 配置对象；
- 模型所需 tokenizer/processor；
- 属于算法行为的采样 scheduler；
- 与 checkpoint 对齐的模块和参数命名。

这里不能感知 runner、dataloader、CLI 或可视化逻辑。

### `bundle.py`

负责具体实现边界：

- 只构造本地 `network` 中明确列出的类；
- 校验组件组合；
- 提供训练和推理共用的原子操作；
- 控制冻结、训练、LoRA 和 checkpoint 保存范围；
- 维护严格的本地 artifact 读写协议。

`ModelBundle.PRETRAINED_SPEC` 可以声明一个受支持的磁盘 artifact 如何映射到本地组件，但不能用于任意动态 import。导出必须由具体 bundle 实现，因为 artifact 的 schema 和校验责任属于该实现。

### Trainer

负责 loss、更新顺序、优化器分组和训练期验证。Trainer 调用 bundle 原子操作，不能再维护一套模型 forward。

### Pipeline

负责推理图和公开输入输出。通用 CLI 根据 `cfg.pipeline.type` 选择推理图，根据
`cfg.inference.task` 选择 I/O 适配；不能通过 trainer 名称猜测任务。

## 组件注册

可执行组件注册到 `MODEL_COMPONENTS`：

```python
from hftrainer.registry import MODEL_COMPONENTS


@MODEL_COMPONENTS.register_module()
class MyTransformer(torch.nn.Module):
    ...
```

类定义必须位于 `hftrainer.models.<implementation>` 下。config 只引用本地注册名：

```python
model = dict(
    type='MyMethodBundle',
    transformer=dict(
        type='MyTransformer',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/my-method/transformer',
        ),
        trainable=True,
        save_ckpt=True,
    ),
)
```

未知名字和点分类路径会被拒绝。不能增加“安装了什么包就从什么包找类”的 fallback。

## 接入公开实现

公开代码可以作为只读参考；许可证允许时，也可以以固定 revision 的修改快照纳入仓库。两种情况都必须：

1. 记录仓库、不可变 revision 和许可证；
2. 保留许可证要求的版权和署名；
3. 按要求给修改文件增加显著变更声明；
4. 按 HFTrainer 的 model/trainer/pipeline 边界重新组织；
5. 把内部 import 改为仓库本地命名空间；
6. 移除通过外部模型框架构造运行对象的路径；
7. 在隔离的开发环境做参考对齐测试，但不能把参考实现变成产品依赖。

不能把纳入或参考改写的代码描述成 HFTrainer 原创。LTX-2.5 已随包附带固定源码 revision、修改说明和单独许可证。

## Artifact 协议

本地 artifact 通常包括：

```text
artifact/
  config.json 或 bundle_config.json
  model.safetensors（或带索引的分片）
  必要的 tokenizer/processor 资源
  manifest.json
```

loader 应根据 artifact 风险校验：

- schema/format 版本；
- 组件类和配置；
- state-dict key 与 tensor shape；
- manifest 中记录的文件或分片哈希；
- tied/shared 参数别名；
- 权重覆盖率，默认拒绝危险的低覆盖加载。

`strict=False` 不是兼容方案。需要格式转换时，应提供明确的转换工具并记录来源格式。

## LoRA

使用 `hftrainer.models.lora.apply_lora`。它注入本地 `LoRALinear`，支持 adapter-only 保存和确定性的推理合并。QLoRA 在仓库拥有并验证本地 4-bit linear 之前不会开放。

## 必须具备的测试

每个新实现至少需要：

1. 符合真实 tensor rank 的 tiny forward；
2. 训练 loss 与 backward；
3. tiny 推理/采样；
4. 保存恢复后输出一致；
5. 错误、缺失、篡改 artifact 的拒绝测试；
6. config import 与 registry 解析；
7. 主动拦截外部模型包后的新进程导入；
8. 存在参考实现时的 key/shape 与数值对齐。

大规模 gated 模型如果无法在本地分配真实权重，可以做 contract/tiny 测试，但文档必须准确说明做过什么、没做过什么。

## Review 清单

- [ ] model/trainer/pipeline/config 使用同一个实现标识；
- [ ] 模型数学位于 `network/`；
- [ ] 没有外部模型 import 或动态逃逸路径；
- [ ] bundle 只导入明确的本地组件；
- [ ] trainer/pipeline 复用 bundle 原子操作；
- [ ] artifact 覆盖率和不匹配可见；
- [ ] 来源与许可证完整；
- [ ] tiny 训练、推理、round-trip 全部通过；
- [ ] 所有 leaf config 能解析到本地组件；
- [ ] wheel 在没有禁用模型包的环境中可安装和导入。
