# Accelerate 集成与按模块隔离

> 本文档是给 framework 维护者和高级使用者看的「**为什么**」文档。
> 普通用户写 config 只需要看 [按模块控制 dtype / GC / trainable](../memory.md) 即可。

## TL;DR

HF-Trainer **没有**把 `ModelBundle` 整体传给 `accelerator.prepare(...)`，
而是 **逐个把 trainable 子模块** + optimizer + dataloader 传进去：

```python
to_prepare = [bundle.motion_transformer, optimizer, train_dl, ...]
prepared = accelerator.prepare(*to_prepare)
```

这种「per-submodule prepare」是有意的设计，但它的副产品是
**bundle 自己直接持有的 `nn.Parameter` / `register_buffer`**
（例如 HyMotion 的 `null_vtxt_feat`、`null_ctxt_input`）会处在
**optimizer / DDP grad sync / Accelerator state 三不管地带**，
必须显式补 3 个对接点才能正常工作。本文档解释这个取舍的成因和正确处理方式。

## 1. `accelerator.prepare` 的工作粒度

Accelerate 是 **module-centric** 的扩展点：

| 调用                                                    | 工作单位         | 实际做了什么                                      |
|--------------------------------------------------------|------------------|--------------------------------------------------|
| `accelerator.prepare(M)`                               | `nn.Module`      | DDP/FSDP/DeepSpeed wrap、`.to(device)`、autocast |
| `accelerator.prepare(opt)`                             | `Optimizer`      | 替换 param refs、配合 mixed precision            |
| `accelerator.prepare(dl)`                              | `DataLoader`     | DistributedSampler                               |
| `accelerator.register_for_checkpointing(obj)`          | 任意 stateful 对象 | save_state / load_state 时调 `obj.state_dict()` |

注意：`prepare(M)` 会**完整处理 M 的 module tree**——`M.named_parameters()`、`M.named_buffers()` 子树里所有 tensor 都覆盖到。
Accelerate 没有「自动遍历进程里所有 Parameter」的能力，**你必须显式把持有目标 Parameter 的根 Module 传给它**，否则它根本看不见。

## 2. 为什么不把 `bundle` 整体 prepare

理论上 `accelerator.prepare(bundle)` 是合法的——bundle 是 `nn.Module`，prepare 会处理它整棵子树（包括 `bundle.null_vtxt_feat`）。**没人这么做是因为多 GPU/混合精度场景下会出 5 个工程问题**：

### 问题 1 — frozen 子模块被 DDP wrap 浪费

DDP 的工作单位是被 wrap 的 `nn.Module` 整体，不区分内部子模块的 trainable 状态：

- `bundle.text_encoder` (e.g. Qwen3-4B, frozen) 也被包进 DDP 通信组
- DDP 给 frozen 模块创建广播 buffer → 浪费 ~8GB / GPU
- forward 路径不经过 frozen 模块 → 默认 hang，必须 `find_unused_parameters=True`，每步 ~5–10% 性能损失

### 问题 2 — bundle 没有统一 forward，DDP grad sync 失效

DDP 的 reducer hook 注册在被 wrap module 的 `__call__` 上。trainer 的真实写法是：

```python
output = self.bundle.motion_transformer(noisy_input, ...)   # 直接拿子模块调
```

如果 wrap 的是 bundle，那这个调用绕过了 wrapper 的 `__call__` →
**reducer 不被触发，多卡 grad 不同步**，多卡训练直接出错。

要修就得给 bundle 加统一 `forward(task='diffusion_step', **kwargs)` dispatcher，
所有 trainer 改写成 `bundle(...)` —— 把当前 trainer/bundle 的解耦推回去。

### 问题 3 — 多 optimizer + 多子模块的引用断裂

Accelerate 的 `prepare(model, optimizer)` 在 *配对* 调用时能自动 patch optimizer 的 param refs。
但当一次性传 `prepare(bundle, opt1, opt2)` 时，Accelerate 没有「opt1 对应 bundle 哪棵子树」的元信息，无法正确对接。
当前设计每个 `optimizer_i` 拿 `bundle.子模块_i.parameters()`，子模块单独 prepare，optimizer 单独 prepare，配对清晰。

### 问题 4 — FSDP shard 全 bundle 后 frozen 模块 load 路径乱套

FSDP 把被 wrap module 的所有 Parameter flatten 成一维向量按 rank 切片。如果 wrap bundle，`bundle.text_encoder` 也被切片：

- frozen 模块每张卡只有 1/N → forward 要 all_gather → 巨大通信开销
- HF `from_pretrained` 提供的 unsharded weight 文件不能直接 `load_state_dict` 给 FSDP-wrapped 模块

`_pre_prepare_load`（commit `5bf5f63`）是建立在「**只 trainable 模块被 FSDP wrap，frozen 模块原样保留**」前提上的。

### 问题 5 — per-module 配置失效

VerMo 这种典型 config：

```python
model = dict(
    type='VerMoBundle',
    audio_tokenizer=dict(module_dtype='fp32'),                                  # 子模块各自要不同 dtype
    motion_transformer=dict(module_dtype='bf16', gradient_checkpointing=True),  # GC 也只针对它
    text_encoder=dict(module_dtype='fp16'),
)
```

bundle 在 `__init__` 时就已经给每个子模块独立施加 `mod.to(dtype)` / `mod.gradient_checkpointing_enable()`。
如果改成 `prepare(bundle)`：

- `mixed_precision='bf16'` 是全局一刀切的，audio_tokenizer 想要 fp32 做不到
- gradient_checkpointing 只能要么全开要么全关
- per-submodule cpu_offload 不可表达

## 3. 副产品：bundle-level orphan tensor

per-submodule prepare 是个工程胜利，但**有一个副产品**：

```
prepare(bundle.motion_transformer)   ← motion_transformer.* 都被覆盖 ✓
（bundle.text_encoder 不 prepare，手动 .to(device)）
                                      ↑
                         bundle.null_vtxt_feat 既不在 motion_transformer
                         也不在 text_encoder 的子树里 → 三不管地带
```

为什么有些 tensor 必须挂在 bundle 上而不是子模块上？因为它们**不属于任何具体子模块**：

- `null_vtxt_feat` / `null_ctxt_input` 是 classifier-free guidance 的可学习 null embedding
  - 不属于 `motion_transformer`：是它的输入预处理产物
  - 不属于 `text_encoder`：CFG 的本意就是绕过 text encoder
  - 跨多个子模块共享，由 bundle 在 `mask_text_cond` 这个共享 API 里使用
- `mean` / `std`：motion 归一化常量，bundle 在多个原子前向函数里用
- UMO 的 `null_source_feat` (trainable)：source-CFG 的 null embedding

## 4. 三处对接点（不是胶带，是接缝）

orphan tensor 必须显式补 3 处对接点才能在分布式训练里正常工作。**每一处都对接到 Accelerate / PyTorch 的标准扩展点**，不是私自捏的 side channel：

### 对接点 A — Optimizer

orphan param 不在任何被 prepare 的子模块里 → optimizer 看不见 → 不被训练。

```python
# from_cfg
_orphan_trainable_params = [
    p for _, p in bundle.named_parameters(recurse=False) if p.requires_grad
]
# trainable_parameters() returns _trainable_modules' params + _orphan_trainable_params
```

> 出处: commit `29947be` (`fix: save/train/sync bundle-level Parameters and Buffers`)

### 对接点 B — DDP gradient sync

orphan param 不在任何 DDP-wrapped 模块里 → backward 时各 rank 的 grad 各算各的，不同步。

```python
def _sync_orphan_param_grads(self):
    """Manual all_reduce over _orphan_trainable_params after backward."""
    if not self._orphan_trainable_params or self.accelerator.num_processes <= 1:
        return
    # 显式 all_reduce
```

> 出处: commit `29947be`

### 对接点 C — Accelerator save_state / load_state

orphan tensor 不被 `accelerator.save_state` / `load_state` 处理 → 每次 full-resume 都被 reset 为构造时 zeros。

```python
class _BundleOrphanCheckpoint:
    """Adapter exposing bundle.named_parameters(recurse=False) and
    named_buffers(recurse=False) via state_dict()/load_state_dict()."""
    ...

# from_cfg, after accelerator.prepare(...)
accelerator.register_for_checkpointing(_BundleOrphanCheckpoint(bundle))
```

`save_state` 把 adapter 的 `state_dict()` 写到 `custom_checkpoint_0.pkl`；
`load_state` 反向读回。完全走 Accelerate 标准 `register_for_checkpointing` 扩展点。

> 出处: commit `9a67a3d`（修复 commit `29947be` 留下的 LOAD 侧缺失）

### Bug 史

直到 2026-04 这个 bug 都未被发现，因为：

- **首次启动**走 `_load(scope='model')` → `bundle.load_state_dict_selective` → 正确从 T2M 1.0 把 `null_vtxt_feat` (norm=10.13) 注入 bundle ✓
- **第二次启动**走 `_load(scope='full')` → `accelerator.load_state` → orphan tensor 不被恢复，留在构造时 zero ❌
- 后续 save 把 zero 持久化到 `model.pt::__bundle_params__`，bug 闭环

audit 结果：35+ 个 HyMotion T2M / M2M / UMO 产线 ckpt 的 `null_vtxt_feat` 都是 0。修复后短跑产物（不触发 auto-resume）的 norm 是 10.13——直接验证 first-load 通路一直是对的，只有 resume 通路坏了。

## 5. 当前代码的预期不变量

写 bundle / runner 代码时请保持下列契约：

| 契约 | 说明 |
|------|-----|
| bundle 直接持有的 `nn.Parameter` 必须出现在 `bundle.named_parameters(recurse=False)` 里 | 三处对接点的扫描入口 |
| 同上，`register_buffer` 出现在 `bundle.named_buffers(recurse=False)` | 适配器一并处理 |
| 在 `from_cfg` 中 prepare 之后才注册 `_BundleOrphanCheckpoint` | bundle 子模块此时已替换为 DDP wrapper，不影响适配器（适配器只看 recurse=False） |
| `_BundleOrphanCheckpoint` 是 Accelerator custom_objects index 0 | 不要在它之前注册其他对象 |
| `model.pt::__bundle_params__` 仅服务于 selective load (`scope='model'`)，不参与 full-resume | 双轨清晰：full-resume 走 custom_checkpoint_0.pkl |

## 6. 长期建议（未实施）

如果要从根上消除 "orphan" 这个概念，可以把 bundle-level Parameter 收进一个专用子模块：

```python
class _SharedParamModule(nn.Module):
    """跨子模块共享的 stateful tensor 容器；本身没有 forward。"""
    def __init__(self, vtxt_dim, ctxt_dim):
        super().__init__()
        self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, vtxt_dim), requires_grad=False)
        self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, ctxt_dim), requires_grad=False)
        self.register_buffer('mean', ...)

class HyMotionM2MBundle(ModelBundle):
    def __init__(self, ...):
        ...
        self.shared_params = _SharedParamModule(...)
        self._save_ckpt_modules.append('shared_params')
```

这样 `shared_params` 走和 `motion_transformer` 完全一样的 prepare/save/load 通路，
**完全消除 orphan 的概念**。代价是要批量改 7+ 个文件里的 `bundle.null_vtxt_feat → bundle.shared_params.null_vtxt_feat`。

如果未来做 bundle 结构 cleanup，建议做这个迁移；当前代码用三处对接点保证正确性。

## 7. 相关文档

- 用户视角的按模块配置语法：[显存与精度](../memory.md)
- ModelBundle 通用约定：[ModelBundle 设计](model_bundle.md)
- Checkpoint save/load 规则：[Checkpoint](checkpoint.md)
- 多 optimizer：[Multi-Optimizer](multi_optimizer.md)
