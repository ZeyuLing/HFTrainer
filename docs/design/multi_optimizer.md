# Multi-Optimizer Design / 多 Optimizer 设计

## Overview / 概述

A single optimizer design cannot support GAN training (separate optimizers for generator and discriminator, alternating updates each step). The design principle: **optimizer construction and update logic are pushed down to the Trainer**; the Runner does not enforce a single optimizer.

单 optimizer 设计无法支持 GAN 训练（G/D 各一个 optimizer，每步交替更新）。设计原则：**optimizer 的构建和更新逻辑都下沉到 Trainer**，Runner 不强制持有单个 optimizer。

---

## Config Layer / Config 层

The `optimizer` field supports either a single dict (standard case) or a named dict (multi-optimizer):

`optimizer` 字段支持单个 dict（常规情况）或命名 dict（多 optimizer）：

```python
# Standard: single optimizer / 常规：单 optimizer
optimizer = dict(type='AdamW', lr=1e-5)

# GAN: multiple optimizers, keys match Trainer attribute names
# GAN：多 optimizer，key 与 Trainer 内部的属性名对应
optimizer = dict(
    generator=dict(type='AdamW', lr=1e-4, betas=(0.0, 0.999)),
    discriminator=dict(type='AdamW', lr=4e-4, betas=(0.0, 0.999)),
)

# Similarly, lr_scheduler supports multiple / 同理，lr_scheduler 也支持多个
lr_scheduler = dict(
    generator=dict(type='cosine_with_warmup', num_warmup_steps=500),
    discriminator=dict(type='constant'),
)
```

---

## Runner Layer / Runner 层

The Runner detects whether `optimizer` is a single dict or named dict, builds accordingly, and passes them to the Trainer:

Runner 检测 `optimizer` 是单 dict 还是命名 dict，分别构建，统一传给 Trainer：

```python
# AccelerateRunner.build_optimizers(cfg, trainer)
if all(isinstance(v, dict) and 'type' in v for v in optimizer_cfg.values()):
    # Multi-optimizer: build for each named module
    # 多 optimizer：为每个命名模块构建
    optimizers = {
        name: build_optimizer(opt_cfg, trainer.bundle.get_module(name).parameters())
        for name, opt_cfg in optimizer_cfg.items()
    }
else:
    # Single optimizer / 单 optimizer
    optimizers = {'default': build_optimizer(optimizer_cfg, trainer.bundle.trainable_parameters())}

# accelerator.prepare supports multiple optimizers
# accelerator.prepare 支持多个 optimizer
prepared = accelerator.prepare(*optimizers.values(), ...)
trainer.set_optimizers(optimizers)
```

---

## Trainer Layer / Trainer 层

`train_step` has full autonomous control over the update order of multiple optimizers. The Runner only builds and prepares; it does not participate in the update logic.

`train_step` 完全自主控制多个 optimizer 的更新顺序，Runner 只负责构建和 prepare，不参与更新逻辑：

```python
class GANTrainer(BaseTrainer):
    def train_step(self, batch, step: int) -> dict:
        # === Train Discriminator ===
        self.opt_d.zero_grad()
        loss_d = self._discriminator_loss(batch)
        self.accelerator.backward(loss_d)
        self.opt_d.step()

        # === Train Generator ===
        self.opt_g.zero_grad()
        loss_g = self._generator_loss(batch)
        self.accelerator.backward(loss_g)
        self.opt_g.step()

        return {'loss_d': loss_d, 'loss_g': loss_g}
```

---

## Checkpoint / Checkpoint 处理

`accelerator.save_state()` automatically saves all optimizer states. `load_scope='full'` also automatically restores them. No special handling needed.

`accelerator.save_state()` 会自动保存所有 optimizer 的 state，`load_scope='full'` 时也自动恢复，无需特殊处理。
