# Accelerate Integration & Per-Module Isolation

> Audience: framework maintainers and advanced users.
> If you only write configs, [Memory & Precision](../memory.md) is enough.

## TL;DR

HF-Trainer **does not** pass `ModelBundle` as a whole to `accelerator.prepare(...)`.
Instead, it prepares **trainable sub-modules individually** plus optimizers and dataloaders:

```python
to_prepare = [bundle.motion_transformer, optimizer, train_dl, ...]
prepared = accelerator.prepare(*to_prepare)
```

This per-submodule prepare is intentional, but it has a side effect:
**any `nn.Parameter` / `register_buffer` declared directly on the bundle**
(e.g. HyMotion's `null_vtxt_feat`, `null_ctxt_input`) ends up
in a three-way blind spot — invisible to the optimizer, to DDP grad sync,
and to Accelerator's state machinery — and needs three explicit hand-off
points to function correctly. This document explains the trade-off and the
correct way to handle the side effect.

## 1. What `accelerator.prepare` actually operates on

Accelerate is a **module-centric** extension surface:

| Call                                            | Granularity      | Effect                                                  |
|-------------------------------------------------|------------------|---------------------------------------------------------|
| `accelerator.prepare(M)`                        | `nn.Module`      | DDP/FSDP/DeepSpeed wrap, `.to(device)`, autocast        |
| `accelerator.prepare(opt)`                      | `Optimizer`      | Param-ref rewrite, mixed-precision integration          |
| `accelerator.prepare(dl)`                       | `DataLoader`     | DistributedSampler installation                         |
| `accelerator.register_for_checkpointing(obj)`   | Any stateful obj | `obj.state_dict()` is called on save_state/load_state   |

`prepare(M)` traverses **the whole module subtree of M** — every tensor in
`M.named_parameters()` and `M.named_buffers()` is covered. There is **no
mechanism in Accelerate that magically discovers Parameters living outside
the prepared subtree**. The root that owns the tensor must be passed
explicitly.

## 2. Why we don't `prepare(bundle)` directly

A `ModelBundle` *is* an `nn.Module`, so `prepare(bundle)` is technically valid
and would happily process `bundle.null_vtxt_feat`. We don't do it because
**multi-GPU / mixed-precision training breaks in five concrete ways**:

### Problem 1 — DDP wraps frozen sub-modules

DDP works at the **wrapped module** level and is blind to the per-parameter
`requires_grad` flag at sync time:

- `bundle.text_encoder` (e.g. Qwen3-4B, frozen) joins the DDP communication group
- DDP allocates broadcast buffers for it — easily ~8GB / GPU wasted
- The forward path skips frozen sub-modules → DDP hangs unless
  `find_unused_parameters=True`, which costs ~5–10% / step

### Problem 2 — Bundle has no unified `forward`, DDP grad sync silently breaks

DDP's reducer hook fires on the wrapped module's `__call__`. The trainer
actually calls sub-modules directly:

```python
output = self.bundle.motion_transformer(noisy_input, ...)   # bypasses bundle.__call__
```

If the wrapper sits on `bundle`, this call **bypasses the wrapper** and the
reducer is never triggered. **Multi-GPU gradients silently desync**.

The only fix is to give bundle a unified `forward(task='diffusion_step', **kw)`
dispatcher and rewrite every trainer to `bundle(...)`. That undoes the
trainer/bundle decoupling that makes the framework usable in the first place.

### Problem 3 — Multi-optimizer + multi-sub-module reference rewrite breaks

Accelerate's `prepare(model, optimizer)` can patch the optimizer's param refs
**because the pairing is explicit in the call**. `prepare(bundle, opt1, opt2)`
gives Accelerate no way to know which optimizer owns which subtree.

The current design pairs each `optimizer_i` with `bundle.子模块_i.parameters()`,
prepares each pair separately, and the rewrite is unambiguous.

### Problem 4 — FSDP shards frozen sub-modules and breaks `from_pretrained`

FSDP flattens all parameters of a wrapped module into a 1-D vector and shards
them by rank. Wrapping `bundle` shards `bundle.text_encoder` too:

- Each rank only holds 1/N of the frozen weights → all-gather every forward,
  enormous comm cost
- HF `from_pretrained` produces unsharded weight files that cannot be loaded
  via `load_state_dict` into an FSDP-wrapped module

`_pre_prepare_load` (commit `5bf5f63`) is built on the assumption that
**only trainable sub-modules are FSDP-wrapped, frozen modules stay intact**.

### Problem 5 — Per-module configuration becomes impossible

A typical VerMo config:

```python
model = dict(
    type='VerMoBundle',
    audio_tokenizer=dict(module_dtype='fp32'),
    motion_transformer=dict(module_dtype='bf16', gradient_checkpointing=True),
    text_encoder=dict(module_dtype='fp16'),
)
```

Bundle's `__init__` already applies `mod.to(dtype)` /
`mod.gradient_checkpointing_enable()` per sub-module. Switching to
`prepare(bundle)`:

- `mixed_precision='bf16'` is global, audio_tokenizer cannot stay fp32
- gradient_checkpointing is all-or-nothing
- Per-sub-module cpu_offload becomes inexpressible

## 3. The side effect: bundle-level orphan tensors

Per-submodule prepare is an engineering win, but **it leaves a gap**:

```
prepare(bundle.motion_transformer)   ← motion_transformer.* covered ✓
(bundle.text_encoder is .to(device)'d manually, not prepared)
                                      ↑
                         bundle.null_vtxt_feat lives on neither subtree
                         → three-way blind spot
```

Why are some tensors owned by the bundle rather than a sub-module? Because
they don't naturally belong to any one of them:

- `null_vtxt_feat` / `null_ctxt_input` — classifier-free guidance null
  embeddings.
  - Not part of `motion_transformer`: they are a preprocessing artifact
    consumed by it.
  - Not part of `text_encoder`: the whole point of CFG is to bypass it.
  - Used by the bundle in `mask_text_cond`, a shared API across trainer/pipeline.
- `mean` / `std` — motion normalization stats used by multiple atomic forward
  functions.
- UMO's `null_source_feat` (trainable) — source-CFG null embedding.

## 4. Three hand-off points (not duct tape — formal seams)

Each hand-off uses an Accelerate / PyTorch standard extension point. They
are not private side channels:

### A — Optimizer

Orphan params live outside any prepared sub-module → optimizer can't see them.

```python
# from_cfg
_orphan_trainable_params = [
    p for _, p in bundle.named_parameters(recurse=False) if p.requires_grad
]
# trainable_parameters() = trainable_modules' params ∪ _orphan_trainable_params
```

> Origin: commit `29947be` (`fix: save/train/sync bundle-level Parameters and Buffers`)

### B — DDP gradient sync

Orphan params are outside any DDP-wrapped module → each rank computes its own
gradient with no reduction.

```python
def _sync_orphan_param_grads(self):
    """Manual all_reduce over _orphan_trainable_params after backward."""
    ...
```

> Origin: commit `29947be`

### C — Accelerator save_state / load_state

`accelerator.save_state` / `load_state` cover only prepared modules and
explicitly registered objects. Without registration, orphan tensors
silently revert to constructor-time zeros after every full resume.

```python
class _BundleOrphanCheckpoint:
    """Adapter exposing bundle.named_parameters(recurse=False) and
    named_buffers(recurse=False) via state_dict()/load_state_dict()."""
    ...

# from_cfg, after accelerator.prepare(...)
accelerator.register_for_checkpointing(_BundleOrphanCheckpoint(bundle))
```

`save_state` → adapter's `state_dict()` written to `custom_checkpoint_0.pkl`.
`load_state` → adapter's `load_state_dict()` reads it back. Standard
`register_for_checkpointing` extension point.

> Origin: commit `9a67a3d` (closes the LOAD-side hole left by `29947be`)

### Bug history

The bug went undetected until 2026-04 because:

- **First startup**: `_load(scope='model')` → `bundle.load_state_dict_selective`
  correctly injects `null_vtxt_feat` (norm=10.13) from T2M 1.0 ✓
- **Subsequent restarts**: `_load(scope='full')` → `accelerator.load_state`
  does not restore orphan tensors → they revert to zeros ❌
- Subsequent save persists the zeros into `model.pt::__bundle_params__`,
  closing the loop.

Audit of 35+ HyMotion T2M / M2M / UMO production checkpoints: every
`null_vtxt_feat` is zero. After the fix, short-run products (which never
trigger auto-resume) have norm 10.13 — proving the first-load path was
always correct; only the resume path was broken.

## 5. Invariants the current code expects

When writing bundle / runner code, preserve the following:

| Invariant | Reason |
|-----------|--------|
| Bundle-level `nn.Parameter` must be reachable via `bundle.named_parameters(recurse=False)` | Scan target for the three hand-offs |
| Same for `register_buffer` via `bundle.named_buffers(recurse=False)` | Adapter handles both |
| `_BundleOrphanCheckpoint` is registered **after** `accelerator.prepare(...)` | Sub-modules are by then DDP-wrapped; the adapter only walks `recurse=False`, so unaffected |
| `_BundleOrphanCheckpoint` is Accelerator custom-object index 0 | Don't register other objects ahead of it |
| `model.pt::__bundle_params__` serves selective load (`scope='model'`) only, **not** full-resume | Two-track design: full-resume goes through `custom_checkpoint_0.pkl` |

## 6. Long-term suggestion (not implemented)

To eliminate the "orphan" concept entirely, collect bundle-level Parameters
into a dedicated sub-module:

```python
class _SharedParamModule(nn.Module):
    """Container for stateful tensors shared across sub-modules; no forward."""
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

`shared_params` then traverses the same prepare/save/load path as
`motion_transformer`, **eliminating the orphan concept**. The cost is renaming
`bundle.null_vtxt_feat → bundle.shared_params.null_vtxt_feat` across 7+ files.

Recommended for any future bundle structure cleanup. Until then, the three
hand-off points guarantee correctness.

## 7. Related docs

- User-facing per-module config syntax: [Memory & Precision](../memory.md)
- General bundle contract: [ModelBundle](model_bundle.md)
- Checkpoint save/load rules: [Checkpoint](checkpoint.md)
- Multi optimizer: [Multi-Optimizer](multi_optimizer.md)
