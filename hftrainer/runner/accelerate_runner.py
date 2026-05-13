"""
AccelerateRunner: the central training orchestrator.

Responsibilities:
  - Build all components from config (bundle, trainer, dataloaders, optimizers, hooks, etc.)
  - Prepare via accelerator.prepare()
  - Drive the training/validation loop
  - Handle checkpoint save / load / auto_resume
"""

import os
import copy
import math
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import DistributedType
from mmengine.config import Config

from hftrainer.utils.logger import get_logger, add_file_handler
from hftrainer.utils.checkpoint_utils import find_latest_checkpoint
from hftrainer.utils.env import collect_env_info
from hftrainer.runner.loops import EpochBasedLoop, IterBasedLoop

logger = get_logger()


class _LegacyOrphanStub:
    """Tiny stub used by ``_ensure_bundle_orphan_custom_ckpt`` to feed
    ``accelerate.checkpointing.save_custom_state`` (which requires a
    ``state_dict()`` method) when synthesising
    ``custom_checkpoint_0.pkl`` from a legacy ``model.pt``.
    """

    def __init__(self, payload: Dict[str, torch.Tensor]):
        self._payload = payload

    def state_dict(self):
        return self._payload


class _BundleOrphanCheckpoint:
    """Adapter that exposes a ModelBundle's orphan tensors to Accelerator's
    standard ``save_state`` / ``load_state`` machinery.

    Why this exists
    ---------------
    ``Accelerator.save_state`` / ``load_state`` only manage state for objects
    explicitly registered with the accelerator — modules/optimizers passed
    through ``accelerator.prepare`` and additional objects passed to
    ``accelerator.register_for_checkpointing``.

    A ``ModelBundle`` typically prepares only its sub-modules (e.g.
    ``motion_transformer``, ``text_encoder``).  ``nn.Parameter`` and
    ``register_buffer`` attributes living **directly on the bundle**
    (e.g. HyMotion's ``null_vtxt_feat`` / ``null_ctxt_input``,
    UMO's ``null_source_feat``) would otherwise sit in a three-way blind
    spot — invisible to the optimizer, to DDP gradient sync, and to
    Accelerator state machinery — and silently revert to constructor-
    time zeros after every full-resume.

    Registering an instance of this class via
    ``accelerator.register_for_checkpointing`` makes those orphan
    tensors round-trip cleanly through the standard
    ``custom_checkpoint_*.pkl`` mechanism with **no patch logic on the
    load path**.
    """

    def __init__(self, bundle: nn.Module):
        # Stored as plain attribute (not nn.Module child) to avoid
        # accidentally exposing the bundle through Accelerator's module tree.
        object.__setattr__(self, '_bundle', bundle)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        bundle = self._bundle
        out: Dict[str, torch.Tensor] = {}
        for name, p in bundle.named_parameters(recurse=False):
            out[name] = p.data.detach().cpu().clone()
        for name, b in bundle.named_buffers(recurse=False):
            out[name] = b.detach().cpu().clone()
        return out

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        bundle = self._bundle
        param_names = {n for n, _ in bundle.named_parameters(recurse=False)}
        buffer_names = {n for n, _ in bundle.named_buffers(recurse=False)}
        for name, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            if name in param_names:
                target = getattr(bundle, name)
                if target.shape != value.shape:
                    logger.warning(
                        f"Bundle orphan param '{name}' shape mismatch "
                        f"(ckpt {tuple(value.shape)} vs model {tuple(target.shape)}); skipped"
                    )
                    continue
                with torch.no_grad():
                    target.data.copy_(value.to(target.device, dtype=target.dtype))
            elif name in buffer_names:
                target = getattr(bundle, name)
                if target.shape != value.shape:
                    logger.warning(
                        f"Bundle orphan buffer '{name}' shape mismatch "
                        f"(ckpt {tuple(value.shape)} vs model {tuple(target.shape)}); skipped"
                    )
                    continue
                target.copy_(value.to(target.device, dtype=target.dtype))
            else:
                logger.warning(
                    f"Bundle orphan checkpoint contained '{name}' which is "
                    f"not a current bundle attribute; skipped"
                )


class AccelerateRunner:
    """
    Main training runner that integrates Accelerate with the hftrainer framework.

    Usage:
        runner = AccelerateRunner.from_cfg(cfg)
        runner.train()
    """

    def __init__(
        self,
        bundle,
        trainer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizers: Dict[str, torch.optim.Optimizer],
        lr_schedulers: Dict[str, Any],
        accelerator: Accelerator,
        hooks: List[Any],
        evaluators: List[Any],
        visualizers: List[Any],
        train_cfg: dict,
        work_dir: str,
        run_dir: Optional[str] = None,
        load_from: Optional[dict] = None,
        auto_resume: bool = False,
        cfg=None,  # original full config for reference
    ):
        self.bundle = bundle
        self.trainer = trainer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.accelerator = accelerator
        self.hooks = hooks
        self.evaluators = evaluators
        self.visualizers = visualizers
        self.train_cfg = train_cfg
        self.work_dir = work_dir
        self.run_dir = run_dir or work_dir
        self.load_from = load_from
        self.auto_resume = auto_resume
        self.cfg = cfg

        self.global_step = 0
        self.current_epoch = 0
        self.max_grad_norm = train_cfg.get('max_grad_norm', None)

        # Bundle-level Parameters outside any sub-module (DDP-unsafe).
        # Assigned by from_cfg() after cls() construction; default to empty.
        self._orphan_trainable_params: List[torch.nn.Parameter] = []

        # Inject accelerator into trainer
        self.trainer.accelerator = accelerator
        self.trainer.runner = self

        # Inject runner into hooks
        for hook in self.hooks:
            hook.runner = self

        # Auto-inherit by_epoch from train_cfg to hooks that have by_epoch=None
        by_epoch = self.train_cfg.get('by_epoch', False)
        for hook in self.hooks:
            if hasattr(hook, 'by_epoch') and hook.by_epoch is None:
                hook.by_epoch = by_epoch

    @classmethod
    def from_cfg(cls, cfg) -> 'AccelerateRunner':
        """Build all components from a config object or dict."""
        if isinstance(cfg, dict):
            cfg = Config(cfg)

        work_dir = getattr(cfg, 'work_dir', 'work_dirs/default')
        os.makedirs(work_dir, exist_ok=True)

        # ── Create timestamped run directory for logs/config/tensorboard ──
        # Checkpoints stay in work_dir (base) so auto_resume can always find them.
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(work_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)

        # ── Build Accelerator ──
        accel_cfg = getattr(cfg, 'accelerator', {})
        if hasattr(accel_cfg, 'to_dict'):
            accel_cfg = accel_cfg.to_dict()
        accel_cfg = copy.deepcopy(accel_cfg)

        # Build FSDP plugin if specified in config
        fsdp_plugin = None
        fsdp_cfg = accel_cfg.pop('fsdp_plugin', None)
        if fsdp_cfg is not None:
            from accelerate import FullyShardedDataParallelPlugin
            if hasattr(fsdp_cfg, 'to_dict'):
                fsdp_cfg = fsdp_cfg.to_dict()
            fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_cfg)

        # Build DeepSpeed plugin if specified in config
        deepspeed_plugin = None
        ds_cfg = accel_cfg.pop('deepspeed_plugin', None)
        if ds_cfg is not None:
            from accelerate import DeepSpeedPlugin
            if hasattr(ds_cfg, 'to_dict'):
                ds_cfg = ds_cfg.to_dict()
            deepspeed_plugin = DeepSpeedPlugin(**ds_cfg)

        # Auto-fallback: bf16 → fp16 if device doesn't support bf16
        requested_mp = accel_cfg.get('mixed_precision', 'no')
        if requested_mp == 'bf16':
            import torch as _torch
            if _torch.cuda.is_available() and not _torch.cuda.is_bf16_supported():
                logger.warning(
                    "bf16 mixed precision requested but not supported on this device. "
                    "Falling back to fp16."
                )
                accel_cfg['mixed_precision'] = 'fp16'

        accelerator = Accelerator(
            mixed_precision=accel_cfg.get('mixed_precision', 'no'),
            gradient_accumulation_steps=accel_cfg.get('gradient_accumulation_steps', 1),
            log_with=accel_cfg.get('log_with', 'tensorboard'),
            project_dir=run_dir,
            fsdp_plugin=fsdp_plugin,
            deepspeed_plugin=deepspeed_plugin,
        )

        # ── File logging, config dump, env info (main process only) ──
        if accelerator.is_main_process:
            add_file_handler(logger, os.path.join(run_dir, 'train.log'))
            try:
                cfg.dump(os.path.join(run_dir, 'config.py'))
                logger.info(f"Config saved to: {os.path.join(run_dir, 'config.py')}")
            except Exception as e:
                logger.warning(f"Could not dump config: {e}")
            logger.info(f"Work dir: {work_dir}")
            logger.info(f"Run dir (logs/tb): {run_dir}")
            logger.info(f"Environment info:\n{collect_env_info()}")

        # ── Build ModelBundle ──
        model_cfg = getattr(cfg, 'model', None)
        assert model_cfg is not None, "cfg.model is required"
        bundle = cls._build_bundle(model_cfg, accelerator)

        # Log model parameter summary
        if accelerator.is_main_process:
            cls._log_model_summary(bundle)

        # ── Build Trainer ──
        trainer_cfg = getattr(cfg, 'trainer', None)
        assert trainer_cfg is not None, "cfg.trainer is required"
        if accelerator.is_main_process:
            logger.info("Building trainer...")
        trainer = cls._build_trainer(trainer_cfg, bundle)

        # ── Build DataLoaders ──
        if accelerator.is_main_process:
            logger.info("Building dataloaders...")
        train_dl_cfg = getattr(cfg, 'train_dataloader', None)
        val_dl_cfg = getattr(cfg, 'val_dataloader', None)
        train_dataloader = cls._build_dataloader(train_dl_cfg) if train_dl_cfg else None
        val_dataloader = cls._build_dataloader(val_dl_cfg) if val_dl_cfg else None

        # ── Build Optimizers ──
        if accelerator.is_main_process:
            logger.info("Building optimizers...")
        optimizer_cfg = getattr(cfg, 'optimizer', None)
        assert optimizer_cfg is not None or train_dataloader is None, "cfg.optimizer is required for training"
        optimizers = cls._build_optimizers(optimizer_cfg, bundle) if optimizer_cfg else {}

        # ── Compute total training steps (needed for some schedulers) ──
        train_cfg_dict = getattr(cfg, 'train_cfg', {})
        if hasattr(train_cfg_dict, 'to_dict'):
            train_cfg_dict = train_cfg_dict.to_dict()
        num_training_steps = cls._compute_num_training_steps(
            train_cfg_dict, train_dataloader, accel_cfg
        )

        # ── Build LR Schedulers ──
        lr_sched_cfg = getattr(cfg, 'lr_scheduler', None)
        lr_schedulers = cls._build_lr_schedulers(lr_sched_cfg, optimizers, num_training_steps)

        # ── Build Hooks ──
        hooks_cfg = getattr(cfg, 'default_hooks', {})
        if hasattr(hooks_cfg, 'to_dict'):
            hooks_cfg = hooks_cfg.to_dict()
        hooks = cls._build_hooks(hooks_cfg)

        # ── Build Evaluators ──
        eval_cfg = getattr(cfg, 'val_evaluator', None)
        evaluators = cls._build_evaluators(eval_cfg)

        # ── Build Visualizers ──
        vis_cfg = getattr(cfg, 'val_visualizer', None)
        visualizers = cls._build_visualizers(vis_cfg)

        # ── Pre-prepare model-only load for FSDP / DeepSpeed ──
        # FSDP and DeepSpeed flatten/shard parameters during prepare(), so
        # model-only checkpoints (which store original-shape tensors) must
        # be loaded BEFORE prepare().  Full-resume checkpoints that contain
        # accelerator state files are handled AFTER prepare() via
        # accelerator.load_state().
        _pre_loaded_model = False
        _pre_loaded_meta = None
        uses_sharding = fsdp_plugin is not None or deepspeed_plugin is not None
        if uses_sharding:
            _pre_loaded_model, _pre_loaded_meta = cls._pre_prepare_load(
                bundle, cfg, work_dir, accelerator,
            )

        # ── Accelerator prepare ──
        # Only prepare trainable modules, not frozen ones
        if accelerator.is_main_process:
            logger.info("Starting accelerator.prepare()...")
        trainable_module_list = [
            getattr(bundle, name) for name in bundle._trainable_modules
            if isinstance(getattr(bundle, name), nn.Module)
        ]

        # Identify bundle-level trainable Parameters (e.g. null_vtxt_feat,
        # null_ctxt_input) that live outside any registered sub-module.
        # These cannot be DDP-wrapped directly, so we manually all_reduce
        # their gradients after each backward pass (see train loop).
        # NOTE: from_cfg is a classmethod so 'self' doesn't exist yet;
        # we store in a local variable and assign to runner after cls().
        _orphan_trainable_params = [
            param
            for _name, param in bundle.named_parameters(recurse=False)
            if param.requires_grad
        ]

        # Move frozen modules to device manually
        for name in bundle._frozen_modules:
            mod = getattr(bundle, name, None)
            if isinstance(mod, nn.Module):
                # Check for meta tensors (e.g. from HF from_pretrained with
                # missing checkpoint keys).  to_empty() materialises them as
                # uninitialised storage on the target device, which is fine
                # because they were already "newly initialised" by HF and
                # will be overwritten by any subsequent load_state_dict.
                has_meta = any(
                    p.device.type == 'meta'
                    for p in mod.parameters()
                ) or any(
                    b.device.type == 'meta'
                    for b in mod.buffers()
                )
                if has_meta:
                    mod.to_empty(device=accelerator.device)
                    # Re-initialise any remaining uninitialised params
                    for p in mod.parameters():
                        if not p.is_complex() and p.requires_grad:
                            torch.nn.init.normal_(p)
                else:
                    mod.to(accelerator.device)

        # Move orphan parameters/buffers (registered directly on the bundle,
        # not inside any sub-module) to the accelerator device.
        _child_params = set()
        for child in bundle.children():
            for p in child.parameters():
                _child_params.add(p.data_ptr())
            for b in child.buffers():
                _child_params.add(b.data_ptr())
        for p in bundle.parameters():
            if p.data_ptr() not in _child_params:
                p.data = p.data.to(accelerator.device)
        for name, buf in bundle.named_buffers():
            if buf.data_ptr() not in _child_params:
                # re-register buffer on device
                parts = name.rsplit('.', 1)
                if len(parts) == 1:
                    bundle.register_buffer(parts[0], buf.to(accelerator.device))
                # nested buffers inside sub-modules are already handled

        # Prepare trainable modules + optimizer + dataloader
        optimizer_list = list(optimizers.values())
        scheduler_list = list(lr_schedulers.values())

        to_prepare = trainable_module_list + optimizer_list
        if train_dataloader is not None:
            to_prepare.append(train_dataloader)
        if val_dataloader is not None:
            to_prepare.append(val_dataloader)
        to_prepare.extend(scheduler_list)

        prepared = accelerator.prepare(*to_prepare)

        # Make Accelerator's save_state / load_state cover bundle-level orphan
        # nn.Parameters and buffers (e.g. null_vtxt_feat / null_ctxt_input /
        # null_source_feat / mean / std).  Without this, those tensors would
        # be invisible to Accelerator's state machinery and silently revert
        # to constructor-time zeros across every full-resume cycle.
        # Index 0 is reserved for this adapter; do not register additional
        # custom-checkpoint objects ahead of it without updating consumers.
        accelerator.register_for_checkpointing(_BundleOrphanCheckpoint(bundle))

        # Unpack prepared objects back
        idx = 0
        for i, name in enumerate(bundle._trainable_modules):
            if isinstance(getattr(bundle, name), nn.Module):
                setattr(bundle, name, prepared[idx])
                idx += 1

        prepared_optimizers = {}
        for key in optimizers:
            prepared_optimizers[key] = prepared[idx]
            idx += 1

        prepared_train_dl = None
        if train_dataloader is not None:
            prepared_train_dl = prepared[idx]
            idx += 1

        prepared_val_dl = None
        if val_dataloader is not None:
            prepared_val_dl = prepared[idx]
            idx += 1

        prepared_schedulers = {}
        for key in lr_schedulers:
            prepared_schedulers[key] = prepared[idx]
            idx += 1

        runner = cls(
            bundle=bundle,
            trainer=trainer,
            train_dataloader=prepared_train_dl,
            val_dataloader=prepared_val_dl,
            optimizers=prepared_optimizers,
            lr_schedulers=prepared_schedulers,
            accelerator=accelerator,
            hooks=hooks,
            evaluators=evaluators,
            visualizers=visualizers,
            train_cfg=train_cfg_dict,
            work_dir=work_dir,
            run_dir=run_dir,
            load_from=getattr(cfg, 'load_from', None),
            auto_resume=getattr(cfg, 'auto_resume', False),
            cfg=cfg,
        )

        # Assign orphan trainable params (bundle-level Parameters outside sub-modules)
        runner._orphan_trainable_params = _orphan_trainable_params

        # Apply pre-loaded meta (global_step / epoch) if we did a pre-prepare load
        if _pre_loaded_model and _pre_loaded_meta is not None:
            runner.global_step = _pre_loaded_meta.get('global_step', 0)
            runner.current_epoch = _pre_loaded_meta.get('current_epoch', 0)
            logger.info(
                f"Restored training position: global_step={runner.global_step}, "
                f"epoch={runner.current_epoch}. Optimizer state was reset."
            )

        # Initialize tensorboard / trackers
        if accelerator.is_main_process:
            try:
                accelerator.init_trackers('training')
            except Exception as e:
                logger.warning(f"Could not init trackers: {e}")

        # If trainer controls optimization, inject optimizers/schedulers into it
        if runner.trainer.trainer_controls_optimization:
            runner.trainer.set_optimizers(runner.optimizers, runner.lr_schedulers)
            logger.info(
                f"trainer_controls_optimization=True: injected "
                f"{list(runner.optimizers.keys())} optimizers into "
                f"{type(runner.trainer).__name__}"
            )

        # Handle checkpoint loading (skipped if pre-prepare load already handled it)
        if not _pre_loaded_model:
            runner._handle_load()

        return runner

    # ─────────────────────────────────────────────────────────────────────────
    # Component builders
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_bundle(model_cfg, accelerator=None):
        """Build ModelBundle from config."""
        from hftrainer.registry import MODEL_BUNDLES
        if hasattr(model_cfg, 'to_dict'):
            model_cfg = model_cfg.to_dict()
        model_cfg = copy.deepcopy(model_cfg)
        return MODEL_BUNDLES.build(model_cfg)

    @staticmethod
    def _pre_prepare_load(bundle, cfg, work_dir, accelerator):
        """Load model-only checkpoint BEFORE accelerator.prepare().

        FSDP and DeepSpeed flatten/shard parameters during prepare(),
        making it impossible to load original-shape state dicts afterwards.
        This method detects model-only checkpoints (no accelerator state
        files) and loads them while the model still has original shapes.

        For full-resume checkpoints (containing ``model.safetensors``),
        loading is deferred to ``_handle_load()`` which runs after
        prepare() and uses ``accelerator.load_state()``.

        **Memory optimisation**: In distributed mode the checkpoint is
        loaded **one rank at a time** (rank 0 first, then the rest in
        sequence).  Each rank frees the raw state-dict immediately after
        ``load_state_dict_selective`` so that at most one copy of the
        checkpoint coexists with the model weights in CPU RAM.  Without
        this serialisation, *all* ranks loading a large ``.pth`` file
        simultaneously can easily exceed host memory (e.g. 8 × 40 GB on
        a 375 GB node).

        Returns:
            (loaded: bool, meta: dict | None)
        """
        from hftrainer.utils.checkpoint_utils import (
            find_latest_checkpoint, load_checkpoint,
        )
        import gc

        auto_resume = getattr(cfg, 'auto_resume', False)
        load_from = getattr(cfg, 'load_from', None)

        # Determine checkpoint path and whether it is model-only
        ckpt_path = None
        is_model_only = False
        is_auto_resume_ckpt = False  # True only when ckpt comes from auto_resume

        if auto_resume:
            latest = find_latest_checkpoint(work_dir)
            if latest:
                has_accel_state = (
                    os.path.exists(os.path.join(latest, 'model.safetensors'))
                    or os.path.exists(os.path.join(latest, 'pytorch_model.bin'))
                )
                if has_accel_state:
                    # Full resume — let _handle_load deal with it after prepare
                    return False, None
                else:
                    ckpt_path = latest
                    is_model_only = True
                    is_auto_resume_ckpt = True
            # else: no checkpoint found — fall through to check load_from

        # When auto_resume found no checkpoint (or is disabled), check load_from
        # so that model-only pretrained weights are loaded BEFORE FSDP wrapping.
        exclude_bundle_keys = None
        if not ckpt_path and load_from is not None:
            if hasattr(load_from, 'to_dict'):
                load_from = load_from.to_dict()
            if isinstance(load_from, str):
                # Plain string path → treat as model-only load
                ckpt_path = load_from
                is_model_only = True
            elif isinstance(load_from, dict):
                scope = load_from.get('load_scope', 'model')
                if scope == 'model':
                    ckpt_path = load_from.get('path', None)
                    is_model_only = True
                    exclude_bundle_keys = load_from.get(
                        'exclude_bundle_keys', None
                    )
                # scope == 'full' is handled after prepare by _handle_load

        if not ckpt_path or not is_model_only:
            return False, None

        # -----------------------------------------------------------
        # Serialised loading: one rank at a time to avoid OOM
        # -----------------------------------------------------------
        is_distributed = torch.distributed.is_initialized()
        num_processes = accelerator.num_processes if is_distributed else 1
        my_rank = accelerator.process_index if is_distributed else 0

        try:
            for loading_rank in range(num_processes):
                if my_rank == loading_rank:
                    logger.info(
                        "[Pre-FSDP] Rank %d/%d: loading model weights "
                        "from %s ...",
                        my_rank, num_processes, ckpt_path,
                    )
                    state_dict = load_checkpoint(ckpt_path, map_location='cpu')
                    bundle.load_state_dict_selective(
                        state_dict,
                        exclude_bundle_keys=exclude_bundle_keys,
                    )
                    del state_dict
                    gc.collect()

                # Barrier: wait for the current loading_rank to finish before
                # the next rank starts, keeping peak CPU RAM bounded.
                if is_distributed:
                    torch.distributed.barrier()

            sep = '=' * 60
            logger.info(sep)
            logger.info(
                f"[Pre-FSDP] Loaded model weights from: {ckpt_path}"
            )
            logger.info(
                "Weights loaded before FSDP sharding. "
                "Optimizer state was reset."
            )
            logger.info(sep)
        except FileNotFoundError:
            logger.warning(
                f"No model checkpoint found at {ckpt_path}, "
                "skipping pre-prepare load"
            )
            return False, None

        # Read meta for step/epoch restoration — only for auto_resume
        # checkpoints (i.e. ckpt_path was set by the auto_resume branch).
        # When load_from with load_scope='model' is used, we are starting a
        # NEW training run from pretrained weights, so global_step/epoch
        # must stay at 0.
        meta = None
        if is_auto_resume_ckpt:
            meta_path = os.path.join(ckpt_path, 'meta.pt')
            if os.path.exists(meta_path):
                meta = torch.load(
                    meta_path, map_location='cpu', weights_only=False,
                )

        return True, meta

    @staticmethod
    def _log_model_summary(bundle):
        """Log per-module parameter counts and trainable/save status.

        Also reports **bundle-level orphan tensors** — ``nn.Parameter`` /
        ``register_buffer`` attached directly to the bundle (e.g.
        ``null_vtxt_feat``, ``null_ctxt_input``, ``mean``, ``std``).  These
        live outside any sub-module and are managed via the
        ``_BundleOrphanCheckpoint`` adapter (see ``docs/design/
        accelerate_integration.md``); without listing them here the summary
        would silently hide ~5K bundle params and several MB of buffers,
        and a missing/zero null embedding would not be visible at startup.
        """
        rows = []
        total_params = 0
        total_trainable = 0

        for name, module in bundle._modules.items():
            if module is None:
                continue
            n_params = sum(p.numel() for p in module.parameters())
            n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            is_trainable = name in bundle._trainable_modules
            is_save = name in bundle._save_ckpt_modules
            rows.append((name, n_params, n_trainable, is_trainable, is_save))
            total_params += n_params
            total_trainable += n_trainable

        # Bundle-level orphan parameters (direct nn.Parameter children).
        # Reported as a single "<bundle-orphan>" pseudo-row; trainable
        # column reflects whichever orphan params have requires_grad=True.
        # Save status is always True because they round-trip via
        # _BundleOrphanCheckpoint -> custom_checkpoint_0.pkl AND model.pt::
        # __bundle_params__.
        orphan_param_pairs = list(bundle.named_parameters(recurse=False))
        if orphan_param_pairs:
            orphan_total = sum(p.numel() for _n, p in orphan_param_pairs)
            orphan_trainable = sum(
                p.numel() for _n, p in orphan_param_pairs if p.requires_grad
            )
            any_trainable = any(p.requires_grad for _n, p in orphan_param_pairs)
            rows.append((
                '<bundle-orphan>',
                orphan_total,
                orphan_trainable,
                any_trainable,
                True,
            ))
            total_params += orphan_total
            total_trainable += orphan_trainable

        if not rows:
            return

        # Compute column widths
        name_w = max(len(r[0]) for r in rows)
        name_w = max(name_w, len('Module'), len('TOTAL'))
        tp_w = max(len(f'{r[1]:,}') for r in rows)
        tp_w = max(tp_w, len('Total Params'), len(f'{total_params:,}'))
        tr_w = max(len(f'{r[2]:,}') for r in rows)
        tr_w = max(tr_w, len('Trainable Params'), len(f'{total_trainable:,}'))
        flag_w = max(len('Trainable'), len('False'))
        save_w = max(len('Save Ckpt'), len('False'))

        header = (f"  {'Module':<{name_w}}  {'Total Params':>{tp_w}}  "
                  f"{'Trainable Params':>{tr_w}}  {'Trainable':<{flag_w}}  "
                  f"{'Save Ckpt':<{save_w}}")
        sep = '  ' + '-' * name_w + '  ' + '-' * tp_w + '  ' + '-' * tr_w + '  ' + '-' * flag_w + '  ' + '-' * save_w

        lines = ['\nModel Summary:', header, sep]
        for name, n_params, n_trainable, is_trainable, is_save in rows:
            lines.append(
                f"  {name:<{name_w}}  {n_params:>{tp_w},}  "
                f"{n_trainable:>{tr_w},}  {str(is_trainable):<{flag_w}}  "
                f"{str(is_save):<{save_w}}"
            )
        lines.append(sep)
        lines.append(
            f"  {'TOTAL':<{name_w}}  {total_params:>{tp_w},}  "
            f"{total_trainable:>{tr_w},}"
        )
        if total_params > 0:
            ratio = total_trainable / total_params * 100
            lines.append(f"  Trainable ratio: {ratio:.2f}%")

        # Bundle-level orphan params/buffers: per-tensor norms.  This is the
        # earliest place at startup where we surface whether null embeddings
        # carry a meaningful pretrained value (norm > 0) or are stuck at
        # zero — the latter being the M2M caption-mode regression we fixed
        # in 9a67a3d.  Showing it here makes that visible without grep'ing
        # logs after EMA init.
        orphan_lines = []
        for n, p in bundle.named_parameters(recurse=False):
            orphan_lines.append(
                f"  <bundle-orphan> {n:<24s} param "
                f"shape={tuple(p.shape)} norm={p.detach().float().norm().item():.4f} "
                f"requires_grad={p.requires_grad}"
            )
        for n, b in bundle.named_buffers(recurse=False):
            orphan_lines.append(
                f"  <bundle-orphan> {n:<24s} buffer "
                f"shape={tuple(b.shape)} norm={b.detach().float().norm().item():.4f}"
            )
        if orphan_lines:
            lines.append('  Bundle orphan tensors:')
            lines.extend(orphan_lines)

        lines.append('')

        logger.info('\n'.join(lines))

    @staticmethod
    def _build_trainer(trainer_cfg, bundle):
        """Build Trainer from config, injecting the bundle."""
        from hftrainer.registry import TRAINERS
        if hasattr(trainer_cfg, 'to_dict'):
            trainer_cfg = trainer_cfg.to_dict()
        trainer_cfg = copy.deepcopy(trainer_cfg)
        return TRAINERS.build(trainer_cfg, default_args={'bundle': bundle})

    @staticmethod
    def _build_dataloader(dl_cfg) -> Optional[DataLoader]:
        """Build DataLoader from config."""
        if dl_cfg is None:
            return None
        from hftrainer.registry import DATASETS
        if hasattr(dl_cfg, 'to_dict'):
            dl_cfg = dl_cfg.to_dict()
        dl_cfg = copy.deepcopy(dl_cfg)

        dataset_cfg = dl_cfg.pop('dataset', None)
        if dataset_cfg is None:
            dataset_cfg = dl_cfg
        elif hasattr(dataset_cfg, 'to_dict'):
            dataset_cfg = dataset_cfg.to_dict()

        batch_size = dl_cfg.pop('batch_size', 1)
        num_workers = dl_cfg.pop('num_workers', 0)
        shuffle = dl_cfg.pop('shuffle', True)
        pin_memory = dl_cfg.pop('pin_memory', False)
        drop_last = dl_cfg.pop('drop_last', False)
        collate_fn = dl_cfg.pop('collate_fn', None)
        persistent_workers = dl_cfg.pop('persistent_workers', False)
        sampler = dl_cfg.pop('sampler', None)

        dataset = DATASETS.build(dataset_cfg)
        if collate_fn is None and hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        if sampler is not None:
            loader_kwargs['sampler'] = sampler
        else:
            loader_kwargs['shuffle'] = shuffle

        if collate_fn is not None:
            loader_kwargs['collate_fn'] = collate_fn

        if num_workers > 0:
            loader_kwargs['persistent_workers'] = persistent_workers

        return DataLoader(dataset, **loader_kwargs)

    @staticmethod
    def _build_optimizers(optimizer_cfg, bundle) -> Dict[str, torch.optim.Optimizer]:
        """
        Build optimizer(s) from config.
        Supports single optimizer dict or named dict for multi-optimizer.
        """
        if hasattr(optimizer_cfg, 'to_dict'):
            optimizer_cfg = optimizer_cfg.to_dict()
        optimizer_cfg = copy.deepcopy(optimizer_cfg)

        # Detect single vs multi-optimizer
        is_multi = (
            isinstance(optimizer_cfg, dict)
            and all(isinstance(v, dict) and 'type' in v for v in optimizer_cfg.values())
            and 'type' not in optimizer_cfg
        )

        if is_multi:
            optimizers = {}
            for name, opt_cfg in optimizer_cfg.items():
                opt_cfg = copy.deepcopy(opt_cfg)
                # Support explicit 'params' key: list of bundle module names
                param_names = opt_cfg.pop('params', None)
                if param_names is not None:
                    params = []
                    for mod_name in param_names:
                        module = getattr(bundle, mod_name, None)
                        if module is not None and isinstance(module, nn.Module):
                            params.extend(
                                param for param in module.parameters()
                                if param.requires_grad
                            )
                        else:
                            raise ValueError(
                                f"Optimizer '{name}' references module '{mod_name}' "
                                f"which does not exist in the bundle or is not nn.Module. "
                                f"Available modules: {bundle._trainable_modules}"
                            )
                else:
                    # Fallback: match optimizer name to bundle module name
                    module = getattr(bundle, name, None)
                    if module is not None and isinstance(module, nn.Module):
                        params = [
                            param for param in module.parameters()
                            if param.requires_grad
                        ]
                    else:
                        raise ValueError(
                            f"Optimizer '{name}' does not match any bundle module. "
                            f"Use 'params' key to specify module names explicitly. "
                            f"Available trainable modules: {bundle._trainable_modules}"
                        )
                optimizers[name] = AccelerateRunner._build_single_optimizer(opt_cfg, params)
            return optimizers
        else:
            params = bundle.trainable_parameters()
            return {'default': AccelerateRunner._build_single_optimizer(optimizer_cfg, params)}

    @staticmethod
    def _build_single_optimizer(opt_cfg: dict, params) -> torch.optim.Optimizer:
        """Build a single optimizer."""
        opt_cfg = copy.deepcopy(opt_cfg)
        opt_type = opt_cfg.pop('type')

        # Import from torch.optim, then transformers.optimization, then registry
        import torch.optim as optim
        if hasattr(optim, opt_type):
            cls = getattr(optim, opt_type)
        else:
            # Try transformers.optimization (Adafactor, etc.)
            try:
                import transformers.optimization as tf_optim
                if hasattr(tf_optim, opt_type):
                    cls = getattr(tf_optim, opt_type)
                else:
                    raise ImportError
            except (ImportError, AttributeError):
                from hftrainer.registry import _import_hf_class
                cls = _import_hf_class(opt_type)
                if cls is None:
                    raise ValueError(f"Unknown optimizer type: {opt_type}")

        return cls(params, **opt_cfg)

    @staticmethod
    def _build_lr_schedulers(sched_cfg, optimizers: dict, num_training_steps: int) -> dict:
        """Build LR scheduler(s)."""
        if sched_cfg is None:
            return {}

        if hasattr(sched_cfg, 'to_dict'):
            sched_cfg = sched_cfg.to_dict()
        sched_cfg = copy.deepcopy(sched_cfg)

        # Detect multi-scheduler
        is_multi = (
            isinstance(sched_cfg, dict)
            and all(isinstance(v, dict) and 'type' in v for v in sched_cfg.values())
            and 'type' not in sched_cfg
        )

        if is_multi:
            schedulers = {}
            for name, s_cfg in sched_cfg.items():
                if name in optimizers:
                    schedulers[name] = AccelerateRunner._build_single_scheduler(
                        s_cfg, optimizers[name], num_training_steps
                    )
            return schedulers
        else:
            optimizer = optimizers.get('default', next(iter(optimizers.values())))
            sched = AccelerateRunner._build_single_scheduler(sched_cfg, optimizer, num_training_steps)
            return {'default': sched}

    @staticmethod
    def _build_single_scheduler(sched_cfg: dict, optimizer, num_training_steps: int):
        """Build a single LR scheduler. Supports HF get_scheduler API."""
        sched_cfg = copy.deepcopy(sched_cfg)
        sched_type = sched_cfg.pop('type')

        # Alias mapping for convenience
        SCHEDULER_ALIASES = {
            'cosine_with_warmup': 'cosine',  # transformers uses 'cosine'
        }
        sched_type = SCHEDULER_ALIASES.get(sched_type, sched_type)

        # Try transformers get_scheduler first
        HF_SCHEDULER_TYPES = {
            'linear', 'cosine', 'cosine_with_restarts', 'polynomial',
            'constant', 'constant_with_warmup', 'inverse_sqrt',
            'reduce_lr_on_plateau', 'cosine_with_min_lr',
            'cosine_warmup_with_min_lr', 'warmup_stable_decay',
        }

        if sched_type in HF_SCHEDULER_TYPES:
            from transformers import get_scheduler
            num_warmup_steps = sched_cfg.pop('num_warmup_steps', 0)
            return get_scheduler(
                name=sched_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **sched_cfg,
            )
        else:
            # Try torch.optim.lr_scheduler
            import torch.optim.lr_scheduler as lr_sched
            if hasattr(lr_sched, sched_type):
                cls = getattr(lr_sched, sched_type)
                return cls(optimizer, **sched_cfg)
            raise ValueError(f"Unknown scheduler type: {sched_type}")

    @staticmethod
    def _compute_num_training_steps(train_cfg: dict, train_dataloader, accel_cfg: dict) -> int:
        """Compute total number of training steps."""
        grad_accum = accel_cfg.get('gradient_accumulation_steps', 1)

        if train_cfg.get('by_epoch', False):
            max_epochs = train_cfg.get('max_epochs', 1)
            if train_dataloader is not None:
                try:
                    return max_epochs * math.ceil(len(train_dataloader) / grad_accum)
                except TypeError:
                    pass
            return max_epochs * 1000  # fallback
        else:
            max_iters = train_cfg.get('max_iters', 10000)
            return math.ceil(max_iters / grad_accum)

    @staticmethod
    def _build_hooks(hooks_cfg: dict) -> list:
        """Build hooks from config."""
        from hftrainer.registry import HOOKS
        hooks = []
        for name, hook_cfg in hooks_cfg.items():
            if hook_cfg is None:
                continue
            if hasattr(hook_cfg, 'to_dict'):
                hook_cfg = hook_cfg.to_dict()
            hook_cfg = copy.deepcopy(hook_cfg)
            hook = HOOKS.build(hook_cfg)
            hooks.append(hook)
        # Sort by priority if available
        hooks.sort(key=lambda h: getattr(h, 'priority', 50))
        return hooks

    @staticmethod
    def _build_evaluators(eval_cfg) -> list:
        """Build evaluator(s) from config."""
        if eval_cfg is None:
            return []
        from hftrainer.registry import EVALUATORS
        if isinstance(eval_cfg, (list, tuple)):
            return [EVALUATORS.build(copy.deepcopy(cfg)) for cfg in eval_cfg]
        if hasattr(eval_cfg, 'to_dict'):
            eval_cfg = eval_cfg.to_dict()
        return [EVALUATORS.build(copy.deepcopy(eval_cfg))]

    @staticmethod
    def _build_visualizers(vis_cfg) -> list:
        """Build visualizer(s) from config."""
        if vis_cfg is None:
            return []
        from hftrainer.registry import VISUALIZERS
        if isinstance(vis_cfg, (list, tuple)):
            return [VISUALIZERS.build(copy.deepcopy(cfg)) for cfg in vis_cfg]
        if hasattr(vis_cfg, 'to_dict'):
            vis_cfg = vis_cfg.to_dict()
        return [VISUALIZERS.build(copy.deepcopy(vis_cfg))]

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint handling
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_load(self):
        """Handle auto_resume and load_from at startup."""
        if self.auto_resume:
            latest = find_latest_checkpoint(self.work_dir)
            if latest:
                # Determine the appropriate load scope.
                # If the checkpoint dir has accelerator state files (model.safetensors
                # or pytorch_model.bin), do a full resume (model + optimizer + meta).
                # Otherwise fall back to model-only selective load — this handles
                # migrated legacy checkpoints that only contain model.pt / meta.pt.
                has_accel_state = (
                    os.path.exists(os.path.join(latest, 'model.safetensors'))
                    or os.path.exists(os.path.join(latest, 'pytorch_model.bin'))
                )
                if has_accel_state:
                    self._load(latest, load_scope='full')
                else:
                    logger.info(
                        f"Checkpoint {latest} has no accelerator state files. "
                        "Loading model weights only (optimizer will be reset)."
                    )
                    self._load(latest, load_scope='model')
                    # Still restore global_step / epoch from meta if available
                    meta_path = os.path.join(latest, 'meta.pt')
                    if os.path.exists(meta_path):
                        meta = torch.load(meta_path, map_location='cpu', weights_only=False)
                        self.global_step = meta.get('global_step', 0)
                        self.current_epoch = meta.get('current_epoch', 0)
                        logger.info(
                            f"Restored training position: global_step={self.global_step}, "
                            f"epoch={self.current_epoch}. Optimizer state was reset."
                        )
                # After resume, patch any zero null embeddings from pretrained.
                self._patch_zero_null_embeddings_from_pretrained()
                return
            else:
                logger.info("auto_resume=True but no checkpoint found. Starting from scratch.")

        if self.load_from is not None:
            load_cfg = self.load_from
            if hasattr(load_cfg, 'to_dict'):
                load_cfg = load_cfg.to_dict()
            path = load_cfg.get('path', load_cfg) if isinstance(load_cfg, dict) else load_cfg
            scope = load_cfg.get('load_scope', 'model') if isinstance(load_cfg, dict) else 'model'
            ebk = load_cfg.get('exclude_bundle_keys', None) if isinstance(load_cfg, dict) else None
            self._load(path, load_scope=scope, exclude_bundle_keys=ebk)
            # After load_from, patch any zero null embeddings from a
            # secondary pretrained source.  This handles the case where
            # load_from points to a checkpoint that lacks good null
            # embeddings (e.g. an unconditioned model with zeros) while
            # a pretrained T2M checkpoint has the correct values.
            self._patch_zero_null_embeddings_from_pretrained()

    def _load(self, path: str, load_scope: str = 'model',
              exclude_bundle_keys=None):
        """
        Load checkpoint with given scope.

        load_scope='model': load model weights only, reset optimizer/scheduler/meta
        load_scope='full':  load everything via accelerator.load_state
        """
        sep = '=' * 60
        if load_scope == 'full':
            logger.info(sep)
            logger.info(f"Resuming from checkpoint: {path}")
            # Bundle-level orphan tensors (null_vtxt_feat, null_ctxt_input,
            # mean, std, ...) round-trip through Accelerator's standard
            # custom-checkpoint mechanism via the
            # ``_BundleOrphanCheckpoint`` adapter that was registered in
            # ``from_cfg``.  For ckpts that pre-date that registration we
            # first synthesise a ``custom_checkpoint_0.pkl`` from
            # ``model.pt::__bundle_params__`` so ``accelerator.load_state``
            # sees a complete, count-matched state directory.
            self._ensure_bundle_orphan_custom_ckpt(path)
            self.accelerator.load_state(path)
            # Try to restore global_step from metadata
            meta_path = os.path.join(path, 'meta.pt')
            if os.path.exists(meta_path):
                meta = torch.load(meta_path, map_location='cpu', weights_only=False)
                self.global_step = meta.get('global_step', 0)
                self.current_epoch = meta.get('current_epoch', 0)
            logger.info(
                f"Resumed: global_step={self.global_step}, epoch={self.current_epoch}. "
                f"Training will continue from step {self.global_step + 1}."
            )

            # Post-resume verification of bundle-level orphan tensors.
            # The Model Summary printed in from_cfg shows their values at
            # bundle-init time (i.e. zeros), which is misleading when the
            # ckpt actually carries patched / pretrained values.  Print the
            # post-load norm here so a missing or unexpectedly-zero null
            # embedding is visible at startup without grep'ing model.pt.
            orphan_lines = []
            for n, p in self.bundle.named_parameters(recurse=False):
                orphan_lines.append(
                    f"  <bundle-orphan post-load> {n:<22s} param "
                    f"shape={tuple(p.shape)} norm={p.detach().float().norm().item():.4f} "
                    f"requires_grad={p.requires_grad}"
                )
            for n, b in self.bundle.named_buffers(recurse=False):
                orphan_lines.append(
                    f"  <bundle-orphan post-load> {n:<22s} buffer "
                    f"shape={tuple(b.shape)} norm={b.detach().float().norm().item():.4f}"
                )
            if orphan_lines:
                logger.info('Post-resume bundle orphan tensors:\n' + '\n'.join(orphan_lines))
            logger.info(sep)
        elif load_scope == 'model':
            from hftrainer.utils.checkpoint_utils import load_checkpoint
            try:
                state_dict = load_checkpoint(path, map_location='cpu')
                self.bundle.load_state_dict_selective(
                    state_dict,
                    exclude_bundle_keys=exclude_bundle_keys,
                )
                logger.info(sep)
                logger.info(f"Loaded model weights from: {path}")
                logger.info("Optimizer and training state reset to initial.")
                logger.info(sep)
            except FileNotFoundError:
                logger.warning(f"No model checkpoint found at {path}, skipping model load")
        else:
            raise ValueError(f"Unknown load_scope: {load_scope}. Expected 'model' or 'full'.")

    def save_checkpoint(self):
        """Save checkpoint. Directory name reflects training mode."""
        by_epoch = self.train_cfg.get('by_epoch', False)
        if by_epoch:
            ckpt_dir = os.path.join(self.work_dir, f'checkpoint-epoch_{self.current_epoch}')
        else:
            ckpt_dir = os.path.join(self.work_dir, f'checkpoint-iter_{self.global_step}')

        if self.accelerator.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        # Save the accelerator state on every process so distributed backends
        # can write their rank-local shards and RNG state.
        self.accelerator.save_state(ckpt_dir)
        self.accelerator.wait_for_everyone()

        # Save selective model weights in a backend-aware way.
        state_dict = self._state_dict_to_save()

        if not self.accelerator.is_main_process:
            return

        import torch
        torch.save(state_dict, os.path.join(ckpt_dir, 'model.pt'))

        # Save meta using completed step / epoch counts for exact resume.
        meta = {'global_step': self.global_step, 'current_epoch': self.current_epoch}
        torch.save(meta, os.path.join(ckpt_dir, 'meta.pt'))

        logger.info(f"Saved checkpoint to: {ckpt_dir}")

        # Manage max_keep_ckpts
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if max_keep_ckpts is set."""
        max_keep = None
        for hook in self.hooks:
            if hasattr(hook, 'max_keep_ckpts'):
                max_keep = hook.max_keep_ckpts
                break
        if max_keep is None:
            return

        import glob, shutil
        pattern = os.path.join(self.work_dir, 'checkpoint-*')
        candidates = [c for c in glob.glob(pattern) if os.path.isdir(c)]
        ckpts = sorted(candidates, key=self._extract_ckpt_order)
        while len(ckpts) > max_keep:
            oldest = ckpts.pop(0)
            if os.path.isdir(oldest):
                shutil.rmtree(oldest)
            logger.info(f"Removed old checkpoint: {oldest}")

    def _ensure_bundle_orphan_custom_ckpt(self, path: str):
        """Legacy-ckpt one-shot migration for the bundle-orphan adapter.

        Background: ``_BundleOrphanCheckpoint`` is registered with
        Accelerator in :meth:`from_cfg`, which means
        ``accelerator.load_state`` expects a matching
        ``custom_checkpoint_0.pkl`` file under ``path``.  Checkpoints
        produced before this registration was added do not have that
        file — Accelerator would then raise ``RuntimeError`` due to
        a count mismatch on custom-objects.

        This helper detects that case and synthesises the missing
        ``custom_checkpoint_0.pkl`` from
        ``model.pt::__bundle_params__`` (where every legacy ckpt
        already stores those orphan tensors), so the standard load
        path can proceed without any bespoke "post-load patch" code.

        Hot-path is untouched: for any ckpt produced after this fix
        the file already exists and this method is a single
        ``os.path.exists`` no-op.
        """
        cust = os.path.join(path, 'custom_checkpoint_0.pkl')
        if os.path.exists(cust):
            return  # produced by the new save path; nothing to do

        # Try to rebuild from legacy model.pt::__bundle_params__
        mpt = os.path.join(path, 'model.pt')
        if not os.path.exists(mpt):
            logger.warning(
                f"Full-resume from {path}: neither custom_checkpoint_0.pkl nor "
                "model.pt is present; bundle-level orphan tensors cannot be "
                "restored.  This will likely cause Accelerator.load_state to "
                "fail with a custom-object count mismatch."
            )
            return
        try:
            blob = torch.load(mpt, map_location='cpu', weights_only=False)
        except Exception as exc:
            logger.warning(
                f"Full-resume from {path}: failed to read legacy model.pt "
                f"({exc}); orphan tensors may not be restored."
            )
            return
        bundle_params = blob.get('__bundle_params__') if isinstance(blob, dict) else None
        if not isinstance(bundle_params, dict):
            bundle_params = {}
        # Write the synthesised custom checkpoint on the main process so
        # Accelerator.load_state finds a count-matched state directory.
        if self.accelerator.is_main_process:
            from accelerate.checkpointing import save_custom_state
            save_custom_state(
                _LegacyOrphanStub(bundle_params),
                path,
                index=0,
                save_on_each_node=False,
            )
            logger.info(
                f"Migrated legacy ckpt: synthesised custom_checkpoint_0.pkl "
                f"from model.pt::__bundle_params__ at {path} "
                f"(keys: {sorted(bundle_params.keys())})"
            )
        self.accelerator.wait_for_everyone()

    def _patch_zero_null_embeddings_from_pretrained(self):
        """Patch all-zero null embeddings from a pretrained checkpoint.

        Handles two scenarios:

        1. **auto_resume**: resumes from work_dir checkpoint that may have
           zero null embeddings (pre-2026-03-27 bug). Falls back to
           ``load_from.path`` to get correct values.

        2. **load_from**: the loaded checkpoint itself may have zero null
           embeddings (e.g. an unconditioned model never trained with text).
           In this case, ``load_from.null_embedding_source`` can specify a
           separate pretrained checkpoint (typically the T2M pretrained) that
           carries the correct values.  Falls back to ``load_from.path`` if
           no explicit source is given.

        The method detects frozen bundle-level nn.Parameters that are
        all-zero and patches them from the resolved source checkpoint.

        Impact: null embeddings are used during CFG training (cond_mask_prob
        > 0) and inference-time CFG.  Zero null embeddings cause the model
        to receive uninformative null conditioning, breaking text guidance.
        """
        if self.load_from is None:
            return

        # Identify candidate params: bundle-level nn.Parameters that are
        # frozen (requires_grad=False) and currently all-zero.
        zero_params = {}
        for name, param in self.bundle.named_parameters(recurse=False):
            if not param.requires_grad and param.detach().abs().max().item() == 0.0:
                zero_params[name] = param

        if not zero_params:
            return

        # Resolve the pretrained checkpoint path.
        # Priority: null_embedding_source > path (from load_from config).
        load_cfg = self.load_from
        if hasattr(load_cfg, 'to_dict'):
            load_cfg = load_cfg.to_dict()

        pretrained_path = None
        if isinstance(load_cfg, dict):
            # First try explicit null_embedding_source — a separate
            # checkpoint known to have good null embedding values.
            pretrained_path = load_cfg.get('null_embedding_source')
            if not pretrained_path:
                pretrained_path = load_cfg.get('path')
        elif isinstance(load_cfg, str):
            pretrained_path = load_cfg

        if not pretrained_path or not isinstance(pretrained_path, str):
            return

        # Load pretrained state dict and extract matching keys.
        try:
            from hftrainer.utils.checkpoint_utils import load_checkpoint
            source_sd = load_checkpoint(pretrained_path, map_location='cpu')
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            logger.warning(
                f"Cannot patch zero null embeddings: failed to load "
                f"pretrained ckpt at {pretrained_path}: {exc}"
            )
            return

        # Source may be flat (key→tensor) or have __bundle_params__.
        patched = []
        for name, param in zero_params.items():
            src_tensor = None
            # Try direct flat key first (legacy T2M checkpoint format).
            if name in source_sd and isinstance(source_sd[name], torch.Tensor):
                src_tensor = source_sd[name]
            # Try __bundle_params__ dict (newer format).
            elif '__bundle_params__' in source_sd:
                bp = source_sd['__bundle_params__']
                if isinstance(bp, dict) and name in bp and isinstance(bp[name], torch.Tensor):
                    src_tensor = bp[name]

            if src_tensor is not None and src_tensor.shape == param.shape:
                if src_tensor.abs().max().item() > 0:
                    param.data.copy_(src_tensor)
                    patched.append(
                        f"{name}: zeros -> norm={src_tensor.float().norm().item():.4f}"
                    )

        if patched:
            logger.warning(
                f"Patched {len(patched)} all-zero frozen parameter(s) from "
                f"pretrained checkpoint ({pretrained_path}):\n"
                + '\n'.join(f'  {p}' for p in patched)
                + '\nThese were likely zeros due to a historical bug where '
                'auto_resume preempted load_from. Future checkpoints will '
                'save the corrected values.'
            )

    def _state_dict_to_save(self) -> Dict[str, dict]:
        """Build a nested state dict for save_ckpt=True modules.

        Also saves bundle-level nn.Parameters (e.g. null_vtxt_feat,
        null_ctxt_input) that live outside any sub-module.  These are
        stored under the key ``'__bundle_params__'``.
        """
        from hftrainer.models.peft_utils import get_lora_state_dict

        state_dict = {'__hftrainer_meta__': self.bundle.checkpoint_metadata()}
        for name in self.bundle._save_ckpt_modules:
            module = getattr(self.bundle, name, None)
            if not isinstance(module, nn.Module):
                continue
            if name in self.bundle._trainable_modules:
                module_state = self._get_module_state_dict(module)
            else:
                module_state = module.state_dict()

            if self.bundle.get_module_checkpoint_format(name) == 'lora':
                module_state = get_lora_state_dict(module, state_dict=module_state)

            state_dict[name] = module_state

        # Save bundle-level nn.Parameters that are not inside any sub-module.
        # Without this, parameters like null_vtxt_feat / null_ctxt_input are
        # lost across checkpoint save/load cycles.
        bundle_params = {}
        for param_name, param in self.bundle.named_parameters(recurse=False):
            bundle_params[param_name] = param.data.clone()
        for buf_name, buf in self.bundle.named_buffers(recurse=False):
            bundle_params[buf_name] = buf.clone()
        if bundle_params:
            state_dict['__bundle_params__'] = bundle_params

        return state_dict

    def _sync_orphan_param_grads(self):
        """All-reduce gradients for bundle-level Parameters not in any DDP module.

        DDP automatically syncs gradients for parameters inside wrapped modules,
        but bundle-level Parameters (e.g. ``null_vtxt_feat``) live outside any
        DDP-wrapped sub-module.  Without explicit sync, their gradients diverge
        across ranks during multi-GPU training.

        This is a no-op when ``_orphan_trainable_params`` is empty or when
        running on a single device (no distributed backend).
        """
        if not self._orphan_trainable_params:
            return
        if self.accelerator.num_processes <= 1:
            return

        import torch.distributed as dist
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        for param in self._orphan_trainable_params:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)

    def _get_module_state_dict(self, module: nn.Module) -> dict:
        """Return a saveable state dict without forcing unnecessary backend imports."""
        if self.accelerator.distributed_type in {
            DistributedType.FSDP,
            DistributedType.DEEPSPEED,
            DistributedType.MEGATRON_LM,
        }:
            return self.accelerator.get_state_dict(module)

        while hasattr(module, 'module') and isinstance(module.module, nn.Module):
            module = module.module
        return module.state_dict()

    @staticmethod
    def _extract_ckpt_order(path):
        """Extract sort key from checkpoint directory name.

        Supports: checkpoint-iter_N, checkpoint-epoch_N, checkpoint-N (legacy).
        Falls back to meta.pt global_step if directory name cannot be parsed.
        """
        import re
        basename = os.path.basename(path)
        # checkpoint-iter_5000 -> 5000
        m = re.match(r'checkpoint-iter_(\d+)$', basename)
        if m:
            return int(m.group(1))
        # checkpoint-epoch_3 -> 3  (epoch number, not directly comparable to iter)
        m = re.match(r'checkpoint-epoch_(\d+)$', basename)
        if m:
            return int(m.group(1))
        # Legacy: checkpoint-5000 -> 5000
        m = re.match(r'checkpoint-(\d+)$', basename)
        if m:
            return int(m.group(1))
        return -1

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        """Start training."""
        assert self.train_dataloader is not None, "train_dataloader is required for training"

        train_cfg = self.train_cfg
        by_epoch = train_cfg.get('by_epoch', False)
        val_interval = train_cfg.get('val_interval', None)

        rank = self.accelerator.process_index
        logger.info(f"[Rank {rank}] train(): entering before_run hooks")
        for hook in self.hooks:
            if hasattr(hook, 'before_run'):
                logger.info(f"[Rank {rank}] before_run -> {type(hook).__name__}")
                hook.before_run()
                logger.info(f"[Rank {rank}] before_run done -> {type(hook).__name__}")
        logger.info(f"[Rank {rank}] all before_run hooks done; entering main training loop (by_epoch={by_epoch})")

        if by_epoch:
            self._train_by_epoch(train_cfg, val_interval)
        else:
            self._train_by_iter(train_cfg, val_interval)

        # Call after_run hooks (CheckpointHook.after_run saves final checkpoint)
        for hook in self.hooks:
            if hasattr(hook, 'after_run'):
                hook.after_run()

        # End trackers (flushes tensorboard)
        try:
            self.accelerator.end_training()
        except Exception:
            pass

        logger.info("Training complete.")

    def _train_by_iter(self, train_cfg: dict, val_interval):
        """Iteration-based training loop."""
        max_iters = train_cfg.get('max_iters', 10000)

        self.bundle.train()
        self.trainer.train()

        loop = IterBasedLoop(
            self.train_dataloader, max_iters,
            val_interval=val_interval or max_iters,
        )

        for step_idx, batch in loop.iter_batches():
            if step_idx < self.global_step:
                continue  # skip already-trained steps when resuming

            # Before-iter hooks
            for hook in self.hooks:
                if hasattr(hook, 'before_train_iter'):
                    hook.before_train_iter(step_idx)

            # Training step
            try:
                with self.accelerator.accumulate(*[
                    getattr(self.bundle, n) for n in self.bundle._trainable_modules
                    if isinstance(getattr(self.bundle, n), nn.Module)
                ]):
                    output = self.trainer.train_step(batch)

                    if not self.trainer.trainer_controls_optimization:
                        # Runner-controlled optimization (default single-loss path)
                        loss = output.get('loss')
                        if loss is not None:
                            self.accelerator.backward(loss)
                            self._sync_orphan_param_grads()
                            if self.max_grad_norm is not None:
                                params = list(self.bundle.trainable_parameters())
                                self.accelerator.clip_grad_norm_(params, self.max_grad_norm)
                            for opt in self.optimizers.values():
                                opt.step()
                                opt.zero_grad()
                            for sched in self.lr_schedulers.values():
                                sched.step()
                    # else: trainer already did backward/step/zero_grad in train_step
            except Exception:
                rank = self.accelerator.process_index
                logger.error(
                    f"[Rank {rank}] Exception in train_step at global_step={step_idx}:\n"
                    + traceback.format_exc()
                )
                raise

            self.global_step = step_idx + 1

            # After-iter hooks (CheckpointHook handles saving at its interval)
            for hook in self.hooks:
                if hasattr(hook, 'after_train_iter'):
                    hook.after_train_iter(step_idx, output)

            # Validation
            if val_interval and self.global_step % val_interval == 0:
                self.val()
                self.bundle.train()
                self.trainer.train()

    def _train_by_epoch(self, train_cfg: dict, val_interval):
        """Epoch-based training loop."""
        max_epochs = train_cfg.get('max_epochs', 100)
        rank = self.accelerator.process_index
        logger.info(f"[Rank {rank}] _train_by_epoch: starting from current_epoch={self.current_epoch}, max_epochs={max_epochs}")

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            logger.info(f"[Rank {rank}] epoch={epoch}: bundle.train()")
            self.bundle.train()
            self.trainer.train()

            for hook in self.hooks:
                if hasattr(hook, 'before_train_epoch'):
                    logger.info(f"[Rank {rank}] before_train_epoch -> {type(hook).__name__}")
                    hook.before_train_epoch(epoch)
                    logger.info(f"[Rank {rank}] before_train_epoch done -> {type(hook).__name__}")

            logger.info(f"[Rank {rank}] starting dataloader iter for epoch={epoch}")
            for batch_idx, batch in enumerate(self.train_dataloader):
                if batch_idx == 0:
                    logger.info(f"[Rank {rank}] first batch received for epoch={epoch}")
                current_step = self.global_step

                # Before-iter hooks
                for hook in self.hooks:
                    if hasattr(hook, 'before_train_iter'):
                        hook.before_train_iter(current_step)

                try:
                    with self.accelerator.accumulate(*[
                        getattr(self.bundle, n) for n in self.bundle._trainable_modules
                        if isinstance(getattr(self.bundle, n), nn.Module)
                    ]):
                        output = self.trainer.train_step(batch)

                        if not self.trainer.trainer_controls_optimization:
                            # Runner-controlled optimization (default single-loss path)
                            loss = output.get('loss')
                            if loss is not None:
                                self.accelerator.backward(loss)
                                self._sync_orphan_param_grads()
                                if self.max_grad_norm is not None:
                                    params = list(self.bundle.trainable_parameters())
                                    self.accelerator.clip_grad_norm_(params, self.max_grad_norm)
                                for opt in self.optimizers.values():
                                    opt.step()
                                    opt.zero_grad()
                                for sched in self.lr_schedulers.values():
                                    sched.step()
                        # else: trainer already did backward/step/zero_grad in train_step
                except Exception:
                    rank = self.accelerator.process_index
                    logger.error(
                        f"[Rank {rank}] Exception in train_step at epoch={epoch} "
                        f"batch={batch_idx} global_step={current_step}:\n"
                        + traceback.format_exc()
                    )
                    raise

                self.global_step = current_step + 1

                for hook in self.hooks:
                    if hasattr(hook, 'after_train_iter'):
                        hook.after_train_iter(current_step, output)

            self.current_epoch = epoch + 1
            for hook in self.hooks:
                if hasattr(hook, 'after_train_epoch'):
                    hook.after_train_epoch(epoch)

            if val_interval and self.current_epoch % val_interval == 0:
                self.val()
                self.bundle.train()
                self.trainer.train()

    # ─────────────────────────────────────────────────────────────────────────
    # Validation loop
    # ─────────────────────────────────────────────────────────────────────────

    def val(self):
        """Run validation loop."""
        if self.val_dataloader is None:
            return

        self.bundle.eval()
        self.trainer.eval()

        # Reset evaluators
        for ev in self.evaluators:
            ev.reset()

        all_outputs = []
        for batch in self.val_dataloader:
            with torch.no_grad():
                output = self.trainer.val_step(batch)

            # Gather across processes
            output = self._gather_output(output)

            for ev in self.evaluators:
                ev.process(output)
            all_outputs.append(output)

        # Compute metrics
        metrics = {}
        for ev in self.evaluators:
            metrics.update(ev.compute())

        if self.accelerator.is_main_process:
            self.log(metrics)

        # Visualize
        if all_outputs and self.accelerator.is_main_process:
            for vis in self.visualizers:
                vis.visualize(all_outputs[-1], step=self.global_step)

        return metrics

    def _gather_output(self, output: dict) -> dict:
        """Gather output tensors across all processes."""
        import torch
        gathered = {}
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                gathered[k] = self.accelerator.gather_for_metrics(v)
            else:
                gathered[k] = v
        return gathered

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics."""
        step = step or self.global_step
        msg_parts = [f"step={step}"]
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                msg_parts.append(f"{k}={v:.4f}")
            else:
                msg_parts.append(f"{k}={v}")
        if self.accelerator.is_main_process:
            logger.info("  ".join(msg_parts))

        if self.accelerator.is_main_process and hasattr(self.accelerator, 'log'):
            try:
                self.accelerator.log(metrics, step=step)
            except Exception:
                pass
