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
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from accelerate import Accelerator
from mmengine.config import Config

from hftrainer.utils.logger import get_logger, add_file_handler
from hftrainer.utils.checkpoint_utils import find_latest_checkpoint
from hftrainer.utils.env import collect_env_info
from hftrainer.runner.loops import EpochBasedLoop, IterBasedLoop

logger = get_logger()


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

        # Inject accelerator into trainer
        self.trainer.accelerator = accelerator

        # Inject runner into hooks
        for hook in self.hooks:
            hook.runner = self

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

        # When running without `accelerate launch`, ensure distributed env vars
        # are not partially set (which would cause init_process_group to fail)
        if 'RANK' not in os.environ and 'WORLD_SIZE' not in os.environ:
            # Single-process mode: set env vars to avoid distributed init
            os.environ.setdefault('RANK', '0')
            os.environ.setdefault('LOCAL_RANK', '0')
            os.environ.setdefault('WORLD_SIZE', '1')
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '29500')

        accelerator = Accelerator(
            mixed_precision=accel_cfg.get('mixed_precision', 'no'),
            gradient_accumulation_steps=accel_cfg.get('gradient_accumulation_steps', 1),
            log_with=accel_cfg.get('log_with', 'tensorboard'),
            project_dir=run_dir,
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
        trainer = cls._build_trainer(trainer_cfg, bundle)

        # ── Build DataLoaders ──
        train_dl_cfg = getattr(cfg, 'train_dataloader', None)
        val_dl_cfg = getattr(cfg, 'val_dataloader', None)
        train_dataloader = cls._build_dataloader(train_dl_cfg) if train_dl_cfg else None
        val_dataloader = cls._build_dataloader(val_dl_cfg) if val_dl_cfg else None

        # ── Build Optimizers ──
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

        # ── Accelerator prepare ──
        # Only prepare trainable modules, not frozen ones
        trainable_module_list = [
            getattr(bundle, name) for name in bundle._trainable_modules
            if isinstance(getattr(bundle, name), nn.Module)
        ]

        # Move frozen modules to device manually
        for name in bundle._frozen_modules:
            mod = getattr(bundle, name, None)
            if isinstance(mod, nn.Module):
                mod.to(accelerator.device)

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

        # Initialize tensorboard / trackers
        if accelerator.is_main_process:
            try:
                accelerator.init_trackers('training')
            except Exception as e:
                logger.warning(f"Could not init trackers: {e}")

        # Handle checkpoint loading
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
    def _log_model_summary(bundle):
        """Log per-module parameter counts and trainable/save status."""
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

        batch_size = dl_cfg.pop('batch_size', 1)
        num_workers = dl_cfg.pop('num_workers', 0)
        shuffle = dl_cfg.pop('shuffle', True)
        pin_memory = dl_cfg.pop('pin_memory', False)
        drop_last = dl_cfg.pop('drop_last', False)
        collate_fn = dl_cfg.pop('collate_fn', None)
        persistent_workers = dl_cfg.pop('persistent_workers', False)
        sampler = dl_cfg.pop('sampler', None)

        dataset = DATASETS.build(dl_cfg)

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
                # Get params for this named module
                module = getattr(bundle, name, None)
                if module is not None and isinstance(module, nn.Module):
                    params = list(module.parameters())
                else:
                    params = bundle.trainable_parameters()
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
                self._load(latest, load_scope='full')
                return
            else:
                logger.info("auto_resume=True but no checkpoint found. Starting from scratch.")

        if self.load_from is not None:
            load_cfg = self.load_from
            if hasattr(load_cfg, 'to_dict'):
                load_cfg = load_cfg.to_dict()
            path = load_cfg.get('path', load_cfg) if isinstance(load_cfg, dict) else load_cfg
            scope = load_cfg.get('load_scope', 'model') if isinstance(load_cfg, dict) else 'model'
            self._load(path, load_scope=scope)

    def _load(self, path: str, load_scope: str = 'model'):
        """
        Load checkpoint with given scope.

        load_scope='model': load model weights only, reset optimizer/scheduler/meta
        load_scope='full':  load everything via accelerator.load_state
        """
        sep = '=' * 60
        if load_scope == 'full':
            logger.info(sep)
            logger.info(f"Resuming from checkpoint: {path}")
            self.accelerator.load_state(path)
            # Try to restore global_step from metadata
            meta_path = os.path.join(path, 'meta.pt')
            if os.path.exists(meta_path):
                meta = torch.load(meta_path, map_location='cpu')
                self.global_step = meta.get('global_step', 0)
                self.current_epoch = meta.get('current_epoch', 0)
            logger.info(
                f"Resumed: global_step={self.global_step}, epoch={self.current_epoch}. "
                f"Training will continue from step {self.global_step + 1}."
            )
            logger.info(sep)
        elif load_scope == 'model':
            import torch
            from hftrainer.utils.checkpoint_utils import load_checkpoint
            # Load model weights only
            ckpt_path = os.path.join(path, 'model.pt') if os.path.isdir(path) else path
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location='cpu')
                self.bundle.load_state_dict_selective(state_dict)
                logger.info(sep)
                logger.info(f"Loaded model weights from: {ckpt_path}")
                logger.info("Optimizer and training state reset to initial.")
                logger.info(sep)
            else:
                logger.warning(f"model.pt not found at {path}, skipping model load")
        else:
            raise ValueError(f"Unknown load_scope: {load_scope}. Expected 'model' or 'full'.")

    def save_checkpoint(self):
        """Save checkpoint at current global_step."""
        if not self.accelerator.is_main_process:
            return

        ckpt_dir = os.path.join(self.work_dir, f'checkpoint-{self.global_step}')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save model weights (selective)
        import torch
        state_dict = self.bundle.state_dict_to_save()
        torch.save(state_dict, os.path.join(ckpt_dir, 'model.pt'))

        # Save full accelerator state (optimizer, scheduler, etc.)
        self.accelerator.save_state(ckpt_dir)

        # Save meta
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

        import glob, re
        pattern = os.path.join(self.work_dir, 'checkpoint-*')
        ckpts = sorted(
            glob.glob(pattern),
            key=lambda p: int(re.search(r'checkpoint-(\d+)', p).group(1))
            if re.search(r'checkpoint-(\d+)', p) else 0
        )
        while len(ckpts) > max_keep:
            import shutil
            oldest = ckpts.pop(0)
            if os.path.isdir(oldest):
                shutil.rmtree(oldest)
            logger.info(f"Removed old checkpoint: {oldest}")

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        """Start training."""
        assert self.train_dataloader is not None, "train_dataloader is required for training"

        train_cfg = self.train_cfg
        by_epoch = train_cfg.get('by_epoch', False)
        val_interval = train_cfg.get('val_interval', None)
        save_interval = train_cfg.get('save_interval', val_interval)

        # Call before_run hooks
        for hook in self.hooks:
            if hasattr(hook, 'before_run'):
                hook.before_run()

        if by_epoch:
            self._train_by_epoch(train_cfg, val_interval, save_interval)
        else:
            self._train_by_iter(train_cfg, val_interval, save_interval)

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

    def _train_by_iter(self, train_cfg: dict, val_interval, save_interval):
        """Iteration-based training loop."""
        max_iters = train_cfg.get('max_iters', 10000)
        grad_accum = getattr(self.accelerator, 'gradient_accumulation_steps', 1)

        self.bundle.train()
        self.trainer.train()

        loop = IterBasedLoop(
            self.train_dataloader, max_iters,
            val_interval=val_interval or max_iters,
            save_interval=save_interval or val_interval or max_iters,
        )

        for global_step, batch in loop.iter_batches():
            if global_step < self.global_step:
                continue  # skip already-trained steps when resuming

            self.global_step = global_step

            # Before-iter hooks
            for hook in self.hooks:
                if hasattr(hook, 'before_train_iter'):
                    hook.before_train_iter(global_step)

            # Training step
            with self.accelerator.accumulate(*[
                getattr(self.bundle, n) for n in self.bundle._trainable_modules
                if isinstance(getattr(self.bundle, n), nn.Module)
            ]):
                output = self.trainer.train_step(batch)
                loss = output.get('loss')
                if loss is not None:
                    self.accelerator.backward(loss)
                    for opt in self.optimizers.values():
                        opt.step()
                        opt.zero_grad()
                    for sched in self.lr_schedulers.values():
                        sched.step()

            # After-iter hooks (CheckpointHook handles saving at its interval)
            for hook in self.hooks:
                if hasattr(hook, 'after_train_iter'):
                    hook.after_train_iter(global_step, output)

            # Validation
            if val_interval and (global_step + 1) % val_interval == 0:
                self.val()
                self.bundle.train()
                self.trainer.train()

        self.global_step = max_iters

    def _train_by_epoch(self, train_cfg: dict, val_interval, save_interval):
        """Epoch-based training loop."""
        max_epochs = train_cfg.get('max_epochs', 100)

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            self.bundle.train()
            self.trainer.train()

            for hook in self.hooks:
                if hasattr(hook, 'before_train_epoch'):
                    hook.before_train_epoch(epoch)

            for batch_idx, batch in enumerate(self.train_dataloader):
                # Before-iter hooks
                for hook in self.hooks:
                    if hasattr(hook, 'before_train_iter'):
                        hook.before_train_iter(self.global_step)

                with self.accelerator.accumulate(*[
                    getattr(self.bundle, n) for n in self.bundle._trainable_modules
                    if isinstance(getattr(self.bundle, n), nn.Module)
                ]):
                    output = self.trainer.train_step(batch)
                    loss = output.get('loss')
                    if loss is not None:
                        self.accelerator.backward(loss)
                        for opt in self.optimizers.values():
                            opt.step()
                            opt.zero_grad()
                        for sched in self.lr_schedulers.values():
                            sched.step()

                self.global_step += 1

                for hook in self.hooks:
                    if hasattr(hook, 'after_train_iter'):
                        hook.after_train_iter(self.global_step, output)

            for hook in self.hooks:
                if hasattr(hook, 'after_train_epoch'):
                    hook.after_train_epoch(epoch)

            if val_interval and (epoch + 1) % val_interval == 0:
                self.val()
                self.bundle.train()
                self.trainer.train()

            if save_interval and (epoch + 1) % save_interval == 0:
                self.save_checkpoint()

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
