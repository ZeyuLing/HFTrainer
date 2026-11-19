# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs, gather_object, set_seed
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader

from hftrainer.optim.optimizers import build_optimizers, uses_muon
from hftrainer.runner.accelerate_runner import AccelerateRunner
from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.models.lora import (
    LoRALinear,
    apply_lora,
    get_lora_state_dict,
    set_lora_state_dict,
)
from hftrainer.models.ltx_video.network.text_encoders.gemma import convert_to_additive_mask
from hftrainer.trainers.ltx_video.native import logger
from hftrainer.trainers.ltx_video.native.config import LtxTrainerConfig
from hftrainer.trainers.ltx_video.native.config_display import print_config
from hftrainer.trainers.ltx_video.native.datasets import PrecomputedDataset
from hftrainer.trainers.ltx_video.native.gpu_utils import free_gpu_memory, get_gpu_memory_gb
from hftrainer.models.ltx_video.component_loader import (
    embedding_weight_paths,
    load_embeddings_processor,
    load_transformer,
    read_video_scale_factors,
    resolve_video_vae_path,
)
from hftrainer.trainers.ltx_video.native.progress import TrainingProgress
from hftrainer.trainers.ltx_video.native.optional_dependencies import require_wandb
from hftrainer.trainers.ltx_video.native.quantization import quantize_model
from hftrainer.trainers.ltx_video.native.sigma_tracker import SigmaBucketTracker
from hftrainer.trainers.ltx_video.native.timestep_samplers import SAMPLERS
from hftrainer.trainers.ltx_video.native.training_state import ConfigFingerprint, RngStates, TrainingState
from hftrainer.trainers.ltx_video.native.training_strategies import get_training_strategy
from hftrainer.trainers.ltx_video.native.validation_runner import ValidationRunner

if TYPE_CHECKING:
    from hftrainer.models.ltx_video.network.loader.registry import Registry

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

IS_MAIN_PROCESS = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"

StepCallback = Callable[[int, int, list[Path]], None]  # (step, total, list[sampled_video_path]) -> None

MEMORY_CHECK_INTERVAL = 200


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


@dataclass(frozen=True)
class TrainingStepOutput:
    """Output from a single training step."""

    loss: Tensor  # [B,] per-element loss (unreduced)
    sigma: Tensor  # [B,] sampled sigma, detached from computational graph


class LtxvTrainer:
    """LTX model preparation, loss, validation and checkpoint lifecycle."""

    trainer_controls_optimization = False

    def __init__(
        self,
        trainer_config: LtxTrainerConfig,
        *,
        component_registry: Registry | None = None,
    ) -> None:
        self._config = trainer_config
        legacy_type = trainer_config.optimization.optimizer_type
        legacy_class = {'adamw': 'AdamW', 'adamw8bit': 'bitsandbytes.optim.AdamW8bit'}[legacy_type]
        self._optimizer_config = dict(type=legacy_class, lr=trainer_config.optimization.learning_rate)
        self._optimizer_config.update(trainer_config.optimizer or {})
        self._optimizer_type = {
            'AdamW': 'adamw',
            'torch.optim.AdamW': 'adamw',
            'bitsandbytes.optim.AdamW8bit': 'adamw8bit',
            'torch.optim.Muon': 'Muon',
        }.get(self._optimizer_config['type'], self._optimizer_config['type'])
        if component_registry is None:
            from hftrainer.models.ltx_video.component_loader import LTXComponentStore

            component_registry = LTXComponentStore().training_registry
        self._component_registry = component_registry
        if IS_MAIN_PROCESS:
            print_config(trainer_config)
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        # The VAE isn't loaded during training (latents are precomputed), so source the
        # video compression factors from the checkpoint's embedded VAE config. This keeps
        # the RoPE positions consistent with the VAE that produced the latents (and with
        # inference), instead of assuming the default 32x32x8 layout.
        self._training_strategy.video_scale_factors = read_video_scale_factors(
            resolve_video_vae_path(self._config.model.model_path, self._config.model.video_vae_path)
        )

        # ValidationRunner loads its own models (text encoder, VAE encoder/decoder, etc.),
        # caches prompt embeddings and conditioning media, then unloads encoders.
        self._validation_runner = ValidationRunner(
            config=self._config.validation,
            model_path=self._config.model.model_path,
            text_encoder_path=self._config.model.text_encoder_path,
            video_vae_path=self._config.model.video_vae_path,
            audio_vae_path=self._config.model.audio_vae_path,
            load_text_encoder_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
            component_registry=self._component_registry,
        )

        self._load_models()
        self._setup_accelerator()
        if uses_muon(self._optimizer_config) and self._accelerator.distributed_type in {
            DistributedType.FSDP, DistributedType.DEEPSPEED,
        }:
            raise ValueError('PyTorch Muon requires unsharded parameters. Use single-device training or DDP.')
        self._collect_trainable_params()
        self._loaded_checkpoint_path: Path | None = None
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._global_step = -1
        self._checkpoint_paths: list[Path] = []
        self._training_state_paths: list[Path] = []
        self._training_state_size_warned = False
        self._sigma_tracker = SigmaBucketTracker()
        self._wandb_run = None

    def train(
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """Compatibility entry point for training from an LTX YAML config."""
        return self.build_runner(
            disable_progress_bars=disable_progress_bars, step_callback=step_callback,
        ).train()

    def build_runner(
        self, *, disable_progress_bars=False, step_callback=None, framework_cfg=None, bundle=None,
    ):
        cfg = self._config
        device = self._accelerator.device
        self._start_mem = get_gpu_memory_gb(device)
        self._peak_mem = self._start_mem
        self._train_start_time = time.time()
        self._disable_progress_bars = disable_progress_bars
        self._step_callback = step_callback
        self._sampled_videos_paths = None

        initial_step, training_state = self._resume_state
        set_seed(cfg.seed)
        self._init_optimizer()
        if training_state is not None and not self._restore_training_state(training_state):
            initial_step, training_state = 0, None
        self._init_wandb(resume_run_id=training_state.wandb_run_id if training_state else None)

        self._init_dataloader()
        data_iter = iter(self._dataloader)
        self._init_timestep_sampler()
        self._accelerator.wait_for_everyone()
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()

        self._remaining_steps = cfg.optimization.steps - initial_step
        if self._remaining_steps <= 0:
            raise ValueError(
                f"No remaining training steps: initial_step={initial_step} >= "
                f"target_steps={cfg.optimization.steps}. Nothing to train."
            )
        self._initial_step = initial_step
        self._progress = TrainingProgress(
            enabled=self._accelerator.is_main_process and not disable_progress_bars,
            total_steps=self._remaining_steps,
        )
        hooks = AccelerateRunner._build_hooks(getattr(framework_cfg, 'default_hooks', {}) or {})
        if bundle is None:
            bundle = ModelBundle()
        bundle.attach_trainable_module('transformer', self._transformer)
        runner = AccelerateRunner(
            bundle=bundle,
            trainer=self,
            train_dataloader=self._dataloader,
            val_dataloader=None,
            optimizers=self._optimizers,
            lr_schedulers=self._lr_schedulers,
            accelerator=self._accelerator,
            hooks=[self, *hooks],
            evaluators=[],
            visualizers=[],
            train_cfg={
                'max_iters': cfg.optimization.steps,
                'count_optimizer_steps': True,
                'max_grad_norm': cfg.optimization.max_grad_norm,
            },
            work_dir=cfg.output_dir,
            cfg=framework_cfg,
            checkpoint_saver=self._save_checkpoint,
            validation=self._validate,
            train_context=lambda: self._progress,
            initial_train_iterator=data_iter,
        )
        runner.global_step = self._global_step = initial_step
        return runner

    @property
    def trainable_parameters(self):
        return self._trainable_params

    def train_step(self, batch):
        output = self._training_step(batch)
        return {'loss': output.loss.mean(), 'sample_losses': output.loss.detach(), 'sigma': output.sigma}

    def before_run(self):
        self._transformer.train()
        self._global_step = self.runner.global_step
        cfg = self._config
        if cfg.validation.interval and not cfg.validation.skip_initial_validation:
            self.runner.val()
        self._accelerator.wait_for_everyone()

    def _validate(self):
        with self._offloaded_optimizer_state():
            self._sampled_videos_paths = self._run_validation(self._progress)

    def after_train_microbatch(self, microbatch_idx, output, completed_step, started_at):
        self._global_step = self.runner.global_step
        cfg = self._config
        if completed_step:
            if cfg.validation.interval and self._global_step % cfg.validation.interval == 0:
                self.runner.val()
            if cfg.checkpoints.interval and self._global_step % cfg.checkpoints.interval == 0:
                self.runner.save_checkpoint()

        self._accelerator.wait_for_everyone()
        if self._step_callback and completed_step:
            self._step_callback(self._global_step, cfg.optimization.steps, self._sampled_videos_paths)
        self._accelerator.wait_for_everyone()

        lr = self._optimizers['default'].param_groups[0]['lr']
        step_time = (time.perf_counter() - started_at) * cfg.optimization.gradient_accumulation_steps
        step_loss = output['loss'].detach().item()
        self._progress.update_training(loss=step_loss, lr=lr, step_time=step_time, advance=completed_step)
        if self._accelerator.is_main_process and completed_step:
            self._sigma_tracker.update(output['sigma'].cpu().tolist(), output['sample_losses'].cpu().tolist())
            metrics = {
                'train/loss': step_loss,
                'train/learning_rate': lr,
                'train/step_time': step_time,
                'train/global_step': self._global_step,
            }
            for name, optimizer in self._optimizers.items():
                if name != 'default':
                    metrics[f'train/learning_rate/{name}'] = optimizer.param_groups[0]['lr']
            metrics.update(self._sigma_tracker.get_metrics())
            self._log_metrics(metrics)

        if self._disable_progress_bars and self._accelerator.is_main_process and completed_step:
            if self._global_step % 20 == 0:
                elapsed = time.time() - self._train_start_time
                total = elapsed / (self._global_step - self._initial_step) * self._remaining_steps
                logger.info(
                    f"Step {self._global_step}/{cfg.optimization.steps} - Loss: {step_loss:.4f}, "
                    f"LR: {lr:.2e}, Time/Step: {step_time:.2f}s, Total Time: {total:.0f}s"
                )
        if microbatch_idx % MEMORY_CHECK_INTERVAL == 0:
            self._peak_mem = max(self._peak_mem, get_gpu_memory_gb(self._accelerator.device))

    def after_run(self):
        cfg = self._config
        elapsed = time.time() - self._train_start_time
        rate = self._remaining_steps / elapsed
        num_processes = self._accelerator.num_processes
        global_batch_size = cfg.optimization.batch_size * num_processes
        stats = TrainingStats(
            total_time_seconds=elapsed,
            steps_per_second=rate,
            samples_per_second=rate * global_batch_size,
            peak_gpu_memory_gb=max(self._peak_mem, get_gpu_memory_gb(self._accelerator.device)),
            num_processes=num_processes,
            global_batch_size=global_batch_size,
        )
        saved_path = self.runner.save_checkpoint()
        if self._accelerator.is_main_process:
            self._log_training_stats(stats)
            if cfg.hub.push_to_hub:
                from hftrainer.trainers.ltx_video.native.hf_hub_utils import push_to_hub

                push_to_hub(saved_path, self._sampled_videos_paths, cfg)
            if self._wandb_run is not None:
                self._log_metrics({
                    'stats/total_time_minutes': elapsed / 60,
                    'stats/steps_per_second': rate,
                    'stats/samples_per_second': stats.samples_per_second,
                    'stats/peak_gpu_memory_gb': stats.peak_gpu_memory_gb,
                })
                self._wandb_run.finish()

        self._accelerator.wait_for_everyone()
        self.runner.result = saved_path, stats

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> TrainingStepOutput:
        """Perform a single training step using the configured strategy."""
        # Apply embedding connectors to transform pre-computed text embeddings
        conditions = batch["conditions"]

        if "video_prompt_embeds" in conditions:
            # New format: separate video/audio features from precompute()
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            # Legacy format: single prompt_embeds tensor — duplicate for both modalities
            video_features = conditions["prompt_embeds"]
            audio_features = conditions["prompt_embeds"]

        mask = conditions["prompt_attention_mask"]
        additive_mask = convert_to_additive_mask(mask, video_features.dtype)
        video_embeds, audio_embeds, attention_mask = self._embeddings_processor.create_embeddings(
            video_features, audio_features, additive_mask
        )

        conditions["video_prompt_embeds"] = video_embeds
        conditions["audio_prompt_embeds"] = audio_embeds
        conditions["prompt_attention_mask"] = attention_mask

        # Use strategy to prepare training inputs (returns ModelInputs with Modality objects)
        model_inputs = self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler)

        # Run transformer forward pass with Modality-based interface
        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )

        # Use strategy to compute loss (returns per-element [B,] for sigma-bucket tracking)
        loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)

        # Sigma comes from whichever modality is generated (video preferred, else audio).
        if model_inputs.video is not None and model_inputs.video.enabled:
            sigma = model_inputs.video.sigma.detach()
        else:
            sigma = model_inputs.audio.sigma.detach()

        return TrainingStepOutput(loss=loss, sigma=sigma)

    def _load_models(self) -> None:
        """Load the transformer and embeddings processor for training."""
        logger.debug("Loading transformer...")
        self._transformer = load_transformer(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            registry=self._component_registry,
        )

        # DDP-safe: LOCAL_RANK is set by accelerate before trainer init. Loading on bare
        # "cuda" would resolve to cuda:0 on every rank and crash with a device mismatch.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        init_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        logger.debug("Loading embeddings processor...")
        self._embeddings_processor = load_embeddings_processor(
            checkpoint_path=embedding_weight_paths(
                self._config.model.model_path,
                self._config.model.text_encoder_path,
            ),
            gemma_model_path=self._config.model.text_encoder_path,
            device=init_device,
            dtype=torch.bfloat16,
            registry=self._component_registry,
        )
        self._embeddings_processor.feature_extractor = None

        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        self._transformer = self._transformer.to(dtype=transformer_dtype)

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.info(f'Quantizing model with "{self._config.acceleration.quantization}". This may take a while...')
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )

        self._transformer.requires_grad_(False)

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode."""
        if self._config.model.training_mode == "lora":
            # For LoRA training, first set up LoRA layers
            self._setup_lora()
        elif self._config.model.training_mode == "full":
            # For full training, unfreeze all transformer parameters
            self._transformer.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        logger.debug(f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        logger.debug(f"Adding LoRA adapter with rank {self._config.lora.rank}")
        self._transformer = apply_lora(
            self._transformer,
            {
                'r': self._config.lora.rank,
                'lora_alpha': self._config.lora.alpha,
                'target_modules': self._config.lora.target_modules,
                'lora_dropout': self._config.lora.dropout,
            },
        )

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config, then resolve resume state."""
        if not self._config.model.load_checkpoint:
            self._resume_state: tuple[int, TrainingState | None] = (0, None)
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(f"⚠️ Could not find checkpoint at {self._config.model.load_checkpoint}")
            self._resume_state = (0, None)
            return

        self._loaded_checkpoint_path = checkpoint_path
        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        else:  # LoRA mode
            self._load_lora_checkpoint(checkpoint_path)

        self._resume_state = self._resolve_resume_state()

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        """Load full model checkpoint."""
        state_dict = load_file(checkpoint_path)
        self._transformer.load_state_dict(state_dict, strict=True)

        logger.info("✅ Full model checkpoint loaded successfully")

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint with DDP/FSDP compatibility."""
        state_dict = load_file(checkpoint_path)
        with safe_open(checkpoint_path, framework='pt', device='cpu') as handle:
            metadata = handle.metadata() or {}
        scaling_folded = metadata.get('hftrainer_lora_scaling_folded') == 'true'

        # Adjust layer names to match internal format.
        # (Weights are saved in ComfyUI-compatible format, with "diffusion_model." prefix)
        state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

        # Load LoRA weights and verify all weights were loaded
        set_lora_state_dict(
            self._transformer,
            state_dict,
            scaling_folded=scaling_folded,
        )

        logger.info("✅ LoRA checkpoint loaded successfully")

    def _resolve_resume_state(self) -> tuple[int, TrainingState | None]:
        """Determine resume state by looking for a training state file next to the loaded checkpoint.
        Returns (initial_step, TrainingState or None).
        If no_resume config is set, no checkpoint loaded, or no state file found: returns (0, None).
        """
        if self._config.checkpoints.no_resume or self._loaded_checkpoint_path is None:
            return 0, None

        state = self._load_training_state(self._loaded_checkpoint_path)
        if state is None:
            return 0, None

        fp = state.config_fingerprint
        cfg = self._config
        mismatches: list[str] = []
        if fp.optimizer_type != self._optimizer_type:
            mismatches.append(f"optimizer_type: {fp.optimizer_type} → {self._optimizer_type}")
        if fp.scheduler_type != cfg.optimization.scheduler_type:
            mismatches.append(f"scheduler_type: {fp.scheduler_type} → {cfg.optimization.scheduler_type}")
        if fp.training_mode != cfg.model.training_mode:
            mismatches.append(f"training_mode: {fp.training_mode} → {cfg.model.training_mode}")
        if (
            cfg.model.training_mode == "lora"
            and cfg.lora is not None
            and fp.lora_rank is not None
            and fp.lora_rank != cfg.lora.rank
        ):
            mismatches.append(f"lora_rank: {fp.lora_rank} → {cfg.lora.rank}")
        if mismatches:
            logger.warning(
                f"⚠️ Training state config mismatch ({', '.join(mismatches)}). "
                "Starting from step 0. Set checkpoints.no_resume=true to silence this warning."
            )
            return 0, None

        if state.global_step < 0:
            logger.warning(f"⚠️ Training state has invalid global_step={state.global_step!r}. Starting from step 0.")
            return 0, None
        logger.info(f"📌 Resuming from step {state.global_step}")
        return state.global_step, state

    @staticmethod
    def _load_training_state(checkpoint_path: Path) -> TrainingState | None:
        """Load training state file that corresponds to a checkpoint weights file."""
        match = re.search(r"step_(\d+)", checkpoint_path.name)
        if not match:
            return None

        step_str = match.group(1)
        state_path = checkpoint_path.parent / f"training_state_step_{step_str}.pt"

        if not state_path.exists():
            return None

        try:
            raw: dict = torch.load(state_path, map_location="cpu", weights_only=False)
            state = TrainingState.from_save_dict(raw)
            logger.info(f"📥 Loaded training state from {state_path}")
            return state
        except Exception as e:
            logger.warning(f"⚠️ Failed to load training state from {state_path}: {e}. Starting from step 0.")
            return None

    def _restore_training_state(self, training_state: TrainingState) -> bool:
        """Restore optimizer, scheduler, and RNG states from a loaded TrainingState.
        Must be called after _init_optimizer() (which calls accelerator.prepare).
        Returns True if restore succeeded, False if it failed (caller should fall back to step 0).
        """
        initial_optimizer_state = self._component_states(self._optimizers)
        initial_scheduler_state = self._component_states(self._lr_schedulers)
        try:
            if training_state.optimizer_state_dict is not None:
                self._restore_component_states(self._optimizers, training_state.optimizer_state_dict)
                logger.debug("Restored optimizer state (full mode)")

            if training_state.lr_scheduler_state_dict is not None and self._lr_schedulers:
                self._restore_component_states(self._lr_schedulers, training_state.lr_scheduler_state_dict)
                logger.debug("Restored LR scheduler state")
        except Exception as e:
            self._restore_component_states(self._optimizers, initial_optimizer_state)
            if initial_scheduler_state is not None:
                self._restore_component_states(self._lr_schedulers, initial_scheduler_state)
            logger.warning(f"⚠️ Failed to restore training state: {e}. Starting from step 0.")
            return False

        rng = training_state.rng_states
        if self._accelerator.num_processes > 1:
            logger.debug("Skipping RNG restore in multi-process mode (only main process state was saved)")
        else:
            if rng.torch_state is not None:
                torch.random.set_rng_state(rng.torch_state)
            if rng.cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng.cuda_state)
            logger.debug("Restored RNG states")

        return True

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""

        # For FSDP + LoRA: Cast entire model to FP32.
        # FSDP requires uniform dtype across all parameters in wrapped modules.
        # In LoRA mode, PEFT creates LoRA params in FP32 while base model is BF16.
        # We cast the base model to FP32 to match the LoRA params.
        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.model.training_mode == "lora":
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self._transformer = self._transformer.to(dtype=torch.float32)

        # Enable gradient checkpointing if requested
        # For PeftModel, we need to access the underlying base model
        transformer = (
            self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
        )

        transformer.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)

        # noinspection PyTypeChecker
        self._transformer = self._accelerator.prepare(self._transformer)

        # Log GPU memory usage after model preparation
        vram_usage_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB")

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._config.training_strategy.get_data_sources()

            self._dataset = PrecomputedDataset(self._config.data.preprocessed_data_root, data_sources=data_sources)
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")

        num_workers = self._config.data.num_dataloader_workers
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, LoRALinear):
                torch.nn.init.kaiming_uniform_(module.lora_A.weight, a=5**0.5)
                torch.nn.init.zeros_(module.lora_B.weight)

    @staticmethod
    def _component_states(components):
        if not components:
            return None
        if len(components) == 1:
            return components['default'].state_dict()
        return {name: component.state_dict() for name, component in components.items()}

    @staticmethod
    def _restore_component_states(components, states):
        if len(components) == 1:
            components['default'].load_state_dict(states)
            return
        if states.keys() != components.keys():
            raise ValueError('Saved and configured optimizer or scheduler groups differ.')
        for name, component in components.items():
            component.load_state_dict(states[name])

    def _init_optimizer(self) -> None:
        """Build configured optimizers and retain LTX's scheduler semantics."""
        optimizers = build_optimizers(self._optimizer_config, self._trainable_params, self._transformer)
        schedulers = {}
        self._optimizers = {}
        for name, optimizer in optimizers.items():
            scheduler = self._create_scheduler(optimizer)
            if scheduler is None:
                self._optimizers[name] = self._accelerator.prepare(optimizer)
            else:
                self._optimizers[name], schedulers[name] = self._accelerator.prepare(optimizer, scheduler)
        self._lr_schedulers = schedulers

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = dict(self._config.optimization.scheduler_params or {})

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=params.pop("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", steps // 4),
                T_mult=params.pop("T_mult", 1),
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", steps // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        # find_unused_parameters=True keeps DDP happy when LoRA targets a branch the forward
        # pass skips (e.g. audio LoRA with `with_audio: false`, or short module patterns like
        # "to_k" that match the audio branch unintentionally). It's a no-op for FSDP and
        # single-GPU runs. The probing cost is paid only on the first step.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )

            local_batch = self._config.optimization.batch_size
            global_batch = self._config.optimization.batch_size * self._accelerator.num_processes
            logger.info(f"Local batch size: {local_batch}, global batch size: {global_batch}")

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin") and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}")

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.acceleration.quantization:
            logger.warning(
                f"FSDP with quantization ({self._config.acceleration.quantization}) may have compatibility issues."
                "Monitor training stability and consider disabling quantization if issues arise."
            )

    @contextlib.contextmanager
    def _offloaded_optimizer_state(self) -> Iterator[None]:
        """Context manager that offloads optimizer state to CPU during validation.
        Opt-in via `acceleration.offload_optimizer_during_validation`. Frees VRAM for
        validation video generation when optimizer state is large (e.g. full fine-tune
        AdamW, high-rank LoRA). No-op for FSDP (sharded state -- manual `.cpu()` breaks
        metadata).
        """
        enabled = (
            self._config.acceleration.offload_optimizer_during_validation
            and self._accelerator.distributed_type != DistributedType.FSDP
        )

        # Track exactly which tensors we move so we don't promote ones that were
        # intentionally on CPU (e.g. AdamW's `step` scalar on recent PyTorch).
        offloaded: list[tuple[dict, str]] = []
        if enabled:
            offloaded_bytes = 0
            for optimizer in self._optimizers.values():
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.is_cuda:
                            offloaded.append((state, k))
                            offloaded_bytes += v.nbytes
            if offloaded:
                logger.info(f"Offloading optimizer state to CPU ({offloaded_bytes / 1e9:.1f} GB)")
                for state, k in offloaded:
                    state[k] = state[k].cpu()

        try:
            yield
        finally:
            device = self._accelerator.device
            for state, k in offloaded:
                state[k] = state[k].to(device)

    def _run_validation(self, progress: TrainingProgress) -> list[Path]:
        """Run distributed validation by delegating to the ValidationRunner.
        Each rank generates its assigned subset of validation samples (round-robin by
        `process_index`/`num_processes`), so all GPUs stay busy and no rank idles long
        enough to trigger NCCL timeouts. Paths are gathered across ranks so rank 0 has
        the full list for W&B logging.
        Under FSDP with multiple processes, ranks pad with extra generate passes
        (same sample, no disk write) so every rank runs the same number of forwards --
        avoids collective mismatch.
        Note: Multi-node training requires a shared filesystem so rank 0 can read
        videos written by other ranks.
        """
        for optimizer in self._optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        num_samples = len(self._config.validation.samples)
        if num_samples == 0:
            return []

        rank = self._accelerator.process_index
        world_size = self._accelerator.num_processes

        rank_indices = list(range(rank, num_samples, world_size))
        work_items: list[tuple[int, bool]] = [(i, True) for i in rank_indices]
        if self._accelerator.distributed_type == DistributedType.FSDP and world_size > 1:
            # FSDP forwards run collective ops; pad short ranks with no-save duplicates so
            # every rank executes the same number of forwards. A rank with empty
            # rank_indices (world_size > num_samples) still pads with sample 0 to stay in
            # sync with the others.
            max_per_rank = math.ceil(num_samples / world_size)
            pad_seed = rank_indices[-1] if rank_indices else 0
            work_items += [(pad_seed, False)] * (max_per_rank - len(work_items))

        # W&B logging is handled by the trainer (after gathering across ranks),
        # so we always pass wandb_run=None to the runner.
        sampled = self._validation_runner.run(
            transformer=self._transformer,
            step=self._global_step,
            output_dir=Path(self._config.output_dir),
            device=self._accelerator.device,
            progress=progress,
            wandb_run=None,
            work_items=work_items,
        )

        if world_size > 1:
            sampled = sorted(gather_object(sampled), key=lambda x: x[0])

        paths = [p for _, p in sampled]

        if (
            self._accelerator.is_main_process
            and paths
            and self._config.wandb.log_validation_videos
            and self._wandb_run is not None
        ):
            self._validation_runner.log_to_wandb(self._wandb_run, paths, self._global_step)

        # Non-main ranks must not reach checkpoint collectives while main is still logging to W&B.
        self._accelerator.wait_for_everyone()

        return paths

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _save_checkpoint(self) -> Path | None:
        """Save the model weights."""
        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        # Prepare paths
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        # Get state dict (collective operation - all processes must participate)
        self._accelerator.wait_for_everyone()
        state_dict = self._accelerator.get_state_dict(self._transformer)

        # Only the main process saves the checkpoint
        if not self._accelerator.is_main_process:
            return None

        save_dir.mkdir(exist_ok=True, parents=True)

        # For LoRA: extract only adapter weights; for full: use as-is
        if is_lora:
            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            # For FSDP, pass the gathered state dict since model params aren't directly accessible
            state_dict = get_lora_state_dict(
                unwrapped,
                state_dict=state_dict if is_fsdp else None,
                fold_scaling=True,
            )

            # Convert to ComfyUI-compatible format (add "diffusion_model." prefix)
            state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}

        # Determine save precision
        save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32

        # Cast to configured precision. safetensors rejects non-contiguous tensors, and casting
        # a non-contiguous parameter keeps it non-contiguous.
        state_dict = {k: v.to(save_dtype).contiguous() if isinstance(v, Tensor) else v for k, v in state_dict.items()}

        # Save to disk with metadata
        save_file(state_dict, saved_weights_path, metadata=self._build_checkpoint_metadata())

        rel_path = saved_weights_path.relative_to(self._config.output_dir)
        logger.info(f"💾 {prefix.capitalize()} weights for step {self._global_step} saved in {rel_path}")

        # The final step may be saved both by the interval hook and at loop exit.
        # Retention counts distinct artifacts, otherwise keep_last_n=1 unlinks
        # the just-written file when its path appears twice in the history.
        self._checkpoint_paths = [
            path for path in self._checkpoint_paths if path != saved_weights_path
        ]
        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()

        self._save_training_state(save_dir)

        return saved_weights_path

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

    def _save_training_state(self, save_dir: Path) -> None:
        """Save training state alongside checkpoint for resume.
        Respects checkpoints.save_training_state config:
        - "full": optimizer + scheduler + RNG + step
        - "minimal": scheduler + RNG + step only
        - "off": skip entirely
        """
        if not self._accelerator.is_main_process:
            return

        mode = self._config.checkpoints.save_training_state
        if mode == "off":
            return

        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        optimizer_state = None
        if mode == "full":
            if is_fsdp:
                logger.warning(
                    "⚠️ save_training_state='full' is not supported with FSDP. "
                    "Saving 'minimal' state (scheduler + RNG only)."
                )
            else:
                optimizer_state = self._component_states(self._optimizers)

        state = TrainingState(
            global_step=self._global_step,
            config_fingerprint=ConfigFingerprint(
                optimizer_type=self._optimizer_type,
                scheduler_type=self._config.optimization.scheduler_type,
                training_mode=self._config.model.training_mode,
                lora_rank=self._config.lora.rank if self._config.lora is not None else None,
            ),
            rng_states=RngStates(
                torch_state=torch.random.get_rng_state(),
                cuda_state=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            ),
            lr_scheduler_state_dict=self._component_states(self._lr_schedulers),
            optimizer_state_dict=optimizer_state,
            wandb_run_id=self._wandb_run.id if self._wandb_run is not None else None,
        )

        state_path = save_dir / f"training_state_step_{self._global_step:05d}.pt"
        tmp_path = state_path.with_suffix(".pt.tmp")
        try:
            torch.save(state.to_save_dict(), tmp_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        tmp_path.rename(state_path)

        file_size_gb = state_path.stat().st_size / (1024**3)
        if file_size_gb > 1.0 and not self._training_state_size_warned:
            self._training_state_size_warned = True
            logger.warning(
                f"⚠️ Training state file is {file_size_gb:.1f} GB (full mode includes optimizer state). "
                f'Set checkpoints.save_training_state="minimal" to save only scheduler/RNG/step (~few KB), '
                f'or "off" to disable entirely.'
            )

        if not self._training_state_paths or self._training_state_paths[-1] != state_path:
            self._training_state_paths.append(state_path)
        self._cleanup_training_states()

        rel_path = state_path.relative_to(self._config.output_dir)
        logger.debug(f"Training state saved to {rel_path}")

    def _cleanup_training_states(self) -> None:
        """Clean up old training state files, using the same keep_last_n as checkpoints."""
        keep_n = self._config.checkpoints.keep_last_n
        if 0 < keep_n < len(self._training_state_paths):
            to_remove = self._training_state_paths[:-keep_n]
            for old_state in to_remove:
                if old_state.exists():
                    old_state.unlink()
                    logger.debug(f"Removed old training state: {old_state}")
            self._training_state_paths = self._training_state_paths[-keep_n:]

    def _build_checkpoint_metadata(self) -> dict[str, str]:
        """Build metadata dictionary for safetensors checkpoint.
        Delegates to the training strategy to get strategy-specific metadata
        that downstream inference pipelines may need.
        Returns:
            Dictionary of string key-value pairs for safetensors metadata.
            Values are converted to strings for safetensors compatibility.
        """
        raw_metadata = self._training_strategy.get_checkpoint_metadata()
        # Convert all values to strings for safetensors compatibility
        metadata = {k: str(v) for k, v in raw_metadata.items()}
        if self._config.model.training_mode == 'lora':
            metadata.update({
                'hftrainer_lora_format': 'effective-b-v1',
                'hftrainer_lora_scaling_folded': 'true',
                'hftrainer_lora_rank': str(self._config.lora.rank),
                'hftrainer_lora_alpha': str(self._config.lora.alpha),
            })
        if metadata:
            logger.info(f"Saving checkpoint metadata: {metadata}")
        return metadata

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not self._accelerator.is_main_process:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

        logger.info(f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _init_wandb(self, resume_run_id: str | None = None) -> None:
        """Initialize Weights & Biases run, resuming an existing run if its id is provided."""
        if not self._config.wandb.enabled or not self._accelerator.is_main_process:
            self._wandb_run = None
            return

        wandb = require_wandb()
        wandb_config = self._config.wandb
        init_kwargs: dict[str, Any] = {
            "project": wandb_config.project,
            "entity": wandb_config.entity,
            "name": Path(self._config.output_dir).name,
            "tags": wandb_config.tags,
            "config": self._config.model_dump(),
        }
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"
        run = wandb.init(**init_kwargs)
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Persist metrics locally even when the optional W&B integration is off."""
        if IS_MAIN_PROCESS:
            path = Path(self._config.output_dir) / "metrics.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)
