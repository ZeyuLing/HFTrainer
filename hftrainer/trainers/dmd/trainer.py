"""DMD trainer aligned with the original distribution-matching formulation."""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from hftrainer.registry import TRAINERS
from hftrainer.trainers.base_trainer import BaseTrainer


@TRAINERS.register_module()
class DMDTrainer(BaseTrainer):
    """
    One-step diffusion distillation trainer.

    Bundle expectations:
      - generator_unet: trainable one-step generator
      - fake_score_unet: trainable score model for generated samples
      - real_score_unet: frozen teacher score model
      - vae / text_encoder / scheduler: frozen support modules

    Optimizer expectations:
      - generator: optimizer for ``generator_unet`` (or legacy ``student``)
      - fake_score: optimizer for ``fake_score_unet`` (or legacy ``discriminator``)

    This implementation follows the core structure of the original DMD paper:
      - distribution-matching loss from the difference of real/fake score estimates
      - regression loss on teacher targets
      - separate training of the fake score network
    """

    trainer_controls_optimization = True

    def __init__(
        self,
        bundle,
        dm_weight: float = 1.0,
        regression_weight: float = 1.0,
        fake_score_weight: float = 1.0,
        online_regression_num_inference_steps: int = 20,
        score_start_step: Optional[int] = None,
        score_warmup_steps: Optional[int] = None,
        score_update_interval: Optional[int] = None,
        score_weight_schedule: str = 'linear',
        disc_start_step: Optional[int] = None,
        disc_warmup_steps: Optional[int] = None,
        disc_update_interval: Optional[int] = None,
        val_prompts: Optional[list] = None,
        num_val_samples: int = 4,
        **kwargs,
    ):
        super().__init__(bundle)
        self.dm_weight = float(dm_weight)
        self.regression_weight = float(regression_weight)
        self.fake_score_weight = float(fake_score_weight)
        self.online_regression_num_inference_steps = max(
            0, int(online_regression_num_inference_steps)
        )

        # Keep backward compatibility with the older discriminator-named knobs.
        score_start_step = disc_start_step if score_start_step is None else score_start_step
        score_warmup_steps = (
            disc_warmup_steps if score_warmup_steps is None else score_warmup_steps
        )
        score_update_interval = (
            disc_update_interval
            if score_update_interval is None else score_update_interval
        )

        self.score_start_step = max(0, int(score_start_step or 0))
        self.score_warmup_steps = max(0, int(score_warmup_steps or 0))
        self.score_update_interval = max(1, int(score_update_interval or 1))
        self.score_weight_schedule = score_weight_schedule
        self.val_prompts = val_prompts or ['a photo of a cat', 'a cinematic sunset']
        self.num_val_samples = max(1, int(num_val_samples))

    def _get_named_optimizer(self, preferred: str, legacy: Optional[str] = None):
        if preferred in self.optimizers:
            return self.get_optimizer(preferred)
        if legacy and legacy in self.optimizers:
            return self.get_optimizer(legacy)
        raise KeyError(
            f"Missing optimizer '{preferred}'. Available: {list(self.optimizers.keys())}"
        )

    def _get_named_scheduler(self, preferred: str, legacy: Optional[str] = None):
        if preferred in self.lr_schedulers:
            return self.get_lr_scheduler(preferred)
        if legacy and legacy in self.lr_schedulers:
            return self.get_lr_scheduler(legacy)
        return None

    @staticmethod
    def _set_requires_grad(module, requires_grad: bool):
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        opt_g = self._get_named_optimizer('generator', legacy='student')
        opt_f = self._get_named_optimizer('fake_score', legacy='discriminator')
        sched_g = self._get_named_scheduler('generator', legacy='student')
        sched_f = self._get_named_scheduler('fake_score', legacy='discriminator')

        texts = batch['text']
        device = next(self.bundle.generator_unet.parameters()).device
        batch_size = len(texts)

        score_schedule = self.get_discriminator_factor(
            base_weight=1.0,
            start_step=self.score_start_step,
            warmup_steps=self.score_warmup_steps,
            schedule=self.score_weight_schedule,
        )
        should_update_fake_score = (
            score_schedule > 0
            and self.should_update_discriminator(
                start_step=self.score_start_step,
                update_interval=self.score_update_interval,
            )
        )

        with torch.no_grad():
            text_embeddings = self.bundle.encode_text(texts).to(device)
            uncond_embeddings = self.bundle.get_unconditional_text_embeddings(
                batch_size, device=device
            )

        generator_noise = batch.get('generator_noise')
        regression_noise = batch.get('regression_noise')
        if regression_noise is not None:
            regression_noise = regression_noise.to(device)
        if generator_noise is not None:
            generator_noise = generator_noise.to(device)
        if generator_noise is None:
            generator_noise = regression_noise
        if generator_noise is None:
            generator_noise = self.bundle.sample_latent_noise(batch_size, device=device)

        loss_dm = torch.tensor(0.0, device=device)
        loss_reg = torch.tensor(0.0, device=device)
        loss_fake_score = torch.tensor(0.0, device=device)
        raw_loss_dm = torch.tensor(0.0, device=device)
        raw_loss_reg = torch.tensor(0.0, device=device)
        raw_loss_fake = torch.tensor(0.0, device=device)
        dm_grad_norm = torch.tensor(0.0, device=device)

        # ── Generator update: DM loss + teacher regression loss ──
        opt_g.zero_grad()

        fake_latents = self.bundle.generate_latents(
            generator_noise,
            text_embeddings,
            uncond_embeddings=uncond_embeddings,
        )

        if score_schedule > 0 and self.dm_weight > 0:
            raw_loss_dm, dm_logs = self.bundle.compute_distribution_matching_loss(
                fake_latents,
                text_embeddings,
                uncond_embeddings=uncond_embeddings,
            )
            dm_grad_norm = dm_logs['dm_grad_norm']
            loss_dm = raw_loss_dm * (self.dm_weight * score_schedule)

        if self.regression_weight > 0:
            reg_texts = batch.get('regression_text') or texts
            if reg_texts == texts:
                reg_text_embeddings = text_embeddings
                reg_uncond_embeddings = uncond_embeddings
            else:
                with torch.no_grad():
                    reg_text_embeddings = self.bundle.encode_text(reg_texts).to(device)
                    reg_uncond_embeddings = self.bundle.get_unconditional_text_embeddings(
                        len(reg_texts), device=device
                    )

            reg_noise = regression_noise
            if reg_noise is None:
                reg_noise = generator_noise

            target_latents = batch.get('regression_target_latents')
            if target_latents is not None:
                target_latents = target_latents.to(device)
            elif self.online_regression_num_inference_steps > 0:
                target_latents = self.bundle.sample_teacher_deterministic(
                    reg_noise,
                    reg_text_embeddings,
                    uncond_embeddings=reg_uncond_embeddings,
                    num_inference_steps=self.online_regression_num_inference_steps,
                )

            if target_latents is not None:
                pred_reg_latents = self.bundle.generate_latents(
                    reg_noise,
                    reg_text_embeddings,
                    uncond_embeddings=reg_uncond_embeddings,
                )
                raw_loss_reg = F.mse_loss(
                    pred_reg_latents.float(),
                    target_latents.float(),
                )
                loss_reg = raw_loss_reg * self.regression_weight

        total_generator_loss = loss_dm + loss_reg
        if total_generator_loss.requires_grad:
            self.accelerator.backward(total_generator_loss)
            opt_g.step()
            if sched_g is not None:
                sched_g.step()

        # ── Fake score update ──
        if should_update_fake_score:
            opt_f.zero_grad()
            self._set_requires_grad(self.bundle.generator_unet, False)
            try:
                raw_loss_fake, _fake_logs = self.bundle.compute_fake_score_loss(
                    fake_latents.detach(),
                    text_embeddings,
                )
                loss_fake_score = raw_loss_fake * (
                    self.fake_score_weight * score_schedule
                )
                self.accelerator.backward(loss_fake_score)
                opt_f.step()
                if sched_f is not None:
                    sched_f.step()
            finally:
                self._set_requires_grad(self.bundle.generator_unet, True)

        return {
            'loss': None,
            'loss_dm': loss_dm.detach(),
            'loss_reg': loss_reg.detach(),
            'loss_fake_score': loss_fake_score.detach(),
            'raw_loss_dm': raw_loss_dm.detach(),
            'raw_loss_reg': raw_loss_reg.detach(),
            'raw_loss_fake_score': raw_loss_fake.detach(),
            'score_schedule': float(score_schedule),
            'dm_grad_norm': dm_grad_norm.detach(),
        }

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        prompts = self.val_prompts[:self.num_val_samples]
        device = next(self.bundle.generator_unet.parameters()).device

        with torch.no_grad():
            text_embeddings = self.bundle.encode_text(prompts).to(device)
            uncond_embeddings = self.bundle.get_unconditional_text_embeddings(
                len(prompts), device=device
            )
            noise = self.bundle.sample_latent_noise(len(prompts), device=device)
            latents = self.bundle.generate_latents(
                noise,
                text_embeddings,
                uncond_embeddings=uncond_embeddings,
            )
            images = self.bundle.decode_latent(latents)
            images = (images / 2 + 0.5).clamp(0, 1)

        return {'preds': images.cpu(), 'prompts': prompts}
