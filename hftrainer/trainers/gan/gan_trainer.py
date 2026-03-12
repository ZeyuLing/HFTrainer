"""GAN trainer with alternating generator / discriminator updates."""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any

from hftrainer.trainers.base_trainer import BaseTrainer
from hftrainer.registry import TRAINERS


@TRAINERS.register_module()
class GANTrainer(BaseTrainer):
    """
    GAN trainer with separate generator and discriminator optimization.

    Supports classic BCE / hinge losses as well as a StyleGAN2-style
    non-saturating logistic objective with optional R1 and path-length
    regularization.

    Expected bundle modules:
      - generator: the generator network (trainable=True)
      - discriminator: the discriminator network (trainable=True)

    Expected optimizers (config keys):
      - generator: optimizer for generator params
      - discriminator: optimizer for discriminator params

    train_step performs two phases:
      Phase 1 -- Train Discriminator:
        - D(real_data) -> real_score
        - G(noise).detach() -> fake_data (no grad to G)
        - D(fake_data) -> fake_score
        - loss_d = adversarial_loss(real_score, fake_score)
        - accelerator.backward(loss_d) -> discriminator.step()

      Phase 2 -- Train Generator (every d_steps_per_g_step):
        - G(noise) -> fake_data (WITH grad to G)
        - D(fake_data) -> fake_score
        - loss_g = generator_loss(fake_score)
        - accelerator.backward(loss_g) -> generator.step()
    """

    trainer_controls_optimization = True

    def __init__(
        self,
        bundle,
        d_steps_per_g_step: int = 1,
        gan_loss_type: str = 'bce',  # 'bce', 'hinge', or 'stylegan2'
        disc_start_step: int = 0,
        disc_warmup_steps: int = 0,
        disc_update_interval: int = 1,
        disc_weight_schedule: str = 'linear',
        r1_gamma: float = 0.0,
        d_reg_interval: int = 1,
        pl_weight: float = 0.0,
        g_reg_interval: int = 1,
        pl_batch_shrink: int = 2,
        pl_decay: float = 0.01,
        num_val_samples: int = 4,
        val_truncation_psi: float = 0.7,
        **kwargs,
    ):
        super().__init__(bundle)
        self.d_steps_per_g_step = max(1, int(d_steps_per_g_step))
        self.gan_loss_type = gan_loss_type
        self.disc_start_step = max(0, int(disc_start_step))
        self.disc_warmup_steps = max(0, int(disc_warmup_steps))
        self.disc_update_interval = max(1, int(disc_update_interval))
        self.disc_weight_schedule = disc_weight_schedule
        self.r1_gamma = float(r1_gamma)
        self.d_reg_interval = max(1, int(d_reg_interval))
        self.pl_weight = float(pl_weight)
        self.g_reg_interval = max(1, int(g_reg_interval))
        self.pl_batch_shrink = max(1, int(pl_batch_shrink))
        self.pl_decay = float(pl_decay)
        self.num_val_samples = max(1, int(num_val_samples))
        self.val_truncation_psi = float(val_truncation_psi)
        self._pl_mean = None

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        opt_g = self.get_optimizer('generator')
        opt_d = self.get_optimizer('discriminator')
        sched_g = self.get_lr_scheduler('generator')
        sched_d = self.get_lr_scheduler('discriminator')

        real_data = batch['real_data']
        bsz = real_data.shape[0]
        device = real_data.device
        current_step = self.get_current_step()

        disc_factor = self.get_discriminator_factor(
            base_weight=1.0,
            start_step=self.disc_start_step,
            warmup_steps=self.disc_warmup_steps,
            schedule=self.disc_weight_schedule,
        )
        should_update_d = (
            disc_factor > 0
            and self.should_update_discriminator(
                start_step=self.disc_start_step,
                update_interval=self.disc_update_interval,
            )
        )
        should_update_g = (
            disc_factor > 0
            and current_step % self.d_steps_per_g_step == 0
        )

        loss_d = torch.tensor(0.0, device=device)
        loss_d_main = torch.tensor(0.0, device=device)
        loss_d_reg = torch.tensor(0.0, device=device)
        loss_g = torch.tensor(0.0, device=device)
        loss_g_main = torch.tensor(0.0, device=device)
        loss_g_reg = torch.tensor(0.0, device=device)

        # ═══════════════════════════════════════════════════════════════
        # Phase 1: Train Discriminator
        # ═══════════════════════════════════════════════════════════════
        if should_update_d:
            opt_d.zero_grad()

            needs_r1 = self.r1_gamma > 0 and self._should_apply_regularizer(
                self.d_reg_interval
            )
            real_data_for_d = real_data.detach().requires_grad_(needs_r1)
            real_score = self.bundle.discriminator(real_data_for_d)

            with torch.no_grad():
                noise = torch.randn(
                    bsz, self.bundle.generator.latent_dim, device=device
                )
                fake_data = self.bundle.generator(noise)
            fake_score = self.bundle.discriminator(fake_data)

            loss_d_main = self._discriminator_loss(real_score, fake_score)
            loss_d = disc_factor * loss_d_main

            if needs_r1:
                loss_d_reg = self._r1_penalty(real_data_for_d, real_score)
                loss_d = loss_d + disc_factor * loss_d_reg

            self.accelerator.backward(loss_d)
            opt_d.step()
            if sched_d is not None:
                sched_d.step()

        # ═══════════════════════════════════════════════════════════════
        # Phase 2: Train Generator (every d_steps_per_g_step iterations)
        # ═══════════════════════════════════════════════════════════════
        if should_update_g:
            opt_g.zero_grad()
            self._set_requires_grad(self.bundle.discriminator, False)
            try:
                noise = torch.randn(
                    bsz, self.bundle.generator.latent_dim, device=device
                )
                needs_pl = self.pl_weight > 0 and self._should_apply_regularizer(
                    self.g_reg_interval
                )
                fake_data, path_latents = self._forward_generator(
                    noise, return_latents=needs_pl
                )
                fake_score = self.bundle.discriminator(fake_data)

                loss_g_main = disc_factor * self._generator_loss(fake_score)
                loss_g = loss_g_main

                if needs_pl:
                    loss_g_reg = disc_factor * self._path_length_regularizer(
                        fake_data, path_latents
                    )
                    loss_g = loss_g + loss_g_reg

                self.accelerator.backward(loss_g)
                opt_g.step()
                if sched_g is not None:
                    sched_g.step()
            finally:
                self._set_requires_grad(self.bundle.discriminator, True)

        # Return loss=None to signal runner: do NOT backward
        return {
            'loss': None,
            'loss_d': loss_d.detach(),
            'loss_d_main': loss_d_main.detach(),
            'loss_d_reg': loss_d_reg.detach(),
            'loss_g': loss_g.detach(),
            'loss_g_main': loss_g_main.detach(),
            'loss_g_reg': loss_g_reg.detach(),
            'disc_factor': float(disc_factor),
        }

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        real_data = batch['real_data']
        bsz = min(self.num_val_samples, real_data.shape[0])
        device = real_data.device
        noise = torch.randn(
            bsz, self.bundle.generator.latent_dim, device=device
        )
        with torch.no_grad():
            if hasattr(self.bundle, 'sample'):
                preds = self.bundle.sample(
                    noise,
                    truncation_psi=self.val_truncation_psi,
                )
            else:
                preds = self.bundle.generator(noise)
        preds = (preds / 2 + 0.5).clamp(0, 1)
        prompts = [f'sample_{i}' for i in range(preds.shape[0])]
        return {'preds': preds.detach().cpu(), 'prompts': prompts}

    def _discriminator_loss(self, real_score, fake_score):
        """Compute discriminator loss."""
        if self.gan_loss_type == 'bce':
            real_loss = F.binary_cross_entropy_with_logits(
                real_score, torch.ones_like(real_score)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_score, torch.zeros_like(fake_score)
            )
            return (real_loss + fake_loss) / 2
        elif self.gan_loss_type == 'hinge':
            return (F.relu(1 - real_score) + F.relu(1 + fake_score)).mean()
        elif self.gan_loss_type == 'stylegan2':
            return (
                F.softplus(fake_score).mean()
                + F.softplus(-real_score).mean()
            )
        else:
            raise ValueError(f"Unknown gan_loss_type: {self.gan_loss_type}")

    def _generator_loss(self, fake_score):
        """Compute generator loss (fool the discriminator)."""
        if self.gan_loss_type == 'bce':
            return F.binary_cross_entropy_with_logits(
                fake_score, torch.ones_like(fake_score)
            )
        elif self.gan_loss_type == 'hinge':
            return -fake_score.mean()
        elif self.gan_loss_type == 'stylegan2':
            return F.softplus(-fake_score).mean()
        else:
            raise ValueError(f"Unknown gan_loss_type: {self.gan_loss_type}")

    def _should_apply_regularizer(self, interval: int) -> bool:
        interval = max(1, int(interval))
        return (self.get_current_step() - 1) % interval == 0

    @staticmethod
    def _set_requires_grad(module, requires_grad: bool):
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _forward_generator(self, noise: torch.Tensor, return_latents: bool = False):
        if not return_latents:
            return self.bundle.generator(noise), None

        try:
            output = self.bundle.generator(noise, return_latents=True)
        except TypeError as exc:
            raise ValueError(
                "Path length regularization requires the generator to support "
                "generator(noise, return_latents=True) -> (images, latents)."
            ) from exc

        if not (isinstance(output, tuple) and len(output) == 2):
            raise ValueError(
                "Expected generator(noise, return_latents=True) to return "
                "(images, latents)."
            )
        return output

    def _r1_penalty(self, real_data: torch.Tensor, real_score: torch.Tensor):
        grads = torch.autograd.grad(
            outputs=real_score.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        penalty = grads.square().reshape(grads.shape[0], -1).sum(dim=1).mean()
        return penalty * (self.r1_gamma * 0.5 * self.d_reg_interval)

    def _path_length_regularizer(
        self,
        fake_data: torch.Tensor,
        path_latents: torch.Tensor,
    ) -> torch.Tensor:
        if path_latents is None:
            raise ValueError(
                "Path length regularization requires generator latents."
            )

        pl_batch = max(1, fake_data.shape[0] // max(1, self.pl_batch_shrink))
        fake_data = fake_data[:pl_batch]

        noise = torch.randn_like(fake_data) / math.sqrt(fake_data[0].numel())
        grads = torch.autograd.grad(
            outputs=(fake_data * noise).sum(),
            inputs=path_latents,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]
        if grads is None:
            return torch.zeros((), device=fake_data.device, dtype=fake_data.dtype)
        grads = grads[:pl_batch]

        if grads.ndim == 2:
            grads = grads.unsqueeze(1)
        grads = grads.reshape(grads.shape[0], grads.shape[1], -1)
        path_lengths = grads.square().sum(dim=2).mean(dim=1).sqrt()

        mean_length = path_lengths.mean().detach()
        if self._pl_mean is None:
            self._pl_mean = torch.zeros(
                (), device=mean_length.device, dtype=mean_length.dtype
            )
        self._pl_mean = self._pl_mean + self.pl_decay * (
            mean_length - self._pl_mean
        )

        penalty = (path_lengths - self._pl_mean).square().mean()
        return penalty * (self.pl_weight * self.g_reg_interval)
