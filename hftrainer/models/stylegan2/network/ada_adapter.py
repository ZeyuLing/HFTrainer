"""Checkpoint-compatible StyleGAN2-ADA components for the HFTrainer boundary.

NVIDIA-derived model mathematics and license are in ada/ and LICENSE.nvidia.
"""
import torch
from hftrainer.registry import MODEL_COMPONENTS
from .ada.networks import Generator, Discriminator


@MODEL_COMPONENTS.register_module()
class StyleGAN2ADAGenerator(Generator):
    def __init__(self, z_dim=512, w_dim=512, img_resolution=1024, img_channels=3,
                 c_dim=0, mapping_kwargs=None, synthesis_kwargs=None, style_mixing_prob=0.9):
        super().__init__(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim,
                         img_resolution=img_resolution, img_channels=img_channels,
                         mapping_kwargs=mapping_kwargs or {}, synthesis_kwargs=synthesis_kwargs or {})
        self.latent_dim = z_dim
        self.style_mixing_prob = style_mixing_prob

    def forward(self, z, c=None, truncation_psi=1., truncation_cutoff=None,
                return_latents=False, noise_mode=None, **kwargs):
        if self.c_dim and c is None:
            raise ValueError('This conditional checkpoint requires class labels')
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        if self.training and self.style_mixing_prob and ws.shape[1] > 1:
            if torch.rand((), device=z.device) < self.style_mixing_prob:
                cutoff = int(torch.randint(1, ws.shape[1], (), device=z.device))
                second = self.mapping(torch.randn_like(z), c, skip_w_avg_update=True)
                ws = torch.cat((ws[:, :cutoff], second[:, cutoff:]), dim=1)
        images = self.synthesis(ws, noise_mode=noise_mode or ('random' if self.training else 'const'), **kwargs)
        return (images, ws) if return_latents else images


@MODEL_COMPONENTS.register_module()
class StyleGAN2ADADiscriminator(Discriminator):
    def __init__(self, c_dim=0, img_resolution=1024, img_channels=3, **kwargs):
        super().__init__(c_dim=c_dim, img_resolution=img_resolution, img_channels=img_channels, **kwargs)

    def forward(self, images, c=None, **kwargs):
        if self.c_dim and c is None:
            raise ValueError('This conditional checkpoint requires class labels')
        return super().forward(images, c, **kwargs)
