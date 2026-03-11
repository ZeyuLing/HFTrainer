"""StyleGAN2 bundle."""

import torch

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class StyleGAN2Bundle(ModelBundle):
    """ModelBundle for StyleGAN2-style adversarial image generation."""

    def __init__(self, generator: dict, discriminator: dict):
        super().__init__()
        self._build_modules({
            'generator': generator,
            'discriminator': discriminator,
        })

    def sample(
        self,
        z: torch.Tensor,
        truncation_psi: float = 1.0,
        truncation_cutoff=None,
        return_latents: bool = False,
    ):
        return self.generator(
            z,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            return_latents=return_latents,
        )

    def discriminate(self, images: torch.Tensor) -> torch.Tensor:
        return self.discriminator(images)
