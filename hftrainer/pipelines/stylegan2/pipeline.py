"""StyleGAN2 inference pipeline."""

import torch

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


@PIPELINES.register_module()
class StyleGAN2Pipeline(BasePipeline):
    """Sample images from a StyleGAN2 bundle."""

    def __init__(self, bundle, truncation_psi: float = 0.7):
        super().__init__(bundle)
        self.truncation_psi = truncation_psi

    def __call__(self, num_samples: int = 1, z: torch.Tensor = None):
        device = next(self.bundle.generator.parameters()).device
        if z is None:
            z = torch.randn(
                num_samples,
                self.bundle.generator.latent_dim,
                device=device,
            )
        with torch.no_grad():
            images = self.bundle.sample(z, truncation_psi=self.truncation_psi)
        return (images / 2 + 0.5).clamp(0, 1)
