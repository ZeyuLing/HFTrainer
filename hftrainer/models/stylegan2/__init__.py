"""StyleGAN2 network and model bundle."""

from .bundle import StyleGAN2Bundle
from .model import StyleGAN2Discriminator, StyleGAN2Generator

__all__ = [
    'StyleGAN2Bundle',
    'StyleGAN2Discriminator',
    'StyleGAN2Generator',
]
