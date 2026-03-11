"""GAN model modules."""

from hftrainer.models.gan.stylegan2 import StyleGAN2Generator, StyleGAN2Discriminator
from hftrainer.models.gan.stylegan2_bundle import StyleGAN2Bundle

__all__ = ['StyleGAN2Generator', 'StyleGAN2Discriminator', 'StyleGAN2Bundle']
