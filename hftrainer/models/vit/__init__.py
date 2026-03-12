"""ViT model package.

This is the package-level re-export, not the bundle implementation itself.
Concrete model internals can live alongside ``bundle.py`` as the package grows.
"""

from .bundle import ViTBundle

__all__ = ['ViTBundle']
