"""Vendored, self-contained T2M-GPT networks.

Ported verbatim from the official T2M-GPT repository
(``ref_repo/T2M-GPT/models/``) and made fully independent of it:

* ``models.*`` absolute imports rewritten as package-relative imports.
* Hard-coded ``torch.zeros(...).cuda()`` codebook initialisation relaxed so
  the modules can be constructed on CPU (the codebook buffer is overwritten by
  the checkpoint on ``load_state_dict``).
* Training-only / evaluator-only files (``modules.py``, ``evaluator_wrapper``,
  ``smpl``, ``rotation2xyz``) are intentionally not vendored — inference only
  needs the VQ-VAE and the GPT transformer.

Public entry points used by :class:`T2MGPTBundle`:

* :class:`HumanVQVAE` — VQ-VAE (Encoder/Decoder + EMA-reset quantizer).
* :class:`Text2Motion_Transformer` — the cross-conditional GPT.
"""

from .t2m_trans import Text2Motion_Transformer
from .vqvae import HumanVQVAE, VQVAE_251

__all__ = ["HumanVQVAE", "VQVAE_251", "Text2Motion_Transformer"]
