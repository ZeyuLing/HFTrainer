"""HyMotion-T2M Bundle: text-to-motion generation via flow matching.

This bundle holds a HunyuanMotionMMDiT transformer and provides atomic
forward functions shared between Trainer and Pipeline:

  - predict_flow()          -- single forward through the transformer
  - decode_motion_from_latent() -- denormalize + FK to 3D keypoints
  - mask_text_cond()        -- classifier-free guidance null masking
  - encode_text()           -- lazy-load text encoder and encode text

Unlike HyMotion-M2M, this bundle does NOT use VACE conditioning.
The input to the transformer is just x_t (motion_dim), not
[x_t, vace_context] (motion_dim * 4).
"""

from __future__ import annotations

import os.path as osp
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Convert length list to boolean mask. (B,) -> (B, max_len)."""
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


def _get_module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@MODEL_BUNDLES.register_module()
class HyMotionT2MBundle(ModelBundle):
    """ModelBundle for HyMotion-T2M text-to-motion generation.

    The only sub-module managed via ``_build_modules`` is
    ``motion_transformer``  (the HunyuanMotionMMDiT).  Auxiliary objects
    (M2MLoss, null embeddings, mean/std buffers) are created directly in
    ``__init__`` as regular attributes.

    Key difference from HyMotion-M2M: NO VACE conditioning.
    input_dim = motion_dim (not motion_dim * 4).
    """

    def __init__(
        self,
        motion_transformer: dict,
        # ----- optional text encoder (lazy-loaded at encode_text time) -----
        text_encoder: Optional[dict] = None,
        # ----- mean / std for normalisation -----
        mean_std_dir: Optional[str] = None,
        # ----- model hyperparams -----
        motion_type: str = 'smpl_22',
        pred_type: str = 'velocity',
        uncondition_mode: bool = False,
        losses_cfg: Optional[dict] = None,
        noise_scheduler_cfg: Optional[dict] = None,
        infer_noise_scheduler_cfg: Optional[dict] = None,
        cond_mask_prob: float = 0.1,
        vtxt_input_dim: int = 768,
        ctxt_input_dim: int = 4096,
        # ----- SMPL body model path (optional; skipped if None) -----
        body_model_path: Optional[str] = None,
    ):
        super().__init__()

        # ---- build trainable module via _build_modules ----
        self._build_modules({'motion_transformer': motion_transformer})

        # ---- hyper-params ----
        self.motion_type = motion_type
        self.pred_type = pred_type
        self.uncondition_mode = uncondition_mode
        self.cond_mask_prob = cond_mask_prob
        self._noise_scheduler_cfg = deepcopy(noise_scheduler_cfg or {'method': 'euler'})
        self._infer_noise_scheduler_cfg = deepcopy(
            infer_noise_scheduler_cfg or {'validation_steps': 50}
        )

        # ---- text encoder config (lazy-loaded) ----
        self._text_encoder_cfg = deepcopy(text_encoder) if text_encoder else None

        # ---- null embeddings for classifier-free guidance ----
        # Zero default; actual values loaded from pretrained checkpoint.
        self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, vtxt_input_dim))
        self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, ctxt_input_dim))

        # ---- mean / std buffers ----
        self._load_mean_std(mean_std_dir)

        # ---- M2M loss (reused for T2M: same velocity / x1 loss) ----
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss
        self.m2m_loss = M2MLoss(**(losses_cfg or {}))

        # ---- SMPL body model (optional for FK losses / decode) ----
        self._body_model_path = body_model_path
        self._body_model: Optional[nn.Module] = None  # lazy

        # ---- store vtxt/ctxt dims for later ----
        self._vtxt_input_dim = vtxt_input_dim
        self._ctxt_input_dim = ctxt_input_dim

        # ---- infer params ----
        self.validation_steps = self._infer_noise_scheduler_cfg.get(
            'validation_steps', 50
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_mean_std(self, mean_std_dir: Optional[str]) -> None:
        if mean_std_dir is not None and osp.isdir(mean_std_dir):
            mean = torch.from_numpy(
                np.load(osp.join(mean_std_dir, 'Mean.npy'))
            ).float()
            std = torch.from_numpy(
                np.load(osp.join(mean_std_dir, 'Std.npy'))
            ).float()
            # Zero-out near-zero std dims (matching official HY-Motion-1.0)
            # These dims are effectively constant and should produce zero after normalization
            std = torch.where(std < 1e-3, torch.zeros_like(std), std)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        else:
            self.register_buffer('mean', torch.zeros(1))
            self.register_buffer('std', torch.ones(1))

    @property
    def body_model(self):
        """Lazy-load SmplxLiteJ24 body model."""
        if self._body_model is None:
            from hftrainer.models.motion.hymotion_m2m.network.smpl_lite import SmplxLiteJ24
            kwargs = {}
            if self._body_model_path is not None:
                kwargs['model_path'] = self._body_model_path
            try:
                self._body_model = SmplxLiteJ24(**kwargs)
                self._body_model.to(_get_module_device(self))
                self._body_model.eval()
            except Exception:
                return None
        return self._body_model

    # ------------------------------------------------------------------
    # Atomic forward functions (shared by Trainer and Pipeline)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> Dict[str, Tensor]:
        """Lazy-load text encoder and encode text to vtxt/ctxt.

        Returns dict with keys: text_vec_raw, text_ctxt_raw, text_ctxt_raw_length.
        """
        device = _get_module_device(self)
        if not hasattr(self, '_text_encoder') or self._text_encoder is None:
            if self._text_encoder_cfg is None:
                raise RuntimeError(
                    'No text_encoder config provided; cannot encode text.'
                )
            from hftrainer.models.motion.hymotion_m2m.network.text_encoder import (
                HYTextModel,
            )
            cfg = deepcopy(self._text_encoder_cfg)
            cfg.pop('type', None)
            self._text_encoder = HYTextModel(**cfg)
        vtxt, ctxt, ctxt_len = self._text_encoder.encode(text)
        return {
            'text_vec_raw': vtxt.to(device),
            'text_ctxt_raw': ctxt.to(device),
            'text_ctxt_raw_length': ctxt_len.to(device),
        }

    def mask_text_cond(
        self,
        vtxt: Tensor,
        ctxt: Tensor,
        force_mask: bool = False,
        cond_mask_prob: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """Apply classifier-free guidance masking to text conditions."""
        bs = vtxt.shape[0]
        if force_mask:
            return (
                self.null_vtxt_feat.expand(*vtxt.shape),
                self.null_ctxt_input.expand(*ctxt.shape),
            )
        if self.training and cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=vtxt.device) * cond_mask_prob
            ).view(bs, 1).bool()
            mask_vtxt = mask
            while mask_vtxt.ndim < vtxt.ndim:
                mask_vtxt = mask_vtxt.unsqueeze(-1)
            vtxt = torch.where(
                mask_vtxt, self.null_vtxt_feat.expand_as(vtxt), vtxt
            )
            mask_ctxt = mask
            while mask_ctxt.ndim < ctxt.ndim:
                mask_ctxt = mask_ctxt.unsqueeze(-1)
            ctxt = torch.where(
                mask_ctxt, self.null_ctxt_input.expand_as(ctxt), ctxt
            )
        return vtxt, ctxt

    def predict_flow(
        self,
        x_input: Tensor,
        ctxt_input: Tensor,
        vtxt_input: Tensor,
        timesteps: Tensor,
        x_mask_temporal: Optional[Tensor] = None,
        ctxt_mask_temporal: Optional[Tensor] = None,
    ) -> Tensor:
        """Single forward pass through the MMDiT transformer.

        Args:
            x_input: noisy motion x_t, shape (B, L, motion_dim).
                     Unlike M2M, this is NOT concatenated with VACE context.
            ctxt_input: token-level text embeddings, (B, Lc, Dc).
            vtxt_input: sentence-level text embeddings, (B, 1, Dv).
            timesteps: diffusion timesteps, (B,).
            x_mask_temporal: (B, L) boolean mask for motion sequence.
            ctxt_mask_temporal: (B, Lc) boolean mask for text tokens.

        Returns:
            Model prediction, shape (B, L, motion_dim).
        """
        return self.motion_transformer(
            x=x_input,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )

    def decode_motion_from_latent(
        self,
        latent: Tensor,
    ) -> Dict[str, Tensor]:
        """Denormalize latent and run FK to get 3D keypoints.

        Returns dict with keys: keypoints3d, rot6d, transl, latent_denorm.
        """
        from hftrainer.models.motion.hymotion_m2m.network.geometry import rot6d_to_rotation_matrix

        std = torch.where(self.std < 1e-3, torch.zeros_like(self.std), self.std)
        latent_denorm = latent * std + self.mean

        B, L = latent_denorm.shape[:2]
        transl = latent_denorm[..., 0:3].clone()
        root_rot6d = latent_denorm[..., 3:9].reshape(B, L, 1, 6).clone()
        body6d = latent_denorm[..., 9:135].reshape(B, L, 21, 6).clone()
        rot6d = torch.cat([root_rot6d, body6d], dim=2)
        root_rotmat = rot6d_to_rotation_matrix(rot6d[:, :, 0, :])

        k3d = None
        if self.body_model is not None:
            try:
                device = latent.device
                betas = torch.zeros(1, 16, device=device)
                k3d_list = []
                for b in range(B):
                    out = self.body_model(
                        body6d[b].to(device),
                        betas,
                        root_rot6d[b].to(device),
                        transl[b].to(device),
                    )
                    k3d_list.append(out)
                k3d = torch.stack(k3d_list, dim=0)
            except Exception:
                k3d = None

        # Ground alignment: offset translation so lowest joint touches Y=0.
        # Matches official HY-Motion-1.0 post-FK processing.
        if k3d is not None:
            # min Y across all joints and frames per batch sample
            min_y = k3d[:, :, :, 1].min(dim=2)[0].min(dim=1)[0]  # (B,)
            transl[:, :, 1] = transl[:, :, 1] - min_y.unsqueeze(1)
            k3d[:, :, :, 1] = k3d[:, :, :, 1] - min_y.unsqueeze(1).unsqueeze(1)

        return {
            'latent_denorm': latent_denorm,
            'keypoints3d': k3d,
            'rot6d': rot6d,
            'transl': transl,
            'root_rotations_mat': root_rotmat,
        }

    def normalize_motion(self, motion: Tensor) -> Tensor:
        """Normalize motion using mean/std buffers.

        Dims with near-zero std (constant in training data) produce 0 after normalization.
        This matches official HY-Motion-1.0 behavior.
        """
        # Safe division: where std==0, output 0 (those dims are constant)
        safe_std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
        result = (motion - self.mean) / safe_std
        # Zero out dims where std was near-zero
        result = torch.where(self.std.unsqueeze(0) < 1e-3, torch.zeros_like(result), result)
        return result

    def denormalize_motion(self, motion: Tensor) -> Tensor:
        """Denormalize motion (matching official HY-Motion-1.0: zeros for near-zero std)."""
        std = torch.where(self.std < 1e-3, torch.zeros_like(self.std), self.std)
        return motion * std + self.mean
