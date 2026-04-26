"""HyMotion-UMO Bundle: UMO-style temporal fusion on frozen MMDiT backbone.

Architecture (UMO = Universal Motion Operator):
  - Frozen backbone: HunyuanMotionMMDiT (0.46B T2M-Lite)
  - Trainable E_ctx: nn.Linear(motion_dim, feat_dim), ~0.2M params
  - Trainable meta_op_embeddings: nn.Embedding(3, motion_dim), ~600 params
  - Meta-operations: {P=Preserve(0), G=Generate(1), E=Edit(2)}

Forward:
  s_tilde = source_motion + meta_op_embedding(tau)   # (B,T,201)
  fused = E_in(x_t) + E_ctx(s_tilde)                 # (B,T,1024)
  output = frozen_backbone(pre_encoded_motion=fused)  # (B,T,201)

Total trainable: ~0.207M (E_ctx: 201*1024+1024, embeddings: 3*201)
"""

from __future__ import annotations

import logging
import os.path as osp
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Convert length list to boolean mask. (B,) -> (B, max_len)."""
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


def _get_module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


# ---------------------------------------------------------------------------
# Meta-operation constants
# ---------------------------------------------------------------------------

META_OP_PRESERVE = 0   # P: frame is known, keep as-is
META_OP_GENERATE = 1   # G: frame needs generation from scratch
META_OP_EDIT = 2       # E: frame needs editing (source + text instruction)

NUM_META_OPS = 3


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@MODEL_BUNDLES.register_module()
class HyMotionUMOBundle(ModelBundle):
    """ModelBundle for UMO-style temporal fusion on frozen HunyuanMotionMMDiT.

    Only ``context_encoder`` and ``meta_op_embeddings`` are trainable.
    The ``motion_transformer`` backbone is fully frozen after weight loading.
    """

    def __init__(
        self,
        motion_transformer: dict,
        # ----- optional text encoder (lazy-loaded) -----
        text_encoder: Optional[dict] = None,
        # ----- mean / std for normalisation -----
        mean_std_dir: Optional[str] = None,
        # ----- model hyperparams -----
        motion_dim: int = 201,
        feat_dim: int = 1024,
        pred_type: str = 'velocity',
        losses_cfg: Optional[dict] = None,
        noise_scheduler_cfg: Optional[dict] = None,
        infer_noise_scheduler_cfg: Optional[dict] = None,
        cond_mask_prob: float = 0.1,
        vtxt_input_dim: int = 768,
        ctxt_input_dim: int = 4096,
        # ----- SMPL body model path (optional) -----
        body_model_path: Optional[str] = None,
    ):
        super().__init__()

        # ---- build backbone via _build_modules (will be frozen later) ----
        self._build_modules({'motion_transformer': motion_transformer})

        # ---- hyper-params ----
        self.motion_dim = motion_dim
        self.feat_dim = feat_dim
        self.pred_type = pred_type
        self.cond_mask_prob = cond_mask_prob
        self._noise_scheduler_cfg = deepcopy(noise_scheduler_cfg or {'method': 'euler'})
        self._infer_noise_scheduler_cfg = deepcopy(
            infer_noise_scheduler_cfg or {'validation_steps': 50}
        )

        # ---- text encoder config (lazy-loaded) ----
        self._text_encoder_cfg = deepcopy(text_encoder) if text_encoder else None

        # ---- UMO-specific trainable modules ----
        # E_ctx: context encoder, same architecture as E_in (input_encoder)
        self.context_encoder = nn.Linear(motion_dim, feat_dim)

        # Meta-operation embeddings: {P=0, G=1, E=2}
        self.meta_op_embeddings = nn.Embedding(NUM_META_OPS, motion_dim)

        # Register UMO modules as trainable + save_ckpt so the runner's
        # optimizer and checkpoint logic picks them up.
        self._trainable_modules.extend(['context_encoder', 'meta_op_embeddings'])
        self._save_ckpt_modules.extend(['context_encoder', 'meta_op_embeddings'])

        # ---- null embeddings for classifier-free guidance ----
        # Frozen: values come from pretrained T2M checkpoint via load_from.
        self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, vtxt_input_dim), requires_grad=False)
        self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, ctxt_input_dim), requires_grad=False)

        # Null source embedding for source-motion CFG dropout (trainable — no pretrained value)
        self.null_source_feat = nn.Parameter(torch.zeros(1, 1, feat_dim))

        # ---- mean / std buffers ----
        self._load_mean_std(mean_std_dir)

        # ---- loss ----
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss
        self.m2m_loss = M2MLoss(**(losses_cfg or {}))

        # ---- SMPL body model (optional) ----
        self._body_model_path = body_model_path
        self._body_model: Optional[nn.Module] = None

        # ---- store dims ----
        self._vtxt_input_dim = vtxt_input_dim
        self._ctxt_input_dim = ctxt_input_dim
        self.validation_steps = self._infer_noise_scheduler_cfg.get('validation_steps', 50)

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
            std = torch.where(std < 1e-3, torch.ones_like(std), std)
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
    # UMO core: init / freeze / fusion
    # ------------------------------------------------------------------

    def init_context_encoder_from_pretrained(self):
        """Copy weights from backbone's input_encoder to context_encoder.

        This initializes E_ctx to be identical to E_in, so that at training
        start, the model behaves as if source = x_t (identity baseline).
        """
        # Handle DDP wrapping: Accelerate may wrap motion_transformer in DDP
        mt = self.motion_transformer
        if hasattr(mt, 'module'):
            mt = mt.module
        input_encoder = mt.input_encoder
        if not isinstance(input_encoder, nn.Linear):
            logger.warning(
                "input_encoder is not nn.Linear, skipping E_ctx init. "
                "Type: %s", type(input_encoder).__name__
            )
            return

        # Handle DDP wrapping for context_encoder as well
        ctx_enc = self.context_encoder
        if hasattr(ctx_enc, 'module'):
            ctx_enc = ctx_enc.module

        with torch.no_grad():
            ctx_enc.weight.copy_(input_encoder.weight)
            ctx_enc.bias.copy_(input_encoder.bias)
        logger.info(
            "Initialized context_encoder from input_encoder weights "
            "(%.3fM params)", sum(p.numel() for p in self.context_encoder.parameters()) / 1e6
        )

    def freeze_backbone(self):
        """Freeze entire backbone, keep only E_ctx + meta_op_embeddings trainable."""
        # Freeze everything in motion_transformer
        for param in self.motion_transformer.parameters():
            param.requires_grad = False

        # Ensure UMO-specific modules are trainable
        for param in self.context_encoder.parameters():
            param.requires_grad = True
        for param in self.meta_op_embeddings.parameters():
            param.requires_grad = True
        # null embeddings for CFG
        self.null_vtxt_feat.requires_grad = True
        self.null_ctxt_input.requires_grad = True
        self.null_source_feat.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Froze backbone: trainable=%.3fM / total=%.3fM (%.2f%%)",
            trainable / 1e6, total / 1e6, 100.0 * trainable / total
        )

    def prepare_umo_input(
        self,
        x_t: Tensor,
        source_motion: Tensor,
        meta_ops: Tensor,
    ) -> Tensor:
        """UMO temporal fusion: E_in(x_t) + E_ctx(source + Emb(τ)).

        Args:
            x_t: noisy motion (B, T, motion_dim), already normalized.
            source_motion: source motion (B, T, motion_dim), already normalized.
                P-frames have clean values, G-frames are zero.
            meta_ops: per-frame meta-operation labels (B, T), long tensor.
                Values: 0=Preserve, 1=Generate, 2=Edit.

        Returns:
            Fused features (B, T, feat_dim) ready for backbone.
        """
        # Add meta-operation embeddings to source motion
        meta_emb = self.meta_op_embeddings(meta_ops)  # (B, T, motion_dim)
        s_tilde = source_motion + meta_emb             # (B, T, motion_dim)

        # Encode through separate encoders
        # Handle DDP wrapping on motion_transformer
        mt = self.motion_transformer
        if hasattr(mt, 'module'):
            mt = mt.module
        motion_feat = mt.input_encoder(x_t.float())  # (B, T, feat_dim)
        context_feat = self.context_encoder(s_tilde.float())               # (B, T, feat_dim)

        # Element-wise add (UMO fusion)
        return motion_feat + context_feat  # (B, T, feat_dim)

    # ------------------------------------------------------------------
    # Forward functions (shared by Trainer and Pipeline)
    # ------------------------------------------------------------------

    def predict_flow(
        self,
        x_t: Tensor,
        source_motion: Tensor,
        meta_ops: Tensor,
        ctxt_input: Tensor,
        vtxt_input: Tensor,
        timesteps: Tensor,
        x_mask_temporal: Optional[Tensor] = None,
        ctxt_mask_temporal: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with UMO temporal fusion.

        Args:
            x_t: noisy motion (B, T, motion_dim).
            source_motion: source motion (B, T, motion_dim).
            meta_ops: (B, T) long, meta-operation labels.
            ctxt_input: text token embeddings (B, Lc, Dc).
            vtxt_input: text vector embeddings (B, 1, Dv).
            timesteps: (B,) diffusion timesteps.
            x_mask_temporal: (B, T) boolean, True=valid.
            ctxt_mask_temporal: (B, Lc) boolean, True=valid.

        Returns:
            Predicted velocity/x1, shape (B, T, motion_dim).
        """
        fused = self.prepare_umo_input(x_t, source_motion, meta_ops)
        return self.motion_transformer(
            x=None,  # not used when pre_encoded_motion is provided
            pre_encoded_motion=fused,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> Dict[str, Tensor]:
        """Lazy-load text encoder and encode text to vtxt/ctxt."""
        device = _get_module_device(self)
        if not hasattr(self, '_text_encoder') or self._text_encoder is None:
            if self._text_encoder_cfg is None:
                raise RuntimeError('No text_encoder config; cannot encode text.')
            from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel
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
            vtxt = torch.where(mask_vtxt, self.null_vtxt_feat.expand_as(vtxt), vtxt)
            mask_ctxt = mask
            while mask_ctxt.ndim < ctxt.ndim:
                mask_ctxt = mask_ctxt.unsqueeze(-1)
            ctxt = torch.where(mask_ctxt, self.null_ctxt_input.expand_as(ctxt), ctxt)
        return vtxt, ctxt

    def normalize_motion(self, motion: Tensor) -> Tensor:
        """Normalize motion using mean/std buffers."""
        std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
        return (motion - self.mean) / std

    def denormalize_motion(self, motion: Tensor) -> Tensor:
        """Denormalize motion."""
        std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
        return motion * std + self.mean

    def decode_motion_from_latent(self, latent: Tensor) -> Dict[str, Tensor]:
        """Denormalize latent and extract components.

        For 201-dim: [transl(3), root_rot6d(6), body_rot6d(126), ric(66)]
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

        # RIC joints (last 66 dims)
        ric_joints = None
        if latent_denorm.shape[-1] >= 201:
            ric_joints = latent_denorm[..., 135:201].reshape(B, L, 22, 3).clone()

        # FK for 3D keypoints
        k3d = None
        if self.body_model is not None:
            try:
                device = latent.device
                betas = torch.zeros(1, 16, device=device)
                k3d_list = []
                for b in range(B):
                    out = self.body_model(
                        body6d[b].to(device), betas,
                        root_rot6d[b].to(device), transl[b].to(device),
                    )
                    k3d_list.append(out)
                k3d = torch.stack(k3d_list, dim=0)
            except Exception:
                k3d = None

        return {
            'latent_denorm': latent_denorm,
            'keypoints3d': k3d,
            'ric_joints': ric_joints,
            'rot6d': rot6d,
            'transl': transl,
            'root_rotations_mat': root_rotmat,
        }
