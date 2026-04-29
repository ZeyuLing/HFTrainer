"""HyMotion-M2M Bundle: motion-to-motion editing via flow matching.

This bundle holds a HunyuanMotionMMDiT transformer and provides atomic
forward functions shared between Trainer and Pipeline:

  - prepare_padding()       -- align src/tgt motions + build masks
  - prepare_vace_input()    -- build VACE conditioning context
  - predict_flow()          -- single forward through the transformer
  - decode_motion_from_latent() -- denormalize + FK to 3D keypoints
  - mask_text_cond()        -- classifier-free guidance null masking
"""

from __future__ import annotations

import os
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
class HyMotionM2MBundle(ModelBundle):
    """ModelBundle for HYMotion-M2M motion-to-motion editing.

    The only sub-module managed via ``_build_modules`` is
    ``motion_transformer``  (the HunyuanMotionMMDiT).  Auxiliary objects
    (M2MLoss, SmplxLiteJ24, null embeddings, mean/std buffers) are created
    directly in ``__init__`` as regular attributes.
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
        uncondition_mode: bool = True,
        losses_cfg: Optional[dict] = None,
        noise_scheduler_cfg: Optional[dict] = None,
        infer_noise_scheduler_cfg: Optional[dict] = None,
        cond_mask_prob: float = 1.0,
        vace_condition_mode: str = 'split_reactive',
        vtxt_input_dim: int = 768,
        ctxt_input_dim: int = 4096,
        # ----- SMPL body model path (optional; skipped if None) -----
        body_model_path: Optional[str] = None,
        # ----- rotation space -----
        rotation_space: str = 'local',
        # ----- KIMODO-style auxiliary losses (j_p / j_v / fk_consistency) -----
        kimodo_aux_loss_cfg: Optional[dict] = None,
    ):
        super().__init__()

        # ---- build trainable module via _build_modules ----
        self._build_modules({'motion_transformer': motion_transformer})

        # ---- hyper-params ----
        self.motion_type = motion_type
        self.pred_type = pred_type
        self.uncondition_mode = uncondition_mode
        self.cond_mask_prob = cond_mask_prob
        self.vace_condition_mode = str(vace_condition_mode or 'split_reactive').strip()
        self.rotation_space = rotation_space
        assert rotation_space in ('local', 'global'), (
            f"rotation_space must be 'local' or 'global', got {rotation_space!r}"
        )
        self._noise_scheduler_cfg = deepcopy(noise_scheduler_cfg or {'method': 'euler'})
        self._infer_noise_scheduler_cfg = deepcopy(
            infer_noise_scheduler_cfg or {'validation_steps': 50}
        )

        # ---- text encoder config (lazy-loaded) ----
        self._text_encoder_cfg = deepcopy(text_encoder) if text_encoder else None

        # ---- null embeddings for classifier-free guidance ----
        # Frozen: values come from pretrained T2M checkpoint via load_from.
        # These represent the learned "no text condition" embedding and should
        # NOT be updated during M2M training — the transformer adapts to them.
        # Freezing also avoids the need to save/sync them across checkpoints.
        self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, vtxt_input_dim), requires_grad=False)
        self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, ctxt_input_dim), requires_grad=False)

        # ---- mean / std buffers ----
        self._load_mean_std(mean_std_dir)

        # ---- M2M loss ----
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss
        self.m2m_loss = M2MLoss(**(losses_cfg or {}))

        # ---- KIMODO-style auxiliary loss (optional, post-hoc) ----
        # Computed by the trainer in addition to M2MLoss; constructed here so
        # weights/config travel with the bundle and survive checkpoint reload.
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )
        self.kimodo_aux_loss = KimodoStyleAuxLoss(**(kimodo_aux_loss_cfg or {}))

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
            # Clamp std to avoid div-by-zero
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
            # Keep text encoder on CPU — it is inference-only and not part of
            # the trainable graph.  Moving an 8B LLM to each rank's GPU would
            # exhaust memory.  encode() uses get_module_device(self) internally
            # so inputs/outputs stay on CPU; we move the returned tensors to
            # the training device below.
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

    def prepare_padding(
        self,
        src_motion: Tensor,
        tgt_motion: Optional[Tensor],
        tgt_length: List[int],
        src_mask: Optional[Tensor] = None,
        src_length: Optional[List[int]] = None,
        ref_pose: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[int], List[int], Tensor]:
        """Pad src/tgt motions to the same length and build tgt_padding_mask.

        Returns:
            (src_motion, src_mask, tgt_motion, src_length, tgt_length,
             tgt_padding_mask)
        """
        device = src_motion.device
        B, L_s, D = src_motion.shape
        L_t = tgt_motion.shape[1] if tgt_motion is not None else L_s
        L_r = ref_pose.shape[1] if ref_pose is not None else 0

        if src_length is None:
            src_length = tgt_length

        max_len = max(L_s, L_t)
        if src_mask is None:
            src_mask = torch.ones_like(src_motion)

        # Pad src
        if L_s < max_len:
            pad = max_len - L_s
            src_motion = F.pad(src_motion, (0, 0, 0, pad))
            src_mask = F.pad(src_mask, (0, 0, 0, pad))

        # Pad tgt
        if tgt_motion is not None and L_t < max_len:
            pad = max_len - L_t
            tgt_motion = F.pad(tgt_motion, (0, 0, 0, pad))
        elif tgt_motion is None:
            tgt_motion = torch.zeros(B, max_len, D, dtype=src_motion.dtype, device=device)

        # Build tgt_padding_mask
        if L_r > 0:
            ref_mask = torch.ones(B, L_r, dtype=torch.bool, device=device)
        else:
            ref_mask = torch.empty(B, 0, dtype=torch.bool, device=device)

        tgt_mask = _length_to_mask(
            torch.tensor(tgt_length, dtype=torch.long, device=device), max_len
        )
        tgt_padding_mask = torch.cat([ref_mask, tgt_mask], dim=1)

        return src_motion, src_mask, tgt_motion, src_length, tgt_length, tgt_padding_mask

    def prepare_vace_input(
        self,
        src_motion: Tensor,
        ref_pose: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Build VACE conditioning context.

        Returns tensor of shape (B, L, 3*D) where D is the motion dim.
        """
        B, L_src, D = src_motion.shape
        if src_mask is None:
            src_mask = torch.ones_like(src_motion)

        inactive = src_motion * (1 - src_mask)
        if self.vace_condition_mode == 'split_reactive':
            reactive = src_motion * src_mask
        elif self.vace_condition_mode == 'clean_zero_mask':
            reactive = torch.zeros_like(src_motion)
        elif self.vace_condition_mode == 'no_inactive':
            # v2 slim VACE — drops the `inactive` channel. Rationale:
            # under mask-aware noise (MAN), `x_t[known] = clean_motion` already
            # carries known-region values into the model, so `inactive` becomes
            # redundant. VACE then only needs to signal: (a) what the pre-edit
            # value was in mask=1 regions (`reactive`, 0 in completion, LQ in
            # editing), and (b) where the mask is. Total vace_context = 2*D.
            # Model input = x_t + reactive + mask = 3*D.
            reactive = src_motion * src_mask
            vace_context = reactive  # (B, L, D) — zero in completion, LQ in editing
            if ref_pose is not None:
                _, L_ref, _ = ref_pose.shape
                src_mask = torch.cat(
                    [torch.zeros(B, L_ref, D, dtype=src_mask.dtype, device=src_mask.device), src_mask],
                    dim=1,
                )
                vace_context = torch.cat([ref_pose, vace_context], dim=1)
            vace_context = torch.cat([vace_context, src_mask], dim=-1)  # (B, L, 2*D)
            return vace_context
        else:
            raise ValueError(f'Unsupported vace_condition_mode: {self.vace_condition_mode}')

        vace_context = torch.cat([inactive, reactive], dim=-1)  # (B, L, 2*D)

        if ref_pose is not None:
            _, L_ref, _ = ref_pose.shape
            ref_pose = torch.cat([ref_pose, torch.zeros_like(ref_pose)], dim=1)
            src_mask = torch.cat(
                [torch.zeros(B, L_ref, D, dtype=src_mask.dtype, device=src_mask.device), src_mask],
                dim=1,
            )
            vace_context = torch.cat([ref_pose, vace_context], dim=1)

        vace_context = torch.cat([vace_context, src_mask], dim=-1)  # (B, L, 3*D)
        return vace_context

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
            x_input: concatenated [x_t, vace_context], shape (B, L, D + 3*D_motion).
            ctxt_input: token-level text embeddings, (B, Lc, Dc).
            vtxt_input: sentence-level text embeddings, (B, 1, Dv).
            timesteps: diffusion timesteps, (B,).
            x_mask_temporal: (B, L) boolean mask for motion sequence.
            ctxt_mask_temporal: (B, Lc) boolean mask for text tokens.

        Returns:
            Model prediction, shape (B, L, D_motion).
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

        When ``rotation_space == 'global'``, the denormalized rot6d is in
        world-frame global rotation.  We convert it back to local (SMPL)
        rotation before FK so that the output NPZ is always SMPL-compatible.
        """
        from hftrainer.models.motion.hymotion_m2m.network.geometry import rot6d_to_rotation_matrix

        std = torch.where(self.std < 1e-3, torch.zeros_like(self.std), self.std)
        latent_denorm = latent * std + self.mean

        B, L = latent_denorm.shape[:2]
        transl = latent_denorm[..., 0:3].clone()

        # Extract rot6d: (B, L, 22, 6)
        rot6d_all = latent_denorm[..., 3:135].reshape(B, L, 22, 6).clone()

        # If trained in global rotation space, convert back to local for SMPL output
        if self.rotation_space == 'global':
            from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
                global_to_local_rot6d_torch,
            )
            rot6d_all = global_to_local_rot6d_torch(rot6d_all)

        root_rot6d = rot6d_all[:, :, 0:1, :]   # (B, L, 1, 6)
        body6d = rot6d_all[:, :, 1:, :]         # (B, L, 21, 6)
        rot6d = rot6d_all
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

        return {
            'latent_denorm': latent_denorm,
            'keypoints3d': k3d,
            'rot6d': rot6d,
            'transl': transl,
            'root_rotations_mat': root_rotmat,
        }

    def normalize_motion(self, motion: Tensor) -> Tensor:
        """Normalize motion using mean/std buffers."""
        return (motion - self.mean) / self.std

    def denormalize_motion(self, motion: Tensor) -> Tensor:
        """Denormalize motion."""
        std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
        return motion * std + self.mean

    def get_bone_offsets(self) -> Tensor:
        """Get bone offsets for FK/IK.

        Attempts to compute from body model first; falls back to pre-computed
        file at ``data/hymotion_m2m_data/bone_offsets_22.pt``.

        Returns:
            bone_offsets: (22, 3) tensor of bone offsets.
        """
        from hftrainer.datasets.motion.motionhub.transforms.fk_utils import SMPL22_PARENTS

        # Try computing from body model
        if self.body_model is not None:
            try:
                J_template = self.body_model.J_template[:22].clone()
                offsets = torch.zeros(22, 3, device=J_template.device, dtype=J_template.dtype)
                offsets[0] = J_template[0]
                for j in range(1, 22):
                    parent = SMPL22_PARENTS[j]
                    offsets[j] = J_template[j] - J_template[parent]
                return offsets
            except Exception:
                pass

        # Fallback: load pre-computed file
        fallback_path = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__))))),
            'data', 'hymotion_m2m_data', 'bone_offsets_22.pt',
        )
        if osp.isfile(fallback_path):
            offsets = torch.load(fallback_path, map_location='cpu')
            return offsets.to(_get_module_device(self))

        raise RuntimeError(
            'Cannot compute bone offsets: body model unavailable and '
            f'fallback file not found at {fallback_path}. '
            'Run `python tools/precompute_bone_offsets.py` first.'
        )
