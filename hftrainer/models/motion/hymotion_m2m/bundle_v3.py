"""HyMotion-M2M v3 Bundle: DSCF (Dual-Stream Condition Fusion) architecture.

This bundle holds a HunyuanMotionMMDiTv3 transformer that uses cross-attention
(instead of VACE input-concat) for motion condition injection. Text and motion
conditions are on equal footing — both delivered via cross-attention with
timestep-adaptive fusion gates.

Key differences from v1 (bundle.py):
  - No VACE conditioning: motion condition handled by MotionCondEncoder inside
    the transformer, injected via cross-attention in every DualCondMMDiTBlock.
  - predict_flow() passes condition_mask + known_motion directly to transformer.
  - No prepare_vace_input() method — the v3 transformer handles everything.
  - RoleEmbedding + TimestepAdaptiveFusionGate are inside the transformer.

Atomic forward functions shared between Trainer and Pipeline:
  - prepare_padding()       -- align src/tgt motions + build masks
  - predict_flow()          -- forward through the v3 transformer
  - decode_motion_from_latent() -- denormalize + FK to 3D keypoints
  - mask_text_cond()        -- classifier-free guidance null masking
  - encode_text()           -- lazy-load text encoder
"""

from __future__ import annotations

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
class HyMotionM2Mv3Bundle(ModelBundle):
    """ModelBundle for HyMotion-M2M v3 (DSCF architecture).

    The only sub-module managed via ``_build_modules`` is
    ``motion_transformer`` (the HunyuanMotionMMDiTv3). Auxiliary objects
    (M2MLoss, SmplxLiteJ24, null embeddings, mean/std buffers) are created
    directly in ``__init__``.

    Architecture highlights:
      - Motion condition via cross-attention (MotionCondEncoder → condition tokens)
      - Text condition via cross-attention (same as v1)
      - Timestep-adaptive fusion gates balance text vs motion condition per block
      - RoleEmbedding provides per-frame KEEP/GENERATE/EDIT signal
      - All above components live INSIDE the motion_transformer
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
        vtxt_input_dim: int = 768,
        ctxt_input_dim: int = 4096,
        # ----- SMPL body model path (optional; skipped if None) -----
        body_model_path: Optional[str] = None,
        # ----- rotation space -----
        rotation_space: str = 'local',
        # ----- KIMODO-style auxiliary losses (j_p / j_v / fk_consistency) -----
        kimodo_aux_loss_cfg: Optional[dict] = None,
        # ----- text attention preservation gradient scale -----
        text_grad_scale: float = 1.0,
    ):
        super().__init__()

        # ---- build trainable module via _build_modules ----
        self._build_modules({'motion_transformer': motion_transformer})

        # ---- hyper-params ----
        self.motion_type = motion_type
        self.pred_type = pred_type
        self.uncondition_mode = uncondition_mode
        self.cond_mask_prob = cond_mask_prob
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
        self.null_vtxt_feat = nn.Parameter(
            torch.zeros(1, 1, vtxt_input_dim), requires_grad=False
        )
        self.null_ctxt_input = nn.Parameter(
            torch.zeros(1, 1, ctxt_input_dim), requires_grad=False
        )

        # ---- mean / std buffers ----
        self._load_mean_std(mean_std_dir)

        # ---- M2M loss ----
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss
        self.m2m_loss = M2MLoss(**(losses_cfg or {}))

        # ---- KIMODO-style auxiliary loss (optional) ----
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

        # ---- text gradient scale ----
        self._text_grad_scale = text_grad_scale

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
        """Apply classifier-free guidance masking to text conditions.

        When force_mask=True, replaces ALL samples with null embeddings.
        When training with cond_mask_prob > 0, randomly masks per-sample.
        """
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

    def predict_flow(
        self,
        x: Tensor,
        ctxt_input: Tensor,
        vtxt_input: Tensor,
        timesteps: Tensor,
        condition_mask: Tensor,
        known_motion: Tensor,
        x_mask_temporal: Optional[Tensor] = None,
        ctxt_mask_temporal: Optional[Tensor] = None,
        edit_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Single forward pass through the v3 MMDiT transformer.

        In DSCF v3, the transformer internally handles:
          - Building scalar_mask from condition_mask
          - Input projection (x_t + scalar_mask → feat_dim)
          - MotionCondEncoder (known_motion + mask → condition tokens)
          - RoleEmbedding (mask → per-frame role signal)
          - DualCondMMDiTBlocks (text cross-attn + motion-cond cross-attn + fusion gates)

        Args:
            x: Noisy motion, shape (B, L, motion_dim).
            ctxt_input: Token-level text embeddings, (B, Lc, ctxt_dim).
            vtxt_input: Sentence-level text embeddings, (B, 1, vtxt_dim).
            timesteps: Diffusion timesteps, (B,).
            condition_mask: (B, L, motion_dim) binary mask (1=generate, 0=known).
            known_motion: (B, L, motion_dim) — motion values; zero where mask=1.
            x_mask_temporal: (B, L) boolean mask for motion sequence (True=valid).
            ctxt_mask_temporal: (B, Lc) boolean mask for text tokens (True=valid).
            edit_mask: (B, L) optional boolean — True where frame is in editing
                mode (has pre-edit values). Used to assign EDIT role.

        Returns:
            Model prediction (velocity), shape (B, L, motion_dim).
        """
        return self.motion_transformer(
            x=x,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            condition_mask=condition_mask,
            known_motion=known_motion,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=ctxt_mask_temporal,
            edit_mask=edit_mask,
        )

    def decode_motion_from_latent(
        self,
        latent: Tensor,
    ) -> Dict[str, Tensor]:
        """Denormalize latent and run FK to get 3D keypoints.

        Returns dict with keys: keypoints3d, rot6d, transl, latent_denorm.

        When ``rotation_space == 'global'``, the denormalized rot6d is in
        world-frame global rotation. We convert back to local (SMPL) rotation
        before FK so output is always SMPL-compatible.
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

    def compute_mask_density(self, src_mask: Tensor) -> Tensor:
        """Compute per-sample mask density.

        src_mask: (B, L, D), 1=generate, 0=known
        Returns: (B,) density in [0, 1]
        """
        return src_mask.mean(dim=(-1, -2))

    def apply_text_attention_preservation(self) -> None:
        """Apply TAP gradient scaling to text-related transformer parameters.

        Called by trainer at initialization. Uses self._text_grad_scale.
        Does nothing if scale >= 1.0.
        """
        if self._text_grad_scale >= 1.0:
            return
        from hftrainer.models.motion.hymotion_m2m.network.condition_routing import (
            TextAttentionPreservation,
        )
        self._tap = TextAttentionPreservation(
            text_grad_scale=self._text_grad_scale,
            refiner_grad_scale=min(self._text_grad_scale * 10, 1.0),
        )
        self._tap.apply(self.motion_transformer)

    # ------------------------------------------------------------------
    # Override checkpoint loading for v1→v3 key remapping
    # ------------------------------------------------------------------

    def load_state_dict_selective(self, state_dict: Dict[str, Any], strict: bool = False):
        """Override to handle v1→v3 transformer key remapping.

        The base class `load_state_dict_selective` splits flat keys by the
        first `.` and passes `{'double_blocks.0.*': ...}` to
        `self.motion_transformer.load_state_dict(...)`. But v3's transformer
        uses `blocks.*` naming, so all v1 keys are "unexpected" and nothing
        loads.

        This override detects v1-style keys (containing `double_blocks.` or
        `single_blocks.`) in the motion_transformer portion and routes them
        through `self.motion_transformer.load_pretrained_backbone()` which
        handles the proper remapping.
        """
        if not state_dict:
            return

        from hftrainer.utils.logger import get_logger
        logger = get_logger()

        # Let base class handle __hftrainer_meta__ and __bundle_params__
        # We extract those first, then handle the rest ourselves.
        checkpoint_meta = {}
        if '__hftrainer_meta__' in state_dict:
            meta = state_dict.pop('__hftrainer_meta__')
            if isinstance(meta, dict):
                checkpoint_meta = meta.get('modules', {}) or {}

        # Restore bundle-level parameters / buffers (null embeddings, mean, std)
        bundle_params = state_dict.pop('__bundle_params__', None)
        if bundle_params and isinstance(bundle_params, dict):
            for pname, pval in bundle_params.items():
                if hasattr(self, pname):
                    attr = getattr(self, pname)
                    if isinstance(attr, nn.Parameter):
                        if attr.shape == pval.shape:
                            attr.data.copy_(pval)
                        else:
                            logger.warning(
                                f"Shape mismatch for bundle param '{pname}': "
                                f"ckpt {tuple(pval.shape)} vs model {tuple(attr.shape)}, skipped"
                            )
                    elif isinstance(attr, Tensor):
                        if attr.shape == pval.shape:
                            attr.copy_(pval)

        if not state_dict:
            return

        # Detect flat vs nested format
        first_val = next(iter(state_dict.values()))
        if isinstance(first_val, Tensor):
            # Flat state dict — split by first '.' into nested
            nested: Dict[str, Dict[str, Tensor]] = {}
            for key, val in state_dict.items():
                parts = key.split('.', 1)
                if len(parts) == 2 and hasattr(self, parts[0]):
                    mod_name, param_name = parts
                    if mod_name not in nested:
                        nested[mod_name] = {}
                    nested[mod_name][param_name] = val
                else:
                    # Bundle-level tensor (mean, std, null embeddings)
                    if hasattr(self, key):
                        attr = getattr(self, key)
                        if isinstance(attr, nn.Parameter):
                            if attr.shape == val.shape:
                                attr.data.copy_(val)
                        elif isinstance(attr, Tensor):
                            if attr.shape == val.shape:
                                attr.copy_(val)
            state_dict = nested

        # Handle motion_transformer specially for v1→v3 remapping
        transformer_sd = state_dict.pop('motion_transformer', None)
        if transformer_sd is not None:
            # Detect if this is a v1-style checkpoint by checking for
            # double_blocks.* or single_blocks.* keys
            has_v1_keys = any(
                k.startswith('double_blocks.') or k.startswith('single_blocks.')
                for k in transformer_sd.keys()
            )
            has_v3_keys = any(
                k.startswith('blocks.') for k in transformer_sd.keys()
            )

            if has_v1_keys and not has_v3_keys:
                # v1→v3 remapping via load_pretrained_backbone
                logger.info(
                    f"[HyMotionM2Mv3Bundle] Detected v1-style checkpoint "
                    f"({len(transformer_sd)} keys). Using load_pretrained_backbone() "
                    f"for v1→v3 key remapping."
                )
                missing, unexpected = self.motion_transformer.load_pretrained_backbone(
                    transformer_sd, strict=False
                )
                if missing:
                    logger.info(
                        f"[v1→v3 remap] {len(missing)} missing keys (expected for "
                        f"v3-specific modules: cond_encoder, role_emb, fusion_gate, "
                        f"cross-attn, input_encoder). First 5: {missing[:5]}"
                    )
                if unexpected:
                    logger.warning(
                        f"[v1→v3 remap] {len(unexpected)} unexpected keys: {unexpected[:5]}"
                    )
            elif has_v3_keys:
                # Already v3-style keys — use normal loading
                logger.info(
                    f"[HyMotionM2Mv3Bundle] Detected v3-style checkpoint "
                    f"({len(transformer_sd)} keys). Using direct load_state_dict."
                )
                # Filter shape mismatches
                target_sd = self.motion_transformer.state_dict()
                for k in list(transformer_sd.keys()):
                    if k in target_sd and isinstance(transformer_sd[k], Tensor):
                        if transformer_sd[k].shape != target_sd[k].shape:
                            logger.warning(
                                f"Shape mismatch for 'motion_transformer.{k}': "
                                f"ckpt {tuple(transformer_sd[k].shape)} vs "
                                f"model {tuple(target_sd[k].shape)}, skipped"
                            )
                            del transformer_sd[k]
                missing, unexpected = self.motion_transformer.load_state_dict(
                    transformer_sd, strict=False
                )
                if missing:
                    logger.warning(
                        f"Missing keys in 'motion_transformer': {missing[:5]}..."
                    )
                if unexpected:
                    logger.warning(
                        f"Unexpected keys in 'motion_transformer': {unexpected[:5]}..."
                    )
            else:
                logger.warning(
                    f"[HyMotionM2Mv3Bundle] Could not detect checkpoint style "
                    f"for motion_transformer ({len(transformer_sd)} keys). "
                    f"Keys sample: {list(transformer_sd.keys())[:3]}. "
                    f"Falling back to direct load_state_dict."
                )
                self.motion_transformer.load_state_dict(transformer_sd, strict=False)

        # Handle remaining modules via standard logic (e.g. text_encoder if present)
        for name, sd in state_dict.items():
            if hasattr(self, name):
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    load_target = module
                    while hasattr(load_target, 'module'):
                        load_target = load_target.module
                    if not strict:
                        target_sd = load_target.state_dict()
                        for k in list(sd.keys()):
                            if k in target_sd and isinstance(sd[k], Tensor):
                                if sd[k].shape != target_sd[k].shape:
                                    del sd[k]
                    load_target.load_state_dict(sd, strict=strict)

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
