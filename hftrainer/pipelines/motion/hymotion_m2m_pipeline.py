"""HyMotion-M2M Pipeline: ODE-based inference for motion-to-motion editing.

Supports **imputation inference** for ``_man`` (mask-aware noise) variants:
when ``replacement_guidance != 'none'``, the pipeline performs per-step
replacement of known (unmasked) regions with clean motion, matching the
training distribution where ``x_t[known] = x1`` (clean).

Replacement guidance modes
--------------------------
- ``"none"`` (default): Standard ODE integration, no per-step replacement.
  Use for standard (non-MAN) models.
- ``"all"``: At every ODE step, replace known regions with ``clean_motion``.
- ``"skip_last"``: Same as ``"all"`` but skip replacement on the final step.
  This is the recommended mode for ``_man`` variants — matches MoGenDiT's
  default ``imputation_mode``.

When ``replacement_guidance != 'none'``, the batch **must** contain a
``clean_motion`` key: the full normalized motion ``(B, T, D)`` **without**
masked regions zeroed.  The initial noise ``y0`` is also set to
``clean_motion`` in known regions (matching training where ``x_t[known] = x1``).

Position constraint support
---------------------------
When ``position_constraints`` is provided in the batch, the pipeline applies
IK projection at each ODE step (after imputation) to enforce world-space
position constraints on specified joints. See
:class:`~hftrainer.pipelines.motion.position_constraint.PositionConstraint`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from hftrainer.registry import PIPELINES
from hftrainer.pipelines.motion.repair_utils import (
    compute_ada_keep_mask,
    compute_strict_adaptive_mask,
    joint_mask_to_dim_mask,
)


def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


def _gaussian_temporal_smooth(
    x: Tensor,
    sigma: float,
    protect_mask: Optional[Tensor] = None,
) -> Tensor:
    """1-D Gaussian temporal smoothing along axis 1 of a (B, T, D) tensor.

    Ported from ``scripts/eval/eval_m2m_v2_all_tasks._gaussian_temporal_smooth``.
    Used to pre-smooth the LQ ``clean_motion`` (the *kept* region the model
    conditions on, and which is copied back into the output) before masked
    imputation. Because partial regeneration keeps the jittery LQ on unmasked
    cells, smoothing there both lowers the residual jitter in the output and
    gives the regenerated region a smooth boundary to blend against.

    Where ``protect_mask > 0.5`` (the *generate* region) values pass through
    unchanged -- smoothing there is pointless (overwritten by the model) and
    would bleed bad values across defect boundaries.
    """
    if sigma <= 0.0:
        return x
    T = x.shape[1]
    radius = max(1, int(round(3.0 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    B, _, D = x.shape
    x_flat = x.permute(0, 2, 1).reshape(B * D, 1, T)
    w = kernel.view(1, 1, -1)
    x_pad = F.pad(x_flat, (radius, radius), mode='replicate')
    y_flat = F.conv1d(x_pad, w)
    y = y_flat.reshape(B, D, T).permute(0, 2, 1).contiguous()
    if protect_mask is not None:
        y = torch.where(protect_mask > 0.5, x, y)
    return y


@PIPELINES.register_module()
class HyMotionM2MPipeline:
    """Inference pipeline for HyMotion-M2M.

    Uses ODE integration to solve the flow matching ODE from noise to clean
    motion, conditioned on source motion (VACE) and optionally text.

    Parameters
    ----------
    bundle : HyMotionM2MBundle
        The model bundle.
    num_steps : int
        Number of ODE integration steps.
    text_guidance_scale : float
        Classifier-free guidance scale for text conditioning.
    replacement_guidance : str
        Controls per-step replacement of unmasked (known) regions during
        ODE integration.  For ``_man`` (mask-aware noise) variants, this
        implements train-consistent imputation: during training,
        ``x_t[known] = x1`` (clean), so replacing known regions with clean
        motion at each step matches the training distribution.

        - ``"none"`` (default): No replacement. Standard ODE integration.
        - ``"all"``: At every ODE step, replace known regions with
          ``clean_motion`` from the batch.
        - ``"skip_last"``: Same as ``"all"`` but skip replacement on the
          final step.  Recommended for ``_man`` variants (matches
          MoGenDiT's default mode).

        When not ``"none"``, the batch must contain ``clean_motion``:
        the full normalized motion (B, T, D) without masked-region zeroing.
    """

    VALID_REPLACEMENT_MODES = ('none', 'all', 'skip_last', 'flow_interp')

    def __init__(
        self,
        bundle,
        num_steps: int = 50,
        text_guidance_scale: float = 1.0,
        replacement_guidance: str = 'none',
        position_constraint_interval: int = 5,
        max_text_len: int = 128,
        sdedit_tau: float = 0.0,
    ):
        if replacement_guidance not in self.VALID_REPLACEMENT_MODES:
            raise ValueError(
                f'replacement_guidance must be one of '
                f'{self.VALID_REPLACEMENT_MODES}, got {replacement_guidance!r}'
            )
        self.bundle = bundle
        self.num_steps = num_steps
        self.text_guidance_scale = text_guidance_scale
        self.replacement_guidance = replacement_guidance
        self.position_constraint_interval = position_constraint_interval
        # IMPORTANT: must match the trainer's max_text_len (default 128) so
        # the context attention mask and positional statistics at inference
        # match what the model saw during training. Using the raw per-sample
        # token length (12-20) instead of padding to 128 was a bug that made
        # captioned inference produce distorted outputs (2026-04-20).
        self.max_text_len = max_text_len
        # SDEdit-style partial-noise start for E9 motion repair. In
        # flow-matching convention (t=0 -> pure noise, t=1 -> clean data),
        # the default inpainting path starts from t=0 on the masked region
        # (full regeneration). SDEdit τ lets us start from t = 1 - τ instead
        # — i.e. the masked region is initialized as `(1-τ)*x_clean + τ*z`
        # and the ODE runs from t=1-τ up to t=1. Smaller τ → more LQ retained,
        # closer to a "cleanup" of defects; larger τ (→1) → full regeneration.
        # Only applied when replacement_guidance != 'none' (requires
        # `clean_motion` in the batch to know what the masked region's LQ is).
        if not (0.0 <= sdedit_tau <= 1.0):
            raise ValueError(
                f'sdedit_tau must be in [0, 1], got {sdedit_tau!r}'
            )
        self.sdedit_tau = float(sdedit_tau)

    @torch.no_grad()
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on a batch.

        Uses midpoint ODE solver for numerical stability (euler diverges).
        """
        return self._inference(batch)

    def _inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Actual inference logic.

        Returns:
            Dict with keys: rot6d, transl, keypoints3d (optional), latent.
        """
        device = next(self.bundle.motion_transformer.parameters()).device

        src_motion = batch['src_motion'].to(device)
        B, T, D = src_motion.shape

        src_mask = batch.get('src_mask')
        if src_mask is not None:
            src_mask = src_mask.to(device)

        src_length = batch.get('src_length')
        if isinstance(src_length, Tensor):
            src_length = src_length.tolist()

        tgt_length = batch.get('tgt_length', src_length)
        if isinstance(tgt_length, Tensor):
            tgt_length = tgt_length.tolist()

        ref_pose = batch.get('ref_pose')
        if ref_pose is not None and not isinstance(ref_pose, Tensor):
            ref_pose = None
        if ref_pose is not None:
            ref_pose = ref_pose.to(device)

        tgt_padding_mask = _length_to_mask(
            torch.tensor(tgt_length, dtype=torch.long, device=device), T
        )

        # Prepare text
        # CRITICAL: must match training-time convention (see
        # HyMotionM2MTrainer._prepare_and_forward):
        #   1. ctxt_input is always padded to max_text_len=128, regardless of
        #      the per-sample caption length.
        #   2. ctxt_mask_temporal marks valid tokens (True = valid, False = pad).
        #   3. Null-caption samples get the *learned* null_ctxt_input broadcast
        #      to the full (1, 128, 4096) shape — NOT a zero tensor with one
        #      active token. Attention masks are all-False for null samples
        #      (training convention in `_length_to_mask(ctxt_length=0, 128)`).
        #   4. All tensors must match the transformer's parameter dtype so
        #      attention math happens in the same precision as training.
        # Earlier code used raw caption seq_len (12-20) for ctxt and
        # zeros-with-first-token-null for CFG, which led to distorted
        # captioned outputs because the model never saw those distributions
        # during training. (2026-04-20)
        pad_len = self.max_text_len
        model_dtype = next(self.bundle.motion_transformer.parameters()).dtype

        def _pad_ctxt(ctxt: Tensor, length_is_valid: bool) -> Tensor:
            """Pad / truncate ctxt to (B, pad_len, D)."""
            if ctxt.shape[1] == pad_len:
                return ctxt
            if ctxt.shape[1] < pad_len:
                return F.pad(ctxt, (0, 0, 0, pad_len - ctxt.shape[1]))
            return ctxt[:, :pad_len]

        if 'text_vec_raw' in batch:
            vtxt_input = batch['text_vec_raw'].to(device=device, dtype=model_dtype)
            ctxt_raw = batch['text_ctxt_raw'].to(device=device, dtype=model_dtype)
            ctxt_input = _pad_ctxt(ctxt_raw, True)
            ctxt_length = batch['text_ctxt_raw_length'].to(device).clamp(max=pad_len)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)
        else:
            # Unconditioned inference: MUST match training convention in
            # HyMotionM2MTrainer._prepare_and_forward lines 212-215:
            #   ctxt_input = null_ctxt_input.expand(B, 1, -1)   ← 1 token
            #   ctxt_length = 1
            #   ctxt_mask = all-True over length 1
            # Earlier code used pad_len (128) here for symmetry with the
            # captioned branch, but the uncond-trained model never saw a
            # 128-token context during training — it always saw a single
            # null token. Feeding 128 repeated null tokens + all-False
            # attention mask is a severe OOD shift that produces catastrophic
            # jitter in output (found 2026-04-21).
            vtxt_input = self.bundle.null_vtxt_feat.to(dtype=model_dtype).expand(B, 1, -1)
            ctxt_input = self.bundle.null_ctxt_input.to(dtype=model_dtype).expand(B, 1, -1).contiguous()
            ctxt_length = torch.ones(B, dtype=torch.long, device=device)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, 1)

        # Prepare VACE context
        vace_context = self.bundle.prepare_vace_input(
            src_motion=src_motion,
            ref_pose=ref_pose,
            src_mask=src_mask,
        )

        do_cfg = self.text_guidance_scale > 1.0 and not self.bundle.uncondition_mode

        # CFG null-branch construction.  The "silent" CFG branch nulls BOTH
        # sentence-level vtxt AND token-level ctxt to match training-time
        # mask_text_cond behavior (which masks both vtxt and ctxt).
        # Previously only vtxt was nulled while ctxt was kept intact, making
        # CFG guidance depend solely on the 768-dim vtxt difference — far too
        # weak for effective caption guidance.  Fixed 2026-05-15.
        if do_cfg:
            null_vtxt = self.bundle.null_vtxt_feat.to(dtype=model_dtype).expand_as(vtxt_input)
            # Expand null_ctxt to match ctxt_input's sequence length so
            # torch.cat along batch dim works correctly.
            null_ctxt = self.bundle.null_ctxt_input.to(dtype=model_dtype).expand(
                ctxt_input.shape[0], ctxt_input.shape[1], -1
            ).contiguous()
            # Build attention mask for the null branch that matches training:
            # during training, mask_text_cond drops text AND the trainer
            # updates ctxt_mask_temporal to [True, False, ..., False] (only
            # position 0 attends).  The old code reused the conditioned mask
            # here (k True positions where k = caption length), creating a
            # train-inference mismatch for the null branch.  Fixed 2026-05-15.
            null_ctxt_mask = torch.zeros_like(ctxt_mask_temporal)
            null_ctxt_mask[:, 0] = True  # Only position 0 is valid

        # ODE function
        ode_cfg = dict(self.bundle._noise_scheduler_cfg)
        ode_cfg.pop('method', None)  # odeint uses it positionally

        def fn(t: Tensor, x: Tensor) -> Tensor:
            x_input = torch.cat([x, vace_context], dim=-1)
            if do_cfg:
                x_input = torch.cat([x_input, x_input], dim=0)
            x_pred = self.bundle.predict_flow(
                x_input=x_input,
                ctxt_input=(
                    ctxt_input if not do_cfg
                    else torch.cat([null_ctxt, ctxt_input], dim=0)
                ),
                vtxt_input=(
                    vtxt_input if not do_cfg
                    else torch.cat([null_vtxt, vtxt_input], dim=0)
                ),
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=(
                    tgt_padding_mask if not do_cfg
                    else tgt_padding_mask.repeat(2, 1)
                ),
                ctxt_mask_temporal=(
                    ctxt_mask_temporal if not do_cfg
                    else torch.cat([null_ctxt_mask, ctxt_mask_temporal], dim=0)
                ),
            )

            if self.bundle.pred_type == 'x1':
                t_eps = 0.05
                if do_cfg:
                    x_pred = (x_pred - torch.cat([x, x], dim=0)) / (1.0 - t).clamp_min(t_eps)
                else:
                    x_pred = (x_pred - x) / (1.0 - t).clamp_min(t_eps)

            if do_cfg:
                pred_basic, pred_text = x_pred.chunk(2, dim=0)
                x_pred = pred_basic + self.text_guidance_scale * (pred_text - pred_basic)
            return x_pred

        # -----------------------------------------------------------------
        # Initial y0 and replacement guidance setup
        # -----------------------------------------------------------------
        z = torch.randn(B, T, D, device=device, dtype=src_motion.dtype)
        t = torch.linspace(0, 1, self.num_steps + 1, device=device, dtype=src_motion.dtype)

        rep_mode = self.replacement_guidance
        use_replacement = (
            rep_mode != 'none'
            and src_mask is not None
            and src_mask.sum() > 0                # has masked regions
            and src_mask.sum() < src_mask.numel()  # has unmasked regions
        )

        if use_replacement:
            # keep_mask: (B, T, D), True = known region (mask=0).
            #
            # ⚠️ 2026-04-24 bug fix ("E13 每段尾帧静止"): exclude PAD frames
            # from the known region even if src_mask=0 there. Rationale: under
            # training distribution, pad frames (idx >= tgt_length) carry
            # src_mask=0 AND src_motion=0 AND attention is masked out by
            # tgt_padding_mask AND loss is masked out. The model never
            # "sees" pad frames during training. If we leave keep_mask=True
            # on pad frames at inference time, the replacement loop below
            # pins x[pad] ← x_clean[pad] = replicate(normalize(synthetic-zero
            # last frame)) every ODE step — i.e. it anchors the entire pad
            # region to the training-set MEAN pose. For cases where the
            # "synthetic zero" is just the training mean (E13 where src_raw
            # is zeros outside the prefix, or any short-clip inference where
            # we pad by replicating a valid end frame), this mean-pose
            # anchor leaks into the valid region via shared LayerNorms /
            # residual paths (pad is key-masked but still flows as a
            # query/value through per-token feedforwards) and pulls the
            # TAIL of the valid region visibly toward static "mean pose".
            # Users reported this as "每段尾帧几乎不动、静止" on E13.
            #
            # Fix: combine src_mask (per-frame-per-dim "is this a known
            # sample?") with tgt_padding_mask ("is this a valid frame in
            # the model's view?"). Pad frames get keep_mask=False → neither
            # y0 init (below) nor per-step replacement touches them. They
            # become ordinary ODE free-evolve tokens — consistent with
            # training where the model is simply not asked about them.
            valid_frame_mask = tgt_padding_mask.unsqueeze(-1)  # (B, T, 1)
            keep_mask = (src_mask < 0.5) & valid_frame_mask
            # clean_motion: full normalized motion WITHOUT masked-region
            # zeroing.  Required for _man (mask-aware noise) imputation.
            assert 'clean_motion' in batch, (
                'replacement_guidance requires "clean_motion" in batch '
                '(full normalized motion without zeroing masked regions)'
            )
            x_clean = batch['clean_motion'].to(device)

            if self.sdedit_tau > 0.0:
                # SDEdit-style partial-noise start on masked region.
                # Flow-matching convention (in this pipeline): t=0 → noise,
                # t=1 → clean, so x_t = (1-t)*z + t*clean. To start from τ
                # noise fraction we set t_init = 1 - τ. The loop below will
                # honor this by skipping ODE steps with t_curr < t_init.
                tau = self.sdedit_tau
                t_init = 1.0 - tau
                x_partial_noised = (1.0 - t_init) * z + t_init * x_clean
                y0 = torch.where(keep_mask, x_clean, x_partial_noised)
                sdedit_t_init = t_init
            else:
                # _man training: x_t[known] = x1 (clean).  Match at t=0:
                y0 = torch.where(keep_mask, x_clean, z)
                sdedit_t_init = None
        else:
            y0 = z
            sdedit_t_init = None

        # -----------------------------------------------------------------
        # Position constraint setup
        # -----------------------------------------------------------------
        position_constraints = batch.get('position_constraints')
        use_pos_constraint = position_constraints is not None and len(position_constraints) > 0
        pos_solver = None
        pos_affected_dims = None

        if use_pos_constraint:
            from hftrainer.pipelines.motion.position_constraint import (
                PositionConstraintSolver,
                get_affected_dims,
            )
            bone_offsets = self.bundle.get_bone_offsets()
            rotation_space = getattr(self.bundle, 'rotation_space', 'local')
            pos_solver = PositionConstraintSolver(
                bone_offsets=bone_offsets,
                rotation_space=rotation_space,
            )
            pos_affected_dims = get_affected_dims(position_constraints)
            pc_interval = self.position_constraint_interval

        # -----------------------------------------------------------------
        # ODE integration
        # -----------------------------------------------------------------
        n_ode_steps = len(t) - 1
        # SDEdit partial-noise start: skip ODE steps whose t_curr < t_init so
        # integration begins at the partial-noise level we initialized y0 at.
        # Step boundaries are inclusive of t_init: we pick the smallest i such
        # that t[i] >= t_init. For tau=0 (default) this yields start_i=0.
        if sdedit_t_init is not None:
            # t is a 1-D tensor, convert to float for the check
            t_vals = t.detach().cpu().tolist()
            start_i = next(
                (i for i, tv in enumerate(t_vals) if tv + 1e-6 >= sdedit_t_init),
                0,
            )
        else:
            start_i = 0
        if use_replacement or use_pos_constraint:
            # Manual Euler with per-step imputation and/or position constraint.
            # Store initial noise for flow_interp mode
            z0 = y0.clone() if rep_mode == 'flow_interp' else None
            x = y0
            for i in range(start_i, n_ode_steps):
                t_curr = t[i]
                dt = t[i + 1] - t[i]
                is_last_step = (i == n_ode_steps - 1)

                v = fn(t_curr, x)
                x = x + v * dt

                # Imputation: force known regions back to expected values.
                if use_replacement:
                    if rep_mode == 'flow_interp' and not is_last_step:
                        t_next = t[i + 1]
                        x_interp = (1 - t_next) * z0 + t_next * x_clean
                        x = torch.where(keep_mask, x_interp, x)
                    elif rep_mode == 'all' or (rep_mode == 'skip_last' and not is_last_step):
                        x = torch.where(keep_mask, x_clean, x)

                # Position constraint projection
                if use_pos_constraint:
                    # Analytic IK (root/2-bone/1-bone): every step
                    # Gradient IK: every pc_interval steps + last step
                    do_gradient_ik = is_last_step or (i % pc_interval == 0)

                    # Denormalize -> IK solve -> renormalize
                    x_denorm = self.bundle.denormalize_motion(x)
                    x_fixed = x_denorm.clone()

                    for b_idx in range(B):
                        frame_constraints = {}  # frame -> list of constraints
                        for c in position_constraints:
                            frame_constraints.setdefault(c.frame, []).append(c)

                        for frame, cs in frame_constraints.items():
                            if frame >= T:
                                continue
                            # Filter by IK type
                            from hftrainer.pipelines.motion.ik_solver import get_ik_strategy
                            analytic_cs = [
                                c for c in cs
                                if get_ik_strategy(c.joint) in ('root', 'two_bone', 'one_bone')
                            ]
                            gradient_cs = [
                                c for c in cs
                                if get_ik_strategy(c.joint) == 'gradient'
                            ]

                            active_cs = analytic_cs
                            if do_gradient_ik:
                                active_cs = active_cs + gradient_cs

                            if active_cs:
                                frame_motion = x_fixed[b_idx, frame]
                                for c in active_cs:
                                    frame_result, _ = pos_solver._solve_single(
                                        frame_motion.unsqueeze(0), [c]
                                    )
                                    frame_motion = frame_result.squeeze(0)
                                x_fixed[b_idx, frame] = frame_motion

                    # Renormalize and selectively replace affected dims
                    x_renorm = self.bundle.normalize_motion(x_fixed)
                    if pos_affected_dims is not None:
                        dim_idx = torch.tensor(pos_affected_dims, device=device)
                        # Only replace affected frames and dims
                        affected_frames = set(c.frame for c in position_constraints if c.frame < T)
                        for f in affected_frames:
                            x[:, f, dim_idx] = x_renorm[:, f, dim_idx]
                    else:
                        x = x_renorm

            # Final hard replacement: ensure known regions are exactly preserved.
            # This is critical for skip_last and flow_interp modes — the model's
            # velocity prediction for known dims is unsupervised (loss is masked),
            # so the last step introduces drift. We do a final replace to guarantee
            # exact preservation.
            if use_replacement and rep_mode in ('skip_last', 'flow_interp'):
                x = torch.where(keep_mask, x_clean, x)

            sampled = x
        else:
            # Standard path: use torchdiffeq if available, else manual Euler.
            try:
                from torchdiffeq import odeint
                method = self.bundle._noise_scheduler_cfg.get('method', 'euler')
                trajectory = odeint(fn, y0, t, method=method)
            except ImportError:
                trajectory = [y0]
                dt = 1.0 / self.num_steps
                x = y0
                for i in range(self.num_steps):
                    t_val = torch.tensor(i * dt, device=device, dtype=src_motion.dtype)
                    v = fn(t_val, x)
                    x = x + v * dt
                    trajectory.append(x)
                trajectory = torch.stack(trajectory, dim=0)

            sampled = trajectory[-1]

        # Decode to motion
        result = self.bundle.decode_motion_from_latent(sampled)
        result['latent'] = sampled
        result['rotation_space'] = getattr(self.bundle, 'rotation_space', 'local')
        return result

    # ------------------------------------------------------------------ #
    # Motion repair (E9): defect detection + masked regeneration.        #
    # ------------------------------------------------------------------ #
    T_PAD_REPAIR = 360  # the context length the model was trained with

    def _repair_forward(
        self,
        motion_norm: Tensor,     # (1, T_pad, D) normalized LQ, zero-padded
        mask135: Tensor,         # (1, T_pad, D) 1=generate, 0=keep
        clean_motion: Tensor,    # (1, T_pad, D) normalized full LQ (no zeroing)
        valid_len: int,
        replacement_guidance: str,
        sdedit_tau: float,
        text_fields: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """One masked-imputation forward pass. Returns normalized latent
        ``(1, T_pad, D)``. Temporarily overrides the instance replacement /
        SDEdit settings so a single pipeline object can serve every repair
        configuration without re-construction."""
        prev_repl, prev_tau = self.replacement_guidance, self.sdedit_tau
        self.replacement_guidance = replacement_guidance
        self.sdedit_tau = float(sdedit_tau)
        try:
            src_motion = motion_norm * (1.0 - mask135)  # inpaint: zero masked
            batch = {
                'src_motion': src_motion,
                'src_mask': mask135,
                'src_length': [valid_len],
                'tgt_length': [valid_len],
                'clean_motion': clean_motion,
            }
            if text_fields is not None:
                batch.update(text_fields)
            out = self._inference(batch)
        finally:
            self.replacement_guidance, self.sdedit_tau = prev_repl, prev_tau
        return out['latent']

    @torch.no_grad()
    def _self_denoise_joint_change(self, mn: Tensor, stage1: Tensor):
        """Physical per-joint change for self-denoise detection.

        Returns ``(jchg, tchg)`` where ``jchg`` is the (T, 22) geodesic angle
        (radians) between the LQ and stage-1 *local* joint rotations and
        ``tchg`` is the (T,) translation L2 change (meters). Measuring in a
        physical space (angle/meters) -- instead of the z-scored channel |Δ|
        used by ``compute_ada_keep_mask`` -- gives the threshold a real meaning
        and avoids the below-noise-floor saturation; magnitude aggregation
        (one scalar per joint) replaces the coverage-amplifying 6-channel OR.
        """
        from hftrainer.models.motion.hymotion_m2m.network.geometry import (
            rot6d_to_rotation_matrix,
        )
        raw_lq = self.bundle.denormalize_motion(mn)        # (1, T, D)
        raw_s1 = self.bundle.denormalize_motion(stage1)
        T = raw_lq.shape[1]
        r6_lq = raw_lq[0, :, 3:135].reshape(T, 22, 6)
        r6_s1 = raw_s1[0, :, 3:135].reshape(T, 22, 6)
        if getattr(self.bundle, 'rotation_space', 'local') == 'global':
            # Compare *local* joint rotations (isolates each joint's own
            # defect; global angles would propagate parent error down the
            # kinematic chain and inflate coverage).
            from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
                global_to_local_rot6d_torch,
            )
            r6_lq = global_to_local_rot6d_torch(r6_lq.unsqueeze(0))[0]
            r6_s1 = global_to_local_rot6d_torch(r6_s1.unsqueeze(0))[0]
        R_lq = rot6d_to_rotation_matrix(r6_lq.reshape(-1, 6)).reshape(T, 22, 3, 3)
        R_s1 = rot6d_to_rotation_matrix(r6_s1.reshape(-1, 6)).reshape(T, 22, 3, 3)
        R_rel = torch.matmul(R_lq.transpose(-1, -2), R_s1)
        tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        jchg = torch.arccos(cos)                            # (T, 22) radians
        tchg = torch.linalg.norm(
            raw_lq[0, :, :3] - raw_s1[0, :, :3], dim=-1)    # (T,) meters
        return jchg.cpu().numpy(), tchg.cpu().numpy()

    @staticmethod
    def _joint_change_to_raw_mask(jchg, tchg, model_dim, joint_thr, trans_thr):
        """Threshold the physical per-joint angle / translation change into a
        ``(T, model_dim)`` raw generate-mask (1=defect). Flags a joint's rot6d
        (and FK-position) channels when its angle change exceeds ``joint_thr``
        (rad); flags translation when ``tchg`` exceeds ``trans_thr`` (m)."""
        T = jchg.shape[0]
        out = np.zeros((T, model_dim), dtype=np.float32)
        jflag = jchg > float(joint_thr)        # (T, 22)
        tflag = tchg > float(trans_thr)        # (T,)
        out[tflag, :3] = 1.0
        for j in range(22):
            out[jflag[:, j], 3 + j * 6: 3 + (j + 1) * 6] = 1.0
            if model_dim >= 198 and j >= 1:
                out[jflag[:, j], 135 + (j - 1) * 3: 135 + j * 3] = 1.0
        return out

    @torch.no_grad()
    def infer_repair(
        self,
        motion: Tensor,
        lengths: Optional[List[int]] = None,
        *,
        # mask source (axis 2)
        mask_source: str = 'self_denoise',     # 'self_denoise' | 'provided'
        adaptive_mask: Optional[Tensor] = None,  # (B,T,22) or (B,T,D), 1=defect
        detect_tau: float = 0.3,                # SDEdit tau for stage-1 projection
        detect_metric: str = 'angle',           # 'angle' (MoGenDIT-style) | 'abs'
        detect_joint_thr_rad: float = 0.15,      # per-joint geodesic angle (rad)
        detect_trans_thr_m: float = 0.05,        # translation change (meters)
        detect_threshold_mode: str = 'abs',     # 'abs' | 'topk_pct' (metric='abs')
        detect_threshold: float = 0.1,
        # mask tightening
        strict_tighten: bool = True,
        strict_dilate: int = 2,
        strict_min_blob: int = 3,
        # the 4 configurable axes
        translation_mode: str = 'lock',         # axis 1: 'lock'|'detected'|'all'
        mask_granularity: str = 'joint',        # axis 4: 'joint'|'frame'
        sdedit_tau: float = 0.5,                 # axis 3: 0=from-scratch, >0=partial
        replacement_guidance: str = 'skip_last',
        presmooth_sigma: float = 0.0,            # Gaussian temporal pre-smooth of kept LQ
        text_fields: Optional[Dict[str, Any]] = None,
        return_mask: bool = True,
    ) -> Dict[str, Any]:
        """Repair defective motion with masked regeneration.

        This is the single canonical entry point for HyMotion-M2M motion
        repair -- call this instead of hand-assembling a batch, so the repair
        recipe lives in one discoverable place.

        The four configurable axes
        --------------------------
        translation_mode : axis 1 -- how the global root translation is treated.
            ``'lock'`` never regenerates translation (M7 convention; keeps the
            global trajectory locked to the input -- recommended, avoids root
            drift). ``'detected'`` regenerates translation only on frames where
            the pelvis is flagged defective. ``'all'`` regenerates translation
            on every valid frame.
        mask_source : axis 2 -- where the defect mask comes from.
            ``'self_denoise'`` (ours): run a stage-1 SDEdit-from-LQ projection
            with this model and threshold ``|LQ - projection|`` (MoGenDIT-style
            ada_denoise but with *our* model). ``'provided'``: use the mask in
            ``adaptive_mask`` (e.g. a MoGenDIT-computed or QC mask). Keeping the
            external mask as a passed-in argument avoids coupling this pipeline
            to other methods' models.
        sdedit_tau : axis 3 -- regeneration strength for masked cells.
            ``0`` starts the masked region from pure noise (full regeneration
            from scratch). ``>0`` starts from ``tau*noise + (1-tau)*LQ`` and only
            runs the last ``tau`` of the ODE (partial re-noise; stays close to
            the input -- gentle cleanup).
        mask_granularity : axis 4 -- spatial extent of regeneration.
            ``'joint'`` regenerates only the flagged joints' channels.
            ``'frame'`` regenerates every joint of any frame that has at least
            one flagged joint (whole-frame regeneration). ``'channel'``
            regenerates only the individual flagged channels (no per-joint OR,
            no strict tightening) -- the MoGenDIT-faithful per-element scheme;
            requires ``mask_source='self_denoise'``.

        Parameters
        ----------
        motion : (B, T, D) or (T, D) tensor
            Raw (un-normalized) LQ motion in the model's representation
            (135-dim, local rotation). Must already be at the model fps.
        lengths : optional list[int]
            Valid frame count per sample (defaults to full T).

        Returns
        -------
        dict with ``motion`` (B,T,135 repaired: transl + rot6d), and when
        ``return_mask``: ``joint_mask`` (B,T,22 bool) and ``mask``
        (B,T,model_dim generate-mask).
        """
        if mask_source not in ('self_denoise', 'provided'):
            raise ValueError(f'mask_source must be self_denoise|provided, got {mask_source!r}')
        if mask_source == 'provided' and adaptive_mask is None:
            raise ValueError("mask_source='provided' requires adaptive_mask")

        device = next(self.bundle.motion_transformer.parameters()).device
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        motion = motion.float().to(device)
        B, T, D = motion.shape
        T_PAD = self.T_PAD_REPAIR
        if T > T_PAD:
            raise NotImplementedError(
                f'infer_repair supports motions up to {T_PAD} frames (got {T}). '
                'For longer sequences, chunk into <=360-frame windows and stitch, '
                'or use the windowed eval path in eval_m2m_v2_all_tasks.py.'
            )
        if lengths is None:
            lengths = [T] * B

        # The model operates on its native motion_dim (198 = 3 transl + 132
        # rot6d + 63 FK joint positions). The caller passes 135-dim (transl +
        # rot6d). We FK-expand 135->198 with the bundle's bone offsets, mirroring
        # the trainer/eval (eval_m2m_v2_all_tasks.motion_135_to_198) so mean/std
        # and the per-channel layout match training. Output is decoded back to
        # 135 (transl + rot6d) which is the canonical caller representation.
        from hftrainer.pipelines.motion.repair_utils import motion_135_to_198
        rot_space = getattr(self.bundle, 'rotation_space', 'local')
        # The model dim is the normalizer width (mean/std), not necessarily a
        # `motion_dim` attribute (which may be unset on some bundles).
        model_dim = int(self.bundle.mean.shape[-1])
        bone_offsets_np = None
        if model_dim >= 198:
            bone_offsets_np = self.bundle.get_bone_offsets().cpu().numpy()  # (22,3)

        local_to_global = None
        if rot_space == 'global':
            from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
                local_to_global_rot6d_torch as local_to_global,
            )

        repaired = motion.clone()
        joint_masks = np.zeros((B, T, 22), dtype=bool)
        mask_dim_out = np.zeros((B, T, model_dim), dtype=np.float32)

        for b in range(B):
            L = int(lengths[b])
            m135 = motion[b].detach().cpu().numpy()              # (T,135) local

            # 135 -> model_dim raw (append FK positions for 198-dim models).
            if model_dim >= 198:
                raw = motion_135_to_198(m135, bone_offsets_np)   # (T,198) local
            else:
                raw = m135.astype(np.float32).copy()
            # local -> global rot6d if the model trains in world frame.
            if local_to_global is not None:
                rl = torch.from_numpy(
                    raw[:, 3:135].reshape(T, 22, 6)).float()
                raw = raw.copy()
                raw[:, 3:135] = local_to_global(rl).reshape(T, 132).numpy()

            mn = self.bundle.normalize_motion(
                torch.from_numpy(raw).float().unsqueeze(0).to(device))  # (1,T,model_dim)
            clean = mn.clone()
            if T < T_PAD:
                # Replicate the last valid frame into the pad region (static
                # hold) rather than zero-padding. Zeros == the normalized MEAN
                # pose, and because pad frames free-evolve under the ODE the
                # last *valid* frame gets pulled toward that mean pose -> a
                # systematic last-frame teleport (jumpLast ~20x the normal
                # per-frame step on ~all clips). A replicated static hold is
                # in-distribution (many training clips end static) and keeps
                # the boundary continuous.
                n_pad = T_PAD - T
                mn = torch.cat([mn, mn[:, -1:].expand(-1, n_pad, -1)], dim=1)
                clean = torch.cat([clean, clean[:, -1:].expand(-1, n_pad, -1)], dim=1)

            # Step A: base defect mask (model_dim, 1=generate).
            if mask_source == 'self_denoise':
                ones = torch.ones_like(mn)
                ones[:, 0, 0] = 0.0  # one keep cell so SDEdit branch activates
                stage1 = self._repair_forward(
                    mn, ones, clean, valid_len=L,
                    replacement_guidance='skip_last', sdedit_tau=detect_tau,
                    text_fields=None,
                )
                if mask_granularity == 'channel':
                    # MoGenDIT-faithful: pure per-channel keep/regenerate in the
                    # normalized space (high_change = |LQ - projection| > thr),
                    # with NO per-joint OR aggregation. Each of the model_dim
                    # channels is decided independently, exactly like MoGenDIT's
                    # official ada_denoise (change_threshold on the 201-dim rep).
                    ch_change = np.abs(
                        mn[0].cpu().numpy() - stage1[0].cpu().numpy())
                    raw_mask = (ch_change > float(detect_threshold)).astype(
                        np.float32)  # (T_PAD, model_dim)
                elif detect_metric == 'angle':
                    # MoGenDIT-style: compare in a *physical* space (per-joint
                    # geodesic angle in radians + translation in meters) so the
                    # threshold has meaning and is not buried under the z-scored
                    # reconstruction noise floor; aggregate by magnitude (one
                    # scalar/joint), not a 6-channel OR.
                    jchg, tchg = self._self_denoise_joint_change(mn, stage1)
                    raw_mask = self._joint_change_to_raw_mask(
                        jchg, tchg, model_dim,
                        joint_thr=detect_joint_thr_rad,
                        trans_thr=detect_trans_thr_m,
                    )
                elif detect_metric == 'abs':
                    raw_mask = compute_ada_keep_mask(
                        mn[0].cpu().numpy(), stage1[0].cpu().numpy(),
                        threshold_mode=detect_threshold_mode,
                        threshold=detect_threshold,
                    )  # (T_PAD, model_dim)
                else:
                    raise ValueError(
                        f'detect_metric must be angle|abs, got {detect_metric!r}')
            else:
                am = adaptive_mask[b]
                am = am.cpu().numpy() if isinstance(am, Tensor) else np.asarray(am)
                raw_mask = np.zeros((T_PAD, model_dim), dtype=np.float32)
                if am.ndim == 2 and am.shape[-1] == 22:
                    jm = am[:T].astype(np.float32)
                    raw_mask[:T, :3] = jm[:, 0:1]
                    for j in range(22):
                        raw_mask[:T, 3 + j * 6:3 + (j + 1) * 6] = jm[:, j:j + 1]
                    if model_dim >= 198:
                        for j in range(1, 22):
                            raw_mask[:T, 135 + (j - 1) * 3:135 + j * 3] = jm[:, j:j + 1]
                else:  # already (T,D-ish): copy what fits
                    cd = min(am.shape[-1], model_dim)
                    raw_mask[:min(am.shape[0], T_PAD), :cd] = \
                        am[:T_PAD, :cd].astype(np.float32)

            if mask_granularity == 'channel':
                # MoGenDIT-faithful path: no joint aggregation, no strict
                # tightening -- the per-channel mask IS the dim mask. Only apply
                # the translation policy and the valid-length guard.
                dim_mask = raw_mask.astype(np.float32).copy()
                dim_mask[L:] = 0.0
                if translation_mode == 'lock':
                    dim_mask[:, :3] = 0.0
                elif translation_mode == 'all':
                    dim_mask[:L, :3] = 1.0
                # 'detected' keeps the per-channel translation decision as-is.
                jflag = (dim_mask[:, 3:135].reshape(T_PAD, 22, 6) >= 0.5).any(-1)
                jflag[L:] = False
            else:
                # Step B: tighten (strict) -> per-joint flag.
                if strict_tighten:
                    tight = compute_strict_adaptive_mask(
                        raw_mask, dilate=strict_dilate, min_blob=strict_min_blob,
                        motion_dim=model_dim,
                        lock_trans=(translation_mode == 'lock'),
                    )
                else:
                    tight = raw_mask
                jflag = (tight[:, 3:135].reshape(T_PAD, 22, 6) >= 0.5).any(-1)
                jflag[L:] = False  # pad frames are always known

                # Step C: granularity (axis 4).
                if mask_granularity == 'frame':
                    frame_hit = jflag.any(axis=-1, keepdims=True)
                    jflag = np.broadcast_to(frame_hit, jflag.shape).copy()
                    jflag[L:] = False
                elif mask_granularity != 'joint':
                    raise ValueError(
                        f'mask_granularity must be joint|frame|channel, '
                        f'got {mask_granularity!r}')

                # Step D: expand to dim mask with translation policy (axis 1).
                dim_mask = joint_mask_to_dim_mask(
                    jflag, motion_dim=model_dim,
                    translation_mode=translation_mode, valid_len=L,
                )
            mask_t = torch.from_numpy(dim_mask).float().unsqueeze(0).to(device)

            # Step D.5: pre-smooth the kept (unmasked) LQ. Partial regeneration
            # keeps jittery LQ on unmasked cells -- both as the conditioning the
            # model sees and as the values copied back into the output -- so the
            # residual corruption jitter survives and seams appear at mask
            # boundaries. A light Gaussian temporal smooth of the kept region
            # lowers that jitter (protect_mask = generate region, left intact).
            mn_e, clean_e = mn, clean
            if presmooth_sigma > 0.0:
                mn_e = _gaussian_temporal_smooth(mn, presmooth_sigma, protect_mask=mask_t)
                clean_e = _gaussian_temporal_smooth(clean, presmooth_sigma, protect_mask=mask_t)

            # Step E: masked regeneration (axis 3 = sdedit_tau).
            latent = self._repair_forward(
                mn_e, mask_t, clean_e, valid_len=L,
                replacement_guidance=replacement_guidance, sdedit_tau=sdedit_tau,
                text_fields=text_fields,
            )
            dec = self.bundle.decode_motion_from_latent(latent)   # local rot6d
            transl = dec['transl'][0, :T]                          # (T,3)
            rot6d = dec['rot6d'][0, :T].reshape(T, 132)            # (T,132) local
            repaired[b] = torch.cat([transl, rot6d], dim=-1)

            joint_masks[b] = jflag[:T]
            mask_dim_out[b] = dim_mask[:T]

        result: Dict[str, Any] = {'motion': repaired}
        if return_mask:
            result['joint_mask'] = torch.from_numpy(joint_masks)
            result['mask'] = torch.from_numpy(mask_dim_out)
        return result
