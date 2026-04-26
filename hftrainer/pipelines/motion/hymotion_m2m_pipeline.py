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

import torch
import torch.nn.functional as F
from torch import Tensor

from hftrainer.registry import PIPELINES


def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


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

        do_cfg = self.text_guidance_scale > 1.0

        # For CFG: prepare null text embeddings for the unconditional branch.
        # Training convention (mask_text_cond): when CFG dropout fires, the
        # ctxt tensor is replaced by `null_ctxt_input.expand_as(ctxt)` BUT
        # `ctxt_mask_temporal` is KEPT AT THE ORIGINAL CAPTION'S LENGTH MASK
        # (only the values change, the attention coverage does not). So we
        # mirror that here: the null branch uses the same ctxt_mask_temporal
        # as the conditioned branch, but with null embedding values. A
        # previous version used `zeros(128) + first-token=null + mask with
        # only first token valid`, which the model never saw during training
        # and produced visibly distorted captioned outputs.
        if do_cfg:
            # null_ctxt must match ctxt_input's token-length (captioned branch
            # uses pad_len tokens, uncond branch uses 1 token — see above).
            # Earlier hard-coded pad_len (128) here crashed when combined
            # with the 1-token uncond branch via torch.cat (2026-04-21).
            ctxt_tokens = ctxt_input.shape[1]
            null_vtxt = self.bundle.null_vtxt_feat.to(dtype=model_dtype).expand(B, 1, -1)
            null_ctxt = self.bundle.null_ctxt_input.to(dtype=model_dtype).expand(B, ctxt_tokens, -1).contiguous()
            null_ctxt_mask = ctxt_mask_temporal  # SAME attention coverage

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
