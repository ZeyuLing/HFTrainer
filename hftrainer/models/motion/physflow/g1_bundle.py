"""PhysFlowG1Bundle: the G1-native HyMotion flow-matching generator wrapped for
the PhysFlow online-adversarial closed loop.

It subclasses :class:`HyMotionT2MBundle` so warm-start / ``from_config`` /
``load_state_dict_selective`` / ``predict_flow`` / ``denormalize_motion`` /
``null_vtxt_feat`` are all inherited unchanged.  On top of that it exposes the
four atomic methods the PhysFlow trainer/reward path expects, but in the 38-d
G1 flow-matching space instead of the KIMODO diffusion space:

  * ``sample_motion``   -- flow-matching ODE sampling from cached dual text
    embeddings (CLIP-L 768 ``vtxt`` + Qwen3 4096 ``ctxt``), returns NORMALIZED
    38-d motion (network space), no grad.
  * ``latents_to_qpos`` -- denormalize + :func:`decode_g1_to_qpos` -> MuJoCo
    qpos numpy ``[B, T, 36]`` (pos3 + quat_wxyz4 + 29 dof), exact, no SMPL
    retarget -- the same qpos layout the frozen judge consumes.
  * ``save_qpos_csv``   -- write qpos ``[T, 36]`` as a header-less, frame-column
    -less CSV, exactly what ``convert_g1_csv_to_proto.py`` parses
    (``--pos-units m --rot-format quat_wxyz --joint-units rad``).
  * ``sft_loss_g1``     -- reward-filtered flow-matching velocity loss toward
    the selected (trackable) sample, with an anchor MSE to a frozen copy of the
    generator captured at first use (anti-collapse, mirrors PhysFlowBundle).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from hftrainer.models.motion.hymotion_t2m.bundle import HyMotionT2MBundle
from hftrainer.models.motion.physflow.g1_repr import decode_g1_to_qpos
from hftrainer.registry import MODEL_BUNDLES


def _len_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    return (torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
            < lengths.unsqueeze(1))


@MODEL_BUNDLES.register_module()
class PhysFlowG1Bundle(HyMotionT2MBundle):
    """G1-native flow-matching generator for the PhysFlow online loop."""

    # Frames the generator was trained at; sampling pads to >= this like eval.
    TRAIN_FRAMES = 360

    def __init__(self, *args, sample_steps: int = 50, sample_guidance: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_steps = int(sample_steps)
        self.sample_guidance = float(sample_guidance)
        # frozen anchor (captured lazily AFTER the checkpoint is loaded so it
        # snapshots the warm-started/fine-tuned policy, not the random head).
        self._anchor_transformer = None

    def _device(self) -> torch.device:
        return next(self.motion_transformer.parameters()).device

    @property
    def _core_transformer(self):
        """The underlying MMDiT, unwrapping accelerate's DDP/FSDP wrapper.

        Under multi-GPU the runner replaces ``bundle.motion_transformer`` with a
        ``DistributedDataParallel`` wrapper. Calling it (``forward``) routes
        through DDP fine, but ATTRIBUTE access (``output_dim``) and ``deepcopy``
        must target the wrapped module, not the DDP shell.
        """
        mt = self.motion_transformer
        return getattr(mt, "module", mt)

    def _maybe_init_anchor(self) -> None:
        if self._anchor_transformer is None:
            try:
                anc = deepcopy(self._core_transformer)
                anc.requires_grad_(False)
                anc.eval()
                # held outside the nn.Module registry (no train / save / DDP wrap)
                object.__setattr__(self, "_anchor_transformer", anc)
            except Exception:
                object.__setattr__(self, "_anchor_transformer", False)  # disable

    # ------------------------------------------------------------ conditioning
    def _pack_ctxt(self, text_ctxt: List[Tensor], ctxt_len: Tensor, device, dtype):
        """List of (seq_i, 4096) -> padded (B, max_seq, 4096) + bool mask."""
        B = len(text_ctxt)
        max_seq = max(int(c.shape[0]) for c in text_ctxt)
        ctxt = torch.zeros(B, max_seq, self._ctxt_input_dim, dtype=dtype, device=device)
        for i, c in enumerate(text_ctxt):
            ctxt[i, :c.shape[0]] = c.to(device, dtype)
        return ctxt, _len_to_mask(ctxt_len.to(device), max_seq)

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_motion(
        self,
        text_vec: Tensor,            # (B, 1, 768)
        text_ctxt: List[Tensor],     # list of (seq_i, 4096)
        ctxt_len: Tensor,            # (B,)
        lengths: Tensor,             # (B,) int target frames
        num_steps: Optional[int] = None,
        guidance: Optional[float] = None,
        initial_noise: Optional[Tensor] = None,
        transformer=None,
        return_initial_noise: bool = False,
    ) -> Tensor:
        """Flow-matching ODE -> NORMALIZED 38-d motion (B, Lmax, 38)."""
        device = self._device()
        dtype = torch.float32
        num_steps = int(num_steps or self.sample_steps)
        guidance = float(guidance if guidance is not None else self.sample_guidance)

        vtxt = text_vec.to(device, dtype)
        ctxt, ctxt_mask = self._pack_ctxt(text_ctxt, ctxt_len, device, dtype)
        lengths = lengths.to(device)
        L = int(lengths.max().item())
        Lp = max(L, self.TRAIN_FRAMES)
        B = vtxt.shape[0]
        core = transformer
        motion_dim = getattr(core, "output_dim", self._core_transformer.output_dim)
        x_mask = _len_to_mask(lengths, Lp)

        def predict(x_input, ctxt_input, vtxt_input, timesteps, x_mask_temporal, ctxt_mask_temporal):
            if core is None:
                return self.predict_flow(
                    x_input=x_input, ctxt_input=ctxt_input, vtxt_input=vtxt_input,
                    timesteps=timesteps, x_mask_temporal=x_mask_temporal,
                    ctxt_mask_temporal=ctxt_mask_temporal)
            return core(
                x=x_input, ctxt_input=ctxt_input, vtxt_input=vtxt_input,
                timesteps=timesteps, x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=ctxt_mask_temporal, mask_density=None,
                task_emb=None)

        do_cfg = guidance > 1.0
        if do_cfg:
            null_vtxt = self.null_vtxt_feat.to(device, dtype).expand_as(vtxt)
            vtxt_cfg = torch.cat([null_vtxt, vtxt], 0)
            ctxt_cfg = torch.cat([ctxt, ctxt], 0)
            ctxt_mask_cfg = torch.cat([ctxt_mask, ctxt_mask], 0)
            x_mask_cfg = x_mask.repeat(2, 1)

        def fn(t_val, x):
            if do_cfg:
                xd = torch.cat([x, x], 0)
                xp = predict(
                    x_input=xd, ctxt_input=ctxt_cfg, vtxt_input=vtxt_cfg,
                    timesteps=t_val.expand(2 * B), x_mask_temporal=x_mask_cfg,
                    ctxt_mask_temporal=ctxt_mask_cfg)
                pu, pt = xp.chunk(2, 0)
                return pu + guidance * (pt - pu)
            return predict(
                x_input=x, ctxt_input=ctxt, vtxt_input=vtxt,
                timesteps=t_val.expand(B), x_mask_temporal=x_mask,
                ctxt_mask_temporal=ctxt_mask)

        if initial_noise is None:
            y0 = torch.randn(B, Lp, motion_dim, device=device, dtype=dtype)
        else:
            y0 = initial_noise.to(device=device, dtype=dtype)
            if y0.shape[0] != B or y0.shape[2] != motion_dim:
                raise ValueError(
                    f"initial_noise shape {tuple(y0.shape)} incompatible with "
                    f"B={B}, motion_dim={motion_dim}")
            if y0.shape[1] < Lp:
                y0 = F.pad(y0, (0, 0, 0, Lp - y0.shape[1]))
            elif y0.shape[1] > Lp:
                y0 = y0[:, :Lp]
        try:
            from torchdiffeq import odeint
            t = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)
            sampled = odeint(fn, y0, t, method='euler')[-1]
        except ImportError:
            x = y0
            dt = 1.0 / num_steps
            for i in range(num_steps):
                x = x + fn(torch.tensor(i * dt, device=device, dtype=dtype), x) * dt
            sampled = x
        sampled = sampled[:, :L, :]  # normalized (B, L, 38)
        if return_initial_noise:
            return sampled, y0
        return sampled

    @torch.no_grad()
    def latents_to_qpos(self, latent: Tensor) -> np.ndarray:
        """Normalized 38-d motion (B, L, 38) -> qpos numpy (B, L, 36)."""
        denorm = self.denormalize_motion(latent.to(self._device()).float())  # (B,L,38)
        qpos = []
        for b in range(denorm.shape[0]):
            qpos.append(decode_g1_to_qpos(denorm[b].cpu()).numpy())
        return np.stack(qpos, axis=0)

    @staticmethod
    def save_qpos_csv(qpos_sample: np.ndarray, csv_path: str) -> None:
        """qpos (T, 36) -> header-less, frame-column-less CSV for the converter."""
        arr = np.asarray(qpos_sample, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None]
        np.savetxt(csv_path, arr, delimiter=",")

    # -------------------------------------------------------------------- loss
    def sft_loss_g1(
        self,
        text_vec: Tensor,            # (B, 1, 768)
        text_ctxt: List[Tensor],     # list of (seq_i, 4096)
        ctxt_len: Tensor,            # (B,)
        target_motion: Tensor,       # (B, L, 38) NORMALIZED selected sample (detached)
        lengths: Tensor,             # (B,)
        good_mask: Optional[Tensor] = None,    # (B,) {0,1}
        sample_weights: Optional[Tensor] = None,  # (B,) reward weights
        anchor_weight: float = 0.0,
        gt_target: Optional[Tensor] = None,    # (B, Lg, 38) NORMALIZED ground-truth motion
        gt_lengths: Optional[Tensor] = None,   # (B,)
        gt_weight: float = 0.0,                # weight of the GT supervised term
    ) -> Dict[str, Tensor]:
        """Reward-filtered flow-matching velocity SFT toward the selected sample.

        x0 ~ N(0, I); t ~ U(0, 1); x_t = (1-t) x0 + t x1; the FM target velocity
        is (x1 - x0).  Loss = ||predict_flow(x_t, t) - (x1 - x0)||^2, masked to
        valid frames, reward-filtered by ``good_mask`` (rejected prompts give 0
        SFT gradient), plus optional anchor MSE to the frozen base generator.

        When ``gt_target`` / ``gt_weight`` are given a ground-truth supervised FM
        term is added.  Both the reward target and the GT target share the SAME
        text conditioning (the batch prompts), so they are concatenated along the
        batch dim and pushed through **one** ``predict_flow`` forward -- this is
        deliberate: under DDP a second grad-forward on the wrapped transformer
        would trip the reducer ("mark ready twice").  The total objective is
        ``reward_sft + gt_weight * gt_sft + anchor_weight * anchor`` (a sum of
        per-group means, not a blended average).
        """
        device = self._device()
        dtype = torch.float32
        vtxt = text_vec.to(device, dtype)
        ctxt, ctxt_mask = self._pack_ctxt(text_ctxt, ctxt_len, device, dtype)
        x1 = target_motion.to(device, dtype).detach()
        lengths = lengths.to(device)
        B, Lr, D = x1.shape

        use_gt = gt_target is not None and gt_weight and gt_weight > 0
        Lg = int(gt_target.shape[1]) if use_gt else 0
        L = max(Lr, Lg)
        if Lr < L:
            x1 = F.pad(x1, (0, 0, 0, L - Lr))
        x_mask = _len_to_mask(lengths, L)

        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=device, dtype=dtype)
        x_t = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1
        v_target = x1 - x0

        if use_gt:
            x1g = gt_target.to(device, dtype).detach()
            gt_lengths = gt_lengths.to(device)
            if Lg < L:
                x1g = F.pad(x1g, (0, 0, 0, L - Lg))
            x_mask_g = _len_to_mask(gt_lengths, L)
            x0g = torch.randn_like(x1g)
            tg = torch.rand(B, device=device, dtype=dtype)
            x_tg = (1.0 - tg.view(B, 1, 1)) * x0g + tg.view(B, 1, 1) * x1g
            v_target_g = x1g - x0g

            # one combined forward over [reward; gt] (DDP-safe single grad-forward)
            pred_cat = self.predict_flow(
                x_input=torch.cat([x_t, x_tg], dim=0),
                ctxt_input=torch.cat([ctxt, ctxt], dim=0),
                vtxt_input=torch.cat([vtxt, vtxt], dim=0),
                timesteps=torch.cat([t, tg], dim=0),
                x_mask_temporal=torch.cat([x_mask, x_mask_g], dim=0),
                ctxt_mask_temporal=torch.cat([ctxt_mask, ctxt_mask], dim=0),
            )
            pred_v, pred_v_g = pred_cat[:B], pred_cat[B:]
        else:
            pred_v = self.predict_flow(
                x_input=x_t, ctxt_input=ctxt, vtxt_input=vtxt,
                timesteps=t, x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask,
            )

        frame_mask = x_mask.unsqueeze(-1).to(dtype)            # (B, L, 1)
        err = (pred_v - v_target) ** 2 * frame_mask
        denom = (frame_mask.sum(dim=(1, 2)) * D).clamp_min(1.0)
        per_sample = err.sum(dim=(1, 2)) / denom               # (B,)

        gm = (good_mask.to(device, dtype) if good_mask is not None
              else torch.ones_like(per_sample))
        w = (sample_weights.to(device, dtype) * gm if sample_weights is not None else gm)
        wsum = w.sum()
        sft = (per_sample * w).sum() / wsum.clamp_min(1e-8) if float(wsum) > 0 \
            else (per_sample * 0.0).sum()

        n_good = gm.sum()
        out: Dict[str, Tensor] = {
            "sft_mse": ((per_sample * gm).sum() / n_good.clamp_min(1.0)).detach(),
            "n_good": n_good.detach(),
        }
        loss = sft

        if use_gt:
            fmg = x_mask_g.unsqueeze(-1).to(dtype)
            errg = (pred_v_g - v_target_g) ** 2 * fmg
            denomg = (fmg.sum(dim=(1, 2)) * D).clamp_min(1.0)
            gt_sft = (errg.sum(dim=(1, 2)) / denomg).mean()
            loss = loss + float(gt_weight) * gt_sft
            out["gt_mse"] = gt_sft.detach()

        if anchor_weight and anchor_weight > 0:
            self._maybe_init_anchor()
            anc = self._anchor_transformer
            if anc:
                with torch.no_grad():
                    base_v = anc(
                        x=x_t, ctxt_input=ctxt, vtxt_input=vtxt, timesteps=t,
                        x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask,
                        mask_density=None, task_emb=None,
                    )
                anchor_err = (pred_v - base_v) ** 2 * frame_mask
                anchor = (anchor_err.sum(dim=(1, 2)) / denom).mean()
                loss = loss + float(anchor_weight) * anchor
                out["anchor_mse"] = anchor.detach()

        out["loss"] = loss
        return out
