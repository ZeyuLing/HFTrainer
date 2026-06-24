"""PhysFlowBundle: KIMODO-G1 generator wrapped for online-adversarial fine-tuning.

Design (validated by scripts/embodied/cursor_physflow_gen_reward_probe.py):
  * The 8B text encoder is NEVER loaded. KIMODO is built with TEXT_ENCODER=dummy
    and we feed pre-extracted ``text_feat`` ([B, 1, 4096]) straight into
    ``Kimodo._generate(...)``, so the dummy encoder is never invoked.
  * Only the inner diffusion denoiser (``kimodo.denoiser.model``) is trainable;
    everything else (motion_rep, diffusion buffers, skeleton) is frozen.
  * The full ``Kimodo`` object is held OUTSIDE the nn.Module registry (so it is
    not duplicated in the bundle state_dict / not DDP-wrapped); the trainable
    inner denoiser is registered normally so the runner can prepare it and the
    optimizer / checkpoint logic picks it up.

Atomic methods (Trainer/Pipeline call these, never ``forward``):
  * ``sample_latents`` -- run DDIM sampling from cached text_feat (no grad).
  * ``decode_latents`` / ``latents_to_qpos`` -- decode for the physics reward.
  * ``sft_loss`` -- reward-weighted x0 supervised loss on the denoiser.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


@MODEL_BUNDLES.register_module()
class PhysFlowBundle(ModelBundle):
    """KIMODO-G1 generator bundle for PhysFlow online-adversarial training."""

    def __init__(
        self,
        kimodo_model: str = "Kimodo-G1-RP-v1",
        checkpoint_dir: Optional[str] = None,
        hf_home: Optional[str] = None,
        device: str = "cuda",
        cfg_weight: Tuple[float, float] = (2.0, 2.0),
        cfg_type: str = "separated",
        sample_diffusion_steps: int = 30,
        offline: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.cfg_weight = list(cfg_weight)
        self.cfg_type = cfg_type
        self.sample_diffusion_steps = int(sample_diffusion_steps)

        hf_home = hf_home or os.path.join(_PROJECT_ROOT, "checkpoints", "kimodo")
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.pop("HUGGINGFACE_CACHE_DIR", None)
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Skip the 8B encoder entirely; conditioning is provided via text_feat.
        os.environ["TEXT_ENCODER"] = "dummy"
        os.environ["TEXT_ENCODER_MODE"] = "local"
        os.environ.setdefault("LOCAL_CACHE", "true")
        if checkpoint_dir:
            os.environ.setdefault("CHECKPOINT_DIR", checkpoint_dir)

        from hftrainer.models.motion.kimodo.network.exports.mujoco import MujocoQposConverter
        from hftrainer.models.motion.kimodo.network.model.load_model import load_model

        dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        kimodo = load_model(kimodo_model, device=dev, eval_mode=True)

        # Hold the full model OUTSIDE nn.Module registration (no duplicate params,
        # no DDP wrap on the composite). object.__setattr__ stores in __dict__.
        object.__setattr__(self, "_kimodo", kimodo)
        object.__setattr__(self, "_converter", MujocoQposConverter(kimodo.skeleton))

        # Freeze everything, then expose the inner denoiser as the trainable module.
        kimodo.requires_grad_(False)
        inner_denoiser = kimodo.denoiser.model  # un-CFG TwoStageDenoiser
        inner_denoiser.requires_grad_(True)
        self.denoiser = inner_denoiser
        self._trainable_modules.append("denoiser")
        self._save_ckpt_modules.append("denoiser")

        # Frozen reference copy of the denoiser (the *base* KIMODO policy). The
        # online-adversarial RAFT loop regresses the trainable denoiser toward
        # its own best-of-N samples, which -- without an anchor -- sharpens the
        # output distribution until sampling diversity collapses and the policy
        # drifts into an untrackable failure mode (observed: diversity gone by
        # step ~100, full collapse by step ~260). Keeping a frozen base and
        # adding an anchor MSE(pred, base_pred) regularizer prevents this drift.
        # Held OUTSIDE the nn.Module registry so it is never trained / saved /
        # DDP-wrapped.
        import copy as _copy

        try:
            base = _copy.deepcopy(inner_denoiser)
            base.requires_grad_(False)
            base.eval()
            object.__setattr__(self, "_base_denoiser", base)
        except Exception:
            object.__setattr__(self, "_base_denoiser", None)

        self.num_base_steps = int(kimodo.diffusion.num_base_steps)
        self.motion_dim = None  # set lazily on first sample
        self.fps = float(kimodo.fps)

    # ------------------------------------------------------------------ utils
    @property
    def kimodo(self):
        return self._kimodo

    def _device(self) -> torch.device:
        return next(self.denoiser.parameters()).device

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_latents(
        self,
        text_feat: torch.Tensor,        # [B, seq, 4096]
        text_pad_mask: torch.Tensor,    # [B, seq] bool
        lengths: torch.Tensor,          # [B] int
        diffusion_steps: Optional[int] = None,
        cfg_weight: Optional[List[float]] = None,
        cfg_type: Optional[str] = None,
    ) -> torch.Tensor:
        """Return denoised normalized latent motion [B, max_frames, D]."""
        device = self._device()
        text_feat = text_feat.to(device)
        text_pad_mask = text_pad_mask.to(device)
        lengths = lengths.to(device)
        max_frames = int(lengths.max().item())

        from hftrainer.models.motion.kimodo.network.motion_rep.feature_utils import length_to_mask

        pad_mask = length_to_mask(lengths).to(device)
        first_heading = torch.zeros(text_feat.shape[0], device=device)

        latent = self._kimodo._generate(
            texts=[""] * text_feat.shape[0],
            max_frames=max_frames,
            num_denoising_steps=diffusion_steps or self.sample_diffusion_steps,
            pad_mask=pad_mask,
            first_heading_angle=first_heading,
            motion_mask=None,
            observed_motion=None,
            cfg_weight=cfg_weight or self.cfg_weight,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            cfg_type=cfg_type or self.cfg_type,
        )
        self.motion_dim = int(latent.shape[-1])
        return latent

    @torch.no_grad()
    def latents_to_qpos(self, latent: torch.Tensor) -> np.ndarray:
        """Decode normalized latent [B, T, D] -> MuJoCo qpos numpy [B, T, 36]."""
        output = self._kimodo.motion_rep.inverse(latent, is_normalized=True, return_numpy=False)
        qpos = self._converter.dict_to_qpos(output, self._device())
        if torch.is_tensor(qpos):
            qpos = qpos.detach().cpu().numpy()
        return qpos

    def save_qpos_csv(self, qpos_sample: np.ndarray, csv_path: str) -> None:
        self._converter.save_csv(qpos_sample, csv_path)

    # -------------------------------------------------------------------- loss
    def sft_loss(
        self,
        text_feat: torch.Tensor,        # [B, seq, 4096]
        text_pad_mask: torch.Tensor,    # [B, seq] bool
        target_latent: torch.Tensor,    # [B, T, D] normalized x0 (detached)
        lengths: torch.Tensor,          # [B]
        sample_weights: Optional[torch.Tensor] = None,  # [B] reward weights
        good_mask: Optional[torch.Tensor] = None,        # [B] {0,1} accept filter
        anchor_weight: float = 0.0,                      # MSE(pred, base_pred) reg
    ) -> Dict[str, torch.Tensor]:
        """Reward-weighted x0 diffusion loss on the trainable denoiser.

        We reconstruct KIMODO's training objective (the inference repo ships no
        loss): sample t ~ U[0, num_base_steps), noise the target via q_sample,
        and regress the denoiser's clean prediction back to the target latent.

        Anti-collapse additions (RAFT/ReST):
          * ``good_mask`` -- only the SFT term for *accepted* (trackable, no-fall)
            targets contributes; rejected prompts get zero SFT weight so the
            policy is never pulled toward motions the robot fails to execute.
          * ``anchor_weight`` -- adds ``MSE(pred, base_pred)`` against the frozen
            base denoiser, keeping the policy from sharpening into a degenerate
            mode (preserves sampling diversity / prevents drift).
        """
        device = self._device()
        text_feat = text_feat.to(device)
        text_pad_mask = text_pad_mask.to(device)
        target = target_latent.to(device).detach()
        lengths = lengths.to(device)
        B, T, _ = target.shape

        # IMPORTANT: sampling mutated the diffusion buffers to a sub-schedule.
        # Reset to the full base schedule so q_sample / t-indexing match training.
        diffusion = self._kimodo.diffusion
        diffusion.calc_diffusion_vars(torch.arange(self.num_base_steps, device=device))

        t = torch.randint(0, self.num_base_steps, (B,), device=device)
        noise = torch.randn_like(target)
        x_t = diffusion.q_sample(target, t, noise)

        from hftrainer.models.motion.kimodo.network.motion_rep.feature_utils import length_to_mask

        pad_mask = length_to_mask(lengths).to(device)
        first_heading = torch.zeros(B, device=device)

        pred_clean = self.denoiser(
            x_t,
            pad_mask,
            text_feat,
            text_pad_mask,
            t,
            first_heading_angle=first_heading,
            motion_mask=None,
            observed_motion=None,
        )

        # masked per-sample MSE over valid frames
        D = pred_clean.shape[-1]
        frame_mask = pad_mask.unsqueeze(-1).to(pred_clean.dtype)  # [B, T, 1]
        err = (pred_clean - target) ** 2 * frame_mask
        denom = (frame_mask.sum(dim=(1, 2)) * D).clamp_min(1.0)
        per_sample = err.sum(dim=(1, 2)) / denom  # [B]

        # ----- reward-filtered SFT weight -----
        if good_mask is not None:
            gm = good_mask.to(device).to(per_sample.dtype)
        else:
            gm = torch.ones_like(per_sample)
        if sample_weights is not None:
            w = sample_weights.to(device).to(per_sample.dtype) * gm
        else:
            w = gm
        wsum = w.sum()
        if float(wsum) > 0:
            sft = (per_sample * w).sum() / wsum.clamp_min(1e-8)
        else:
            # No accepted target this step: contribute zero SFT gradient (but keep
            # the graph so the optimizer/clip path stays well-defined). The anchor
            # term below still pulls the policy back toward base.
            sft = (per_sample * 0.0).sum()

        n_good = gm.sum()
        sft_mse_logged = (per_sample * gm).sum() / n_good.clamp_min(1.0)
        out: Dict[str, torch.Tensor] = {
            "sft_mse": sft_mse_logged.detach(),
            "n_good": n_good.detach(),
        }

        loss = sft
        if anchor_weight and anchor_weight > 0 and self._base_denoiser is not None:
            base = self._base_denoiser.to(device)
            with torch.no_grad():
                base_pred = base(
                    x_t,
                    pad_mask,
                    text_feat,
                    text_pad_mask,
                    t,
                    first_heading_angle=first_heading,
                    motion_mask=None,
                    observed_motion=None,
                )
            anchor_err = (pred_clean - base_pred) ** 2 * frame_mask
            anchor_per = anchor_err.sum(dim=(1, 2)) / denom
            anchor = anchor_per.mean()
            loss = loss + float(anchor_weight) * anchor
            out["anchor_mse"] = anchor.detach()

        out["loss"] = loss
        return out
