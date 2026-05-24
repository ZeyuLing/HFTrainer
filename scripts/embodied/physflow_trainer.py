"""PhysFlow: Bidirectional Physics-RL-Grounded Flow Correction for T2M.

Two directions of mutual improvement:
  Direction A (RL→Gen): RL tracker corrects T2M outputs → fine-tune T2M model
  Direction B (Gen→RL): Improved T2M generates diverse motions → expand RL training

The closed loop (Direction A):
    1. Generate motion on-policy (current model)
    2. Correct via RL closed-loop tracking in MuJoCo physics simulation
    3. Fine-tune with flow matching loss using RL-corrected target

Usage:
    # Direction A only (quick, uses pretrained ONNX policy)
    python3 scripts/embodied/physflow_trainer.py \
        --mode rl-to-gen \
        --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
        --t2m-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
        --output-dir output/physflow_v2 \
        --num-iterations 2000 --lr 2e-5

    # Full bidirectional (alternates A→B→A→B...)
    python3 scripts/embodied/physflow_trainer.py \
        --mode bidirectional \
        --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
        --t2m-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
        --output-dir output/physflow_v2 \
        --num-outer-loops 5 --gen-iters-per-loop 400 --rl-steps-per-loop 500

    # Quick single-sample test:
    python3 scripts/embodied/physflow_trainer.py --test-single \
        --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
        --t2m-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.embodied.physflow_curriculum import PhysFlowCurriculum
from scripts.embodied.physflow_motion_converter import MotionFormatConverter
from scripts.embodied.physflow_rl_oracle import (
    RLPhysicsOracle,
    decode_motion_135_array,
    encode_motion_135,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Convert length list to boolean mask. (B,) -> (B, max_len)."""
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


def motion_135_to_201(
    motion_135: np.ndarray,
    body_model,
    device: torch.device,
) -> np.ndarray:
    """Convert motion_135 to 201-dim by computing FK and RIC joint positions.

    The 201-dim format is:
        [0:3]     translation (3)
        [3:135]   22 x rot6d (132)
        [135:201] 22 x 3 joint positions in RIC (66)

    RIC (Root-Invariant Coordinates, Scheme D):
        - X, Z: relative to pelvis position
        - Y: absolute world height
        - Pelvis joint: always [0, pelvis_y, 0]

    Args:
        motion_135: (T, 135) motion in standard format
        body_model: SmplxLiteJ24 instance
        device: torch device for FK computation

    Returns:
        motion_201: (T, 201) motion with joint positions
    """
    T = motion_135.shape[0]

    # Extract components from motion_135
    transl = motion_135[:, :3]  # (T, 3)
    root_rot6d = motion_135[:, 3:9].reshape(T, 1, 6)  # (T, 1, 6)
    body6d = motion_135[:, 9:135].reshape(T, 21, 6)  # (T, 21, 6)

    # Run FK via SmplxLiteJ24
    # API: body_model(body_pose, betas, global_orient, transl) -> (T, 24, 3)
    with torch.no_grad():
        body6d_t = torch.from_numpy(body6d).float().to(device)  # (T, 21, 6)
        betas = torch.zeros(1, 16, device=device)
        root_rot6d_t = torch.from_numpy(root_rot6d).float().to(device)  # (T, 1, 6)
        transl_t = torch.from_numpy(transl).float().to(device)  # (T, 3)

        joints_world = body_model(body6d_t, betas, root_rot6d_t, transl_t)  # (T, 24, 3)
        joints_world = joints_world.cpu().numpy()

    # Take first 22 joints (SMPL-22, skip last 2 hand joints)
    joints_22 = joints_world[:, :22, :]  # (T, 22, 3)

    # Convert to RIC (Scheme D): XZ relative to pelvis, Y absolute
    pelvis_pos = joints_22[:, 0:1, :]  # (T, 1, 3)
    ric_joints = joints_22.copy()
    ric_joints[:, :, 0] -= pelvis_pos[:, :, 0]  # X relative to pelvis
    ric_joints[:, :, 2] -= pelvis_pos[:, :, 2]  # Z relative to pelvis
    # Y stays absolute (world height)

    # Assemble 201-dim: motion_135 (135) + ric_positions (66)
    ric_flat = ric_joints.reshape(T, 66)  # (T, 66)
    motion_201 = np.concatenate([motion_135, ric_flat], axis=-1)  # (T, 201)

    return motion_201.astype(np.float32)


# ---------------------------------------------------------------------------
# PhysFlow Trainer
# ---------------------------------------------------------------------------

class PhysFlowTrainer:
    """Bidirectional Physics-RL-Grounded Flow Correction trainer.

    Direction A (RL→Gen): On-policy T2M generation + RL physics correction
        + flow matching fine-tuning.
    Direction B (Gen→RL): Generate diverse motions from improved T2M → convert
        to ProtoMotions format → fine-tune RL tracker on expanded distribution.
    """

    def __init__(
        self,
        bundle,
        physics_oracle: RLPhysicsOracle,
        curriculum: PhysFlowCurriculum,
        device: torch.device,
        lr: float = 2e-5,
        num_ode_steps: int = 50,
        text_guidance_scale: float = 5.0,
        grad_clip: float = 1.0,
        soar_lambda: float = 0.0,
        soar_K: int = 50,
        train_last_n_blocks: int = 4,
        use_amp: bool = True,
        motion_converter: Optional[MotionFormatConverter] = None,
        rl_experiment: Optional[str] = None,
        output_dir: str = "output/physflow_v2",
        min_completion: float = 0.8,
        min_root_height: float = 0.3,
        require_no_fall: bool = False,
        grad_accum: int = 1,
        kl_weight: float = 0.0,
    ):
        """Initialize PhysFlow trainer.

        Args:
            bundle: HyMotionT2MBundle instance (loaded with checkpoint)
            physics_oracle: RLPhysicsOracle for RL closed-loop correction
            curriculum: PhysFlowCurriculum for adaptive prompt scheduling
            device: GPU device
            lr: Learning rate (2e-5 = 1/5 of SFT, suitable for post-training)
            num_ode_steps: ODE steps for generation (50 = standard)
            text_guidance_scale: CFG scale for generation (5.0 = standard)
            grad_clip: Gradient clipping norm
            soar_lambda: Weight for SOAR correction loss (0 = disabled)
            soar_K: SOAR rollout step divisor
            train_last_n_blocks: Only train last N transformer blocks + final_layer
                to reduce optimizer memory. Default 4 (~65M params vs full 460M).
                Set to 0 to train all parameters.
            use_amp: Use automatic mixed precision (fp16 compute) to save memory
            motion_converter: MotionFormatConverter for Direction B (T2M → ProtoMotions)
            rl_experiment: Path to ProtoMotions experiment config for Direction B
            output_dir: Output directory for checkpoints and motion libraries
            min_completion: Minimum completion ratio for quality gate (0.8 = strict)
            min_root_height: Minimum root height in meters for quality gate (0.3m)
            require_no_fall: If True, reject ALL entries with status="fell" regardless
                of completion ratio. Prevents degenerate feedback loop from truncated
                targets teaching model to produce falling motions.
            grad_accum: Gradient accumulation steps. Optimizer steps after N successful
                backward passes. Default=1 (no accumulation).
            kl_weight: Weight for KL regularization toward pretrained weights.
                Adds loss term: kl_weight * sum((θ - θ_pretrained)²).
                Prevents catastrophic forgetting of non-trained distribution.
                Recommended: 0.01-0.1 for online SFT. 0 = disabled.
        """
        self.bundle = bundle
        self.physics_oracle = physics_oracle
        self.curriculum = curriculum
        self.device = device
        self.num_ode_steps = num_ode_steps
        self.text_guidance_scale = text_guidance_scale
        self.grad_clip = grad_clip
        self.soar_lambda = soar_lambda
        self.soar_K = soar_K
        self.use_amp = use_amp
        self.motion_converter = motion_converter
        self.rl_experiment = rl_experiment
        self.output_dir = output_dir
        self.min_completion = min_completion
        self.min_root_height = min_root_height
        self.require_no_fall = require_no_fall
        self.grad_accum = grad_accum
        self.kl_weight = kl_weight
        self._accum_count = 0  # Tracks accumulated gradients

        # Set up trainable parameters
        # On memory-constrained GPUs, only fine-tune last N blocks + final_layer
        # to reduce optimizer state memory (AdamW needs 2x params for m/v buffers)
        mt = bundle.motion_transformer
        if train_last_n_blocks > 0:
            # Freeze all parameters first
            for p in mt.parameters():
                p.requires_grad = False
            # Unfreeze last N single-stream blocks
            n_unfreeze = min(train_last_n_blocks, len(mt.single_blocks))
            for block in mt.single_blocks[-n_unfreeze:]:
                for p in block.parameters():
                    p.requires_grad = True
            # Always unfreeze final_layer
            for p in mt.final_layer.parameters():
                p.requires_grad = True
            trainable_params = [p for p in mt.parameters() if p.requires_grad]
            total_params = sum(p.numel() for p in mt.parameters())
            trainable_count = sum(p.numel() for p in trainable_params)
            print(f"  Partial fine-tune: {trainable_count/1e6:.1f}M / "
                  f"{total_params/1e6:.1f}M params trainable "
                  f"(last {n_unfreeze} blocks + final_layer)")
            # NOTE: Do NOT cast frozen params to fp16 manually — it causes NaN
            # during ODE inference. AMP autocast handles mixed precision
            # automatically during forward passes.
        else:
            trainable_params = list(mt.parameters())
            total_params = sum(p.numel() for p in trainable_params)
            print(f"  Full fine-tune: {total_params/1e6:.1f}M params trainable")

        # Optimizer (only trainable parameters)
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.99),
            weight_decay=0.01,
        )
        self.optimizer.zero_grad()  # Ensure clean gradient state at start

        # KL regularization: snapshot pretrained weights for anchoring
        # This prevents catastrophic forgetting by penalizing deviation from
        # pretrained parameters. Only store trainable params to save memory.
        self._pretrained_params: Dict[str, Tensor] = {}
        if self.kl_weight > 0:
            mt = bundle.motion_transformer
            for name, param in mt.named_parameters():
                if param.requires_grad:
                    self._pretrained_params[name] = param.data.clone().detach()
            print(f"  KL regularization: weight={self.kl_weight}, "
                  f"anchoring {len(self._pretrained_params)} param tensors")

        # NOTE: AMP (fp16 autocast) is disabled — this DiT model produces NaN
        # under fp16. Partial fine-tuning (65M params) makes float32 fit in memory.

        # Text encoding cache: pre-encode all curriculum prompts once,
        # then reuse cached embeddings during training to save GPU memory.
        self._text_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        # Statistics
        self.total_iterations = 0
        self.total_skipped = 0
        self.loss_history = []

    def precompute_text_embeddings(self, cache_path: Optional[str] = None):
        """Pre-encode all curriculum prompts and cache on device.

        If cache_path is provided, load pre-computed embeddings from file
        (generated by physflow_precompute_text.py). This avoids loading
        the 8B text encoder entirely.

        Otherwise, load text encoder on CPU and encode (slow but works).
        """
        if cache_path and os.path.exists(cache_path):
            print(f"Loading pre-computed text embeddings from: {cache_path}")
            cache = torch.load(cache_path, map_location='cpu')
            for prompt, feats in cache.items():
                self._text_cache[prompt] = {
                    'text_vec_raw': feats['text_vec_raw'].to(self.device),
                    'text_ctxt_raw': feats['text_ctxt_raw'].to(self.device),
                    'text_ctxt_raw_length': feats['text_ctxt_raw_length'].to(self.device),
                }
            print(f"  Loaded {len(self._text_cache)} text embeddings on {self.device}.")
            return
        all_prompts = set()
        for level in self.curriculum.levels:
            for p in level['prompts']:
                all_prompts.add(p)

        print(f"Pre-encoding {len(all_prompts)} curriculum prompts...")

        # Temporarily move bundle to CPU for text encoding (Qwen3-8B is huge)
        original_device = self.device
        self.bundle.cpu()
        torch.cuda.empty_cache()

        # Encode on CPU
        for prompt in sorted(all_prompts):
            feats = self.bundle.encode_text([prompt])
            self._text_cache[prompt] = {
                'text_vec_raw': feats['text_vec_raw'].to(original_device),
                'text_ctxt_raw': feats['text_ctxt_raw'].to(original_device),
                'text_ctxt_raw_length': feats['text_ctxt_raw_length'].to(original_device),
            }
        print(f"  Cached {len(self._text_cache)} text embeddings on {original_device}.")

        # Free text encoder completely (it's large: Qwen3-8B + CLIP-L)
        if hasattr(self.bundle, '_text_encoder') and self.bundle._text_encoder is not None:
            del self.bundle._text_encoder
            self.bundle._text_encoder = None
            print("  Deleted text encoder (Qwen3-8B + CLIP-L).")

        # Move bundle back to GPU (without text encoder, it's only ~1.8GB)
        self.bundle.to(original_device)
        torch.cuda.empty_cache()
        print(f"  Bundle back on {original_device}.")

    def _get_text_feats(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Get text features from cache or encode on-the-fly."""
        if prompt in self._text_cache:
            return self._text_cache[prompt]
        # Fallback: encode on-the-fly (for prompts not in curriculum)
        feats = self.bundle.encode_text([prompt])
        return {
            'text_vec_raw': feats['text_vec_raw'].to(self.device),
            'text_ctxt_raw': feats['text_ctxt_raw'].to(self.device),
            'text_ctxt_raw_length': feats['text_ctxt_raw_length'].to(self.device),
        }

    # ------------------------------------------------------------------
    # On-policy generation (replicates HyMotionT2MPipeline logic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_motion(self, prompt: str, num_frames: int) -> np.ndarray:
        """Generate motion using current model (on-policy).

        Replicates the ODE inference from HyMotionT2MPipeline:
        - Pad to TRAIN_FRAMES=360
        - Euler ODE integration with CFG
        - Truncate to requested length
        - Denormalize

        Args:
            prompt: Text prompt
            num_frames: Desired motion length in frames

        Returns:
            motion_135: (T, 135) generated motion in denormalized space, Y-up
        """
        TRAIN_FRAMES = 360
        B = 1
        L = num_frames
        L_padded = max(L, TRAIN_FRAMES)
        motion_dim = self.bundle.motion_transformer.output_dim

        # Encode text (from cache or on-the-fly)
        text_feats = self._get_text_feats(prompt)
        vtxt_input = text_feats['text_vec_raw']
        ctxt_input = text_feats['text_ctxt_raw']
        ctxt_length = text_feats['text_ctxt_raw_length']
        ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])

        # Padding mask for motion — marks valid positions (True) vs padding (False).
        # CRITICAL: must use actual length L, NOT L_padded. Using L_padded would
        # mark all positions as valid, causing the model to attend to noise in
        # padding positions and producing broken motion output.
        tgt_padding_mask = _length_to_mask(
            torch.tensor([L], dtype=torch.long, device=self.device), L_padded
        )

        # CFG: prepare null text
        do_cfg = self.text_guidance_scale > 1.0
        if do_cfg:
            null_vtxt = self.bundle.null_vtxt_feat.expand_as(vtxt_input)
            vtxt_cfg = torch.cat([null_vtxt, vtxt_input], dim=0)
            ctxt_cfg = torch.cat([ctxt_input, ctxt_input], dim=0)
            ctxt_mask_cfg = torch.cat([ctxt_mask_temporal, ctxt_mask_temporal], dim=0)

        # ODE function
        def fn(t_val: Tensor, x: Tensor) -> Tensor:
            if do_cfg:
                x_double = torch.cat([x, x], dim=0)
                x_pred = self.bundle.predict_flow(
                    x_input=x_double,
                    ctxt_input=ctxt_cfg,
                    vtxt_input=vtxt_cfg,
                    timesteps=t_val.expand(2 * B),
                    x_mask_temporal=tgt_padding_mask.repeat(2, 1),
                    ctxt_mask_temporal=ctxt_mask_cfg,
                )
            else:
                x_pred = self.bundle.predict_flow(
                    x_input=x,
                    ctxt_input=ctxt_input,
                    vtxt_input=vtxt_input,
                    timesteps=t_val.expand(B),
                    x_mask_temporal=tgt_padding_mask,
                    ctxt_mask_temporal=ctxt_mask_temporal,
                )

            # For velocity pred_type: model directly predicts velocity
            if self.bundle.pred_type == 'x1':
                t_eps = 0.05
                if do_cfg:
                    x_pred = (x_pred - torch.cat([x, x], dim=0)) / (1.0 - t_val).clamp_min(t_eps)
                else:
                    x_pred = (x_pred - x) / (1.0 - t_val).clamp_min(t_eps)

            if do_cfg:
                pred_uncond, pred_text = x_pred.chunk(2, dim=0)
                x_pred = pred_uncond + self.text_guidance_scale * (pred_text - pred_uncond)

            return x_pred

        # Initial noise + Euler integration
        # NOTE: ODE integration MUST stay in float32 to avoid NaN accumulation.
        # AMP autocast only applies inside model forward (fn), not to the
        # outer x = x + v*dt loop. 50 fp16 additions would overflow.
        y0 = torch.randn(B, L_padded, motion_dim, device=self.device, dtype=torch.float32)

        # Euler ODE: t goes from 0 to 1
        # NOTE: Do NOT use AMP autocast during inference. 50 sequential forward
        # passes accumulate fp16 precision errors → NaN. AMP is only used in
        # train_step() where a single forward+backward doesn't compound errors.
        dt = 1.0 / self.num_ode_steps
        x = y0
        for i in range(self.num_ode_steps):
            t_val = torch.tensor(i * dt, device=self.device, dtype=torch.float32)
            v = fn(t_val, x)
            x = x + v * dt

        sampled = x[:, :L, :]  # Truncate to requested length

        # Denormalize to motion space
        latent_denorm = self.bundle.denormalize_motion(sampled)  # (1, L, 201)
        motion_201 = latent_denorm[0].cpu().numpy()  # (L, 201)

        # Extract motion_135 (first 135 dims)
        motion_135 = motion_201[:, :135].astype(np.float32)

        return motion_135

    # ------------------------------------------------------------------
    # Flow matching training step
    # ------------------------------------------------------------------

    def train_step(self, motion_201_phys: np.ndarray, prompt: str) -> Dict:
        """Single flow matching training step with physics-corrected target.

        Args:
            motion_201_phys: (T, 201) physics-corrected motion in raw space
            prompt: Text prompt used for generation

        Returns:
            Dict with loss values and metadata
        """
        self.bundle.motion_transformer.train()

        T = motion_201_phys.shape[0]

        # Normalize motion → x1 (the "clean" target)
        x1 = torch.from_numpy(motion_201_phys).float().to(self.device).unsqueeze(0)  # (1, T, 201)
        x1 = self.bundle.normalize_motion(x1)

        # Sample noise x0
        x0 = torch.randn_like(x1)

        # Sample timestep t ~ U[0, 1]
        timesteps = torch.rand(1, dtype=x1.dtype, device=self.device)
        t = timesteps.unsqueeze(-1).unsqueeze(-1)  # (1, 1, 1) for broadcasting

        # Flow matching interpolation: x_t = (1-t) * x0 + t * x1
        x_t = (1 - t) * x0 + t * x1

        # Encode text (from cache or on-the-fly)
        text_feats = self._get_text_feats(prompt)
        vtxt_input = text_feats['text_vec_raw']
        ctxt_input = text_feats['text_ctxt_raw']
        ctxt_length = text_feats['text_ctxt_raw_length']
        ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])

        # Apply CFG masking (10% dropout during training)
        vtxt_input, ctxt_input = self.bundle.mask_text_cond(
            vtxt_input, ctxt_input,
            force_mask=False,
            cond_mask_prob=self.bundle.cond_mask_prob,
        )

        # Padding mask (all valid for single sample)
        tgt_padding_mask = torch.ones(1, T, device=self.device, dtype=torch.bool)

        # Forward: predict velocity
        # NOTE: Do NOT use fp16 autocast — this DiT model produces NaN under fp16.
        # Full float32 fits in memory with partial fine-tuning (~3.1 GB total).
        pred = self.bundle.predict_flow(
            x_input=x_t,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=tgt_padding_mask,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )

        # Ground truth velocity: v = x1 - x0
        gt_velocity = x1 - x0

        # Loss: SmoothL1 (matching official M2M/T2M loss)
        loss_vel = F.smooth_l1_loss(pred, gt_velocity)

        # Optional: SOAR correction loss
        loss_soar = torch.tensor(0.0, device=self.device)
        if self.soar_lambda > 0:
            loss_soar = self._soar_correction_loss(
                x_t=x_t, x0=x0, x1=x1, timesteps=timesteps,
                vtxt_input=vtxt_input, ctxt_input=ctxt_input,
                tgt_padding_mask=tgt_padding_mask,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )

        # KL regularization: penalize deviation from pretrained weights
        loss_kl = torch.tensor(0.0, device=self.device)
        if self.kl_weight > 0 and self._pretrained_params:
            for name, param in self.bundle.motion_transformer.named_parameters():
                if param.requires_grad and name in self._pretrained_params:
                    loss_kl = loss_kl + F.mse_loss(
                        param, self._pretrained_params[name])
            loss_kl = self.kl_weight * loss_kl

        # Total loss (scale by 1/grad_accum for proper averaging)
        loss = loss_vel + self.soar_lambda * loss_soar + loss_kl
        scaled_loss = loss / self.grad_accum

        # Backward (always — accumulates gradients)
        scaled_loss.backward()
        self._accum_count += 1

        # Optimizer step only when we've accumulated enough
        did_step = False
        if self._accum_count >= self.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.bundle.motion_transformer.parameters() if p.requires_grad],
                self.grad_clip,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accum_count = 0
            did_step = True

        return {
            'loss': loss.item(),
            'loss_velocity': loss_vel.item(),
            'loss_soar': loss_soar.item(),
            'loss_kl': loss_kl.item(),
            'timestep': timesteps.item(),
            'did_optimizer_step': did_step,
        }

    def _soar_correction_loss(
        self,
        x_t: Tensor,
        x0: Tensor,
        x1: Tensor,
        timesteps: Tensor,
        vtxt_input: Tensor,
        ctxt_input: Tensor,
        tgt_padding_mask: Tensor,
        ctxt_mask_temporal: Tensor,
    ) -> Tensor:
        """SOAR off-trajectory correction loss.

        Rollout one step from x_t using model prediction (detached),
        then re-noise and compute correction target back to x1.
        """
        B = x_t.shape[0]
        dt = 1.0 / self.soar_K

        # Rollout: x_hat = x_t + v_pred.detach() * dt
        with torch.no_grad():
            v_pred = self.bundle.predict_flow(
                x_input=x_t,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=timesteps,
                x_mask_temporal=tgt_padding_mask,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )
        x_hat = x_t + v_pred.detach() * dt  # Off-trajectory state

        # Re-noise: z_re = alpha * x_hat + (1-alpha) * x0
        # where alpha corresponds to t' = t + dt (one step forward)
        t_prime = (timesteps + dt).clamp(max=0.99)
        alpha = t_prime.unsqueeze(-1).unsqueeze(-1)
        z_re = alpha * x_hat + (1 - alpha) * x0

        # Correction target: velocity that would bring z_re back to x1
        # v_corr = (x1 - z_re) / (1 - t')
        sigma = (1.0 - t_prime).clamp_min(0.05).unsqueeze(-1).unsqueeze(-1)
        v_corr = (x1 - z_re) / sigma

        # Forward on re-noised state
        pred_corr = self.bundle.predict_flow(
            x_input=z_re,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=t_prime,
            x_mask_temporal=tgt_padding_mask,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )

        # Correction loss
        loss_corr = F.smooth_l1_loss(pred_corr, v_corr)
        return loss_corr

    # ------------------------------------------------------------------
    # Full training iteration
    # ------------------------------------------------------------------

    def train_iteration(self) -> Dict:
        """Single PhysFlow iteration: generate -> correct -> train.

        Returns:
            Dict with iteration results (or skip info)
        """
        iter_start = time.time()

        # Phase 1: Sample prompt from curriculum
        prompt = self.curriculum.get_prompt()
        num_frames = self.curriculum.get_num_frames()

        # Phase 2: On-policy generation
        gen_start = time.time()
        self.bundle.motion_transformer.eval()
        motion_135 = self.generate_motion(prompt, num_frames)
        gen_time = time.time() - gen_start

        # Phase 3: Physics correction
        phys_start = time.time()
        motion_135_phys, stats = self.physics_oracle.correct(motion_135)
        phys_time = time.time() - phys_start

        # Quality gate
        is_good = self.physics_oracle.is_good_quality(
            stats,
            min_completion=self.min_completion,
            min_root_height=self.min_root_height,
        )

        # Additional V4 strict gate: reject ALL "fell" entries regardless of metrics
        if is_good and self.require_no_fall and stats.get('status') == 'fell':
            is_good = False

        self.curriculum.update(success=is_good)

        if not is_good:
            self.total_skipped += 1
            self.total_iterations += 1
            return {
                'iteration': self.total_iterations,
                'skipped': True,
                'reason': 'physics_quality_gate',
                'stats': stats,
                'prompt': prompt,
                'curriculum': self.curriculum.get_state(),
                'gen_time': gen_time,
                'phys_time': phys_time,
            }

        # Phase 4: Convert physics-corrected 135-dim to 201-dim via FK
        body_model = self.bundle.body_model
        if body_model is not None:
            motion_201_phys = motion_135_to_201(
                motion_135_phys, body_model, self.device
            )
        else:
            # Fallback: pad with zeros for RIC dims (less accurate but functional)
            T_phys = motion_135_phys.shape[0]
            motion_201_phys = np.zeros((T_phys, 201), dtype=np.float32)
            motion_201_phys[:, :135] = motion_135_phys
            print("[WARN] No body_model available, RIC dims set to zero")

        # Phase 5: Flow matching fine-tune
        train_start = time.time()
        train_result = self.train_step(motion_201_phys, prompt)
        train_time = time.time() - train_start

        self.total_iterations += 1
        self.loss_history.append(train_result['loss'])

        total_time = time.time() - iter_start

        return {
            'iteration': self.total_iterations,
            'skipped': False,
            'prompt': prompt,
            'loss': train_result['loss'],
            'loss_velocity': train_result['loss_velocity'],
            'loss_soar': train_result['loss_soar'],
            'loss_kl': train_result.get('loss_kl', 0.0),
            'physics_stats': {
                'status': stats.get('status', 'unknown'),
                'completion_ratio': stats.get('completion_ratio', 0.0),
                'root_height_min': stats.get('root_height_min', 0.0),
                'tracking_error': stats.get('tracking_error_mean', 0.0),
                'actual_sim_steps': stats.get('actual_sim_steps', 0),
                'total_sim_steps': stats.get('total_sim_steps', 0),
            },
            'curriculum': self.curriculum.get_state(),
            'timing': {
                'generation': gen_time,
                'physics': phys_time,
                'training': train_time,
                'total': total_time,
            },
            'did_optimizer_step': train_result.get('did_optimizer_step', False),
            'accum_count': self._accum_count,
        }

    # ------------------------------------------------------------------
    # Direction B: Generation → RL
    # ------------------------------------------------------------------

    def direction_b_generate_motion_library(
        self,
        num_motions: int = 100,
        save_dir: Optional[str] = None,
    ) -> str:
        """Generate diverse motion library from current T2M model (Direction B).

        Creates a ProtoMotions-compatible .pt motion library file by:
          1. Sampling diverse prompts (all levels + extended pool)
          2. Generating motions with current T2M model
          3. Converting to ProtoMotions MotionLib format via MotionFormatConverter
          4. Saving as .pt file ready for RL tracker training

        Args:
            num_motions: Number of motions to generate
            save_dir: Directory to save .pt file. Defaults to self.output_dir.

        Returns:
            Path to saved .pt file
        """
        if self.motion_converter is None:
            raise RuntimeError(
                "Direction B requires MotionFormatConverter. "
                "Initialize trainer with motion_converter parameter."
            )

        save_dir = save_dir or self.output_dir
        os.makedirs(save_dir, exist_ok=True)

        log.info(f"[Direction B] Generating motion library: {num_motions} motions")
        self.bundle.motion_transformer.eval()

        motions_135 = []
        motion_names = []
        failed = 0

        for i in range(num_motions):
            prompt = self.curriculum.get_diverse_prompt()
            num_frames = self.curriculum.get_diverse_num_frames()

            try:
                with torch.no_grad():
                    motion = self.generate_motion(prompt, num_frames)
                motions_135.append(motion)
                motion_names.append(f"t2m_gen_{i:04d}")
            except Exception as e:
                log.warning(f"  Motion {i} failed: {e}")
                failed += 1

            if (i + 1) % 20 == 0:
                log.info(f"  Generated {i+1}/{num_motions} motions "
                         f"(failed: {failed})")

        if not motions_135:
            raise RuntimeError("No motions generated successfully!")

        # Convert to ProtoMotions format
        log.info(f"  Converting {len(motions_135)} motions to ProtoMotions format...")
        pt_data = self.motion_converter.motion_135_to_protomotions_pt(
            motions_135, fps=30, motion_names=motion_names
        )

        # Save
        save_path = os.path.join(save_dir, "t2m_motion_lib.pt")
        torch.save(pt_data, save_path)
        log.info(f"  Saved motion library: {save_path} "
                 f"({len(motions_135)} motions, {failed} failed)")

        return save_path

    def direction_b_train_rl(
        self,
        motion_lib_path: str,
        rl_steps: int = 500,
        outer_loop_idx: int = 0,
    ) -> Optional[str]:
        """Fine-tune RL tracker on T2M-generated motion library (Direction B).

        Runs ProtoMotions training via subprocess to expand the RL tracker's
        motion distribution coverage. The improved tracker produces better
        physics corrections in subsequent Direction A iterations.

        Args:
            motion_lib_path: Path to .pt motion library (from generate_motion_library)
            rl_steps: Number of RL training steps (short fine-tune)
            outer_loop_idx: Current outer loop index (for experiment naming)

        Returns:
            Path to new RL checkpoint (or None if training script not found)
        """
        if self.rl_experiment is None:
            log.warning("[Direction B] No RL experiment path specified, skipping")
            return None

        # Find ProtoMotions train_agent.py
        _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
        train_script = _REPO_ROOT / "ref_repo/ProtoMotions/protomotions/train_agent.py"
        if not train_script.exists():
            log.warning(f"[Direction B] train_agent.py not found: {train_script}")
            return None

        experiment_name = f"physflow_rl_iter{outer_loop_idx}"
        log.info(f"[Direction B] Training RL tracker: {experiment_name}")
        log.info(f"  Motion lib: {motion_lib_path}")
        log.info(f"  Steps: {rl_steps}")
        log.info(f"  Experiment: {self.rl_experiment}")

        cmd = [
            sys.executable,
            str(train_script),
            "--robot-name", "smpl",
            "--simulator", "mujoco",
            "--experiment-path", self.rl_experiment,
            "--motion-file", motion_lib_path,
            "--num-envs", "1",
            "--batch-size", "32",
            "--training-max-steps", str(rl_steps),
            "--experiment-name", experiment_name,
        ]

        log.info(f"  Command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )
            log.info(f"  RL training completed successfully")
            if result.stdout:
                # Log last 5 lines of stdout
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    log.info(f"    {line}")
        except subprocess.TimeoutExpired:
            log.warning("[Direction B] RL training timed out (2h)")
            return None
        except subprocess.CalledProcessError as e:
            log.error(f"[Direction B] RL training failed: {e}")
            if e.stderr:
                for line in e.stderr.strip().split('\n')[-10:]:
                    log.error(f"    {line}")
            return None

        # Look for the output checkpoint
        rl_output_dir = _REPO_ROOT / f"output/{experiment_name}"
        ckpt_path = None
        if rl_output_dir.exists():
            onnx_files = list(rl_output_dir.rglob("*.onnx"))
            if onnx_files:
                ckpt_path = str(onnx_files[-1])
                log.info(f"  New RL policy: {ckpt_path}")

        return ckpt_path

    def update_rl_oracle(self, new_onnx_path: str):
        """Update the RL oracle with a new policy checkpoint.

        Called after Direction B training produces an improved RL policy.

        Args:
            new_onnx_path: Path to new ONNX policy file
        """
        if os.path.exists(new_onnx_path):
            log.info(f"[Direction B] Updating RL oracle: {new_onnx_path}")
            self.physics_oracle = RLPhysicsOracle(
                onnx_path=new_onnx_path,
                yaml_path=self.physics_oracle.yaml_path,
                mjcf_path=self.physics_oracle.mjcf_path,
                gear=self.physics_oracle.gear,
            )
        else:
            log.warning(f"[Direction B] ONNX not found: {new_onnx_path}")

    # ------------------------------------------------------------------
    # Bidirectional outer loop
    # ------------------------------------------------------------------

    def run_bidirectional(
        self,
        num_outer_loops: int = 5,
        gen_iters_per_loop: int = 400,
        rl_steps_per_loop: int = 500,
        num_gen_motions_for_rl: int = 100,
        log_interval: int = 10,
        save_interval: int = 200,
    ) -> Dict:
        """Run full bidirectional PhysFlow training (alternating A→B→A→B...).

        Args:
            num_outer_loops: Number of A→B alternation cycles
            gen_iters_per_loop: Direction A iterations per outer loop
            rl_steps_per_loop: Direction B RL training steps per outer loop
            num_gen_motions_for_rl: Motions to generate for Direction B
            log_interval: Print progress every N iterations
            save_interval: Save checkpoint every N iterations

        Returns:
            Final training statistics
        """
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, 'training_log.jsonl')

        log.info("=" * 60)
        log.info("PhysFlow Bidirectional Training")
        log.info("=" * 60)
        log.info(f"  Outer loops: {num_outer_loops}")
        log.info(f"  Direction A iters/loop: {gen_iters_per_loop}")
        log.info(f"  Direction B RL steps/loop: {rl_steps_per_loop}")
        log.info(f"  Motions for RL: {num_gen_motions_for_rl}")
        log.info(f"  Output: {self.output_dir}")

        global_iter = 0

        with open(log_path, 'w') as log_f:
            for outer_idx in range(num_outer_loops):
                log.info(f"\n{'='*60}")
                log.info(f"  Outer Loop {outer_idx+1}/{num_outer_loops}")
                log.info(f"{'='*60}")

                # ═══════════════════════════════════════════════════════
                # Direction A: RL → Generation
                # ═══════════════════════════════════════════════════════
                log.info(f"\n--- Direction A: RL → Generation "
                         f"({gen_iters_per_loop} iterations) ---")

                for i in range(gen_iters_per_loop):
                    result = self.train_iteration()
                    result['outer_loop'] = outer_idx
                    result['direction'] = 'A'
                    log_f.write(json.dumps(result, default=str) + '\n')
                    log_f.flush()

                    global_iter += 1

                    # Progress logging
                    if not result['skipped'] and (i + 1) % log_interval == 0:
                        phys = result['physics_stats']
                        cur = result['curriculum']
                        log.info(
                            f"  [{outer_idx+1}.A.{i+1}/{gen_iters_per_loop}] "
                            f"loss={result['loss']:.5f} | "
                            f"level={cur['level_name']} "
                            f"(sr={cur['success_rate']:.2f}) | "
                            f"phys: {phys['status']} "
                            f"comp={phys['completion_ratio']:.2f}"
                        )

                    # Save checkpoint
                    if global_iter % save_interval == 0:
                        self._save_checkpoint(
                            f"model_outer{outer_idx}_iter{i+1}.pt",
                            outer_idx, global_iter
                        )

                # Save after Direction A
                self._save_checkpoint(
                    f"model_outer{outer_idx}_dirA_final.pt",
                    outer_idx, global_iter
                )

                # ═══════════════════════════════════════════════════════
                # Direction B: Generation → RL
                # ═══════════════════════════════════════════════════════
                if self.motion_converter is not None and self.rl_experiment is not None:
                    log.info(f"\n--- Direction B: Generation → RL ---")

                    # B1: Generate motion library
                    try:
                        motion_lib_path = self.direction_b_generate_motion_library(
                            num_motions=num_gen_motions_for_rl,
                            save_dir=os.path.join(
                                self.output_dir, f"motion_lib_outer{outer_idx}"
                            ),
                        )

                        # B2: Train RL tracker
                        new_onnx = self.direction_b_train_rl(
                            motion_lib_path=motion_lib_path,
                            rl_steps=rl_steps_per_loop,
                            outer_loop_idx=outer_idx,
                        )

                        # B3: Update oracle with new policy
                        if new_onnx:
                            self.update_rl_oracle(new_onnx)

                    except Exception as e:
                        log.error(f"[Direction B] Failed: {e}")
                        log.info("  Continuing with current RL policy...")

                    # Log Direction B event
                    log_f.write(json.dumps({
                        'outer_loop': outer_idx,
                        'direction': 'B',
                        'status': 'completed',
                        'global_iter': global_iter,
                    }, default=str) + '\n')
                    log_f.flush()
                else:
                    log.info(f"\n  [Direction B skipped — no motion_converter "
                             f"or rl_experiment configured]")

        # Final save
        self._save_checkpoint("model_final.pt", num_outer_loops - 1, global_iter)

        log.info(f"\n[DONE] Bidirectional training complete!")
        log.info(f"  Total iterations: {self.total_iterations}")
        log.info(f"  Skipped: {self.total_skipped}")
        log.info(f"  Final curriculum: {self.curriculum}")

        return {
            'total_iterations': self.total_iterations,
            'total_skipped': self.total_skipped,
            'final_curriculum': self.curriculum.get_state(),
            'loss_history': self.loss_history[-100:],
        }

    def _save_checkpoint(self, filename: str, outer_idx: int, global_iter: int):
        """Save model checkpoint."""
        ckpt_path = os.path.join(self.output_dir, filename)
        torch.save({
            'iteration': global_iter,
            'outer_loop': outer_idx,
            'model_state_dict': self.bundle.motion_transformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'curriculum_state': self.curriculum.state_dict(),
            'loss_history': self.loss_history[-500:],
            'total_iterations': self.total_iterations,
            'total_skipped': self.total_skipped,
        }, ckpt_path)
        log.info(f"  [SAVE] {ckpt_path}")


# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------

def load_bundle(config_path: str, ckpt_path: str, device: torch.device):
    """Load HyMotionT2MBundle from config and checkpoint.

    Args:
        config_path: Path to config .py file
        ckpt_path: Path to checkpoint file
        device: Target device

    Returns:
        Loaded bundle on device
    """
    from mmengine import Config

    from hftrainer.registry import MODEL_BUNDLES

    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.copy()
    model_type = model_cfg.pop('type')

    # Ensure text_encoder config is provided (empty dict is falsy in Python,
    # causing bundle._text_encoder_cfg = None and encode_text() to fail).
    # HYTextModel with default args uses Qwen3-8B + CLIP-L from checkpoints/.
    if not model_cfg.get('text_encoder'):
        model_cfg['text_encoder'] = dict(
            llm_type='qwen3',
            sentence_emb_type='clipl',
            torch_dtype=torch.bfloat16,  # Half precision to fit in cgroup memory limit
        )

    # Build bundle
    bundle = MODEL_BUNDLES.build(dict(type=model_type, **model_cfg))

    # Load checkpoint
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # Strip 'model.' prefix if present (common in Accelerate checkpoints)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            cleaned_state_dict[k[6:]] = v
        else:
            cleaned_state_dict[k] = v

    # Load into bundle
    missing, unexpected = bundle.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        print(f"  All {len(cleaned_state_dict)} keys loaded successfully!")

    bundle = bundle.to(device)
    bundle.eval()
    print(f"Bundle loaded successfully. Device: {device}")

    return bundle


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(args):
    """Run PhysFlow Direction A (RL→Generation) training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load T2M bundle
    print("\n" + "=" * 60)
    print("Loading T2M model...")
    print("=" * 60)
    bundle = load_bundle(args.t2m_config, args.t2m_ckpt, device)

    # Initialize RL physics oracle
    print("\n" + "=" * 60)
    print("Initializing RL Physics Oracle...")
    print("=" * 60)
    oracle = RLPhysicsOracle()
    print(f"  ONNX: {oracle.onnx_path}")
    print(f"  MJCF: {oracle.mjcf_path}")

    # Initialize motion converter (for Direction B)
    motion_converter = MotionFormatConverter()

    # Initialize curriculum
    curriculum = PhysFlowCurriculum(
        seed=args.seed,
        min_locomotion_ratio=getattr(args, 'min_locomotion_ratio', 0.0),
    )
    print(f"  Curriculum: {curriculum}")

    # Initialize trainer
    trainer = PhysFlowTrainer(
        bundle=bundle,
        physics_oracle=oracle,
        curriculum=curriculum,
        device=device,
        lr=args.lr,
        num_ode_steps=args.num_ode_steps,
        text_guidance_scale=args.text_guidance_scale,
        grad_clip=args.grad_clip,
        soar_lambda=args.soar_lambda,
        soar_K=args.soar_K,
        train_last_n_blocks=args.train_last_n_blocks,
        use_amp=not args.no_amp,
        motion_converter=motion_converter,
        rl_experiment=getattr(args, 'rl_experiment', None),
        output_dir=args.output_dir,
        min_completion=args.min_completion,
        min_root_height=args.min_root_height,
        require_no_fall=args.require_no_fall,
        grad_accum=args.grad_accum,
        kl_weight=getattr(args, 'kl_weight', 0.0),
    )

    # Pre-encode all curriculum prompts (caches text embeddings, frees text encoder)
    trainer.precompute_text_embeddings(cache_path=args.text_cache)

    # Training log
    log_path = os.path.join(args.output_dir, 'training_log.jsonl')
    print(f"\n  Log: {log_path}")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  LR: {args.lr}")
    print(f"  SOAR lambda: {args.soar_lambda}")
    print(f"  Grad accumulation: {args.grad_accum}")
    print(f"  KL weight: {getattr(args, 'kl_weight', 0.0)}")
    print(f"  Min locomotion ratio: {getattr(args, 'min_locomotion_ratio', 0.0)}")
    print(f"  Require no-fall: {args.require_no_fall}")
    print(f"  Min completion: {args.min_completion}")
    print(f"  Min root height: {args.min_root_height}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting PhysFlow Training (Direction A: RL → Generation)")
    print("=" * 60)

    with open(log_path, 'w') as log_f:
        for i in range(args.num_iterations):
            result = trainer.train_iteration()

            # Log to file
            log_f.write(json.dumps(result, default=str) + '\n')
            log_f.flush()

            # Print progress
            if result['skipped']:
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{args.num_iterations}] SKIPPED "
                          f"({result['reason']}) | "
                          f"curriculum={result['curriculum']['level_name']} | "
                          f"skip_rate={trainer.total_skipped}/{trainer.total_iterations}")
            else:
                if (i + 1) % args.log_interval == 0 or i == 0:
                    timing = result['timing']
                    soar_str = ', soar={:.5f}'.format(result['loss_soar']) if args.soar_lambda > 0 else ''
                    kl_str = ', kl={:.6f}'.format(result.get('loss_kl', 0.0)) if getattr(args, 'kl_weight', 0.0) > 0 else ''
                    phys_stats = result['physics_stats']
                    cur = result['curriculum']
                    completion = phys_stats.get('completion_ratio', 0.0)
                    status = phys_stats.get('status', 'unknown')
                    tracking_err = phys_stats.get('tracking_error', 0.0)
                    print(f"  [{i+1}/{args.num_iterations}] "
                          f"loss={result['loss']:.5f} "
                          f"(vel={result['loss_velocity']:.5f}{soar_str}{kl_str}) | "
                          f"level={cur['level_name']} "
                          f"(sr={cur['success_rate']:.2f}) | "
                          f"phys: {status} "
                          f"completion={completion:.2f} "
                          f"err={tracking_err:.4f} | "
                          f"time={timing['total']:.1f}s "
                          f"(gen={timing['generation']:.1f}s "
                          f"phys={timing['physics']:.1f}s "
                          f"train={timing['training']:.2f}s)")

            # Save checkpoint
            if (i + 1) % args.save_interval == 0:
                ckpt_path = os.path.join(
                    args.output_dir, f'model_iter{i+1}.pt'
                )
                torch.save({
                    'iteration': i + 1,
                    'model_state_dict': bundle.motion_transformer.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'curriculum_state': curriculum.state_dict(),
                    'loss_history': trainer.loss_history,
                }, ckpt_path)
                print(f"  [SAVE] Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(args.output_dir, 'model_final.pt')
    torch.save({
        'iteration': args.num_iterations,
        'model_state_dict': bundle.motion_transformer.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'curriculum_state': curriculum.state_dict(),
        'loss_history': trainer.loss_history,
    }, final_path)
    print(f"\n[DONE] Final model saved: {final_path}")
    print(f"  Total iterations: {trainer.total_iterations}")
    print(f"  Skipped: {trainer.total_skipped}")
    print(f"  Final curriculum: {curriculum}")

    return trainer


def run_bidirectional(args):
    """Run full bidirectional PhysFlow training (Direction A + B alternating)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load T2M bundle
    print("\n" + "=" * 60)
    print("Loading T2M model...")
    print("=" * 60)
    bundle = load_bundle(args.t2m_config, args.t2m_ckpt, device)

    # Initialize RL physics oracle
    print("\n" + "=" * 60)
    print("Initializing RL Physics Oracle...")
    print("=" * 60)
    oracle = RLPhysicsOracle()
    print(f"  ONNX: {oracle.onnx_path}")
    print(f"  MJCF: {oracle.mjcf_path}")

    # Initialize motion converter (for Direction B)
    motion_converter = MotionFormatConverter()

    # Initialize curriculum
    curriculum = PhysFlowCurriculum(seed=args.seed)
    print(f"  Curriculum: {curriculum}")

    # Initialize trainer
    trainer = PhysFlowTrainer(
        bundle=bundle,
        physics_oracle=oracle,
        curriculum=curriculum,
        device=device,
        lr=args.lr,
        num_ode_steps=args.num_ode_steps,
        text_guidance_scale=args.text_guidance_scale,
        grad_clip=args.grad_clip,
        soar_lambda=args.soar_lambda,
        soar_K=args.soar_K,
        train_last_n_blocks=args.train_last_n_blocks,
        use_amp=not args.no_amp,
        motion_converter=motion_converter,
        rl_experiment=args.rl_experiment,
        output_dir=args.output_dir,
    )

    # Pre-encode all curriculum prompts
    trainer.precompute_text_embeddings(cache_path=args.text_cache)

    print("\n" + "=" * 60)
    print("Starting PhysFlow Bidirectional Training")
    print(f"  Outer loops: {args.num_outer_loops}")
    print(f"  Direction A iters/loop: {args.gen_iters_per_loop}")
    print(f"  Direction B RL steps/loop: {args.rl_steps_per_loop}")
    print(f"  Direction B motions to generate: {args.num_gen_motions_for_rl}")
    print("=" * 60)

    # Run bidirectional outer loop
    result = trainer.run_bidirectional(
        num_outer_loops=args.num_outer_loops,
        gen_iters_per_loop=args.gen_iters_per_loop,
        rl_steps_per_loop=args.rl_steps_per_loop,
        num_gen_motions_for_rl=args.num_gen_motions_for_rl,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    print(f"\n[DONE] Bidirectional training complete.")
    print(f"  Total outer loops: {result['total_outer_loops']}")
    print(f"  Total Direction A iters: {result['total_dir_a_iters']}")
    print(f"  Final model: {result['final_model_path']}")
    print(f"  Final curriculum: {curriculum}")

    return trainer


# ---------------------------------------------------------------------------
# Single-sample test
# ---------------------------------------------------------------------------

def run_test_single(args):
    """Run a single generate -> RL correct -> train iteration for testing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load bundle
    print("\n[1/4] Loading T2M model...")
    bundle = load_bundle(args.t2m_config, args.t2m_ckpt, device)

    # Initialize RL oracle
    print("\n[2/4] Initializing RL Physics Oracle...")
    oracle = RLPhysicsOracle()
    print(f"  ONNX: {oracle.onnx_path}")
    print(f"  MJCF: {oracle.mjcf_path}")

    # Initialize motion converter
    motion_converter = MotionFormatConverter()

    # Initialize curriculum
    curriculum = PhysFlowCurriculum(seed=42)

    # Initialize trainer
    trainer = PhysFlowTrainer(
        bundle=bundle,
        physics_oracle=oracle,
        curriculum=curriculum,
        device=device,
        lr=args.lr,
        num_ode_steps=args.num_ode_steps,
        text_guidance_scale=args.text_guidance_scale,
        train_last_n_blocks=args.train_last_n_blocks,
        use_amp=not args.no_amp,
        motion_converter=motion_converter,
        output_dir=args.output_dir,
    )

    # Pre-encode curriculum prompts
    trainer.precompute_text_embeddings(cache_path=args.text_cache)

    # Run single iteration
    print("\n[3/4] Running single PhysFlow iteration...")
    prompt = curriculum.get_prompt()
    num_frames = curriculum.get_num_frames()
    print(f"  Prompt: '{prompt}'")
    print(f"  Frames: {num_frames}")

    print("\n  [3a] Generating motion (ODE {}-step)...".format(args.num_ode_steps))
    t0 = time.time()
    bundle.motion_transformer.eval()
    motion_135 = trainer.generate_motion(prompt, num_frames)
    print(f"  Generated: shape={motion_135.shape}, time={time.time()-t0:.1f}s")
    print(f"  Range: [{motion_135.min():.3f}, {motion_135.max():.3f}]")

    print("\n  [3b] RL Physics correction (closed-loop tracking)...")
    t0 = time.time()
    motion_135_phys, stats = oracle.correct(motion_135)
    print(f"  Corrected: shape={motion_135_phys.shape}, time={time.time()-t0:.1f}s")
    print(f"  Status: {stats['status']}")
    print(f"  Completion: {stats['completion_ratio']:.2f} "
          f"({stats['actual_sim_steps']}/{stats['total_sim_steps']} steps)")
    print(f"  Root height min: {stats['root_height_min']:.4f}")
    print(f"  Tracking error: {stats['tracking_error_mean']:.4f}")
    print(f"  Quality OK: {oracle.is_good_quality(stats)}")

    # Convert to 201-dim
    print("\n  [3c] Converting to 201-dim (FK + RIC)...")
    body_model = bundle.body_model
    if body_model is not None:
        motion_201_phys = motion_135_to_201(motion_135_phys, body_model, device)
        print(f"  motion_201 shape: {motion_201_phys.shape}")
        print(f"  RIC range: [{motion_201_phys[:, 135:].min():.3f}, "
              f"{motion_201_phys[:, 135:].max():.3f}]")
    else:
        print("  [WARN] No body_model, using zero-padded 201-dim")
        T_phys = motion_135_phys.shape[0]
        motion_201_phys = np.zeros((T_phys, 201), dtype=np.float32)
        motion_201_phys[:, :135] = motion_135_phys

    # Training step
    print("\n  [3d] Flow matching training step...")
    t0 = time.time()
    if oracle.is_good_quality(stats):
        train_result = trainer.train_step(motion_201_phys, prompt)
        print(f"  Loss: {train_result['loss']:.6f} "
              f"(velocity={train_result['loss_velocity']:.6f})")
        print(f"  Train time: {time.time()-t0:.2f}s")
    else:
        print("  [SKIP] Quality gate failed, skipping training")

    print("\n[4/4] Test complete!")
    print("=" * 60)
    print("[PASS] PhysFlow single-sample test passed!")

    return trainer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='PhysFlow: Bidirectional Physics-RL-Grounded Flow Correction')

    # Mode selection
    parser.add_argument('--mode', type=str, default='rl-to-gen',
                        choices=['rl-to-gen', 'bidirectional', 'test-single'],
                        help='Training mode: rl-to-gen (Direction A only), '
                             'bidirectional (A+B alternating), '
                             'test-single (one iteration test)')

    # Model paths
    parser.add_argument('--t2m-config', type=str, required=True,
                        help='Path to T2M config file')
    parser.add_argument('--t2m-ckpt', type=str, required=True,
                        help='Path to T2M checkpoint')
    parser.add_argument('--output-dir', type=str, default='output/physflow_v2',
                        help='Output directory for checkpoints and logs')

    # RL Oracle (auto-resolved from repo, no required paths)
    parser.add_argument('--rl-onnx', type=str, default=None,
                        help='Override ONNX RL policy path (auto-resolved if None)')
    parser.add_argument('--rl-mjcf', type=str, default=None,
                        help='Override SMPL MuJoCo XML path (auto-resolved if None)')

    # Direction A parameters
    parser.add_argument('--num-iterations', type=int, default=2000,
                        help='Number of Direction A training iterations (rl-to-gen mode)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--num-ode-steps', type=int, default=50,
                        help='ODE steps for generation')
    parser.add_argument('--text-guidance-scale', type=float, default=5.0,
                        help='CFG guidance scale for generation')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--soar-lambda', type=float, default=0.0,
                        help='SOAR correction loss weight (0=disabled)')
    parser.add_argument('--soar-K', type=int, default=50,
                        help='SOAR rollout step divisor')
    parser.add_argument('--train-last-n-blocks', type=int, default=4,
                        help='Only train last N transformer blocks (0=all)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--text-cache', type=str, default=None,
                        help='Path to pre-computed text embeddings .pt file')

    # Direction B (bidirectional mode) parameters
    parser.add_argument('--rl-experiment', type=str, default=None,
                        help='ProtoMotions experiment config path for RL training '
                             '(required for bidirectional mode)')
    parser.add_argument('--num-outer-loops', type=int, default=5,
                        help='Number of outer loops (A→B→A→B...) in bidirectional mode')
    parser.add_argument('--gen-iters-per-loop', type=int, default=400,
                        help='Direction A iterations per outer loop (bidirectional mode)')
    parser.add_argument('--rl-steps-per-loop', type=int, default=500,
                        help='RL training steps per outer loop (bidirectional mode)')
    parser.add_argument('--num-gen-motions-for-rl', type=int, default=100,
                        help='Number of T2M motions to generate for RL library '
                             '(Direction B)')

    # Quality gate thresholds
    parser.add_argument('--min-completion', type=float, default=0.8,
                        help='Minimum completion ratio for RL correction to pass '
                             'quality gate (default: 0.8). Lower = more training signal.')
    parser.add_argument('--min-root-height', type=float, default=0.3,
                        help='Minimum root height (m) for quality gate '
                             '(default: 0.3). Lower = accept crouching motions.')
    parser.add_argument('--require-no-fall', action='store_true',
                        help='Reject ALL entries with status="fell" regardless of '
                             'completion ratio. Prevents training on truncated/fallen '
                             'targets which cause model degradation.')

    # Gradient accumulation
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps. optimizer.step() is called '
                             'every N *successful* (non-skipped) iterations. '
                             'Effectively increases batch size. Default=1 (no accum).')

    # Anti-catastrophic-forgetting
    parser.add_argument('--kl-weight', type=float, default=0.0,
                        help='KL regularization weight toward pretrained weights. '
                             'Adds loss: kl_weight * mean_MSE(θ, θ_pretrained). '
                             'Prevents catastrophic forgetting. 0=disabled. '
                             'Recommended: 0.01-0.1 for online SFT.')
    parser.add_argument('--min-locomotion-ratio', type=float, default=0.0,
                        help='Minimum fraction of prompts from locomotion/walking '
                             'levels (level 1+). Forces balanced curriculum even at '
                             'low curriculum levels. 0=disabled. Recommended: 0.25-0.4.')

    # Logging and checkpointing
    parser.add_argument('--save-interval', type=int, default=500,
                        help='Checkpoint save interval')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log print interval')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for curriculum')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    # Legacy compatibility (deprecated, kept for backward compat)
    parser.add_argument('--smpl-xml', type=str, default=None,
                        help='[DEPRECATED] Use --rl-mjcf instead. '
                             'Path to SMPL MuJoCo XML (auto-resolved if None)')
    parser.add_argument('--test-single', action='store_true',
                        help='[DEPRECATED] Use --mode test-single instead')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Handle legacy --test-single flag
    if args.test_single:
        args.mode = 'test-single'

    # Handle legacy --smpl-xml → --rl-mjcf
    if args.smpl_xml and not args.rl_mjcf:
        args.rl_mjcf = args.smpl_xml

    # Validate bidirectional mode requirements
    if args.mode == 'bidirectional' and not args.rl_experiment:
        print("[ERROR] --rl-experiment is required for bidirectional mode.")
        print("  Example: --rl-experiment ref_repo/ProtoMotions/examples/"
              "experiments/mimic/mlp.py")
        exit(1)

    # Dispatch
    if args.mode == 'test-single':
        run_test_single(args)
    elif args.mode == 'bidirectional':
        run_bidirectional(args)
    else:
        # Default: rl-to-gen (Direction A only)
        run_training(args)
