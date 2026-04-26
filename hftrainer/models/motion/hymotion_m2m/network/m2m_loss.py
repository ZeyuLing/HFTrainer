from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class M2MLoss(nn.Module):
    def __init__(
        self,
        loss_type: str = "smooth_l1",
        velocity_weight: float = 1.0,
        x1_weight: float = 1.0,
        keypoints3d_weight: float = 1.0,
        translation_weight: float = 1.0,
        motion_smoothness_weight: float = 0.0,
        fk_loss_start_step: int = 0,
        trans_dim_weight: float = 1.0,
        trans_dims: int = 3,
        fk_consistency_weight: float = 0.0,
        fk_consistency_warmup_steps: int = 1000,
    ):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.x1_weight = x1_weight
        self.keypoints3d_weight = keypoints3d_weight
        self.translation_weight = translation_weight
        self.motion_smoothness_weight = motion_smoothness_weight
        self.fk_loss_start_step = fk_loss_start_step
        self.trans_dim_weight = trans_dim_weight
        self.trans_dims = trans_dims
        self.fk_consistency_weight = fk_consistency_weight
        self.fk_consistency_warmup_steps = fk_consistency_warmup_steps

        if loss_type == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        elif loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif loss_type == "mse":
            self.loss_fn = F.mse_loss
        elif loss_type == "l2":
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(
        self,
        pred_vel=None,
        gt_vel=None,
        pred_x1=None,
        gt_x1=None,
        pred_keypoints3d=None,
        gt_keypoints3d=None,
        pred_translation=None,
        gt_translation=None,
        global_step: Optional[int] = None,
        data_mask_temporal: Optional[Tensor] = None,
        generation_mask: Optional[Tensor] = None,
        fk_consistency_loss: Optional[Tensor] = None,
    ):
        """
        pred_vel: (B, L, D)
        gt_vel: (B, L, D)
        pred_x1: (B, L, D)
        gt_x1: (B, L, D)
        pred_keypoints3d: (B, L, J, 3)
        gt_keypoints3d: (B, L, J, 3)
        pred_translation: (B, L, 3)
        gt_translation: (B, L, 3)
        data_mask_temporal: (B, L) — padding mask (1=valid frame, 0=pad)
        generation_mask: (B, L, D) — optional, 1=generation region, 0=known.
            When provided, velocity/x1 losses are computed only on generation
            regions (mask-aware noise training).
        """
        loss_dict = {}
        assert data_mask_temporal is not None, "data_mask_temporal is required"

        if pred_vel is not None and gt_vel is not None:
            # velocity loss: (B, L, D) -> (B, L) -> scalar
            # Apply per-dimension weighting: upweight translation dims (first trans_dims)
            # to compensate for the 3/135 dimension ratio imbalance
            vel_per_dim = self.loss_fn(pred_vel, gt_vel, reduction="none")  # (B, L, D)
            if self.trans_dim_weight != 1.0:
                dim_weights = torch.ones(vel_per_dim.shape[-1], device=vel_per_dim.device)
                dim_weights[:self.trans_dims] = self.trans_dim_weight
                vel_per_dim = vel_per_dim * dim_weights
            # When generation_mask is provided (mask-aware noise), only compute
            # loss on generation regions. Otherwise fall back to per-frame mean.
            if generation_mask is not None:
                gen_mask = generation_mask.to(vel_per_dim.device)  # (B, L, D)
                # Combine with temporal padding mask
                combined = gen_mask * data_mask_temporal.unsqueeze(-1).to(vel_per_dim.device)
                mask_sum = torch.clamp(combined.sum(), min=1.0)
                loss_dict["velocity"] = self.velocity_weight * (vel_per_dim * combined).sum() / mask_sum
            else:
                loss_dict["velocity"] = self.velocity_weight * vel_per_dim.mean(dim=-1)
                # 确保 data_mask_temporal 与 loss_dict["velocity"] 在同一设备上
                data_mask_temporal_vel = data_mask_temporal.to(loss_dict["velocity"].device)
                mask_sum_vel = torch.clamp(data_mask_temporal_vel.sum(), min=1.0)
                loss_dict["velocity"] = (loss_dict["velocity"] * data_mask_temporal_vel).sum() / mask_sum_vel

        if pred_x1 is not None and gt_x1 is not None:
            # x1 loss: (B, L, D) -> (B, L) -> scalar
            # Apply same per-dimension weighting as velocity loss
            x1_per_dim = self.loss_fn(pred_x1, gt_x1, reduction="none")  # (B, L, D)
            if self.trans_dim_weight != 1.0:
                dim_weights = torch.ones(x1_per_dim.shape[-1], device=x1_per_dim.device)
                dim_weights[:self.trans_dims] = self.trans_dim_weight
                x1_per_dim = x1_per_dim * dim_weights
            if generation_mask is not None:
                gen_mask = generation_mask.to(x1_per_dim.device)
                combined = gen_mask * data_mask_temporal.unsqueeze(-1).to(x1_per_dim.device)
                mask_sum = torch.clamp(combined.sum(), min=1.0)
                loss_dict["x1"] = self.x1_weight * (x1_per_dim * combined).sum() / mask_sum
            else:
                loss_dict["x1"] = self.x1_weight * x1_per_dim.mean(dim=-1)
                # 确保 data_mask_temporal 与 loss_dict["x1"] 在同一设备上
                data_mask_temporal_x1 = data_mask_temporal.to(loss_dict["x1"].device)
                mask_sum_x1 = torch.clamp(data_mask_temporal_x1.sum(), min=1.0)
                loss_dict["x1"] = (loss_dict["x1"] * data_mask_temporal_x1).sum() / mask_sum_x1

        if (global_step is None and self.fk_loss_start_step == 0) or (
            global_step is not None and global_step >= self.fk_loss_start_step
        ):
            if pred_keypoints3d is not None and gt_keypoints3d is not None:
                # 计算局部关键点（相对于根节点）
                local_keypoints3d = pred_keypoints3d[:, :, 1:22] - pred_keypoints3d[:, :, 0:1, :]
                local_keypoints3d_gt = gt_keypoints3d[:, :, 1:22] - gt_keypoints3d[:, :, 0:1, :]
                # keypoints3d loss: (B, L, 21, 3) -> (B, L, 21) -> (B, L) -> scalar
                loss_dict["keypoints3d"] = self.keypoints3d_weight * self.loss_fn(
                    local_keypoints3d, local_keypoints3d_gt, reduction="none"
                ).sum(dim=-1).mean(dim=-1)
                # 确保 data_mask_temporal 与 loss_dict["keypoints3d"] 在同一设备上
                data_mask_temporal_kp = data_mask_temporal.to(loss_dict["keypoints3d"].device)
                mask_sum_kp = torch.clamp(data_mask_temporal_kp.sum(), min=1.0)
                loss_dict["keypoints3d"] = (loss_dict["keypoints3d"] * data_mask_temporal_kp).sum() / mask_sum_kp

            if pred_translation is not None and gt_translation is not None and self.translation_weight > 0.0:
                # translation loss: (B, L, 3) -> (B, L) -> scalar
                loss_dict["translation"] = self.translation_weight * self.loss_fn(
                    pred_translation, gt_translation, reduction="none"
                ).mean(dim=-1)
                # 确保 data_mask_temporal 与 loss_dict["translation"] 在同一设备上
                data_mask_temporal_trans = data_mask_temporal.to(loss_dict["translation"].device)
                mask_sum_trans = torch.clamp(data_mask_temporal_trans.sum(), min=1.0)
                loss_dict["translation"] = (loss_dict["translation"] * data_mask_temporal_trans).sum() / mask_sum_trans
        elif global_step is None and self.fk_loss_start_step > 0:
            raise ValueError("global_step is None and fk_loss_start_step is not 0")

        # Motion smoothness loss: penalize deviation in frame-to-frame velocity
        # (temporal difference) between predicted and GT motion. Inspired by
        # KIMODO's velocity loss (γ_vel=2). This operates on the denoised x1
        # space, not the flow velocity.
        if self.motion_smoothness_weight > 0.0 and gt_x1 is not None and pred_x1 is not None:
            # Compute frame-to-frame velocity (temporal difference)
            pred_motion_vel = pred_x1[:, 1:] - pred_x1[:, :-1]  # (B, L-1, D)
            gt_motion_vel = gt_x1[:, 1:] - gt_x1[:, :-1]  # (B, L-1, D)
            smooth_per_dim = self.loss_fn(pred_motion_vel, gt_motion_vel, reduction="none")
            smooth_loss = smooth_per_dim.mean(dim=-1)  # (B, L-1)
            # Mask: both frame t and t+1 must be valid
            smooth_mask = data_mask_temporal[:, 1:] * data_mask_temporal[:, :-1]
            smooth_mask = smooth_mask.to(smooth_loss.device)
            mask_sum_smooth = torch.clamp(smooth_mask.sum(), min=1.0)
            loss_dict["smoothness"] = self.motion_smoothness_weight * (
                smooth_loss * smooth_mask
            ).sum() / mask_sum_smooth

        # FK consistency loss: penalizes inconsistency between rotation/translation
        # and position channels in 198-dim motion. Passed in pre-computed by trainer.
        if (self.fk_consistency_weight > 0.0
                and fk_consistency_loss is not None):
            warmup = 1.0
            if (self.fk_consistency_warmup_steps > 0
                    and global_step is not None
                    and global_step < self.fk_consistency_warmup_steps):
                warmup = global_step / self.fk_consistency_warmup_steps
            loss_dict["fk_consistency"] = (
                self.fk_consistency_weight * warmup * fk_consistency_loss
            )

        return loss_dict
