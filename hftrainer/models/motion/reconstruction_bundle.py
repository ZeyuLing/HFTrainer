"""Lightweight bundles for tokenizer/VAE reconstruction evaluation."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.models.motion.prism.autoencoder_kl_2d import AutoencoderKLPrism2DTK  # noqa: F401
from hftrainer.models.motion.vermo import VQVAEWanMotion2DTK  # noqa: F401
from hftrainer.motion.processing.smpl_processor import SMPLPoseProcessor  # noqa: F401
from hftrainer.registry import MODEL_BUNDLES


_SMPL22_PROCESSOR_CFG: Dict[str, Any] = {
    "type": "SMPLPoseProcessor",
    "trainable": False,
    "save_ckpt": False,
    "do_normalize": True,
    "stats_file": "data/statistic/smplx55_stats_hymotion_aug.json",
    "rot_type": "rotation_6d",
    "transl_type": "abs_rel",
    "smpl_type": "smpl_22",
    "smpl_model": {
        "type": "SmplxLiteV437Coco17",
        "model_path": "checkpoints/smpl_models/smplx",
        "smplx2smpl_path": "checkpoints/smpl_models/smplx2smpl_sparse.pt",
        "coco17_regressor_path": "checkpoints/smpl_models/smpl_coco17_J_regressor.pt",
        "smplx_verts437_path": "checkpoints/smpl_models/smplx_verts437.pt",
        "gender": "neutral",
        "num_betas": 10,
    },
}


def _copy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import copy

    return copy.deepcopy(cfg)


@MODEL_BUNDLES.register_module()
class PrismReconstructionBundle(ModelBundle):
    """PRISM motion VAE plus SMPL processor for reconstruction-only eval."""

    def __init__(
        self,
        vae: dict,
        smpl_pose_processor: dict,
    ):
        super().__init__()
        self._build_modules(
            {
                "vae": vae,
                "smpl_pose_processor": smpl_pose_processor,
            }
        )
        self.use_static = bool(getattr(self.vae.config, "use_static", False))

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        vae_path: Optional[str] = None,
        smpl_pose_processor_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # The released PRISM iter15000 package is a raw checkpoint package. For
        # tokenizer reconstruction we only need the released motion VAE. The
        # package model card records that iter15000 must use wanmo_vae2d_aug.
        root = os.path.abspath(os.path.expanduser(pretrained_model_name_or_path))
        if vae_path is None:
            candidate = os.path.join(root, "vae")
            vae_path = candidate if os.path.isdir(candidate) else "checkpoints/wanmo_vae2d_aug"

        return {
            "vae": {
                "type": "AutoencoderKLPrism2DTK",
                "trainable": False,
                "save_ckpt": False,
                "module_dtype": "fp32",
                "from_pretrained": {"pretrained_model_name_or_path": vae_path},
            },
            "smpl_pose_processor": smpl_pose_processor_cfg or _copy_cfg(_SMPL22_PROCESSOR_CFG),
        }

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    def to_device(self, device: str | torch.device):
        self.to(device)
        return self


@MODEL_BUNDLES.register_module()
class VermoReconstructionBundle(ModelBundle):
    """VerMo motion VQ-VAE plus SMPL processor for reconstruction-only eval."""

    def __init__(
        self,
        motion_tokenizer: dict,
        smpl_pose_processor: dict,
    ):
        super().__init__()
        self._build_modules(
            {
                "motion_tokenizer": motion_tokenizer,
                "smpl_pose_processor": smpl_pose_processor,
            }
        )
        self.use_static = bool(getattr(self.motion_tokenizer.config, "use_static", False))

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        smpl_pose_processor_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "motion_tokenizer": {
                "type": "VQVAEWanMotion2DTK",
                "trainable": False,
                "save_ckpt": False,
                "module_dtype": "fp32",
                "from_pretrained": {
                    "pretrained_model_name_or_path": os.path.abspath(
                        os.path.expanduser(pretrained_model_name_or_path)
                    )
                },
            },
            "smpl_pose_processor": smpl_pose_processor_cfg or _copy_cfg(_SMPL22_PROCESSOR_CFG),
        }

    @property
    def device(self) -> torch.device:
        return next(self.motion_tokenizer.parameters()).device

    def to_device(self, device: str | torch.device):
        self.to(device)
        return self
