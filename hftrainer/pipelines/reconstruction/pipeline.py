"""Tokenizer / VAE reconstruction pipelines for motion leaderboards.

The classes in this module intentionally route reconstruction through
hftrainer-native ModelBundle objects. They do not import raw upstream
repositories or external helper checkouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import numpy as np
import torch

from hftrainer.models.motion.prism.gaussian_distribution import (
    DiagonalGaussianDistributionNd,
)
from hftrainer.motion.representation.rotation import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


@dataclass
class ReconstructionResult:
    """Single-clip reconstruction result."""

    motion: np.ndarray
    metadata: Dict[str, Any]


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return int(((value + multiple - 1) // multiple) * multiple)


def _tensor_shape(value: Any) -> Optional[list[int]]:
    if value is None or not hasattr(value, "shape"):
        return None
    return [int(v) for v in value.shape]


def _hml263_to_mogents_2d(x: torch.Tensor) -> torch.Tensor:
    """Build the MoGenTS joint-group 2D input from normalized HML263."""
    if x.ndim != 3 or x.shape[-1] != 263:
        raise ValueError(f"expected normalized HML263 [B,T,263], got {tuple(x.shape)}")
    bsz, nframes, _ = x.shape
    out = torch.zeros((bsz, nframes, 22, 12), dtype=x.dtype, device=x.device)
    out[:, :, 0, :4] = x[:, :, :4]
    out[:, :, 0, 4:8] = x[:, :, -4:]

    cursor = 4
    pos = x[:, :, cursor: cursor + 21 * 3].reshape(bsz, nframes, 21, 3)
    cursor += 21 * 3
    rot = x[:, :, cursor: cursor + 21 * 6].reshape(bsz, nframes, 21, 6)
    cursor += 21 * 6
    vel = x[:, :, cursor: cursor + 22 * 3].reshape(bsz, nframes, 22, 3)

    out[:, :, 1:, :3] = pos
    out[:, :, 1:, 3:9] = rot
    out[:, :, :, 9:12] = vel
    return out


def _wan_motion_padded_len(value: int, scale: int = 4) -> int:
    if value <= 1:
        return int(value)
    return 1 + _round_up(value - 1, scale)


def _processor_transl_dim(processor) -> int:
    return 6 if getattr(processor, "transl_type", "abs") == "abs_rel" else 3


def _prepare_motion135_array(motion: np.ndarray, pad_multiple: int = 1) -> tuple[np.ndarray, int, int]:
    arr = np.asarray(motion, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 135:
        raise ValueError(f"expected motion135 shape (T,135), got {arr.shape}")
    if len(arr) < 1:
        raise ValueError("empty motion clip")
    if not np.isfinite(arr).all():
        raise ValueError("input contains non-finite values")

    orig_len = int(arr.shape[0])
    padded_len = _wan_motion_padded_len(orig_len, pad_multiple)
    if padded_len > orig_len:
        arr = np.concatenate(
            [arr, np.repeat(arr[-1:], padded_len - orig_len, axis=0)],
            axis=0,
        ).astype(np.float32)
    return arr, orig_len, padded_len


def _motion135_to_processor_motion(
    motion135: np.ndarray,
    processor,
    device: torch.device,
) -> torch.Tensor:
    """Convert repo-canonical motion135 row-6D to processor-native motion_138."""
    arr = np.asarray(motion135, dtype=np.float32)
    transl = arr[:, :3]
    rot6d_row = torch.from_numpy(arr[:, 3:135]).to(device=device).reshape(-1, 22, 6)
    rotmat = rotation_6d_to_matrix(rot6d_row, convention="row")
    rot6d_col = matrix_to_rotation_6d(rotmat, convention="column").reshape(arr.shape[0], 22 * 6)

    transl_block = processor.convert_transl(transl, getattr(processor, "transl_type", None))
    transl_block = torch.from_numpy(np.asarray(transl_block, dtype=np.float32)).to(device=device)
    return torch.cat([transl_block, rot6d_col.to(dtype=transl_block.dtype)], dim=-1).unsqueeze(0)


def _processor_motion_to_motion135(
    motion_vec: torch.Tensor,
    processor,
    orig_len: int,
    use_rollout_trans: bool | str = True,
) -> np.ndarray:
    """Convert processor-native motion_138 column-6D back to motion135 row-6D."""
    transl_dim = _processor_transl_dim(processor)
    motion_vec = motion_vec[:, :orig_len].float()
    transl = processor.inv_convert_transl(
        motion_vec[..., :transl_dim],
        getattr(processor, "transl_type", None),
        use_rollout=use_rollout_trans,
    )
    pose = motion_vec[..., transl_dim:]
    if pose.shape[-1] < 22 * 6:
        raise ValueError(f"processor motion has too few pose channels: {pose.shape[-1]}")
    pose = pose[..., : 22 * 6].reshape(*pose.shape[:-1], 22, 6)
    rotmat = rotation_6d_to_matrix(pose, convention="column")
    rot6d_row = matrix_to_rotation_6d(rotmat, convention="row").reshape(
        motion_vec.shape[0], motion_vec.shape[1], 22 * 6
    )
    out = torch.cat([transl, rot6d_row], dim=-1)
    arr = out.detach().cpu().numpy()[0].astype(np.float32)
    if arr.shape != (orig_len, 135):
        raise ValueError(f"motion135 conversion produced {arr.shape}, expected {(orig_len, 135)}")
    if not np.isfinite(arr).all():
        raise ValueError("reconstruction contains non-finite values")
    return arr


def _append_static_tokens_if_needed(bundle, motion_vec: torch.Tensor) -> torch.Tensor:
    if not bool(getattr(bundle, "use_static", False)):
        return motion_vec
    static = bundle.smpl_pose_processor.get_static_joint_mask_from_motion(motion_vec)
    return torch.cat([motion_vec, static], dim=-1)


def _processor_motion_to_grid(motion_vec: torch.Tensor) -> torch.Tensor:
    if motion_vec.shape[-1] % 6:
        raise ValueError(f"processor motion dim must be divisible by 6, got {motion_vec.shape[-1]}")
    return motion_vec.reshape(motion_vec.shape[0], motion_vec.shape[1], motion_vec.shape[2] // 6, 6)


class BaseReconstructionPipeline(BasePipeline):
    """Base class for deterministic motion auto-encoding pipelines."""

    method = ""
    representation = ""
    feature_dim = 0
    pad_multiple = 1

    def __init__(
        self,
        bundle,
        device: Optional[str] = None,
        latent_mode: str = "mean",
        **kwargs,
    ):
        super().__init__(bundle, **kwargs)
        self.latent_mode = latent_mode
        if device is not None:
            self.to(device)

    def to(self, device):
        if hasattr(self.bundle, "to_device"):
            self.bundle.to_device(device)
        else:
            self.bundle.to(device)
        return self

    @property
    def device(self) -> torch.device:
        if hasattr(self.bundle, "device"):
            return torch.device(self.bundle.device)
        return next(self.bundle.parameters()).device

    @torch.no_grad()
    def reconstruct(self, motion: np.ndarray) -> ReconstructionResult:
        raise NotImplementedError

    def _prepare_motion(self, motion: np.ndarray) -> tuple[torch.Tensor, int, int]:
        arr = np.asarray(motion, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.feature_dim:
            raise ValueError(
                f"{self.method} expected motion shape (T,{self.feature_dim}), got {arr.shape}"
            )
        if len(arr) < 1:
            raise ValueError("empty motion clip")
        if not np.isfinite(arr).all():
            raise ValueError("input contains non-finite values")

        orig_len = int(arr.shape[0])
        padded_len = _round_up(orig_len, self.pad_multiple)
        if padded_len > orig_len:
            pad = np.repeat(arr[-1:], padded_len - orig_len, axis=0)
            arr = np.concatenate([arr, pad], axis=0).astype(np.float32)

        x = torch.from_numpy(arr).to(self.device).unsqueeze(0)
        mean = self.bundle.mean.to(device=x.device, dtype=x.dtype)
        std = self.bundle.std.to(device=x.device, dtype=x.dtype).clamp_min(1e-8)
        return (x - mean) / std, orig_len, padded_len

    def _finish(
        self,
        recon_norm: torch.Tensor,
        orig_len: int,
        padded_len: int,
        metadata: Dict[str, Any],
    ) -> ReconstructionResult:
        recon = self.bundle.denormalize(recon_norm.float())[:, :orig_len]
        arr = recon.detach().cpu().numpy()[0].astype(np.float32)
        if arr.shape != (orig_len, self.feature_dim):
            raise ValueError(
                f"{self.method} reconstructed bad shape {arr.shape}; "
                f"expected {(orig_len, self.feature_dim)}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("reconstruction contains non-finite values")
        metadata = {
            "method": self.method,
            "representation": self.representation,
            "frames": orig_len,
            "padded_frames": padded_len,
            "pad_multiple": self.pad_multiple,
            **metadata,
        }
        return ReconstructionResult(motion=arr, metadata=metadata)

    def _select_latent(self, latent: torch.Tensor, dist) -> torch.Tensor:
        if self.latent_mode == "sample":
            return latent
        if self.latent_mode == "mean":
            loc = getattr(dist, "loc", None)
            if loc is None:
                raise ValueError(f"{self.method} latent distribution has no loc")
            return loc
        raise ValueError(f"unsupported latent mode: {self.latent_mode}")

    def _reconstruct_mld_vae(self, vae, motion: np.ndarray) -> ReconstructionResult:
        norm, orig_len, padded_len = self._prepare_motion(motion)
        lengths = [orig_len]
        latent, dist = vae.encode(norm[:, :orig_len], lengths)
        z = self._select_latent(latent, dist)
        recon_norm = vae.decode(z, lengths)
        return self._finish(
            recon_norm,
            orig_len,
            padded_len,
            {
                "latent_mode": self.latent_mode,
                "latent_shape": _tensor_shape(z),
            },
        )

    def __call__(self, motion: np.ndarray) -> ReconstructionResult:
        return self.reconstruct(motion)


@PIPELINES.register_module()
class T2MGPTReconstructionPipeline(BaseReconstructionPipeline):
    """T2M-GPT VQ-VAE reconstruction on HumanML3D-263."""

    BUNDLE_CLS = "hftrainer.models.motion.t2mgpt.T2MGPTBundle"
    method = "t2mgpt"
    representation = "hml263"
    feature_dim = 263
    pad_multiple = 4

    @torch.no_grad()
    def reconstruct(self, motion: np.ndarray) -> ReconstructionResult:
        norm, orig_len, padded_len = self._prepare_motion(motion)
        indices = self.bundle.vqvae.encode(norm)
        recon_norm = self.bundle.vqvae.forward_decoder(indices)
        return self._finish(
            recon_norm,
            orig_len,
            padded_len,
            {
                "token_shape": _tensor_shape(indices),
                "num_tokens": int(indices.numel()),
            },
        )


@PIPELINES.register_module()
class MotionGPTReconstructionPipeline(BaseReconstructionPipeline):
    """MotionGPT VQ-VAE reconstruction on HumanML3D-263."""

    BUNDLE_CLS = "hftrainer.models.motion.motiongpt.MotionGPTBundle"
    method = "motiongpt"
    representation = "hml263"
    feature_dim = 263
    pad_multiple = 4

    @torch.no_grad()
    def reconstruct(self, motion: np.ndarray) -> ReconstructionResult:
        norm, orig_len, padded_len = self._prepare_motion(motion)
        indices, _ = self.bundle.vae.encode(norm)
        recon_norm = self.bundle.vae.decode(indices)
        return self._finish(
            recon_norm,
            orig_len,
            padded_len,
            {
                "token_shape": _tensor_shape(indices),
                "num_tokens": int(indices.numel()),
            },
        )


@PIPELINES.register_module()
class MoMaskReconstructionPipeline(BaseReconstructionPipeline):
    """MoMask RVQ-VAE reconstruction on HumanML3D-263."""

    BUNDLE_CLS = "hftrainer.models.motion.momask.MoMaskBundle"
    method = "momask"
    representation = "hml263"
    feature_dim = 263
    pad_multiple = 4

    @torch.no_grad()
    def reconstruct(self, motion: np.ndarray) -> ReconstructionResult:
        norm, orig_len, padded_len = self._prepare_motion(motion)
        indices, _ = self.bundle.vq_model.encode(norm)
        recon_norm = self.bundle.vq_model.forward_decoder(indices)
        return self._finish(
            recon_norm,
            orig_len,
            padded_len,
            {
                "token_shape": _tensor_shape(indices),
                "num_tokens": int(indices.numel()),
            },
        )


@PIPELINES.register_module()
class MoGenTSReconstructionPipeline(BaseReconstructionPipeline):
    """MoGenTS dual-stream RVQ-VAE reconstruction on HumanML3D-263."""

    BUNDLE_CLS = "hftrainer.models.motion.mogents.MoGenTSBundle"
    method = "mogents"
    representation = "hml263"
    feature_dim = 263
    pad_multiple = 4

    @torch.no_grad()
    def reconstruct(self, motion: np.ndarray) -> ReconstructionResult:
        norm, orig_len, padded_len = self._prepare_motion(motion)
        x2d = _hml263_to_mogents_2d(norm)
        indices_1d, _, indices_2d, _ = self.bundle.vq_model.encode(norm, x2d)
        recon_norm, recon_fused = self.bundle.vq_model.forward_decoder(indices_1d, indices_2d)
        if recon_norm is None:
            raise RuntimeError("MoGenTS forward_decoder returned no 1D reconstruction")
        return self._finish(
            recon_norm,
            orig_len,
            padded_len,
            {
                "decoder_output": "primary_1d",
                "fused_shape": _tensor_shape(recon_fused),
                "token_shape_1d": _tensor_shape(indices_1d),
                "token_shape_2d": _tensor_shape(indices_2d),
                "num_tokens_1d": int(indices_1d.numel()),
                "num_tokens_2d": int(indices_2d.numel()),
            },
        )


@PIPELINES.register_module()
class MLDReconstructionPipeline(BaseReconstructionPipeline):
    """MLD latent VAE reconstruction on HumanML3D-263."""

    BUNDLE_CLS = "hftrainer.models.motion.mld.MLDBundle"
    method = "mld"
    representation = "hml263"
    feature_dim = 263
    pad_multiple = 1

    @torch.no_grad()
    def reconstruct(self, motion: np.ndarray) -> ReconstructionResult:
        return self._reconstruct_mld_vae(self.bundle.vae, motion)


@PIPELINES.register_module()
class MotionLCMReconstructionPipeline(BaseReconstructionPipeline):
    """MotionLCM latent VAE reconstruction on HumanML3D-263."""

    BUNDLE_CLS = "hftrainer.models.motion.motionlcm.MotionLCMBundle"
    method = "motionlcm"
    representation = "hml263"
    feature_dim = 263
    pad_multiple = 1

    @torch.no_grad()
    def reconstruct(self, motion: np.ndarray) -> ReconstructionResult:
        return self._reconstruct_mld_vae(self.bundle.vae, motion)


@PIPELINES.register_module()
class MotionGPT3ReconstructionPipeline(BaseReconstructionPipeline):
    """MotionGPT3 motion VAE reconstruction on HumanML3D-263."""

    BUNDLE_CLS = "hftrainer.models.motion.motiongpt3.MotionGPT3Bundle"
    method = "motiongpt3"
    representation = "hml263"
    feature_dim = 263
    pad_multiple = 1

    @torch.no_grad()
    def reconstruct(self, motion: np.ndarray) -> ReconstructionResult:
        return self._reconstruct_mld_vae(self.bundle.model.vae, motion)


@PIPELINES.register_module()
class MotionStreamerReconstructionPipeline(BaseReconstructionPipeline):
    """MotionStreamer Causal-TAE reconstruction on MotionStreamer-272."""

    BUNDLE_CLS = "hftrainer.models.motion.motionstreamer.MotionStreamerBundle"
    method = "motionstreamer"
    representation = "ms272"
    feature_dim = 272
    pad_multiple = 4

    @torch.no_grad()
    def reconstruct(self, motion: np.ndarray) -> ReconstructionResult:
        norm, orig_len, padded_len = self._prepare_motion(motion)
        recon_norm, mu, logvar = self.bundle.tae(norm)
        return self._finish(
            recon_norm,
            orig_len,
            padded_len,
            {
                "latent_mu_shape": _tensor_shape(mu),
                "latent_logvar_shape": _tensor_shape(logvar),
            },
        )


@PIPELINES.register_module()
class PrismReconstructionPipeline(BaseReconstructionPipeline):
    """PRISM motion VAE reconstruction on motion135 / native motion_138."""

    BUNDLE_CLS = "hftrainer.models.motion.reconstruction_bundle.PrismReconstructionBundle"
    method = "prism"
    representation = "motion135"
    feature_dim = 135
    pad_multiple = 4

    @torch.no_grad()
    def reconstruct(self, motion: np.ndarray) -> ReconstructionResult:
        arr, orig_len, padded_len = _prepare_motion135_array(motion, self.pad_multiple)
        processor = self.bundle.smpl_pose_processor
        motion_vec = _motion135_to_processor_motion(arr, processor, self.device)
        norm = processor.normalize(motion_vec)
        norm = _append_static_tokens_if_needed(self.bundle, norm)
        grid = _processor_motion_to_grid(norm)

        device_type = grid.device.type
        with torch.autocast(device_type, enabled=False):
            latent_params = self.bundle.vae.encode(grid.float())
            latents = DiagonalGaussianDistributionNd(latent_params).mode()
            recon_grid = self.bundle.vae.decode(latents.float())

        recon_norm = recon_grid.reshape(recon_grid.shape[0], recon_grid.shape[1], -1)
        recon_norm = recon_norm[..., : processor.mean.numel()]
        recon_motion = processor.denormalize(recon_norm)
        use_rollout = getattr(self.bundle.vae.config, "use_rollout_trans", True)
        out = _processor_motion_to_motion135(recon_motion, processor, orig_len, use_rollout)
        return ReconstructionResult(
            motion=out,
            metadata={
                "method": self.method,
                "representation": self.representation,
                "native_representation": "motion_138",
                "frames": orig_len,
                "padded_frames": padded_len,
                "pad_multiple": self.pad_multiple,
                "latent_shape": _tensor_shape(latents),
            },
        )


@PIPELINES.register_module()
class VermoReconstructionPipeline(BaseReconstructionPipeline):
    """VerMo motion VQ-VAE reconstruction on motion135 / native motion_138."""

    BUNDLE_CLS = "hftrainer.models.motion.reconstruction_bundle.VermoReconstructionBundle"
    method = "vermo"
    representation = "motion135"
    feature_dim = 135
    pad_multiple = 4

    @torch.no_grad()
    def reconstruct(self, motion: np.ndarray) -> ReconstructionResult:
        arr, orig_len, padded_len = _prepare_motion135_array(motion, self.pad_multiple)
        processor = self.bundle.smpl_pose_processor
        tokenizer = self.bundle.motion_tokenizer
        motion_vec = _motion135_to_processor_motion(arr, processor, self.device)
        norm = processor.normalize(motion_vec)
        norm = _append_static_tokens_if_needed(self.bundle, norm)
        grid = _processor_motion_to_grid(norm)

        indices = tokenizer.encode(grid, flatten=False).indices
        recon_grid = tokenizer.decode(
            indices,
            flatten=False,
            is_indices=True,
            K=int(grid.shape[2]),
        )
        recon_norm = recon_grid.reshape(recon_grid.shape[0], recon_grid.shape[1], -1)
        recon_norm = recon_norm[..., : processor.mean.numel()]
        recon_motion = processor.denormalize(recon_norm)
        use_rollout = getattr(tokenizer.config, "use_rollout_trans", True)
        out = _processor_motion_to_motion135(recon_motion, processor, orig_len, use_rollout)
        return ReconstructionResult(
            motion=out,
            metadata={
                "method": self.method,
                "representation": self.representation,
                "native_representation": "motion_138",
                "frames": orig_len,
                "padded_frames": padded_len,
                "pad_multiple": self.pad_multiple,
                "token_shape": _tensor_shape(indices),
                "num_tokens": int(indices.numel()),
            },
        )


_PIPELINE_BY_METHOD: dict[str, Type[BaseReconstructionPipeline]] = {
    "t2mgpt": T2MGPTReconstructionPipeline,
    "motiongpt": MotionGPTReconstructionPipeline,
    "momask": MoMaskReconstructionPipeline,
    "mogents": MoGenTSReconstructionPipeline,
    "mld": MLDReconstructionPipeline,
    "motionlcm": MotionLCMReconstructionPipeline,
    "motiongpt3": MotionGPT3ReconstructionPipeline,
    "motionstreamer": MotionStreamerReconstructionPipeline,
    "prism": PrismReconstructionPipeline,
    "vermo": VermoReconstructionPipeline,
}


def get_reconstruction_pipeline_cls(method: str) -> Type[BaseReconstructionPipeline]:
    key = method.lower().replace("-", "").replace("_", "")
    aliases = {
        "t2mgpt": "t2mgpt",
        "t2mgptvavae": "t2mgpt",
        "motiongpt": "motiongpt",
        "momask": "momask",
        "mogents": "mogents",
        "mld": "mld",
        "motionlcm": "motionlcm",
        "motiongpt3": "motiongpt3",
        "motionstreamer": "motionstreamer",
        "prism": "prism",
        "vermo": "vermo",
    }
    canonical = aliases.get(key)
    if canonical is None or canonical not in _PIPELINE_BY_METHOD:
        raise KeyError(f"no hftrainer reconstruction pipeline for method: {method}")
    return _PIPELINE_BY_METHOD[canonical]
