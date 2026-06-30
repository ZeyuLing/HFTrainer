"""MotionLab ModelBundle.

Runtime code is vendored under ``hftrainer.models.motion.motionlab.network`` and
artifacts live under ``checkpoints/baselines/motionlab``.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ARTIFACT = _REPO_ROOT / "checkpoints" / "baselines" / "motionlab"
_NS = "hftrainer.models.motion.motionlab.network.rfmotion."


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    """Resolve a Hugging Face Hub model repo id to a local snapshot directory."""
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))
    except Exception:
        return local


def _install_legacy_aliases() -> None:
    pkg = importlib.import_module("hftrainer.models.motion.motionlab.network.rfmotion")
    sys.modules.setdefault("rfmotion", pkg)


@contextlib.contextmanager
def _cwd(path: Path):
    import os

    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _rewrite_namespace(value):
    if isinstance(value, str):
        return value.replace("rfmotion.", _NS)
    if isinstance(value, list):
        return [_rewrite_namespace(v) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_namespace(v) for k, v in value.items()}
    return value


def _load_cfg(
    artifact_dir: Path,
    checkpoint: Path,
    cfg_file: Path,
    cfg_assets: Path,
    cfg_from_checkpoint: bool,
    clip_path: str,
):
    _install_legacy_aliases()
    from hftrainer.models.motion.motionlab.network.rfmotion.config import get_module_config

    ckpt = torch.load(str(checkpoint), map_location="cpu")
    ckpt_cfg = None
    if isinstance(ckpt, dict):
        ckpt_cfg = ckpt.get("datamodule_hyper_parameters", {}).get("cfg")

    with _cwd(artifact_dir):
        if cfg_from_checkpoint and ckpt_cfg is not None:
            cfg = OmegaConf.create(_rewrite_namespace(ckpt_cfg))
            cfg_assets_obj = OmegaConf.load(str(cfg_assets))
            cfg = OmegaConf.merge(cfg, cfg_assets_obj)
        else:
            cfg_base = OmegaConf.load(str(artifact_dir / "configs" / "base.yaml"))
            cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(str(cfg_file)))
            cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
            cfg_assets_obj = OmegaConf.load(str(cfg_assets))
            cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets_obj)

    cfg.DEBUG = False
    cfg.ACCELERATOR = "gpu"
    cfg.DEVICE = [0]
    cfg.TRAIN.STAGE = "diffusion"
    cfg.TRAIN.ABLATION.VAE = False
    cfg.DATASET.NFEATS = 263
    cfg.DATASET.NJOINTS = 22
    cfg.model.denoiser.params.nfeats = 263
    cfg.METRIC.TYPE = []
    cfg.TEST.CHECKPOINTS = str(checkpoint)
    if clip_path:
        cfg.model.clip_path = clip_path
        cfg.model.text_encoder.params.modelpath = clip_path
    return cfg


@MODEL_BUNDLES.register_module()
class MotionLabBundle(ModelBundle):
    """MotionLab text-to-motion bundle for HumanML3D-263 generation."""

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        checkpoint: Optional[str] = None,
        cfg: Optional[str] = None,
        cfg_assets: Optional[str] = None,
        cfg_from_checkpoint: bool = True,
        clip_path: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        _install_legacy_aliases()
        artifact = Path(artifact_dir or _DEFAULT_ARTIFACT).resolve()
        checkpoint = Path(checkpoint or artifact / "motionflow.ckpt").resolve()
        cfg = Path(cfg or artifact / "configs" / "config_rfmotion_text.yaml").resolve()
        cfg_assets = Path(cfg_assets or artifact / "configs" / "assets.yaml").resolve()

        from hftrainer.models.motion.motionlab.network.rfmotion.config import instantiate_from_config
        from hftrainer.models.motion.motionlab.network.rfmotion.models.operator.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )

        self.cfg = _load_cfg(
            artifact,
            checkpoint,
            cfg,
            cfg_assets,
            cfg_from_checkpoint=cfg_from_checkpoint,
            clip_path=clip_path,
        )
        resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
        text_encoder = instantiate_from_config(self.cfg.model.text_encoder).eval().to(resolved_device)
        denoiser = instantiate_from_config(self.cfg.model.denoiser).eval().to(resolved_device)
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.cfg.model.noise_scheduler.params.num_train_timesteps
        )

        ckpt = torch.load(str(checkpoint), map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        den_state = {
            k.replace("denoiser.", "", 1): v
            for k, v in state.items()
            if k.startswith("denoiser.")
        }
        missing, unexpected = denoiser.load_state_dict(den_state, strict=False)
        self.load_report = {"missing": len(missing), "unexpected": len(unexpected)}

        self.text_encoder = text_encoder
        self.denoiser = denoiser
        self.scheduler = scheduler

        for name in ("Mean.npy", "Std.npy", "mean_motion.npy", "std_motion.npy"):
            if not (artifact / name).exists():
                raise FileNotFoundError(f"MotionLab artifact missing {name}: {artifact / name}")
        self.register_buffer("mean", torch.from_numpy(np.load(artifact / "Mean.npy").astype(np.float32)), persistent=True)
        self.register_buffer("std", torch.from_numpy(np.load(artifact / "Std.npy").astype(np.float32)), persistent=True)
        self.register_buffer("mean_motion", torch.from_numpy(np.load(artifact / "mean_motion.npy").astype(np.float32)), persistent=True)
        self.register_buffer("std_motion", torch.from_numpy(np.load(artifact / "std_motion.npy").astype(np.float32)), persistent=True)

    def to_device(self, device):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.text_encoder.to(device)
        self.denoiser.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.mean_motion = self.mean_motion.to(device)
        self.std_motion = self.std_motion.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def denormalize(self, motion_263: torch.Tensor) -> torch.Tensor:
        return motion_263 * self.std.to(motion_263) + self.mean.to(motion_263)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not (path / "motionflow.ckpt").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        if path.is_dir() and (path / "motionflow.ckpt").exists():
            return cls(artifact_dir=str(path), **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use MotionLabPipeline.infer_t2m for inference.")
