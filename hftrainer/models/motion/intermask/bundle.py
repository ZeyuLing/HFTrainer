"""InterMask ModelBundle backed by native hftrainer runtime modules."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ROOT = _REPO_ROOT / "checkpoints" / "intermask"
_CLIP_VERSION = "ViT-L/14@336px"


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    if local.exists():
        return local
    if "/" not in name_or_path:
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path))
    except Exception:
        return local


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@MODEL_BUNDLES.register_module()
class InterMaskBundle(ModelBundle):
    """InterMask masked-token text-to-motion bundle."""

    SUPPORTED_DATASETS = {"interhuman", "interx"}

    def __init__(
        self,
        dataset_name: str = "interhuman",
        artifact_root: Optional[str] = None,
        device: Optional[str] = None,
        cond_scale: float = 2.0,
        time_steps: int = 20,
        topk_filter_thres: float = 0.9,
        temperature: float = 1.0,
        load_model: bool = True,
        **kwargs,
    ):
        super().__init__()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported InterMask dataset {dataset_name!r}")
        self.dataset_name = dataset_name
        self.artifact_root = Path(artifact_root) if artifact_root else _DEFAULT_ROOT / f"hftrainer_{dataset_name}"
        self.device_name = device
        self.cond_scale = float(cond_scale)
        self.time_steps = int(time_steps)
        self.topk_filter_thres = float(topk_filter_thres)
        self.temperature = float(temperature)
        self.vq_model = None
        self.transformer = None
        self._load_report = None
        self._mean = None
        self._std = None
        if load_model:
            self.load_model()

    @property
    def device(self) -> torch.device:
        if self.vq_model is not None:
            return next(self.vq_model.parameters()).device
        return torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    @property
    def load_report(self):
        if self._load_report is None:
            self.load_model()
        return self._load_report

    @property
    def vq_opt_path(self) -> Path:
        return self.artifact_root / "vq_default" / "opt.txt"

    @property
    def trans_opt_path(self) -> Path:
        return self.artifact_root / "trans_default" / "opt.txt"

    @property
    def vq_ckpt_path(self) -> Path:
        return self.artifact_root / "vq_default" / "model" / "best_fid.tar"

    @property
    def trans_ckpt_path(self) -> Path:
        return self.artifact_root / "trans_default" / "model" / "best_fid.tar"

    def _load_stats(self) -> None:
        if self.dataset_name == "interhuman":
            mean = self.artifact_root / "stats" / "global_mean.npy"
            std = self.artifact_root / "stats" / "global_std.npy"
        else:
            mean = self.artifact_root / "stats" / "interx_mean.npy"
            std = self.artifact_root / "stats" / "interx_std.npy"
        if mean.exists() and std.exists():
            self._mean = np.load(str(mean)).astype(np.float32)
            self._std = np.load(str(std)).astype(np.float32)

    def load_model(self):
        if self.vq_model is not None and self.transformer is not None:
            return self.vq_model, self.transformer
        for path in (self.vq_opt_path, self.trans_opt_path, self.vq_ckpt_path, self.trans_ckpt_path):
            if not path.exists():
                raise FileNotFoundError(f"InterMask artifact file missing: {path}")
        device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        from .network.models.mask_transformer.transformer import MaskTransformer
        from .network.models.vq.model import RVQVAE
        from .network.utils.get_opt import get_opt

        vq_opt = get_opt(str(self.vq_opt_path), device)
        trans_opt = get_opt(str(self.trans_opt_path), device)
        trans_opt.num_tokens = vq_opt.nb_code
        trans_opt.code_dim = vq_opt.code_dim
        dim_pose = 12 if self.dataset_name == "interhuman" else 6
        vq = RVQVAE(
            vq_opt,
            dim_pose,
            vq_opt.nb_code,
            vq_opt.code_dim,
            vq_opt.code_dim,
            vq_opt.down_t,
            vq_opt.stride_t,
            vq_opt.width,
            vq_opt.depth,
            vq_opt.dilation_growth_rate,
            vq_opt.vq_act,
            vq_opt.vq_norm,
        )
        vq_ckpt = torch.load(str(self.vq_ckpt_path), map_location="cpu")
        vq_key = "vq_model" if "vq_model" in vq_ckpt else "net"
        vq_missing, vq_unexpected = vq.load_state_dict(vq_ckpt[vq_key], strict=False)
        trans = MaskTransformer(
            code_dim=trans_opt.code_dim,
            cond_mode="text",
            latent_dim=trans_opt.latent_dim,
            ff_size=trans_opt.ff_size,
            num_layers=trans_opt.n_layers,
            num_heads=trans_opt.n_heads,
            dropout=trans_opt.dropout,
            clip_dim=768,
            cond_drop_prob=trans_opt.cond_drop_prob,
            clip_version=_CLIP_VERSION,
            opt=trans_opt,
        )
        trans_ckpt = torch.load(str(self.trans_ckpt_path), map_location=device)
        trans_key = "t2m_transformer" if "t2m_transformer" in trans_ckpt else "trans"
        trans_missing, trans_unexpected = trans.load_state_dict(trans_ckpt[trans_key], strict=False)
        if vq_unexpected:
            raise RuntimeError(f"Unexpected InterMask VQ keys: {vq_unexpected[:8]}")
        if trans_unexpected:
            raise RuntimeError(f"Unexpected InterMask transformer keys: {trans_unexpected[:8]}")
        allowed_vq_missing = all(k.startswith("decoder.conv") or k.startswith("decoder.resnets") for k in vq_missing)
        allowed_trans_missing = all(k.startswith("clip_") for k in trans_missing)
        if not allowed_vq_missing:
            raise RuntimeError(f"Unexpected InterMask VQ missing keys: {vq_missing[:8]}")
        if not allowed_trans_missing:
            raise RuntimeError(f"Unexpected InterMask transformer missing keys: {trans_missing[:8]}")
        vq.to(device).eval()
        trans.to(device).eval()
        self.vq_model = vq
        self.transformer = trans
        self._load_stats()
        self._load_report = {
            "vq_missing": sorted(vq_missing),
            "transformer_missing": sorted(trans_missing),
            "vq_checkpoint": str(self.vq_ckpt_path),
            "transformer_checkpoint": str(self.trans_ckpt_path),
        }
        return self.vq_model, self.transformer

    def _denormalize_motion(self, motion: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            return motion
        return motion * self._std + self._mean

    @torch.no_grad()
    def generate(
        self,
        texts: str | Sequence[str],
        motion_len: int = 90,
        seed: Optional[int] = None,
        return_numpy: bool = True,
        cond_scale: Optional[float] = None,
        time_steps: Optional[int] = None,
        topk_filter_thres: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        """Generate InterMask motion.

        InterHuman returns ``(B, T, 2, 262)``. InterX returns ``(B, T, 56, 12)``.
        """
        prompts = [texts] if isinstance(texts, str) else list(texts)
        if not prompts:
            raise ValueError("InterMaskBundle.generate needs at least one text prompt")
        vq, trans = self.load_model()
        _seed_everything(seed)
        device = next(trans.parameters()).device
        motion_lens = torch.full((len(prompts),), int(motion_len), dtype=torch.long, device=device)
        ids_length = motion_lens // 4
        ids = trans.generate(
            prompts,
            ids_length,
            int(time_steps or self.time_steps),
            float(cond_scale if cond_scale is not None else self.cond_scale),
            topk_filter_thres=float(topk_filter_thres if topk_filter_thres is not None else self.topk_filter_thres),
            temperature=float(temperature if temperature is not None else self.temperature),
        )
        ids1, ids2 = ids[:, : ids.shape[1] // 2], ids[:, ids.shape[1] // 2 :]
        motion1 = vq.forward_decoder(ids1.unsqueeze(-1).to(device))
        motion2 = vq.forward_decoder(ids2.unsqueeze(-1).to(device))
        motion = torch.cat([motion1, motion2], dim=-1).detach().cpu().numpy()
        if self.dataset_name == "interhuman":
            motion = motion.reshape(motion.shape[0], motion.shape[1], 2, -1)
            motion = self._denormalize_motion(motion)
        else:
            motion = motion.reshape(motion.shape[0], motion.shape[1], 56, 12)
        if return_numpy:
            return motion
        return torch.from_numpy(motion)

    def forward(self, texts: str | Sequence[str], **kwargs):
        return self.generate(texts, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not path.exists():
            path = _maybe_download_hub(pretrained_model_name_or_path, path)
        if path.is_dir():
            kwargs.setdefault("artifact_root", str(path))
            if "dataset_name" not in kwargs:
                name = path.name.lower()
                if "interx" in name:
                    kwargs["dataset_name"] = "interx"
                elif "interhuman" in name:
                    kwargs["dataset_name"] = "interhuman"
        return cls(**kwargs)
