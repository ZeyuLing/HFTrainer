"""InterGen ModelBundle backed by native hftrainer runtime modules."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES


_REPO_ROOT = Path(__file__).resolve().parents[4]
_NETWORK_ROOT = Path(__file__).resolve().parent / "network"
_DEFAULT_ARTIFACT = _REPO_ROOT / "checkpoints" / "intergen" / "hftrainer_interhuman"
_DEFAULT_CKPT = _DEFAULT_ARTIFACT / "intergen.ckpt"
_DEFAULT_MEAN = _DEFAULT_ARTIFACT / "global_mean.npy"
_DEFAULT_STD = _DEFAULT_ARTIFACT / "global_std.npy"
_DEFAULT_CONFIG = _NETWORK_ROOT / "configs" / "model.yaml"


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
class InterGenBundle(ModelBundle):
    """InterGen two-person text-to-motion bundle (InterHuman native-262)."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        device: Optional[str] = None,
        sampling_strategy: Optional[str] = None,
        cfg_weight: Optional[float] = None,
        load_model: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CKPT
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
        self.mean_path = Path(mean_path) if mean_path else _DEFAULT_MEAN
        self.std_path = Path(std_path) if std_path else _DEFAULT_STD
        self.device_name = device
        self.sampling_strategy = sampling_strategy
        self.cfg_weight = cfg_weight
        self._model = None
        self._load_report = None
        mean = np.load(str(self.mean_path)).astype(np.float32)
        std = np.load(str(self.std_path)).astype(np.float32)
        if mean.shape != (262,) or std.shape != (262,):
            raise ValueError(f"InterGen stats must be 262-dim, got {mean.shape} and {std.shape}")
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)
        if load_model:
            self.load_model()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def model(self):
        return self.load_model()

    @property
    def load_report(self):
        if self._load_report is None:
            self.load_model()
        return self._load_report

    def load_model(self):
        if self._model is not None:
            return self._model
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"InterGen checkpoint missing: {self.checkpoint_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"InterGen config missing: {self.config_path}")
        from .network.configs import get_config
        from .network.models import InterGen

        cfg = get_config(str(self.config_path))
        cfg.defrost()
        cfg.CHECKPOINT = str(self.checkpoint_path)
        cfg.MEAN_PATH = str(self.mean_path)
        cfg.STD_PATH = str(self.std_path)
        if self.sampling_strategy is not None:
            cfg.STRATEGY = self.sampling_strategy
        if self.cfg_weight is not None:
            cfg.CFG_WEIGHT = float(self.cfg_weight)
        cfg.freeze()
        model = InterGen(cfg)
        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu")
        state = dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        for key in list(state.keys()):
            if key.startswith("model."):
                state[key.replace("model.", "", 1)] = state.pop(key)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval()
        target_device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(target_device)
        self.to(target_device)
        self._model = model
        self._load_report = {
            "missing": sorted(missing),
            "unexpected": sorted(unexpected),
            "checkpoint": str(self.checkpoint_path),
        }
        return self._model

    @torch.no_grad()
    def generate(
        self,
        texts: str | Sequence[str],
        motion_len: int = 210,
        seed: Optional[int] = None,
        return_numpy: bool = True,
    ):
        """Generate denormalized InterHuman native-262 motion.

        Returns an array shaped ``(B, T, 2, 262)`` when ``return_numpy=True``.
        """
        prompts: List[str] = [texts] if isinstance(texts, str) else list(texts)
        if not prompts:
            raise ValueError("InterGenBundle.generate needs at least one text prompt")
        model = self.load_model()
        _seed_everything(seed)
        device = next(model.parameters()).device
        batch = {
            "motion_lens": torch.full((len(prompts),), int(motion_len), dtype=torch.long, device=device),
            "text": prompts,
        }
        out = model.forward_test(batch)["output"].reshape(len(prompts), int(motion_len), 2, -1)
        out = out * self.std.to(device) + self.mean.to(device)
        return out.detach().cpu().numpy() if return_numpy else out

    def forward(self, texts: str | Sequence[str], **kwargs):
        return self.generate(texts, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not path.exists():
            path = _maybe_download_hub(pretrained_model_name_or_path, path)
        if path.is_dir():
            meta_path = path / "intergen_config.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                kwargs.setdefault("sampling_strategy", meta.get("sampling_strategy"))
                kwargs.setdefault("cfg_weight", meta.get("cfg_weight"))
            kwargs.setdefault("checkpoint_path", str(path / "intergen.ckpt"))
            kwargs.setdefault("mean_path", str(path / "global_mean.npy"))
            kwargs.setdefault("std_path", str(path / "global_std.npy"))
        return cls(**kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        out = Path(save_directory)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_type": "intergen",
            "representation": "interhuman_native_262",
            "sampling_strategy": self.sampling_strategy,
            "cfg_weight": self.cfg_weight,
        }
        (out / "intergen_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return str(out)
