"""TMR HumanML3D evaluator wrapper.

This module wraps the official TMR implementation
(``ref_repo/TMR``) as a reusable evaluator for HumanML3D/Guo 263-dim
features. It is intentionally a bridge evaluator: the model code and weights
remain external, while the repository gets one stable entry point and one JSON
metric schema.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from hftrainer.registry import EVALUATORS

from .humanml3d_263 import read_h3d_texts
from .t2m_metrics import activation_stats, calc_frechet, diversity

_REPO = Path(__file__).resolve().parents[3]
_TMR_ROOT = _REPO / "ref_repo/TMR"
_DEFAULT_MODEL_DIRS = [
    _REPO / "checkpoints/evaluators/tmr_humanml3d_guoh3dfeats",
    _TMR_ROOT / "models/tmr_humanml3d_guoh3dfeats",
]
_GT_ROOT = _REPO / "ref_repo/CondMDI/dataset/HumanML3D"
_SPLIT = _REPO / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt"


class MissingEvaluatorAssets(FileNotFoundError):
    """Raised when a bridged evaluator has no local pretrained assets."""


def _prepend_once(path: Path) -> None:
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)


def _pick_existing_dir(paths: Sequence[Path]) -> Path:
    for p in paths:
        if (p / "config.json").exists():
            return p
    return paths[0]


def _first_full_caption(text_file: Path) -> Optional[str]:
    texts = read_h3d_texts(text_file)
    for t in texts:
        if t["f_tag"] == 0.0 and t["to_tag"] == 0.0:
            return str(t["caption"])
    return str(texts[0]["caption"]) if texts else None


@EVALUATORS.register_module()
class TMRHumanML3DEvaluator:
    """Official TMR retrieval evaluator for HumanML3D-263 predictions."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        tmr_root: Optional[str] = None,
        device: str = "cuda",
        ckpt_name: str = "last",
        batch_size: int = 128,
    ):
        self.tmr_root = Path(tmr_root) if tmr_root else _TMR_ROOT
        self.model_dir = Path(model_dir) if model_dir else _pick_existing_dir(_DEFAULT_MODEL_DIRS)
        self.device = device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        self.ckpt_name = ckpt_name
        self.batch_size = batch_size
        self._model = None
        self._text_model = None
        self._normalizer = None
        self._get_sim_matrix = None
        self._all_contrastive_metrics = None
        self._collate_x_dict = None

    @staticmethod
    def asset_help() -> Dict[str, object]:
        return {
            "evaluator": "tmr_humanml3d",
            "required_model_dir": str(_DEFAULT_MODEL_DIRS[0]),
            "fallback_model_dir": str(_DEFAULT_MODEL_DIRS[1]),
            "prepare": [
                "cd ref_repo/TMR && bash prepare/download_pretrain_models.sh",
                "mkdir -p checkpoints/evaluators",
                "cp -a ref_repo/TMR/models/tmr_humanml3d_guoh3dfeats checkpoints/evaluators/",
            ],
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not (self.model_dir / "config.json").exists():
            raise MissingEvaluatorAssets(
                f"TMR model config not found at {self.model_dir}. "
                f"Prepare assets with: {self.asset_help()['prepare']}"
            )
        if not self.tmr_root.exists():
            raise MissingEvaluatorAssets(f"TMR source tree not found: {self.tmr_root}")

        _prepend_once(self.tmr_root)
        import src.prepare  # noqa: F401
        from hydra.utils import instantiate
        from src.config import read_config
        from src.data.collate import collate_x_dict
        from src.load import load_model_from_cfg
        from src.model.metrics import all_contrastive_metrics
        from src.model.tmr import get_sim_matrix

        cfg = read_config(str(self.model_dir))
        # The official config stores stats and annotation assets as paths relative
        # to the TMR repo root. Keep the checkpoint run_dir absolute, but resolve
        # those auxiliary paths under ``ref_repo/TMR``.
        cfg.data.text_to_token_emb.preload = False
        old_cwd = os.getcwd()
        try:
            os.chdir(self.tmr_root)
            self._text_model = instantiate(cfg.data.text_to_token_emb, device=self.device)
            self._model = load_model_from_cfg(
                cfg, ckpt_name=self.ckpt_name, eval_mode=True, device=self.device
            )
            self._normalizer = instantiate(cfg.data.motion_loader.normalizer)
        finally:
            os.chdir(old_cwd)
        self._collate_x_dict = collate_x_dict
        self._get_sim_matrix = get_sim_matrix
        self._all_contrastive_metrics = all_contrastive_metrics

    @torch.no_grad()
    def encode_motion(self, motions: Sequence[np.ndarray]) -> np.ndarray:
        self._ensure_loaded()
        latents = []
        for i in range(0, len(motions), self.batch_size):
            x_dicts = []
            for motion in motions[i : i + self.batch_size]:
                x = torch.from_numpy(np.asarray(motion, dtype=np.float32)).to(torch.float)
                x = self._normalizer(x).to(self.device)
                x_dicts.append({"x": x, "length": int(len(x))})
            batch = self._collate_x_dict(x_dicts)
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            latents.append(self._model.encode(batch, sample_mean=True).detach().cpu().numpy())
        return np.concatenate(latents, axis=0)

    @torch.no_grad()
    def encode_text(self, captions: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        latents = []
        for i in range(0, len(captions), self.batch_size):
            x_dicts = self._text_model(list(captions[i : i + self.batch_size]))
            batch = self._collate_x_dict(x_dicts)
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            latents.append(self._model.encode(batch, sample_mean=True).detach().cpu().numpy())
        return np.concatenate(latents, axis=0)

    def build_samples_from_dir(
        self,
        pred_dir: str,
        gt_root: str = str(_GT_ROOT),
        texts_dir: Optional[str] = None,
        split_file: str = str(_SPLIT),
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        pred_p = Path(pred_dir)
        gt_p = Path(gt_root)
        texts_p = Path(texts_dir) if texts_dir else gt_p / "texts"
        ids = [x.strip() for x in Path(split_file).read_text().splitlines() if x.strip()]
        samples: List[Dict[str, object]] = []
        for sid in ids:
            pred_file = pred_p / f"{sid}.npy"
            gt_file = gt_p / "new_joint_vecs" / f"{sid}.npy"
            if not (pred_file.exists() and gt_file.exists()):
                continue
            caption = _first_full_caption(texts_p / f"{sid}.txt")
            if not caption:
                continue
            pred = np.load(pred_file)
            gt = np.load(gt_file)
            if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] != 263 or gt.shape[1] != 263:
                continue
            if len(pred) < 1 or len(gt) < 1:
                continue
            samples.append({"name": sid, "caption": caption, "pred": pred, "gt": gt})
            if max_samples and len(samples) >= max_samples:
                break
        return samples

    def evaluate(
        self,
        captions: Sequence[str],
        real_motions: Sequence[np.ndarray],
        pred_motions: Sequence[np.ndarray],
    ) -> Dict[str, object]:
        self._ensure_loaded()
        text_lat = self.encode_text(captions)
        real_lat = self.encode_motion(real_motions)
        pred_lat = self.encode_motion(pred_motions)
        sim_real = self._get_sim_matrix(
            torch.from_numpy(text_lat), torch.from_numpy(real_lat)
        ).cpu().numpy()
        sim_pred = self._get_sim_matrix(
            torch.from_numpy(text_lat), torch.from_numpy(pred_lat)
        ).cpu().numpy()
        real_ret = self._all_contrastive_metrics(sim_real, rounding=None)
        pred_ret = self._all_contrastive_metrics(sim_pred, rounding=None)
        mu_r, cov_r = activation_stats(real_lat)
        mu_p, cov_p = activation_stats(pred_lat)
        return {
            "status": "ok",
            "n_samples": int(len(captions)),
            "retrieval_real": real_ret,
            "retrieval_pred": pred_ret,
            "fid_latent": float(calc_frechet(mu_r, cov_r, mu_p, cov_p)),
            "diversity_real": float(diversity(real_lat)),
            "diversity_pred": float(diversity(pred_lat)),
        }

    def evaluate_dir(
        self,
        pred_dir: str,
        gt_root: str = str(_GT_ROOT),
        texts_dir: Optional[str] = None,
        split_file: str = str(_SPLIT),
        max_samples: Optional[int] = None,
    ) -> Dict[str, object]:
        samples = self.build_samples_from_dir(
            pred_dir, gt_root=gt_root, texts_dir=texts_dir,
            split_file=split_file, max_samples=max_samples,
        )
        if not samples:
            return {
                "status": "no_samples",
                "n_samples": 0,
                "pred_dir": str(pred_dir),
                "config": {"evaluator": "tmr_humanml3d"},
            }
        res = self.evaluate(
            [str(s["caption"]) for s in samples],
            [s["gt"] for s in samples],
            [s["pred"] for s in samples],
        )
        res["config"] = {
            "evaluator": "tmr_humanml3d",
            "model_dir": str(self.model_dir),
            "tmr_root": str(self.tmr_root),
            "pred_dir": str(pred_dir),
            "protocol": "full_clip_primary_caption",
        }
        return res
