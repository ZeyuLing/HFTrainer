"""MotionCLIP SMPL-135 text-to-motion evaluator.

This wraps the MotionCLIP evaluator checkpoint behind the same public evaluator
surface used by :class:`MotionStreamer272Evaluator`. Inputs are annotation-keyed
MotionCLIP-135 motions:

``translation + 22 joints * column-major 6D rotation``.

Use ``scripts/eval/convert_row135_npz_to_motionclip_col.py`` to convert the
canonical repository ``motion135/*.npz`` outputs into this evaluator format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch

from hftrainer.registry import EVALUATORS

from scripts.eval.eval_motionclip_table1_dirs import (
    _collect_available,
    _compute_metrics,
    _load_annotation_entries,
)
from scripts.eval.eval_with_motionclip_evaluator import encode_dataset, load_motionclip

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CKPT = _REPO / "checkpoints/motion_clip/motionclip_base_1p_aug_hq"
_DEFAULT_CLIP = _REPO / "checkpoints/clip-vit-base-patch32"
_DEFAULT_STATS = _REPO / "data/statistic/smplx55_stats_hymotion_aug.json"
_DEFAULT_ANNO = (
    _REPO
    / "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    "gt_motionclip_selected_20260622/"
    "test_hml3d_official272_gtlen_motionclip_selected_caption.json"
)


@EVALUATORS.register_module()
class MotionCLIP135Evaluator:
    """Reusable wrapper for MotionCLIP SMPL-135 retrieval/FID metrics.

    The historical script default L2-normalized embeddings before every metric.
    This class defaults to ``l2_normalize=False`` so FID/MM are computed in the
    raw MotionCLIP projection space. Set ``l2_normalize=True`` only for legacy
    comparisons with older diagnostic runs.
    """

    def __init__(
        self,
        evaluator_ckpt: Optional[str] = None,
        clip_pretrained: Optional[str] = None,
        stats_file: Optional[str] = None,
        anno_file: Optional[str] = None,
        data_dir: str = ".",
        caption_key: str = "hierarchical_caption",
        device: str = "cuda",
        forward_batch_size: int = 32,
        chunk_size: int = 32,
        n_repeats: int = 20,
        seed: int = 0,
        min_frames: int = 1,
        max_frames: int = 300,
        l2_normalize: bool = False,
    ) -> None:
        self.evaluator_ckpt = Path(evaluator_ckpt) if evaluator_ckpt else _DEFAULT_CKPT
        self.clip_pretrained = Path(clip_pretrained) if clip_pretrained else _DEFAULT_CLIP
        self.stats_file = Path(stats_file) if stats_file else _DEFAULT_STATS
        self.anno_file = Path(anno_file) if anno_file else _DEFAULT_ANNO
        self.data_dir = Path(data_dir)
        self.caption_key = caption_key
        self.device = torch.device(
            device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        )
        self.forward_batch_size = int(forward_batch_size)
        self.chunk_size = int(chunk_size)
        self.n_repeats = int(n_repeats)
        self.seed = int(seed)
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.l2_normalize = bool(l2_normalize)
        self._bundle = None
        self._entries = None

    def _ensure_loaded(self):
        if self._bundle is None:
            self._bundle = load_motionclip(
                self.evaluator_ckpt,
                self.device,
                clip_pretrained=str(self.clip_pretrained),
                stats_file=str(self.stats_file),
            )
        return self._bundle

    def _load_entries(self):
        if self._entries is None:
            self._entries = _load_annotation_entries(
                self.anno_file,
                self.data_dir,
                self.caption_key,
                self.min_frames,
                self.max_frames,
            )
        return self._entries

    def evaluate_dir(
        self,
        pred_dir: str | Path,
        real_dir: str | Path,
        *,
        method: str = "prediction",
        l2_normalize: Optional[bool] = None,
    ) -> dict:
        """Evaluate one prediction directory against one real directory."""

        bundle = self._ensure_loaded()
        use_l2 = self.l2_normalize if l2_normalize is None else bool(l2_normalize)
        names, caps, real, pred, lengths, len_mismatch = _collect_available(
            self._load_entries(),
            Path(real_dir),
            Path(pred_dir),
            self.max_frames,
        )
        if not names:
            raise RuntimeError(f"No aligned MotionCLIP-135 samples in {pred_dir}")

        text_real, motion_real = encode_dataset(
            bundle,
            caps,
            real,
            lengths,
            self.device,
            forward_batch_size=self.forward_batch_size,
            max_frames=self.max_frames,
            l2_normalize=use_l2,
        )
        if method.lower() in {"real", "gt", "gt_real"} and Path(pred_dir) == Path(real_dir):
            motion_pred = motion_real
        else:
            _, motion_pred = encode_dataset(
                bundle,
                caps,
                pred,
                lengths,
                self.device,
                forward_batch_size=self.forward_batch_size,
                max_frames=self.max_frames,
                l2_normalize=use_l2,
            )

        metrics = _compute_metrics(
            text_real,
            motion_real,
            motion_pred,
            chunk_size=self.chunk_size,
            n_repeats=self.n_repeats,
            seed=self.seed,
            l2_normalize=use_l2,
        )
        metrics.update(
            {
                "method": method,
                "pred_dir": str(pred_dir),
                "real_dir": str(real_dir),
                "anno_file": str(self.anno_file),
                "min_frames": self.min_frames,
                "max_frames": self.max_frames,
                "length_mismatch": int(len_mismatch),
                "names": names,
            }
        )
        return metrics

    def evaluate_dirs(
        self,
        pred_dirs: Mapping[str, str | Path],
        real_dir: str | Path,
        *,
        l2_normalize: Optional[bool] = None,
    ) -> dict[str, dict]:
        """Evaluate multiple named prediction directories with one loaded model."""

        return {
            method: self.evaluate_dir(
                pred_dir,
                real_dir,
                method=method,
                l2_normalize=l2_normalize,
            )
            for method, pred_dir in pred_dirs.items()
        }

