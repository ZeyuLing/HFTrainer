"""MotionStreamer 272-dim text-to-motion evaluator.

Wraps the public MotionStreamer ``Evaluator_272`` (``epoch=99.ckpt``: a
DistilBERT text encoder + ACTOR motion encoder, latent 256) behind a reusable
:class:`MotionStreamer272Evaluator`. This is the evaluator backing the
HumanML3D (272) rows of the PRISM / HYMotion-M2M papers.

Predictions are scored in the native 272-dim feature space; HumanML3D-263
baselines must first be retargeted to SMPL and re-encoded to 272 (see
``scripts/eval/hml263_to_smpl_ik.py`` + ``scripts/data/convert_motion135_to_h3d272.py``).

Example
-------
>>> ev = MotionStreamer272Evaluator(device="cuda")
>>> metrics = ev.evaluate_dir("outputs/.../mdm_272")          # full pipeline
>>> metrics["fid"], metrics["r_precision_pred"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from hftrainer.registry import EVALUATORS

from .networks import ActorAgnosticEncoder, DistilbertActorAgnosticEncoder
from .t2m_metrics import aggregate_t2m_metrics

_REPO = Path(__file__).resolve().parents[3]
# Weights live under the framework's own ``checkpoints/`` tree (never ref_repo);
# CephFS is extremely slow (~1.4 MB/s cold) for the 518 MB checkpoint, so prefer a
# /dev/shm mirror when present (see ``scripts/eval/_cache_272_data.sh``).
_SHM_CKPT = Path("/dev/shm/eval272_epoch99.ckpt")
_CKPT_DIR = _REPO / "checkpoints/evaluators/motionstreamer_272"
_DEFAULT_CKPT = _SHM_CKPT if _SHM_CKPT.exists() else _CKPT_DIR / "epoch99.ckpt"
_DEFAULT_DATA_ROOT = _REPO / "data/evaluators/humanml3d_272"
_DISTILBERT = "distilbert-base-uncased"  # standard HF model (cached locally)

MAX_MOTION_LENGTH = 300
MIN_MOTION_LENGTH = 60  # 30 fps
UNIT_LENGTH = 4
NFEATS = 272


class _Stub:
    """Absorbs any pickled object whose source class is unavailable."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _tolerant_pickle_module():
    """A ``pickle``-compatible shim whose Unpickler tolerates missing modules.

    The public ``epoch=99.ckpt`` is a PyTorch-Lightning checkpoint whose
    ``hyper_parameters`` pickle references the original ``mld`` package. We only
    need the tensor ``state_dict``, so unknown classes (``mld``,
    ``pytorch_lightning``, …) are replaced with a no-op stub — keeping the loader
    fully independent of ``ref_repo``.
    """
    import pickle
    import types

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub

    shim = types.ModuleType("hftrainer_tolerant_pickle")
    shim.Unpickler = _Unpickler
    shim.load = lambda f, **kw: _Unpickler(f, **kw).load()
    shim.loads = pickle.loads
    shim.Pickler = pickle.Pickler
    shim.dump = pickle.dump
    shim.dumps = pickle.dumps
    return shim


@EVALUATORS.register_module()
class MotionStreamer272Evaluator:
    """Reusable wrapper around the MotionStreamer-272 retrieval evaluator."""

    def __init__(
        self,
        evaluator_ckpt: Optional[str] = None,
        data_root: Optional[str] = None,
        device: str = "cuda",
    ):
        self.evaluator_ckpt = Path(evaluator_ckpt) if evaluator_ckpt else _DEFAULT_CKPT
        self.data_root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
        self.device = torch.device(
            device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        )
        self._textenc = None
        self._motenc = None
        self._mean = None
        self._std = None

    # ------------------------------------------------------------------ encoders
    def _ensure_loaded(self) -> None:
        if self._textenc is not None:
            return
        textenc = DistilbertActorAgnosticEncoder(
            _DISTILBERT, num_layers=4, latent_dim=256
        )
        motenc = ActorAgnosticEncoder(
            nfeats=NFEATS, vae=True, num_layers=4, latent_dim=256, max_len=MAX_MOTION_LENGTH
        )
        sd = torch.load(
            str(self.evaluator_ckpt), map_location="cpu",
            pickle_module=_tolerant_pickle_module(), weights_only=False,
        )["state_dict"]
        textenc.load_state_dict(
            {k[len("textencoder.") :]: v for k, v in sd.items() if k.startswith("textencoder.")},
            strict=True,
        )
        motenc.load_state_dict(
            {k[len("motionencoder.") :]: v for k, v in sd.items() if k.startswith("motionencoder.")},
            strict=True,
        )
        self._textenc = textenc.eval().to(self.device)
        self._motenc = motenc.eval().to(self.device)
        self._mean = np.load(self.data_root / "mean_std" / "Mean.npy")
        self._std = np.load(self.data_root / "mean_std" / "Std.npy")

    def _standardize_pad(self, arrs: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(arrs), MAX_MOTION_LENGTH, NFEATS), dtype=np.float32)
        for i, a in enumerate(arrs):
            t = min(len(a), MAX_MOTION_LENGTH)
            out[i, :t] = (a[:t] - self._mean) / self._std
        return out

    @torch.no_grad()
    def encode_text(self, captions: Sequence[str], batch_size: int = 32) -> np.ndarray:
        self._ensure_loaded()
        embs = []
        for i in range(0, len(captions), batch_size):
            embs.append(self._textenc(list(captions[i : i + batch_size])).loc.cpu().numpy())
        return np.concatenate(embs, axis=0)

    @torch.no_grad()
    def encode_motion(
        self, motions: Sequence[np.ndarray], lengths: Sequence[int], batch_size: int = 32
    ) -> np.ndarray:
        """Encode raw (un-standardized) 272-dim motions to evaluator embeddings."""
        self._ensure_loaded()
        feats = self._standardize_pad(motions)
        lens = np.asarray(lengths, dtype=np.int64)
        embs = []
        for i in range(0, len(feats), batch_size):
            mb = torch.from_numpy(feats[i : i + batch_size]).to(self.device).float()
            lb = torch.from_numpy(lens[i : i + batch_size]).to(self.device).long()
            embs.append(self._motenc(mb, lb).loc.cpu().numpy())
        return np.concatenate(embs, axis=0)

    # --------------------------------------------------------------------- pairs
    def load_test_pairs(self, fps: int = 30):
        """Mirror ``humanml3d_272/dataset_eval_t2m.py`` (name, caption, gt, m_length)."""
        motion_dir = self.data_root / "motion_data"
        text_dir = self.data_root / "texts"
        split = (self.data_root / "split" / "test.txt").read_text().splitlines()
        pairs = []
        for name in split:
            name = name.strip()
            if not name:
                continue
            m_file = motion_dir / f"{name}.npy"
            t_file = text_dir / f"{name}.txt"
            if not (m_file.exists() and t_file.exists()):
                continue
            motion = np.load(m_file)
            if len(motion) < MIN_MOTION_LENGTH or len(motion) >= MAX_MOTION_LENGTH:
                continue
            for line in t_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("#")
                if len(parts) < 4:
                    continue
                caption = parts[0]
                f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
                t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
                if f_tag == 0.0 and t_tag == 0.0:
                    m = motion
                else:
                    m = motion[int(f_tag * fps) : int(t_tag * fps)]
                    if len(m) < MIN_MOTION_LENGTH or len(m) >= MAX_MOTION_LENGTH:
                        continue
                ml = (len(m) // UNIT_LENGTH) * UNIT_LENGTH
                if ml < MIN_MOTION_LENGTH:
                    continue
                pairs.append((name, caption, m[:ml], ml))
        return pairs

    # ------------------------------------------------------------------ evaluate
    def evaluate(
        self,
        captions: Sequence[str],
        real_motions: Sequence[np.ndarray],
        pred_motions: Sequence[np.ndarray],
        lengths: Sequence[int],
        pred_lengths: Optional[Sequence[int]] = None,
        n_repeats: int = 20,
        batch_size: int = 32,
        seed: int = 0,
    ) -> Dict[str, object]:
        """Score aligned (caption, real_272, pred_272, length) triples.

        ``lengths`` are the **ground-truth** motion lengths used to encode the real
        motions. ``pred_lengths`` (optional) are the per-sample generated lengths
        used to encode the predictions; if omitted, predictions reuse ``lengths``.
        This mirrors the released ``evaluation_transformer_motionmillion`` protocol,
        where GT is encoded at its full ``m_length`` and the prediction at its own
        generated length (so a short autoregressive sample is **not** allowed to
        truncate the GT reference).
        """
        text_emb = self.encode_text(captions, batch_size)
        real_emb = self.encode_motion(real_motions, lengths, batch_size)
        pred_emb = self.encode_motion(
            pred_motions, lengths if pred_lengths is None else pred_lengths, batch_size
        )
        return aggregate_t2m_metrics(
            text_emb, real_emb, pred_emb, n_repeats=n_repeats, chunk=batch_size, seed=seed
        )

    def evaluate_dir(
        self,
        pred_dir: str,
        n_repeats: int = 20,
        batch_size: int = 32,
        seed: int = 0,
        gt_only: bool = False,
    ) -> Dict[str, object]:
        """Score a directory of ``<name>.npy`` 272-dim predictions against GT."""
        pred_dir_p = Path(pred_dir)
        pairs = self.load_test_pairs()
        captions: List[str] = []
        real_motions: List[np.ndarray] = []
        pred_motions: List[np.ndarray] = []
        lengths: List[int] = []
        pred_lengths: List[int] = []
        skipped_no_pred = 0
        for name, caption, gt, ml in pairs:
            if gt_only:
                pred = gt
            else:
                pf = pred_dir_p / f"{name}.npy"
                if not pf.exists():
                    skipped_no_pred += 1
                    continue
                pred = np.load(pf)
                pred_ml = (len(pred) // UNIT_LENGTH) * UNIT_LENGTH
                if pred_ml < MIN_MOTION_LENGTH:
                    skipped_no_pred += 1
                    continue
                pred = pred[:pred_ml]
            captions.append(caption)
            real_motions.append(gt)
            pred_motions.append(pred)
            # GT encoded at full length; prediction at its own generated length.
            lengths.append(ml)
            pred_lengths.append(min(ml, len(pred)))

        metrics = self.evaluate(
            captions, real_motions, pred_motions, lengths,
            pred_lengths=pred_lengths,
            n_repeats=n_repeats, batch_size=batch_size, seed=seed,
        )
        metrics["skipped_no_pred"] = int(skipped_no_pred)
        metrics["config"] = {
            "evaluator": "motionstreamer_272",
            "evaluator_ckpt": str(self.evaluator_ckpt),
            "data_root": str(self.data_root),
            "pred_dir": None if gt_only else str(pred_dir_p),
            "gt_only": gt_only,
            "batch_size": batch_size,
        }
        return metrics
